#!/usr/bin/env python
"""
Building-level feature ablation.

The C1 claim is that reactive risk is coupled at the BUILDING level. That motivates
the operational action "inspect the whole building", but it does not by itself show
that building-level FEATURES improve the screening model. This script tests it directly.

We add a layer M1b = M1 + past-only, leakage-safe building-level features (aggregates
over the OTHER systems in the same building, ending at the anchor quarter t):
  * bld_lab_y1_excl   : prior-year UPM labour of the rest of the building (excl focal cell)
  * bld_lab_cum_excl  : cumulative UPM labour of the rest of the building (excl focal cell)
  * bld_cnt_w4_excl   : trailing-4q UPM count of the rest of the building (excl focal cell)
  * bld_n_active_sys  : number of systems active in the building that quarter
  * bld_ppm_upm_ratio : building-level cumulative PPM:UPM count ratio
  * bld_active_share  : building mean of the per-cell active-quarter share

All are derived from columns that are already past-only, so no future information is
introduced. We report LOUO top-10% lift / capture for M0, M1, M1b and the paired
per-campus delta (M1b - M1) with a campus-clustered bootstrap CI.

Interpretation (stated in advance):
  * delta small/zero  -> building coupling is an INSPECTION-UNIT insight, not extra
    predictive signal (a cell's own history already encodes its building's risk level).
  * delta meaningful  -> building features ALSO add tested predictive information.

Outputs -> results/metrics/building_ablation.json
           results/tables/building_ablation_louo.csv
"""
from __future__ import annotations
import os
N_JOBS = int(os.environ.get("N_JOBS", "3"))
os.environ.setdefault("FMSCREEN_THREADS", str(max(2, 24 // max(N_JOBS, 1))))
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, features as FE, engine as E, validation as V, metrics as MET

BLD_FEATURES = ["bld_lab_y1_excl", "bld_lab_cum_excl", "bld_cnt_w4_excl",
                "bld_n_active_sys", "bld_ppm_upm_ratio", "bld_active_share"]


def add_building_features(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    g = p.groupby([C.COL_UNIV, C.COL_BUILDING, "period_q"], observed=True)
    # sums over the building-quarter (all systems), then subtract the focal cell
    tot_lab_y1 = g["upm_labour_y1"].transform("sum")
    tot_lab_cum = g["upm_labour_cum"].transform("sum")
    tot_cnt_w4 = g["upm_cnt_w4"].transform("sum")
    p["bld_lab_y1_excl"] = (tot_lab_y1 - p["upm_labour_y1"]).clip(lower=0)
    p["bld_lab_cum_excl"] = (tot_lab_cum - p["upm_labour_cum"]).clip(lower=0)
    p["bld_cnt_w4_excl"] = (tot_cnt_w4 - p["upm_cnt_w4"]).clip(lower=0)
    p["bld_n_active_sys"] = g[C.COL_SYSTEM].transform("size").astype(float)
    tot_ppm = g["ppm_cnt_cum"].transform("sum")
    tot_upm = g["upm_cnt_cum"].transform("sum")
    p["bld_ppm_upm_ratio"] = (tot_ppm / tot_upm.replace(0, np.nan)).fillna(0.0)
    p["bld_active_share"] = g["active_share_hist"].transform("mean")
    return p


def run_campus(u: str, panel_path: str):
    # register M1b inside the worker (loky workers do not see the parent's edit)
    FE.LAYERS.setdefault("M1b", (FE.LAYERS["M1"][0] + BLD_FEATURES, FE.LAYERS["M1"][1]))
    panel = pd.read_parquet(panel_path)
    ucol = panel[C.COL_UNIV].astype("string").to_numpy()
    train, test = panel[ucol != u], panel[ucol == u]
    out = {"held_out_university": u}
    for layer in ("M0", "M1", "M1b"):
        r = E.evaluate_split(train, test, layer, "gbdt", "severity_labour", pctl=75,
                             compute_ci=False)
        if r is None:
            return None
        out[f"{layer}_lift"] = r["lift_top10"]
        out[f"{layer}_capture"] = r["capture_top10"]
        out[f"{layer}_prauc"] = r.get("pr_auc", float("nan"))
    bl = E.evaluate_baselines(train, test, "severity_labour", pctl=75)
    out["best_rule_lift"] = bl["best_rule_lift_top10"]
    out["base_rate"] = bl["base_rate"]
    return out


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    panel = add_building_features(panel)
    # register M1b = M1 + building features (premiums still merged via needs_prem)
    m1_num, m1_cat = FE.LAYERS["M1"]
    FE.LAYERS["M1b"] = (m1_num + BLD_FEATURES, m1_cat)
    aug_path = C.DATA_PROCESSED / "panel_quarter_bld.parquet"
    panel.to_parquet(aug_path)

    campuses = V.universities(panel)
    parts = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_campus)(u, str(aug_path)) for u in campuses)
    rows = [r for r in parts if r]
    df = pd.DataFrame(rows)
    df.to_csv(C.TABLES / "building_ablation_louo.csv", index=False)

    def med(col):
        return float(np.median(df[col]))

    # paired per-campus deltas + campus-clustered bootstrap CI
    def boot_ci(delta):
        rng = np.random.default_rng(42)
        b = np.array([np.median(rng.choice(delta, len(delta), replace=True)) for _ in range(2000)])
        return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]

    d_m1b_m1 = (df["M1b_lift"] - df["M1_lift"]).to_numpy()
    d_m1b_rule = (df["M1b_lift"] - df["best_rule_lift"]).to_numpy()
    d_m1_rule = (df["M1_lift"] - df["best_rule_lift"]).to_numpy()
    # the cell-level reactive-burden increment, on the same paired footing as M1b - M1,
    # so the two are directly comparable in the paper.
    d_m1_m0 = (df["M1_lift"] - df["M0_lift"]).to_numpy()

    out = {
        "n_campuses": int(len(df)),
        "median_lift": {"M0": med("M0_lift"), "M1": med("M1_lift"), "M1b": med("M1b_lift")},
        "median_capture": {"M0": med("M0_capture"), "M1": med("M1_capture"), "M1b": med("M1b_capture")},
        "delta_M1_minus_M0": {"median": float(np.median(d_m1_m0)), "ci95": boot_ci(d_m1_m0),
                              "n_campuses_M1_gt_M0": int((d_m1_m0 > 0).sum())},
        "delta_M1b_minus_M1": {"median": float(np.median(d_m1b_m1)), "ci95": boot_ci(d_m1b_m1),
                               "n_campuses_M1b_gt_M1": int((d_m1b_m1 > 0).sum())},
        "delta_M1b_minus_rule": {"median": float(np.median(d_m1b_rule)), "ci95": boot_ci(d_m1b_rule),
                                 "n_campuses_beat_rule": int((d_m1b_rule > 0).sum())},
        "delta_M1_minus_rule": {"median": float(np.median(d_m1_rule)), "ci95": boot_ci(d_m1_rule),
                                "n_campuses_beat_rule": int((d_m1_rule > 0).sum())},
        "M1b_frac_meet_2x": float((df["M1b_lift"] >= 2).mean()),
        "building_features": BLD_FEATURES,
    }
    json.dump(out, open(C.METRICS / "building_ablation.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))
    print("\nper-campus:\n", df[["held_out_university", "M0_lift", "M1_lift", "M1b_lift",
                                 "best_rule_lift"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
