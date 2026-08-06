#!/usr/bin/env python
"""
Phase 4a — Leave-One-University-Out cross-campus transfer.

Runs the full config grid (occurrence + severity p50/p75/p90; layers M0/M1/M1_nosys;
models gbdt/logreg) for each held-out campus, with per-fold simple-rule baselines,
clustered-by-BUILDING bootstrap CIs (university clustering is degenerate with a
single held-out campus), and held-out predictions saved for the occurrence-vs-
severity decomposition and heterogeneity analysis.

Aggregates across campuses (median/IQR) and applies the LOCKED sufficiency
criterion. Outputs -> results/metrics/louo_folds.csv, louo_summary.csv,
data/processed/pred_louo/*.parquet

Env: N_JOBS (campus-level parallelism, default 3), CONFIG_SET (full|primary).
"""
from __future__ import annotations
import os
N_JOBS = int(os.environ.get("N_JOBS", "3"))
os.environ.setdefault("FMSCREEN_THREADS", str(max(2, 24 // max(N_JOBS, 1))))

import sys, json, time
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, validation as V, runner as R

PRED_DIR = C.DATA_PROCESSED / "pred_louo"; PRED_DIR.mkdir(parents=True, exist_ok=True)

# configs whose held-out predictions we persist (for Phase 5)
SCORE_IDS = {
    "occurrence|pNone|M1|gbdt",
    "severity_labour|p75|M1|gbdt", "severity_labour|p50|M1|gbdt", "severity_labour|p90|M1|gbdt",
    "severity_labour|p75|M1|logreg",
}


def run_campus(u: str, config_set: str) -> list[dict]:
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    ucol = panel[C.COL_UNIV].astype("string").to_numpy()
    train = panel[ucol != u]
    test = panel[ucol == u]
    if config_set == "primary":
        cfgs = R.standard_configs(targets=[("occurrence", None), ("severity_labour", 75)],
                                  layers=["M0", "M1"], models=["gbdt", "logreg"])
    else:
        cfgs = R.standard_configs()
    t0 = time.time()
    rows, preds = R.run_split(train, test, cfgs, regime="louo",
                              extra={"held_out_university": u},
                              score_cfg_ids=SCORE_IDS, n_boot=400, ci_cluster="building")
    if preds is not None and len(preds):
        preds.to_parquet(PRED_DIR / f"{u}.parquet", index=False)
    print(f"[louo] campus {u}: {len(rows)} configs, {len(test)} test rows "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rows


def main():
    config_set = os.environ.get("CONFIG_SET", "full")
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    campuses = V.universities(panel)
    print(f"[louo] {len(campuses)} campuses, config_set={config_set}, "
          f"N_JOBS={N_JOBS}, threads/worker={os.environ['FMSCREEN_THREADS']}", flush=True)

    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_campus)(u, config_set) for u in campuses)
    rows = [r for sub in results for r in sub]
    df = pd.DataFrame(rows)
    df.to_csv(C.METRICS / "louo_folds.csv", index=False)

    summary = R.summarize_across_folds(df, fold_col="held_out_university")
    summary = summary.sort_values(["target_kind", "pctl", "layer", "model"])
    summary.to_csv(C.METRICS / "louo_summary.csv", index=False)

    # headline: primary analysis = severity_labour p75, known-system, LOUO
    prim = df[(df.target_kind == "severity_labour") & (df.pctl == 75)]
    head = {}
    for model in ("gbdt", "logreg"):
        for layer in ("M0", "M1"):
            sub = prim[(prim.model == model) & (prim.layer == layer)]
            if len(sub) == 0:
                continue
            srow = summary[(summary.target_kind == "severity_labour") & (summary.pctl == 75)
                           & (summary.model == model) & (summary.layer == layer)]
            head[f"{model}_{layer}"] = {
                "lift_median": float(sub["lift_top10"].median()),
                "lift_iqr": [float(sub["lift_top10"].quantile(.25)), float(sub["lift_top10"].quantile(.75))],
                "lift_min": float(sub["lift_top10"].min()), "lift_max": float(sub["lift_top10"].max()),
                "capture_median": float(sub["capture_top10"].median()),
                "frac_folds_meet_2x": float((sub["lift_top10"] >= 2).mean()),
                "frac_folds_beat_rule": float(sub["beats_best_rule"].mean()),
                "frac_folds_sufficient": float(sub["sufficient"].mean()),
                "MEETS_SUFFICIENCY": bool(srow["MEETS_SUFFICIENCY"].iloc[0]) if len(srow) else None,
            }
    json.dump(head, open(C.METRICS / "louo_primary_headline.json", "w"), indent=2)
    print("[louo] PRIMARY (severity p75) headline:\n", json.dumps(head, indent=2), flush=True)
    print(f"[louo] wrote {len(df)} fold-config rows -> results/metrics/louo_folds.csv", flush=True)


if __name__ == "__main__":
    main()
