#!/usr/bin/env python
"""
Optional M2 — does work-order TEXT add screening value?

Work-order descriptions are rich (median 8 words). We build interpretable,
past-only text features: per cell-quarter counts of keyword groups (urgency,
failure, leak/water, comfort, electrical, repair) summed over the previous year
(last 4 quarters, ending at t). M2 = M1 + text. We compare M2 vs M1 on the
temporal split and on LOUO (primary config). If text adds little, the honest
finding is that counts/burden/system already capture the signal.

Outputs -> data/processed/panel_quarter_text.parquet,
           results/metrics/m2_text_results.csv + m2_summary.json
"""
from __future__ import annotations
import os
N_JOBS = int(os.environ.get("N_JOBS", "3"))
os.environ.setdefault("FMSCREEN_THREADS", str(max(2, 24 // max(N_JOBS, 1))))
import sys, json, re
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, io as F, validation as V, engine as E

KEYWORDS = {
    "urgency":   r"emergenc|urgent|asap|immediat|priority|critical",
    "failure":   r"fail|broken|break|not work|inoperat|malfunc|fault|down\b|out of service",
    "leak":      r"leak|flood|water damage|burst|overflow|drip|seepage",
    "comfort":   r"no heat|no cool|too hot|too cold|temperature|hvac|thermostat|air flow",
    "electrical":r"power|outage|no power|tripped|breaker|short circuit|electrical|lighting",
    "repair":    r"repair|replace|fix|restore|rebuild",
}
TEXT_COLS = [f"text_{k}_y1" for k in KEYWORDS]


def build_text_panel() -> pd.DataFrame:
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    cols = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, C.COL_START, C.COL_WODESC]
    raw = F.load_raw(usecols=cols)
    raw = F.add_time_keys(raw, anchor=C.COL_START)
    raw[C.COL_SYSTEM] = raw[C.COL_SYSTEM].astype("string").str.strip().str.upper()
    raw = raw[raw[C.COL_BUILDING].notna() & raw[C.COL_SYSTEM].notna() & raw["period_q"].notna()]
    desc = raw[C.COL_WODESC].astype("string").str.lower().fillna("")
    hits = {f"text_{k}": desc.str.contains(rx, regex=True, na=False).astype("int32")
            for k, rx in KEYWORDS.items()}
    agg = pd.DataFrame({C.COL_UNIV: raw[C.COL_UNIV].astype("string"),
                        C.COL_BUILDING: raw[C.COL_BUILDING].astype("string"),
                        C.COL_SYSTEM: raw[C.COL_SYSTEM].astype("string"),
                        "period_q": raw["period_q"].astype("int64"), **hits})
    cellkey = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, "period_q"]
    per = agg.groupby(cellkey, observed=True).sum().reset_index()

    panel[C.COL_UNIV] = panel[C.COL_UNIV].astype("string")
    panel[C.COL_BUILDING] = panel[C.COL_BUILDING].astype("string")
    panel[C.COL_SYSTEM] = panel[C.COL_SYSTEM].astype("string")
    m = panel.merge(per, on=cellkey, how="left")
    for k in KEYWORDS:
        m[f"text_{k}"] = m[f"text_{k}"].fillna(0.0)
    # past-only rolling last-4-quarter sums (ending at t inclusive)
    g = m.sort_values(cellkey[:-1] + ["period_q"]).groupby(cellkey[:-1], observed=True)
    m = m.sort_values(cellkey[:-1] + ["period_q"])
    for k in KEYWORDS:
        m[f"text_{k}_y1"] = (m.groupby(cellkey[:-1], observed=True)[f"text_{k}"]
                             .transform(lambda s: s.rolling(4, min_periods=1).sum()))
    m = m.reset_index(drop=True)
    m.to_parquet(C.DATA_PROCESSED / "panel_quarter_text.parquet", index=False)
    return m


def run_campus(u: str) -> list[dict]:
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter_text.parquet")
    ucol = panel[C.COL_UNIV].astype("string").to_numpy()
    train, test = panel[ucol != u], panel[ucol == u]
    rows = []
    for layer in ("M1", "M2"):
        r = E.evaluate_split(train, test, layer, "gbdt", "severity_labour", pctl=75,
                             n_boot=200, ci_cluster="building")
        if r:
            rows.append({"regime": "louo", "held_out_university": u, "layer": layer,
                         **{k: v for k, v in r.items() if not k.startswith("_")}})
    return rows


def main():
    print("[m2] building text-augmented panel ...", flush=True)
    panel = build_text_panel()
    print(f"[m2] text panel {panel.shape}; text cols {TEXT_COLS}", flush=True)

    # temporal split: M1 vs M2
    tr, te = V.temporal_split(panel)
    trows = []
    for layer in ("M1", "M2"):
        r = E.evaluate_split(tr, te, layer, "gbdt", "severity_labour", pctl=75,
                             n_boot=300, ci_cluster="university")
        trows.append({"regime": "temporal", "layer": layer, **{k: v for k, v in r.items() if not k.startswith("_")}})
    print("[m2] temporal M1 vs M2:", flush=True)
    for r in trows:
        print(f"   {r['layer']}: lift@10={r['lift_top10']:.3f} cap={r['capture_top10']:.3f} PR-AUC={r['pr_auc']:.3f}", flush=True)

    # LOUO: M1 vs M2
    campuses = V.universities(panel)
    res = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(run_campus)(u) for u in campuses)
    lrows = [r for sub in res for r in sub]
    allrows = trows + lrows
    pd.DataFrame(allrows).to_csv(C.METRICS / "m2_text_results.csv", index=False)

    ldf = pd.DataFrame(lrows)
    summary = {"temporal": {r["layer"]: {"lift": r["lift_top10"], "pr_auc": r["pr_auc"]} for r in trows}}
    for layer in ("M1", "M2"):
        s = ldf[ldf.layer == layer]
        summary[f"louo_{layer}"] = {"lift_median": float(s["lift_top10"].median()),
                                    "capture_median": float(s["capture_top10"].median()),
                                    "prauc_median": float(s["pr_auc"].median())}
    summary["m2_minus_m1_louo_lift_median"] = float(summary["louo_M2"]["lift_median"] - summary["louo_M1"]["lift_median"])
    json.dump(summary, open(C.METRICS / "m2_summary.json", "w"), indent=2)
    print("[m2] summary:\n", json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
