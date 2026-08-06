#!/usr/bin/env python
"""
Secondary analysis — high-COST reactive burden target (Gate 2 = both; Plan 13).

Cost is inflation-deflated (CPI-U, base 2021) and far less complete than labour,
so it is reported as SECONDARY. We repeat the LOUO severity analysis on the cost
target (next-period UPM deflated cost > system p75), for M1 gbdt/logreg, and
report per-campus results + how labour-vs-cost rankings agree.

Outputs -> results/metrics/cost_louo_folds.csv + cost_summary.json
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
from fmscreen import config as C, validation as V, runner as R


def run_campus(u: str) -> list[dict]:
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    ucol = panel[C.COL_UNIV].astype("string").to_numpy()
    train, test = panel[ucol != u], panel[ucol == u]
    # cost completeness of the held-out campus (share of UPM cost>0 among active)
    cost_pos = float((test["upm_cost"] > 0).mean())
    cfgs = R.standard_configs(targets=[("severity_cost", 75)], layers=["M1"], models=["gbdt", "logreg"])
    rows, _ = R.run_split(train, test, cfgs, regime="louo_cost",
                          extra={"held_out_university": u, "cost_pos_share": cost_pos},
                          n_boot=300, ci_cluster="building")
    return rows


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    campuses = V.universities(panel)
    results = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(run_campus)(u) for u in campuses)
    rows = [r for sub in results for r in sub]
    df = pd.DataFrame(rows)
    df.to_csv(C.METRICS / "cost_louo_folds.csv", index=False)

    summary = {}
    for model in ("gbdt", "logreg"):
        sub = df[(df.model == model)]
        if len(sub) == 0:
            continue
        summary[model] = {
            "n_campuses_evaluable": int(len(sub)),
            "lift_median": float(sub["lift_top10"].median()),
            "lift_iqr": [float(sub["lift_top10"].quantile(.25)), float(sub["lift_top10"].quantile(.75))],
            "capture_median": float(sub["capture_top10"].median()),
            "frac_folds_meet_2x": float((sub["lift_top10"] >= 2).mean()),
            "frac_folds_beat_rule": float(sub["beats_best_rule"].mean()),
            "median_cost_pos_share": float(sub["cost_pos_share"].median()),
        }
    json.dump(summary, open(C.METRICS / "cost_summary.json", "w"), indent=2)
    print("[cost] LOUO severity-cost summary:\n", json.dumps(summary, indent=2), flush=True)
    if len(df):
        print(df[["held_out_university", "model", "cost_pos_share", "base_rate",
                  "lift_top10", "capture_top10", "best_rule_lift_top10", "sufficient"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
