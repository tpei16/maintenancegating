#!/usr/bin/env python
"""
Sensitivity — recent-activity cell-inclusion rule (locked design decision).

The main analysis uses the known-system rule (all known building-system cells,
including dormant ones). This sensitivity repeats the PRIMARY LOUO analysis on the
recent-activity subset (cells active in the previous 4 quarters), which removes
the easy dormant-cell negatives and raises the base rate — a harder, honest test
that isolates screening among currently-active cells.

Outputs -> results/metrics/recent_activity_louo.csv + recent_activity_summary.json
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
    panel = panel[panel["active_prev4"] == 1]          # recent-activity rule
    ucol = panel[C.COL_UNIV].astype("string").to_numpy()
    train, test = panel[ucol != u], panel[ucol == u]
    cfgs = R.standard_configs(targets=[("occurrence", None), ("severity_labour", 75)],
                              layers=["M0", "M1"], models=["gbdt", "logreg"])
    rows, _ = R.run_split(train, test, cfgs, regime="louo_recent",
                          extra={"held_out_university": u}, n_boot=300, ci_cluster="building")
    return rows


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    campuses = V.universities(panel)
    results = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(run_campus)(u) for u in campuses)
    df = pd.DataFrame([r for sub in results for r in sub])
    df.to_csv(C.METRICS / "recent_activity_louo.csv", index=False)

    summary = {}
    for (tk, pc), sub in df.groupby(["target_kind", "pctl"], dropna=False):
        for model in ("gbdt", "logreg"):
            for layer in ("M0", "M1"):
                s = sub[(sub.model == model) & (sub.layer == layer)]
                if len(s) == 0:
                    continue
                summary[f"{tk}_p{pc}_{model}_{layer}"] = {
                    "base_rate_median": float(s["base_rate"].median()),
                    "lift_median": float(s["lift_top10"].median()),
                    "capture_median": float(s["capture_top10"].median()),
                    "frac_meet_2x": float((s["lift_top10"] >= 2).mean()),
                    "frac_beat_rule": float(s["beats_best_rule"].mean()),
                }
    json.dump(summary, open(C.METRICS / "recent_activity_summary.json", "w"), indent=2)
    prim = {k: v for k, v in summary.items() if k.startswith("severity_labour_p75")}
    print("[recent] severity p75 (recent-activity rule) vs known-system:\n",
          json.dumps(prim, indent=2), flush=True)


if __name__ == "__main__":
    main()
