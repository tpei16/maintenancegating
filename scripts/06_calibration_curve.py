#!/usr/bin/env python
"""
Phase 4c — local-history calibration curve (contribution C2).

For each held-out campus, add 0/5/10/20% of its EARLIEST local data to the
training set (other universities) and test on a FIXED later window of that campus
(one-quarter embargo between augmentation and test). Produces the calibration
curve: performance vs volume of local history.

Configs: severity_labour p75 (primary) + occurrence; models gbdt + logreg; M1.
Building-clustered bootstrap CIs at each point. Outputs ->
results/metrics/calibration_curve.csv + calibration_summary.csv
"""
from __future__ import annotations
import os
N_JOBS = int(os.environ.get("N_JOBS", "3"))
os.environ.setdefault("FMSCREEN_THREADS", str(max(2, 24 // max(N_JOBS, 1))))
import sys, time
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, validation as V, engine as E

CONFIGS = [("severity_labour", 75), ("occurrence", None)]
MODELS = ["gbdt", "logreg"]


def run_campus(u: str) -> list[dict]:
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    rows = []
    t0 = time.time()
    for target, pctl in CONFIGS:
        for model in MODELS:
            for frac, trn, tst, naug in V.local_calibration(panel, u):
                r = E.evaluate_split(trn, tst, "M1", model, target, pctl=(pctl or 75),
                                     n_boot=300, ci_cluster="building",
                                     ci_metrics=("lift_top10",))
                if r is None:
                    continue
                rows.append({"held_out_university": u, "target_kind": target, "pctl": pctl,
                             "model": model, "frac": frac, "aug_rows": naug,
                             **{k: v for k, v in r.items() if not k.startswith("_")}})
    print(f"[calib] campus {u}: {len(rows)} points ({time.time()-t0:.0f}s)", flush=True)
    return rows


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    campuses = V.universities(panel)
    print(f"[calib] {len(campuses)} campuses, N_JOBS={N_JOBS}", flush=True)
    results = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_campus)(u) for u in campuses)
    df = pd.DataFrame([r for sub in results for r in sub])
    df.to_csv(C.METRICS / "calibration_curve.csv", index=False)

    # aggregate across campuses: median lift at each fraction (per target, model)
    agg = (df.groupby(["target_kind", "pctl", "model", "frac"], dropna=False, observed=True)
             .agg(lift_median=("lift_top10", "median"),
                  lift_q1=("lift_top10", lambda s: s.quantile(.25)),
                  lift_q3=("lift_top10", lambda s: s.quantile(.75)),
                  capture_median=("capture_top10", "median"),
                  n_campuses=("held_out_university", "nunique"))
             .reset_index().sort_values(["target_kind", "pctl", "model", "frac"]))
    agg.to_csv(C.METRICS / "calibration_summary.csv", index=False)

    # crispness: lift gain from frac 0 -> 0.20 per (target, model)
    print("[calib] median lift by fraction (severity p75):", flush=True)
    sev = agg[(agg.target_kind == "severity_labour")]
    print(sev[["model", "frac", "lift_median", "lift_q1", "lift_q3", "capture_median"]].to_string(index=False), flush=True)
    for (tk, pc, model), sub in df.groupby(["target_kind", "pctl", "model"], dropna=False, observed=True):
        piv = sub.groupby("frac")["lift_top10"].median()
        if 0.0 in piv.index and 0.20 in piv.index:
            print(f"  {tk} p{pc} {model}: lift@frac0={piv[0.0]:.2f} -> frac0.20={piv[0.20]:.2f} "
                  f"(delta {piv[0.20]-piv[0.0]:+.3f})", flush=True)


if __name__ == "__main__":
    main()
