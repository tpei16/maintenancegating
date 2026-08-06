#!/usr/bin/env python
"""
Temporal-by-campus transfer: the strictest deployment test.

LOUO answers "trained on other campuses, does it work on a new one?"; the pre-COVID
temporal split answers "trained on the past, does it work on the future?". This test
combines both: for each held-out campus, train on the OTHER campuses' data through 2016
and test the held-out campus in 2017-2019 -- held-out in BOTH space and time, matching a
genuine forward-deployment scenario. We report per-campus top-10% lift, capture, whether
it clears the two-fold bar, and whether boosted M1 beats the best simple rule.

Outputs -> results/metrics/temporal_louo.json
           results/tables/temporal_louo.csv
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
from fmscreen import config as C, engine as E, validation as V

TRAIN_END = 2016
TEST_YEARS = (2017, 2018, 2019)
MIN_TEST_POS = 30


def run_campus(u: str, panel_path: str):
    panel = pd.read_parquet(panel_path)
    ucol = panel[C.COL_UNIV].astype("string").to_numpy()
    yr = panel["year"].to_numpy()
    train = panel[(ucol != u) & (yr <= TRAIN_END)]
    test = panel[(ucol == u) & np.isin(yr, TEST_YEARS)]
    r = E.evaluate_split(train, test, "M1", "gbdt", "severity_labour", pctl=75, compute_ci=False)
    if r is None or r["n_pos"] < MIN_TEST_POS:
        return {"held_out_university": u, "evaluable": False,
                "n_test": int(len(test)), "n_pos": int(0 if r is None else r["n_pos"])}
    bl = E.evaluate_baselines(train, test, "severity_labour", pctl=75)
    return {"held_out_university": u, "evaluable": True,
            "n_test": int(r["n"]), "n_pos": int(r["n_pos"]), "base_rate": r["base_rate"],
            "lift_top10": r["lift_top10"], "capture_top10": r["capture_top10"],
            "pr_auc": r.get("pr_auc", float("nan")),
            "best_rule_lift": bl["best_rule_lift_top10"],
            "meets_2x": bool(r["lift_top10"] >= 2.0),
            "beats_rule": bool(r["lift_top10"] > bl["best_rule_lift_top10"])}


def main():
    panel_path = str(C.DATA_PROCESSED / "panel_quarter.parquet")
    panel = pd.read_parquet(panel_path)
    campuses = V.universities(panel)
    parts = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(run_campus)(u, panel_path) for u in campuses)
    df = pd.DataFrame(parts)
    df.to_csv(C.TABLES / "temporal_louo.csv", index=False)
    ev = df[df["evaluable"]]
    out = {
        "design": "train other campuses through 2016; test held-out campus 2017-2019",
        "n_campuses_total": int(len(df)),
        "n_campuses_evaluable": int(len(ev)),
        "lift_median": float(ev["lift_top10"].median()) if len(ev) else float("nan"),
        "lift_iqr": [float(ev["lift_top10"].quantile(.25)), float(ev["lift_top10"].quantile(.75))] if len(ev) else None,
        "capture_median": float(ev["capture_top10"].median()) if len(ev) else float("nan"),
        "n_meet_2x": int(ev["meets_2x"].sum()) if len(ev) else 0,
        "n_beat_rule": int(ev["beats_rule"].sum()) if len(ev) else 0,
        "per_campus": df.to_dict("records"),
    }
    json.dump(out, open(C.METRICS / "temporal_louo.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "per_campus"}, indent=2))
    print("\nper-campus:\n", df.to_string(index=False))


if __name__ == "__main__":
    main()
