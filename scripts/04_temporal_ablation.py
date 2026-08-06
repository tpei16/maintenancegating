#!/usr/bin/env python
"""
Phase 3 — within-sample TEMPORAL validation + model-layer ablation.

Train <= 2018, test 2019-2021 (multiple universities in test, so university-
clustered CIs are valid here). Full grid: occurrence + severity p50/p75/p90;
layers M0 / M1 / M1_nosys; models gbdt / logreg. Plus simple-rule baselines.

Answers: RQ1b (does record-only screening clear the bar within-sample?),
RQ1c (do reactive-burden features in M1 beat M0?), and the taxonomy contribution
(M1 vs M1_nosys). Outputs -> results/metrics/temporal_results.csv + ablation.json
"""
from __future__ import annotations
import os
os.environ.setdefault("FMSCREEN_THREADS", "24")
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, validation as V, runner as R


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    train, test = V.temporal_split(panel)
    print(f"[temporal] train(<= {C.TEMPORAL_TRAIN_END_YEAR}) n={len(train)}, "
          f"test n={len(test)}, test universities={test[C.COL_UNIV].nunique()}", flush=True)

    cfgs = R.standard_configs()  # full grid
    rows, _ = R.run_split(train, test, cfgs, regime="temporal",
                          extra={"split": "train<=2018_test>=2019"},
                          n_boot=400, ci_cluster="university")
    df = pd.DataFrame(rows)
    df.to_csv(C.METRICS / "temporal_results.csv", index=False)

    # ablation summaries on the primary severity p75 target
    def get(model, layer, target="severity_labour", pctl=75):
        sub = df[(df.model == model) & (df.layer == layer) & (df.target_kind == target) & (df.pctl == pctl)]
        return sub.iloc[0] if len(sub) else None

    abl = {}
    for model in ("gbdt", "logreg"):
        m0, m1, m1ns = get(model, "M0"), get(model, "M1"), get(model, "M1_nosys")
        abl[model] = {
            "M0_lift": float(m0["lift_top10"]) if m0 is not None else None,
            "M1_lift": float(m1["lift_top10"]) if m1 is not None else None,
            "M1_nosys_lift": float(m1ns["lift_top10"]) if m1ns is not None else None,
            "M1_minus_M0 (reactive-burden gain)": (float(m1["lift_top10"] - m0["lift_top10"])
                                                   if (m0 is not None and m1 is not None) else None),
            "M1_minus_M1nosys (taxonomy gain)": (float(m1["lift_top10"] - m1ns["lift_top10"])
                                                 if (m1 is not None and m1ns is not None) else None),
            "best_rule_lift": float(m1["best_rule_lift_top10"]) if m1 is not None else None,
            "M1_beats_rule": bool(m1["beats_best_rule"]) if m1 is not None else None,
        }
    json.dump(abl, open(C.METRICS / "temporal_ablation.json", "w"), indent=2)
    print("[temporal] ablation (severity p75):\n", json.dumps(abl, indent=2), flush=True)

    # quick view
    view = df[(df.target_kind == "severity_labour") & (df.pctl == 75)][
        ["model", "layer", "base_rate", "lift_top10", "lift_top10_ci_lo", "lift_top10_ci_hi",
         "capture_top10", "pr_auc", "best_rule_lift_top10", "sufficient"]]
    print(view.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
