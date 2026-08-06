#!/usr/bin/env python
"""
Phase 5 — occurrence-vs-severity decomposition + heterogeneity.

Consumes the saved LOUO held-out predictions (data/processed/pred_louo/*.parquet).

(1) Occurrence-vs-severity decomposition (RQ3a): the easy occurrence target vs the
    hard severity target (p50/p75/p90), per campus and aggregated. If occurrence is
    predictable above base rate while severity has a lower ceiling, records carry
    signal about WHETHER UPM occurs with a lower ceiling for HOW SEVERE.
(2) System-level heterogeneity (RQ3b): per-system screening performance under the
    primary config (severity p75, gbdt, M1) — which systems screen well/poorly.

Outputs -> results/metrics/decomposition.csv, heterogeneity_by_system.csv +
results/metrics/decomposition_summary.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import metrics as MET


def load_preds() -> pd.DataFrame:
    files = sorted((C.DATA_PROCESSED / "pred_louo").glob("*.parquet"))
    if not files:
        raise SystemExit("No LOUO predictions found — run 05_louo_transfer.py first.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def metrics_for(y, s):
    m = MET.core_metrics(y.to_numpy(), s.to_numpy(), k=C.TOPK_BUDGET)
    return m


def main():
    preds = load_preds()
    print(f"[decomp] predictions {preds.shape}, cfg_ids={preds['cfg_id'].nunique()}", flush=True)

    # ---- (1) occurrence-vs-severity decomposition (gbdt M1) ----
    decomp_cfgs = {
        "occurrence": "occurrence|pNone|M1|gbdt",
        "severity_p50": "severity_labour|p50|M1|gbdt",
        "severity_p75": "severity_labour|p75|M1|gbdt",
        "severity_p90": "severity_labour|p90|M1|gbdt",
    }
    rows = []
    for label, cid in decomp_cfgs.items():
        sub = preds[preds.cfg_id == cid]
        if len(sub) == 0:
            continue
        for u, g in sub.groupby("held_out_university", observed=True):
            m = metrics_for(g["y"], g["score"])
            rows.append({"target": label, "cfg_id": cid, "held_out_university": u,
                         "base_rate": m["base_rate"], "lift_top10": m["lift_top10"],
                         "capture_top10": m["capture_top10"], "pr_auc": m["pr_auc"],
                         "roc_auc": m["roc_auc"], "n": m["n"], "n_pos": m["n_pos"]})
    dec = pd.DataFrame(rows)
    dec.to_csv(C.METRICS / "decomposition.csv", index=False)

    summary = {}
    for label in decomp_cfgs:
        sub = dec[dec.target == label]
        if len(sub) == 0:
            continue
        summary[label] = {
            "base_rate_median": float(sub["base_rate"].median()),
            "lift_median": float(sub["lift_top10"].median()),
            "lift_iqr": [float(sub["lift_top10"].quantile(.25)), float(sub["lift_top10"].quantile(.75))],
            "capture_median": float(sub["capture_top10"].median()),
            "prauc_median": float(sub["pr_auc"].median()),
            "rocauc_median": float(sub["roc_auc"].median()),
        }
    json.dump(summary, open(C.METRICS / "decomposition_summary.json", "w"), indent=2)
    print("[decomp] occurrence-vs-severity (median across campuses):\n",
          json.dumps(summary, indent=2), flush=True)

    # ---- (2) system-level heterogeneity (primary: severity p75 gbdt M1) ----
    prim = preds[preds.cfg_id == "severity_labour|p75|M1|gbdt"]
    het_rows = []
    for syscode, g in prim.groupby("system", observed=True):
        if g["y"].sum() < 20 or len(g) < 200:
            continue
        m = metrics_for(g["y"], g["score"])
        # per-campus stability of this system
        per_campus = []
        for u, gg in g.groupby("held_out_university", observed=True):
            if gg["y"].sum() >= 10 and len(gg) >= 100:
                per_campus.append(MET.lift_at_topk(gg["y"].to_numpy(), gg["score"].to_numpy(), C.TOPK_BUDGET))
        het_rows.append({"system": syscode, "n_cells": int(len(g)), "n_pos": int(g["y"].sum()),
                         "base_rate": m["base_rate"], "lift_top10": m["lift_top10"],
                         "capture_top10": m["capture_top10"], "pr_auc": m["pr_auc"],
                         "n_campuses_evaluable": len(per_campus),
                         "lift_median_across_campuses": float(np.median(per_campus)) if per_campus else np.nan,
                         "frac_campuses_meet_2x": float(np.mean([l >= 2 for l in per_campus])) if per_campus else np.nan})
    het = pd.DataFrame(het_rows).sort_values("lift_top10", ascending=False)
    het.to_csv(C.METRICS / "heterogeneity_by_system.csv", index=False)
    print("[decomp] system heterogeneity (severity p75):\n",
          het[["system", "n_cells", "base_rate", "lift_top10", "capture_top10",
               "frac_campuses_meet_2x"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
