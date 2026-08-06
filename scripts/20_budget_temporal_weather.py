#!/usr/bin/env python
"""
Inspection budget, temporal restructuring, and weather ablation.

(1) Inspection-budget curve: lift & capture at top 5/10/15/20% (severity p75,
    GBDT M1) from the saved LOUO predictions.
(2) Restructured PRIMARY temporal split: train <= 2016, test 2017-2019 (pre-COVID).
(3) COVID sensitivity: same train, test 2017-2019 vs 2020-2021 separately.
(4) Weather ablation: M1 vs M1_noweather on the pre-COVID split.

Outputs -> results/metrics/budget_curve.csv, temporal_precovid.csv,
           covid_sensitivity.json, weather_ablation.json
"""
from __future__ import annotations
import os
os.environ.setdefault("FMSCREEN_THREADS", "24")
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, engine as E, metrics as MET


def budget_curve():
    files = sorted((C.DATA_PROCESSED / "pred_louo").glob("*.parquet"))
    preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    sub = preds[preds.cfg_id == "severity_labour|p75|M1|gbdt"]
    rows = []
    for u, g in sub.groupby("held_out_university", observed=True):
        y, s = g["y"].to_numpy(), g["score"].to_numpy()
        for k in (0.05, 0.10, 0.15, 0.20):
            rows.append({"held_out_university": u, "budget": k,
                         "lift": MET.lift_at_topk(y, s, k),
                         "capture": MET.recall_at_topk(y, s, k),
                         "precision": MET.precision_at_topk(y, s, k)})
    df = pd.DataFrame(rows)
    df.to_csv(C.METRICS / "budget_curve.csv", index=False)
    agg = df.groupby("budget").agg(lift_median=("lift", "median"),
                                   capture_median=("capture", "median")).reset_index()
    print("[budget] median across campuses:\n", agg.to_string(index=False), flush=True)
    return agg


def temporal_precovid():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    train = panel[panel["year"] <= 2016]
    test_pre = panel[(panel["year"] >= 2017) & (panel["year"] <= 2019)]
    test_cov = panel[(panel["year"] >= 2020) & (panel["year"] <= 2021)]
    print(f"[temporal-precovid] train n={len(train)}, test17-19 n={len(test_pre)}, "
          f"test20-21 n={len(test_cov)}", flush=True)

    rows = []
    for model in ("gbdt", "logreg"):
        for layer in ("M0", "M1"):
            r = E.evaluate_split(train, test_pre, layer, model, "severity_labour", 75,
                                 n_boot=400, ci_cluster="university")
            rows.append({"split": "precovid_2017_2019", "model": model, "layer": layer,
                         **{k: v for k, v in r.items() if not k.startswith("_")}})
    df = pd.DataFrame(rows)
    df.to_csv(C.METRICS / "temporal_precovid.csv", index=False)

    # COVID sensitivity: GBDT M1 on 2017-19 vs 2020-21
    cov = {}
    for tag, te in (("test_2017_2019", test_pre), ("test_2020_2021", test_cov)):
        r = E.evaluate_split(train, te, "M1", "gbdt", "severity_labour", 75,
                             n_boot=300, ci_cluster="university")
        bl = E.evaluate_baselines(train, te, "severity_labour", 75)
        cov[tag] = {"base_rate": r["base_rate"], "lift_top10": r["lift_top10"],
                    "capture_top10": r["capture_top10"], "pr_auc": r["pr_auc"],
                    "best_rule_lift": bl["best_rule_lift_top10"]}
    json.dump(cov, open(C.METRICS / "covid_sensitivity.json", "w"), indent=2)
    print("[covid] sensitivity:\n", json.dumps(cov, indent=2), flush=True)

    # weather ablation on the pre-COVID split
    wabl = {}
    for layer in ("M1", "M1_noweather"):
        r = E.evaluate_split(train, test_pre, layer, "gbdt", "severity_labour", 75,
                             compute_ci=False)
        wabl[layer] = {"lift_top10": r["lift_top10"], "capture_top10": r["capture_top10"],
                       "pr_auc": r["pr_auc"]}
    wabl["weather_contribution_lift"] = wabl["M1"]["lift_top10"] - wabl["M1_noweather"]["lift_top10"]
    json.dump(wabl, open(C.METRICS / "weather_ablation.json", "w"), indent=2)
    print("[weather] ablation:\n", json.dumps(wabl, indent=2), flush=True)
    return df


def main():
    budget_curve()
    temporal_precovid()
    print("[20] done", flush=True)


if __name__ == "__main__":
    main()
