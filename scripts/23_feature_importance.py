#!/usr/bin/env python
"""
T1.3 — Feature importance for the boosted M1 screening model.

Reviewers of an ML-in-engineering paper expect to see WHICH features drive the
gradient-boosted model, not only that the M1 layer beats M0 as a group. We compute
exact TreeSHAP attributions (XGBoost `pred_contribs`) for the primary boosted M1
model on the temporal test window (train <= 2018, test > 2018), plus the model's
gain-based importance as a cross-check.

The narrative predictions to confirm (Improvement Plan T1.3):
  * prior-year / recent UPM burden and counts should dominate (this is WHY simple
    rules are competitive);
  * train-derived reactive-premium features should appear in the top tier (this is
    WHY M1 improves on M0);
  * weather/season features should rank low (consistent with the weather ablation
    that moved lift by only -0.05).

Outputs -> results/metrics/feature_importance.json
           results/tables/feature_importance.csv
"""
from __future__ import annotations
import os
os.environ.setdefault("FMSCREEN_THREADS", "8")
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, features as FE, models as M, validation as V

PRETTY = {
    "upm_labour_y1": "Prior-year UPM labor", "upm_cost_y1": "Prior-year UPM cost",
    "ppm_labour_y1": "Prior-year PPM labor", "upm_labour_cum": "Cumulative UPM labor",
    "ppm_labour_cum": "Cumulative PPM labor",
    "upm_cnt_w1": "UPM count, last quarter", "upm_cnt_w2": "UPM count, last 2 quarters",
    "upm_cnt_w4": "UPM count, last 4 quarters", "ppm_cnt_w1": "PPM count, last quarter",
    "ppm_cnt_w2": "PPM count, last 2 quarters", "ppm_cnt_w4": "PPM count, last 4 quarters",
    "upm_cnt_cum": "Cumulative UPM count", "ppm_cnt_cum": "Cumulative PPM count",
    "ppm_upm_ratio": "PPM:UPM ratio", "time_since_upm": "Quarters since last UPM",
    "time_since_ppm": "Quarters since last PPM", "periods_since_first": "Cell age (quarters)",
    "active_share_hist": "Active-quarter share (data density)", "active_cum": "Cumulative active quarters",
    "quarter": "Calendar quarter (season)", "min_temp": "Min temperature", "max_temp": "Max temperature",
    "precip": "Precipitation", "humidity": "Humidity",
    "upm_freq_trend": "UPM-frequency trend", "ppm_activity_trend": "PPM-activity trend",
    "react_burden_trend": "Reactive-burden trend",
    "prem_labour_sys": "Reactive labor premium (system)", "prem_cost_sys": "Reactive cost premium (system)",
    "SystemCode": "System type (UNIFORMAT II)",
}
WEATHER_SEASON = {"min_temp", "max_temp", "precip", "humidity", "quarter"}
PREMIUM = {"prem_labour_sys", "prem_cost_sys"}


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    train, test = V.temporal_split(panel)              # train<=2018, test>2018
    art = FE.assemble_Xy(train, test, "M1", "severity_labour", pctl=75)
    Xtr, ytr, Xte, yte = art["X_train"], art["y_train"], art["X_test"], art["y_test"]
    feats = list(Xtr.columns)
    print(f"[fit] n_train={len(ytr)} pos={int(ytr.sum())} | n_test={len(yte)} "
          f"pos={int(yte.sum())} | {len(feats)} features", flush=True)

    model = M.make_gbdt(art["num_cols"], art["cat_cols"])
    model.fit(Xtr, ytr)
    booster = model.get_booster()

    # ---- exact TreeSHAP via pred_contribs ----
    try:
        dte = xgb.DMatrix(Xte, enable_categorical=True)
        contribs = booster.predict(dte, pred_contribs=True)   # (n, n_feat+1), last=bias
    except Exception as e:
        print(f"[shap] GPU contribs failed ({e}); retry on CPU", flush=True)
        booster.set_param({"device": "cpu"})
        dte = xgb.DMatrix(Xte, enable_categorical=True)
        contribs = booster.predict(dte, pred_contribs=True)
    shap_vals = contribs[:, :-1]
    mean_abs = np.abs(shap_vals).mean(axis=0)
    # map booster feature order -> our columns (DMatrix preserves df column order)
    fmap = booster.feature_names or feats
    shap_by_feat = {fmap[i]: float(mean_abs[i]) for i in range(len(fmap))}

    # ---- gain-based importance (cross-check) ----
    gain = booster.get_score(importance_type="gain")
    total_gain = sum(gain.values()) or 1.0
    gain_by_feat = {f: float(gain.get(f, 0.0)) / total_gain for f in feats}

    rows = []
    s_tot = sum(shap_by_feat.values()) or 1.0
    for f in feats:
        rows.append({"feature": f, "label": PRETTY.get(f, f),
                     "mean_abs_shap": shap_by_feat.get(f, 0.0),
                     "shap_share": shap_by_feat.get(f, 0.0) / s_tot,
                     "gain_share": gain_by_feat.get(f, 0.0),
                     "group": ("weather_season" if f in WEATHER_SEASON else
                               "premium" if f in PREMIUM else
                               "taxonomy" if f == "SystemCode" else "record_history")})
    imp = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp.to_csv(C.TABLES / "feature_importance.csv", index=False)

    top15 = imp.head(15)
    grp_share = imp.groupby("group")["shap_share"].sum().to_dict()
    summary = {
        "n_test": int(len(yte)), "n_test_pos": int(yte.sum()),
        "top15": top15[["rank", "feature", "label", "mean_abs_shap", "shap_share", "gain_share", "group"]].to_dict("records"),
        "group_shap_share": {k: float(v) for k, v in grp_share.items()},
        "weather_season_total_shap_share": float(imp[imp.group == "weather_season"]["shap_share"].sum()),
        "premium_total_shap_share": float(imp[imp.group == "premium"]["shap_share"].sum()),
        "premium_best_rank": int(imp[imp.group == "premium"]["rank"].min()),
        "top_feature": imp.iloc[0]["feature"],
    }
    json.dump(summary, open(C.METRICS / "feature_importance.json", "w"), indent=2, default=float)
    print("[top-15 by mean|SHAP|]\n" +
          top15[["rank", "label", "mean_abs_shap", "shap_share", "gain_share", "group"]].to_string(index=False),
          flush=True)
    print(f"\n[groups] SHAP share: {json.dumps({k: round(v,3) for k,v in grp_share.items()})}", flush=True)
    print(f"[check] weather+season share={summary['weather_season_total_shap_share']:.3f}; "
          f"premium best rank={summary['premium_best_rank']}", flush=True)


if __name__ == "__main__":
    main()
