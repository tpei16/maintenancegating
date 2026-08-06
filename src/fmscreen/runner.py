"""
Experiment runner helpers — config grid, per-split execution, aggregation.

A "config" is (target_kind, pctl, layer, model). evaluate a list of configs on a
single (train, test) split, attach the matching simple-rule baseline (which
depends only on target/pctl), and optionally collect held-out predictions for
downstream decomposition / heterogeneity.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from . import engine as E

# ---- standard config grid ----
TARGETS = [("occurrence", None), ("severity_labour", 75),
           ("severity_labour", 50), ("severity_labour", 90)]
LAYERS = ["M0", "M1", "M1_nosys"]
MODELS = ["gbdt", "logreg"]


def standard_configs(targets=None, layers=None, models=None):
    targets = targets or TARGETS
    layers = layers or LAYERS
    models = models or MODELS
    cfgs = []
    for tk, pc in targets:
        for layer in layers:
            for model in models:
                cfgs.append({"target_kind": tk, "pctl": pc, "layer": layer, "model": model})
    return cfgs


def _cfg_id(c):
    return f"{c['target_kind']}|p{c['pctl']}|{c['layer']}|{c['model']}"


def run_split(train: pd.DataFrame, test: pd.DataFrame, configs: list[dict],
              regime: str, extra: dict, score_cfg_ids: set[str] | None = None,
              n_boot: int = 400, ci_cluster: str = "university") -> tuple[list[dict], pd.DataFrame | None]:
    """Run all configs on one split. Returns (metric_rows, predictions_df_or_None)."""
    score_cfg_ids = score_cfg_ids or set()
    rows, pred_frames = [], []
    # baseline cache keyed by (target_kind, pctl)
    base_cache = {}
    for c in configs:
        key = (c["target_kind"], c["pctl"])
        if key not in base_cache:
            base_cache[key] = E.evaluate_baselines(train, test, c["target_kind"],
                                                   pctl=(c["pctl"] or 75))
        bl = base_cache[key]
        want_scores = _cfg_id(c) in score_cfg_ids
        r = E.evaluate_split(train, test, c["layer"], c["model"], c["target_kind"],
                             pctl=(c["pctl"] or 75), n_boot=n_boot, ci_cluster=ci_cluster,
                             return_scores=want_scores)
        if r is None:
            continue
        row = {"regime": regime, **extra, "target_kind": c["target_kind"],
               "pctl": c["pctl"], "layer": c["layer"], "model": c["model"],
               "best_rule_name": bl["best_rule_name"],
               "best_rule_lift_top10": bl["best_rule_lift_top10"],
               "beats_best_rule": bool(r["lift_top10"] > bl["best_rule_lift_top10"]),
               "meets_2x": bool(r["lift_top10"] >= C.SUFFICIENCY_MIN_LIFT),
               "sufficient": bool(r["lift_top10"] >= C.SUFFICIENCY_MIN_LIFT and
                                  r["lift_top10"] > bl["best_rule_lift_top10"])}
        row.update({k: v for k, v in r.items() if not k.startswith("_")})
        # attach individual rule lifts
        for rname, rv in bl["rules_topk"].items():
            row[f"lift_{rname}"] = rv["lift_top10"]
        rows.append(row)
        if want_scores:
            pred_frames.append(pd.DataFrame({
                "regime": regime, **{k: extra[k] for k in extra},
                "cfg_id": _cfg_id(c), "target_kind": c["target_kind"], "pctl": c["pctl"],
                "layer": c["layer"], "model": c["model"],
                "y": r["_y"], "score": r["_scores"],
                "university": r["_test_univ"], "system": r["_test_system"],
                "building": r["_test_building"],
            }))
    preds = pd.concat(pred_frames, ignore_index=True) if pred_frames else None
    return rows, preds


def summarize_across_folds(df: pd.DataFrame, fold_col: str = "held_out_university") -> pd.DataFrame:
    """Aggregate per-config across folds: median/IQR of lift & capture + sufficiency stability."""
    out = []
    gcols = ["target_kind", "pctl", "layer", "model"]
    for keys, sub in df.groupby(gcols, dropna=False, observed=True):
        lifts = sub["lift_top10"].to_numpy()
        caps = sub["capture_top10"].to_numpy()
        praucs = sub["pr_auc"].to_numpy()
        n_folds = len(sub)
        n_meets = int((sub["lift_top10"] >= C.SUFFICIENCY_MIN_LIFT).sum())
        n_beats = int(sub["beats_best_rule"].sum())
        n_suff = int(sub["sufficient"].sum())
        # no extreme single-institution dependence: drop the best fold, still majority?
        if n_folds >= 2:
            order = np.argsort(-lifts)
            drop_best = lifts[order[1:]]
            stable_wo_best = (drop_best >= C.SUFFICIENCY_MIN_LIFT).mean() if len(drop_best) else 0.0
        else:
            stable_wo_best = float(lifts[0] >= C.SUFFICIENCY_MIN_LIFT)
        out.append({
            **dict(zip(gcols, keys)),
            "n_folds": n_folds,
            "lift_median": float(np.median(lifts)), "lift_q1": float(np.quantile(lifts, .25)),
            "lift_q3": float(np.quantile(lifts, .75)), "lift_min": float(np.min(lifts)),
            "lift_max": float(np.max(lifts)),
            "capture_median": float(np.median(caps)),
            "prauc_median": float(np.nanmedian(praucs)),
            "frac_folds_meet_2x": n_meets / n_folds,
            "frac_folds_beat_rule": n_beats / n_folds,
            "frac_folds_sufficient": n_suff / n_folds,
            "stable_excl_best_fold": float(stable_wo_best),
            # locked criterion: sufficient in the MAJORITY of folds, robust to dropping best
            "MEETS_SUFFICIENCY": bool(n_suff / n_folds > C.STABILITY_MIN_FRACTION and stable_wo_best > 0.5),
        })
    return pd.DataFrame(out)
