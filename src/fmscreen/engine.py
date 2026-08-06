"""
Evaluation engine — one entry point used by every validation regime.

evaluate_split(train, test, layer, model, target_kind, pctl) re-derives
fold-local thresholds/premiums, fits the model, and returns the full metric
bundle (top-k lift with clustered CIs, PR-AUC, calibration) plus the held-out
scores for downstream analysis. evaluate_baselines() scores the simple rules on
the same test set and operating point.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C
from . import features as FE
from . import models as M
from . import metrics as MET
from . import baselines as BL


def evaluate_split(train: pd.DataFrame, test: pd.DataFrame, layer: str,
                   model_name: str, target_kind: str, pctl: int = 75,
                   ci_cluster: str = "university", n_boot: int = 400,
                   compute_ci: bool = True, return_scores: bool = False,
                   ci_metrics: tuple = ("lift_top10", "capture_top10")) -> dict | None:
    art = FE.assemble_Xy(train, test, layer, target_kind, pctl)
    ytr, yte = art["y_train"], art["y_test"]
    if len(yte) == 0 or ytr.sum() == 0 or ytr.sum() == len(ytr) or yte.sum() == 0:
        return None  # degenerate fold (no positives to learn or to score)

    model = M.MODEL_FACTORY[model_name](art["num_cols"], art["cat_cols"])
    scores = M.fit_predict(model, art["X_train"], ytr, art["X_test"])

    out = MET.core_metrics(yte, scores, k=C.TOPK_BUDGET)
    out.update({"layer": layer, "model": model_name, "target": target_kind,
                "pctl": pctl, "n_train": int(len(ytr)),
                "train_base_rate": art["train_base_rate"]})
    if compute_ci:
        clusters = art["test_univ"] if ci_cluster == "university" else art["test_building"]
        for met in ci_metrics:
            ci = MET.bootstrap_ci(yte, scores, clusters, metric=met,
                                  k=C.TOPK_BUDGET, n_boot=n_boot, seed=C.RANDOM_SEED)
            out[f"{met}_ci_lo"], out[f"{met}_ci_hi"] = ci["ci_lo"], ci["ci_hi"]
    if return_scores:
        out["_scores"] = scores
        out["_y"] = yte
        out["_test_univ"] = art["test_univ"]
        out["_test_system"] = art["test_system"]
        out["_test_building"] = art["test_building"]
    return out


def evaluate_baselines(train: pd.DataFrame, test: pd.DataFrame, target_kind: str,
                       pctl: int = 75, k: float = C.TOPK_BUDGET) -> dict:
    """Score every simple rule on the test set at the top-k budget and natural points."""
    _, yte, _ = FE.make_labels(train, test, target_kind, pctl)
    yte = np.asarray(yte)
    base = float(yte.mean())
    res = {"base_rate": base, "rules_topk": {}, "rules_natural": {}}
    for name, sc in BL.rule_scores(test).items():
        res["rules_topk"][name] = {
            "lift_top10": MET.lift_at_topk(yte, sc, k),
            "prec_top10": MET.precision_at_topk(yte, sc, k),
            "recall_top10": MET.recall_at_topk(yte, sc, k),
        }
    for name, flag in BL.rule_natural_flags(test).items():
        flagged_frac = float(flag.mean())
        sel = flag == 1
        prec = float(yte[sel].mean()) if sel.any() else 0.0
        rec = float(yte[sel].sum() / yte.sum()) if yte.sum() > 0 else 0.0
        res["rules_natural"][name] = {
            "flagged_fraction": flagged_frac, "precision": prec, "recall": rec,
            "lift": prec / base if base > 0 else float("nan"),
        }
    # best simple rule by top-k lift (the bar the model must beat)
    best = max(res["rules_topk"].items(), key=lambda kv: (kv[1]["lift_top10"]
              if not np.isnan(kv[1]["lift_top10"]) else -1))
    res["best_rule_name"] = best[0]
    res["best_rule_lift_top10"] = best[1]["lift_top10"]
    return res
