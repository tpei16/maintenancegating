"""
Evaluation metrics.

Screening performance is reported as LIFT over base rate at a fixed top-k
inspection budget, never as a bare precision. At budget k, precision-lift and
recall are linked: lift = recall / k. Headline metrics carry clustered
bootstrap confidence intervals (clustered by university or building) so a
conclusion never rests on a single institution.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss)


def _topk_mask(scores: np.ndarray, k: float) -> np.ndarray:
    """Boolean mask of the top-k fraction by score (ties broken by stable order)."""
    n = len(scores)
    m = max(1, int(np.ceil(k * n)))
    order = np.argsort(-scores, kind="stable")
    mask = np.zeros(n, dtype=bool)
    mask[order[:m]] = True
    return mask


def precision_at_topk(y: np.ndarray, scores: np.ndarray, k: float) -> float:
    mask = _topk_mask(scores, k)
    return float(y[mask].mean()) if mask.any() else 0.0


def recall_at_topk(y: np.ndarray, scores: np.ndarray, k: float) -> float:
    mask = _topk_mask(scores, k)
    pos = y.sum()
    return float(y[mask].sum() / pos) if pos > 0 else 0.0


def lift_at_topk(y: np.ndarray, scores: np.ndarray, k: float) -> float:
    base = y.mean()
    if base <= 0:
        return float("nan")
    return precision_at_topk(y, scores, k) / base


def core_metrics(y: np.ndarray, scores: np.ndarray, k: float = 0.10) -> dict:
    """Full metric bundle for one (y, scores) at budget k and k/2."""
    y = np.asarray(y).astype(float)
    scores = np.asarray(scores).astype(float)
    base = float(y.mean())
    out = {
        "n": int(len(y)),
        "n_pos": int(y.sum()),
        "base_rate": base,
        "prec_top5": precision_at_topk(y, scores, 0.05),
        "prec_top10": precision_at_topk(y, scores, 0.10),
        "recall_top10": recall_at_topk(y, scores, 0.10),
        "lift_top5": lift_at_topk(y, scores, 0.05),
        "lift_top10": lift_at_topk(y, scores, 0.10),
        "capture_top10": recall_at_topk(y, scores, 0.10),  # == recall@10%
    }
    if 0 < y.sum() < len(y):
        out["pr_auc"] = float(average_precision_score(y, scores))
        out["roc_auc"] = float(roc_auc_score(y, scores))
        if scores.min() >= 0 and scores.max() <= 1:
            out["brier"] = float(brier_score_loss(y, scores))
    else:
        out["pr_auc"] = float("nan")
        out["roc_auc"] = float("nan")
    return out


def _metric_func(metric: str, k: float):
    return {
        "lift_top10": lambda yy, ss: lift_at_topk(yy, ss, k),
        "lift_top5": lambda yy, ss: lift_at_topk(yy, ss, 0.05),
        "prec_top10": lambda yy, ss: precision_at_topk(yy, ss, k),
        "recall_top10": lambda yy, ss: recall_at_topk(yy, ss, k),
        "capture_top10": lambda yy, ss: recall_at_topk(yy, ss, k),
        "pr_auc": lambda yy, ss: (average_precision_score(yy, ss)
                                  if 0 < yy.sum() < len(yy) else np.nan),
    }[metric]


def bootstrap_ci(y: np.ndarray, scores: np.ndarray, clusters: np.ndarray,
                 metric: str = "lift_top10", k: float = 0.10,
                 n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> dict:
    """Cluster bootstrap CI for a top-k metric (resample whole clusters)."""
    y = np.asarray(y).astype(float)
    scores = np.asarray(scores).astype(float)
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    # a cluster bootstrap needs >= 2 clusters; with one cluster every resample is
    # identical -> a spurious zero-width CI. Return NaN instead (audit fix).
    if len(uniq) < 2:
        pt = _metric_func(metric, k)(y, scores)
        return {"point": float(pt), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "metric": metric, "n_boot": 0, "note": "single-cluster: CI undefined"}
    idx_by_cluster = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)

    func = _metric_func(metric, k)
    point = func(y, scores)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        chosen = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_cluster[c] for c in chosen])
        boots[b] = func(y[rows], scores[rows])
    boots = boots[~np.isnan(boots)]
    lo = float(np.quantile(boots, (1 - ci) / 2)) if len(boots) else float("nan")
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2)) if len(boots) else float("nan")
    return {"point": float(point), "ci_lo": lo, "ci_hi": hi,
            "metric": metric, "n_boot": int(len(boots))}


def calibration_points(y: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> dict:
    """Reliability-curve points (equal-width score bins) + Brier score."""
    y = np.asarray(y).astype(float)
    scores = np.asarray(scores).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(scores, edges) - 1, 0, n_bins - 1)
    mean_pred, frac_pos, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        if m.any():
            mean_pred.append(float(scores[m].mean()))
            frac_pos.append(float(y[m].mean()))
            counts.append(int(m.sum()))
    brier = float(brier_score_loss(y, scores)) if (scores.min() >= 0 and scores.max() <= 1) else float("nan")
    return {"mean_pred": mean_pred, "frac_pos": frac_pos, "counts": counts, "brier": brier}
