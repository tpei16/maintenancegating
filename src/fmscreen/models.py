"""
Model layer.

Established supervised models, used intentionally: logistic regression as the
interpretable baseline and a gradient-boosted tree (HistGradientBoosting) for
performance. Class imbalance handled by class weighting; thresholds are tuned
downstream at the top-k operating point. The headline is information value, not
algorithmic novelty.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

# per-worker thread budget (set by parallel drivers to avoid oversubscription)
_THREADS = int(os.environ.get("FMSCREEN_THREADS", "0")) or -1
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import xgboost as xgb

from . import config as C

# time-since sentinel (999) is clipped for the linear model so it isn't an outlier
_CLIP_COLS = ["time_since_upm", "time_since_ppm"]


def _clip_sentinels(X):
    X = X.copy()
    for c in _CLIP_COLS:
        if c in X.columns:
            X[c] = np.minimum(X[c].astype(float), 40.0)  # cap at 10 years of quarters
    return X


def make_logreg(num_cols, cat_cols, seed: int = C.RANDOM_SEED) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("clip", FunctionTransformer(_clip_sentinels, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_cols),
        ],
        remainder="drop",
    )
    clf = LogisticRegression(max_iter=2000, class_weight="balanced",
                             C=1.0, solver="lbfgs", n_jobs=_THREADS)
    return Pipeline([("pre", pre), ("clf", clf)])


class BalancedXGB(xgb.XGBClassifier):
    """XGBoost classifier that auto-sets scale_pos_weight = n_neg/n_pos at fit
    time (class balancing) and transparently falls back from GPU to CPU."""

    def fit(self, X, y, **kw):
        pos = float(np.sum(y))
        neg = float(len(y) - pos)
        self.set_params(scale_pos_weight=(neg / max(pos, 1.0)))
        try:
            return super().fit(X, y, **kw)
        except Exception as e:  # GPU unavailable/contended -> CPU
            if "cuda" in str(e).lower() or "device" in str(e).lower() or "gpu" in str(e).lower():
                self.set_params(device="cpu")
                return super().fit(X, y, **kw)
            raise


def make_gbdt(num_cols, cat_cols, seed: int = C.RANDOM_SEED) -> BalancedXGB:
    """Gradient-boosted trees on GPU (XGBoost/CUDA), native categorical support."""
    return BalancedXGB(
        device="cuda",
        tree_method="hist",
        enable_categorical=True,
        n_estimators=500,
        learning_rate=0.06,
        max_depth=6,
        min_child_weight=5.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        gamma=0.0,
        eval_metric="logloss",
        n_jobs=_THREADS,
        random_state=seed,
        verbosity=0,
    )


def make_gbdt_cpu(num_cols, cat_cols, seed: int = C.RANDOM_SEED) -> HistGradientBoostingClassifier:
    """CPU cross-check: HistGradientBoosting with native categorical support."""
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.08, max_iter=400, max_leaf_nodes=63,
        min_samples_leaf=50, l2_regularization=1.0, max_bins=255,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        class_weight="balanced", categorical_features="from_dtype", random_state=seed,
    )


def fit_predict(model, X_train: pd.DataFrame, y_train, X_test: pd.DataFrame) -> np.ndarray:
    """Fit and return positive-class probabilities on X_test."""
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


MODEL_FACTORY = {
    "logreg": make_logreg,
    "gbdt": make_gbdt,          # GPU XGBoost (primary)
    "gbdt_cpu": make_gbdt_cpu,  # CPU HistGBT (robustness cross-check)
}
