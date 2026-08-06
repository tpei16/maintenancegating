"""
Fold-aware feature assembly + leakage-safe labels.

Two things MUST be derived from the training fold only, never the full data:
  1. Severity thresholds  (system-specific percentile of positive next-period
     UPM burden) -> used to binarize the hard severity target.
  2. Reactive-premium features (UPM-to-PPM burden ratio by system) -> merged as
     model features.

This module exposes `assemble_Xy(train, test, layer, target, pctl)` which returns
ready X/y plus clustering keys and the train-derived artifacts, so every
validation split re-derives thresholds and premiums internally.

University identity is NEVER a feature (Section 16). System type IS a feature.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C

# --------------------------------------------------------------------------- #
# Feature-set definitions (model layers M0 / M1)
# --------------------------------------------------------------------------- #
# M0 = system type + basic record history (counts, recency, ratio, density,
#      season, weather). "Whether basic record history carries screening signal."
M0_NUMERIC = [
    "upm_cnt_w1", "upm_cnt_w2", "upm_cnt_w4",
    "ppm_cnt_w1", "ppm_cnt_w2", "ppm_cnt_w4",
    "upm_cnt_cum", "ppm_cnt_cum", "ppm_upm_ratio",
    "time_since_upm", "time_since_ppm",
    "periods_since_first", "active_share_hist", "active_cum",
    "quarter", "min_temp", "max_temp", "precip", "humidity",
]
# M1 adds reactive-burden features (static magnitudes + dynamic trends + premiums)
M1_EXTRA_NUMERIC = [
    "upm_labour_y1", "ppm_labour_y1", "upm_cost_y1",
    "upm_labour_cum", "ppm_labour_cum",
    "upm_freq_trend", "ppm_activity_trend", "react_burden_trend",
    "prem_labour_sys", "prem_cost_sys",   # train-derived premiums (merged in)
]
CAT_FEATURES = [C.COL_SYSTEM]

# M2 work-order-text features (past-only rolling counts of interpretable keyword
# groups in the cell's last-year descriptions; built by scripts/14_text_model.py).
TEXT_COLS = ["text_urgency_y1", "text_failure_y1", "text_leak_y1",
             "text_comfort_y1", "text_electrical_y1", "text_repair_y1"]

LAYERS = {
    "M0": (M0_NUMERIC, CAT_FEATURES),
    "M1": (M0_NUMERIC + M1_EXTRA_NUMERIC, CAT_FEATURES),
    # taxonomy-ablation: drop the standardized system code to quantify how much
    # the FMUCO taxonomy reduces the transfer gap (RQ2b).
    "M1_nosys": (M0_NUMERIC + M1_EXTRA_NUMERIC, []),
    "M0_nosys": (M0_NUMERIC, []),
    # optional text layer (requires the text-augmented panel)
    "M2": (M0_NUMERIC + M1_EXTRA_NUMERIC + TEXT_COLS, CAT_FEATURES),
    # weather ablation: M1 minus the four weather/season-context columns
    "M1_noweather": ([c for c in (M0_NUMERIC + M1_EXTRA_NUMERIC)
                      if c not in ("min_temp", "max_temp", "precip", "humidity")], CAT_FEATURES),
}


# --------------------------------------------------------------------------- #
# Leakage-safe labels
# --------------------------------------------------------------------------- #
def severity_thresholds(train: pd.DataFrame, target: str = "labour",
                        pctl: int = 75) -> tuple[pd.Series, float]:
    """System-specific percentile of POSITIVE next-period UPM burden (train only)."""
    col = "upm_labour_next" if target == "labour" else "upm_cost_next"
    pos = train[train[col] > 0]
    thr = pos.groupby(C.COL_SYSTEM, observed=True)[col].quantile(pctl / 100.0)
    global_thr = float(pos[col].quantile(pctl / 100.0)) if len(pos) else 0.0
    return thr, global_thr


def apply_severity(df: pd.DataFrame, thr: pd.Series, global_thr: float,
                   target: str = "labour") -> np.ndarray:
    col = "upm_labour_next" if target == "labour" else "upm_cost_next"
    t = df[C.COL_SYSTEM].map(thr).astype("float64").fillna(global_thr)
    return ((df[col].to_numpy() > t.to_numpy()) & (df[col].to_numpy() > 0)).astype(int)


def make_labels(train: pd.DataFrame, test: pd.DataFrame, target_kind: str,
                pctl: int = 75):
    """target_kind in {'occurrence','severity_labour','severity_cost'}."""
    if target_kind == "occurrence":
        return (train["occurrence_next"].to_numpy().astype(int),
                test["occurrence_next"].to_numpy().astype(int), None)
    target = "labour" if target_kind == "severity_labour" else "cost"
    thr, gthr = severity_thresholds(train, target=target, pctl=pctl)
    return (apply_severity(train, thr, gthr, target),
            apply_severity(test, thr, gthr, target),
            {"thr": thr.to_dict(), "global_thr": gthr})


# --------------------------------------------------------------------------- #
# Train-derived reactive-premium features (Section 14)
# --------------------------------------------------------------------------- #
def reactive_premiums(train: pd.DataFrame) -> pd.DataFrame:
    """Per-system UPM-to-PPM burden ratio (labour & cost), computed on train only.

    premium = mean labour per UPM work-order / mean labour per PPM work-order.
    """
    g = train.groupby(C.COL_SYSTEM, observed=True).agg(
        upm_lab=("upm_labour", "sum"), upm_n=("upm_count", "sum"),
        ppm_lab=("ppm_labour", "sum"), ppm_n=("ppm_count", "sum"),
        upm_cst=("upm_cost", "sum"), ppm_cst=("ppm_cost", "sum"),
    )
    upm_lab_per = g["upm_lab"] / g["upm_n"].replace(0, np.nan)
    ppm_lab_per = g["ppm_lab"] / g["ppm_n"].replace(0, np.nan)
    upm_cst_per = g["upm_cst"] / g["upm_n"].replace(0, np.nan)
    ppm_cst_per = g["ppm_cst"] / g["ppm_n"].replace(0, np.nan)
    prem = pd.DataFrame({
        "prem_labour_sys": upm_lab_per / ppm_lab_per,
        "prem_cost_sys": upm_cst_per / ppm_cst_per,
    })
    # fallbacks: systems with no PPM/UPM history -> global medians
    prem["prem_labour_sys"] = prem["prem_labour_sys"].replace([np.inf, -np.inf], np.nan)
    prem["prem_cost_sys"] = prem["prem_cost_sys"].replace([np.inf, -np.inf], np.nan)
    return prem


def _merge_premiums(df: pd.DataFrame, prem: pd.DataFrame,
                    glob: dict) -> pd.DataFrame:
    out = df.merge(prem, left_on=C.COL_SYSTEM, right_index=True, how="left")
    out["prem_labour_sys"] = out["prem_labour_sys"].fillna(glob["prem_labour_sys"])
    out["prem_cost_sys"] = out["prem_cost_sys"].fillna(glob["prem_cost_sys"])
    return out


# --------------------------------------------------------------------------- #
# Assemble X / y for one split
# --------------------------------------------------------------------------- #
def assemble_Xy(train: pd.DataFrame, test: pd.DataFrame, layer: str,
                target_kind: str, pctl: int = 75) -> dict:
    num_cols, cat_cols = LAYERS[layer]
    ytr, yte, thr_art = make_labels(train, test, target_kind, pctl)

    tr, te = train, test
    needs_prem = any(c in num_cols for c in ("prem_labour_sys", "prem_cost_sys"))
    if needs_prem:
        prem = reactive_premiums(train)
        glob = {"prem_labour_sys": float(np.nanmedian(prem["prem_labour_sys"])),
                "prem_cost_sys": float(np.nanmedian(prem["prem_cost_sys"]))}
        tr = _merge_premiums(train, prem, glob)
        te = _merge_premiums(test, prem, glob)

    Xtr = tr[num_cols + cat_cols].copy()
    Xte = te[num_cols + cat_cols].copy()
    # ensure SystemCode is category with shared categories (train+test union)
    if C.COL_SYSTEM in cat_cols:
        cats = pd.api.types.union_categoricals(
            [Xtr[C.COL_SYSTEM].astype("category"), Xte[C.COL_SYSTEM].astype("category")]
        ).categories
        for X in (Xtr, Xte):
            X[C.COL_SYSTEM] = pd.Categorical(X[C.COL_SYSTEM], categories=cats)

    return {
        "X_train": Xtr, "y_train": ytr, "X_test": Xte, "y_test": yte,
        "num_cols": num_cols, "cat_cols": cat_cols,
        "test_univ": test[C.COL_UNIV].to_numpy(),
        "test_building": (test[C.COL_UNIV].astype("string") + "/" +
                          test[C.COL_BUILDING].astype("string")).to_numpy(),
        "test_system": test[C.COL_SYSTEM].to_numpy(),
        "base_rate": float(np.mean(yte)),
        "train_base_rate": float(np.mean(ytr)),
        "thresholds": thr_art,
    }
