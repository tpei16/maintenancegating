"""
Validation regimes.

  * temporal_split        : train earlier years, test later years (within-sample).
  * louo_folds            : leave-one-university-out cross-institutional transfer.
  * local_calibration     : add 0/5/10/20% of a held-out campus's EARLIEST data to
                            training; test on a FIXED later window of that campus
                            (the local-history calibration curve — primary novelty).

Every split returns raw (train_df, test_df); the fold-aware thresholds and
premiums are re-derived inside features.assemble_Xy for each split.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config as C


def temporal_split(panel: pd.DataFrame, train_end_year: int = C.TEMPORAL_TRAIN_END_YEAR):
    train = panel[panel["year"] <= train_end_year]
    test = panel[panel["year"] > train_end_year]
    return train, test


def universities(panel: pd.DataFrame) -> list[str]:
    return sorted(panel[C.COL_UNIV].astype("string").unique().tolist(), key=lambda s: int(s))


def louo_folds(panel: pd.DataFrame, min_test_pos: int = 30):
    """Yield (held_out_university, train_df, test_df) for each campus."""
    u_col = panel[C.COL_UNIV].astype("string")
    for u in universities(panel):
        test = panel[u_col.to_numpy() == u]
        train = panel[u_col.to_numpy() != u]
        yield u, train, test


def local_calibration(panel: pd.DataFrame, target_univ: str,
                      fractions=C.CALIB_FRACTIONS, test_boundary_frac: float = 0.20):
    """Local-history calibration generator for one held-out campus.

    The test window is FIXED across all fractions: the held-out campus's rows
    AFTER the `test_boundary_frac` chronological cut. Augmentation adds the
    earliest `frac` of the campus's rows (period_q <= q_cut(frac)), always
    disjoint from and earlier than the fixed test window.

    Yields (frac, train_df, test_df, n_aug_rows).
    """
    u_col = panel[C.COL_UNIV].astype("string").to_numpy()
    others = panel[u_col != target_univ]
    tgt = panel[u_col == target_univ].sort_values("period_q")
    if len(tgt) == 0:
        return
    periods = tgt["period_q"].to_numpy()
    # chronological boundary periods by row-count quantile
    q_test = np.quantile(periods, test_boundary_frac, method="lower")
    test = tgt[tgt["period_q"] > q_test]
    if len(test) == 0:
        # fall back: use the last 40% of cells by period if boundary too tight
        q_test = np.quantile(periods, 0.6, method="lower")
        test = tgt[tgt["period_q"] > q_test]

    for frac in fractions:
        if frac <= 0:
            aug = tgt.iloc[0:0]
        else:
            q_aug = np.quantile(periods, frac, method="lower")
            # ONE-QUARTER EMBARGO: an augmentation anchor at t predicts t+1, so cap
            # q_aug at q_test-1 to guarantee every augmentation label period (t+1)
            # is <= q_test and therefore strictly before any test anchor (> q_test).
            q_aug = min(q_aug, q_test - 1)
            aug = tgt[tgt["period_q"] <= q_aug]
        train = pd.concat([others, aug], axis=0) if len(aug) else others
        yield frac, train, test, int(len(aug))
