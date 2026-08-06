"""
Practical rule-based baselines.

Simple rules a facility team could already run without analytics. Mandatory in
every comparison and foregrounded so the paper does not read as blindly
promoting machine learning. Rankable rules are scored for top-k evaluation;
fixed-set rules also report their natural operating point (flagged fraction).
All inputs are past-only panel columns (computed through period t).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def rule_scores(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Continuous ranking scores for each rule (higher = more likely flagged)."""
    return {
        # >= 3 UPM events in the previous year (rankable by the count itself)
        "rule_upm_count_prevyear": panel["upm_cnt_w4"].to_numpy(dtype=float),
        # top quintile by previous-year UPM cost
        "rule_upm_cost_prevyear": panel["upm_cost_y1"].to_numpy(dtype=float),
        # top quintile by previous-year UPM labour
        "rule_upm_labour_prevyear": panel["upm_labour_y1"].to_numpy(dtype=float),
        # rising UPM frequency AND declining PPM activity (combined trend score)
        "rule_trend_up_ppm_down": (panel["upm_freq_trend"].to_numpy(dtype=float)
                                   - panel["ppm_activity_trend"].to_numpy(dtype=float)),
    }


def rule_natural_flags(panel: pd.DataFrame) -> dict[str, np.ndarray]:
    """Fixed-set rules at their NATURAL operating point (binary 0/1)."""
    return {
        "rule_upm_count_prevyear>=3": (panel["upm_cnt_w4"].to_numpy() >= 3).astype(int),
        "rule_trend_up_ppm_down_flag": ((panel["upm_freq_trend"].to_numpy() > 0) &
                                        (panel["ppm_activity_trend"].to_numpy() < 0)).astype(int),
        # quintile rules at their natural 20% operating point
        "rule_upm_cost_top_quintile": _top_quantile_flag(panel["upm_cost_y1"].to_numpy(dtype=float), 0.20),
        "rule_upm_labour_top_quintile": _top_quantile_flag(panel["upm_labour_y1"].to_numpy(dtype=float), 0.20),
    }


def _top_quantile_flag(x: np.ndarray, q: float) -> np.ndarray:
    n = len(x)
    m = max(1, int(np.ceil(q * n)))
    order = np.argsort(-x, kind="stable")
    flag = np.zeros(n, dtype=int)
    flag[order[:m]] = 1
    return flag
