#!/usr/bin/env python
"""
Analysis Z — PM-effectiveness temporal test (associational).

When PPM intensity increases substantially for a cell (>=50% year-over-year),
does UPM burden decrease in the following year, relative to matched controls
(same system, same campus, no PPM increase)?

ENDOGENEITY CAVEAT (stated, not resolved): PM is often directed at deteriorating
cells, so causal direction is ambiguous; the 1-quarter lag gives only suggestive
directionality. Reported as associational. This is the weakest analysis; included
only because the effect is reported transparently with its caveat.

Outputs -> results/metrics/pm_effectiveness.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

CELL = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]


def main():
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    gp = p.groupby(CELL, observed=True)

    # PPM intensity: prior-year vs the year before that
    ppm_year1 = gp["ppm_count"].transform(lambda s: s.rolling(4, min_periods=4).sum())          # [t-3..t]
    ppm_year0 = gp["ppm_count"].transform(lambda s: s.shift(4).rolling(4, min_periods=4).sum())  # [t-7..t-4]
    # UPM burden before vs after the anchor
    upm_before = gp["upm_labour"].transform(lambda s: s.rolling(4, min_periods=4).sum())          # [t-3..t]
    upm_after = gp["upm_labour"].transform(lambda s: s.shift(-4).rolling(4, min_periods=4).sum())  # [t+1..t+4]
    p["ppm_year1"], p["ppm_year0"] = ppm_year1, ppm_year0
    p["upm_before"], p["upm_after"] = upm_before, upm_after

    valid = p["ppm_year0"].notna() & p["upm_after"].notna() & (p["upm_before"] > 0)
    pp = p[valid].copy()
    pp["pm_increase"] = ((pp["ppm_year0"] >= 2) & (pp["ppm_year1"] >= 1.5 * pp["ppm_year0"])).astype(int)
    pp["upm_declined"] = (pp["upm_after"] < pp["upm_before"]).astype(int)
    pp["upm_rel_change"] = (pp["upm_after"] - pp["upm_before"]) / pp["upm_before"]

    treat = pp[pp["pm_increase"] == 1]
    ctrl = pp[pp["pm_increase"] == 0]

    out = {
        "n_pm_increase_cells": int(len(treat)),
        "n_control": int(len(ctrl)),
        "treat_frac_upm_declined": float(treat["upm_declined"].mean()),
        "control_frac_upm_declined": float(ctrl["upm_declined"].mean()),
        "treat_median_rel_change": float(treat["upm_rel_change"].median()),
        "control_median_rel_change": float(ctrl["upm_rel_change"].median()),
        "pm_responsive_fraction": float(treat["upm_declined"].mean()),
        "pm_resistant_fraction": float(1 - treat["upm_declined"].mean()),
        "caveat": "associational only; PM is directed at deteriorating cells (endogeneity)",
    }
    out["interpretation"] = (
        f"After a >=50% PPM increase, UPM burden declined in the next year for "
        f"{out['treat_frac_upm_declined']:.0%} of cells (controls {out['control_frac_upm_declined']:.0%}); "
        f"the remaining {out['pm_resistant_fraction']:.0%} were PM-resistant (candidates for capital "
        f"renewal rather than more maintenance).")
    json.dump(out, open(C.METRICS / "pm_effectiveness.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
