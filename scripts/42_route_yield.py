#!/usr/bin/env python
"""
42_route_yield.py -- retrospective outcome rate on each route of the nested replay.

The routing itself is defined in 40_outer_s.py: for each outer campus j every gate
quantity is recomputed from the other campuses only, S^(-j) is frozen, and campus
j's units are routed with (R, S^(-j)). This script re-uses that derivation verbatim
and adds the quantity the routing tables do not carry: the share of units on each
route that went on to experience the defined next-quarter high-burden outcome.

The three rates answer "what did the gate's decision select for":
  - maintenance-decision queue (high R, S >= 0.67)
  - condition-verification queue (high R, S < 0.67)
  - low-risk units (R < 0.90), the comparator base

Outputs:
  results/metrics/route_yield.json
"""
from __future__ import annotations
import sys, json, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C

# 40_outer_s.py is not an importable module name (leading digit), so load it by path.
_spec = importlib.util.spec_from_file_location("outer_s", ROOT / "scripts" / "40_outer_s.py")
outer_s = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(outer_s)

R_CUT, S_CUT = outer_s.R_CUT, outer_s.S_CUT


def main():
    prim = outer_s.load_preds()
    trace = outer_s.prior_trace_table()
    campuses = sorted(prim["held_out_university"].astype(str).unique())

    # identical outer-fold derivation to 40_outer_s.py
    routed = []
    for j in campuses:
        dev = prim[prim.held_out_university.astype(str) != j]
        tdev = trace[trace.campus != j]
        Smap = {s: v["S"] for s, v in outer_s.gates_for(dev, tdev).items()}
        sub = prim[prim.held_out_university.astype(str) == j]
        sub = sub[sub.system.isin(Smap)].copy()
        sub["S"] = sub.system.map(Smap)
        routed.append(sub[["held_out_university", "system", "y", "R", "S"]])
    u = pd.concat(routed, ignore_index=True)

    hiR, hiS = u.R >= R_CUT, u.S >= S_CUT
    pma, ver, low = u[hiR & hiS], u[hiR & ~hiS], u[~hiR]

    # per-campus top-decile precision, for the median/IQR quoted in the text
    prec = (u[hiR].groupby("held_out_university")["y"].mean().sort_values())

    out = {
        "design": "nested outer-campus replay (see 40_outer_s.py); outcome = next-quarter "
                  "UPM labour above the system-specific p75 threshold",
        "n_routed_units": int(len(u)),
        "pma_n": int(len(pma)),
        "pma_outcome_rate": round(float(pma.y.mean()), 4),
        "pma_nna": round(float(1.0 / pma.y.mean()), 1),
        "verify_n": int(len(ver)),
        "verify_outcome_rate": round(float(ver.y.mean()), 4),
        "verify_nna": round(float(1.0 / ver.y.mean()), 1),
        # share of the high-risk list the gate clears vs diverts (the "licence" figure)
        "pma_share_of_highrisk": round(len(pma) / (len(pma) + len(ver)), 4),
        "verify_share_of_highrisk": round(len(ver) / (len(pma) + len(ver)), 4),
        "highrisk_rate_ratio": round(float(pma.y.mean() / ver.y.mean()), 2),
        "lowrisk_n": int(len(low)),
        "lowrisk_outcome_rate": round(float(low.y.mean()), 4),
        "louo_prec_top10_median": round(float(prec.median()), 4),
        "louo_prec_iqr": [round(float(prec.quantile(0.25)), 4),
                          round(float(prec.quantile(0.75)), 4)],
    }
    json.dump(out, open(C.METRICS / "route_yield.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
