#!/usr/bin/env python
"""
55_route_yield_corrected.py -- retrospective outcome rate on each route of the
CORRECTED nested replay.

42_route_yield.py computes the same three rates on the single-nested
construction of 40_outer_s.py, which is the construction the paper reports as
leaky.  Its route counts (pma 48910, verify 2571) were superseded by
45_certification_corrected.py (pma 49801, verify 1872), but its outcome rates
were not recomputed, so the manuscript quoted counts from one run and rates from
another.  This script closes that gap.

Nothing is refitted.  The corrected per-campus certificates are read from
cert_corrected.json, the risk percentile is the decision-time one (R_quarter),
and every scored unit is routed by exactly the rule in 45's route(): a stratum
with no held-out gate evidence fails data sufficiency by default and routes as
uncertified.  The only quantity added is mean(y) on each route.

Outputs:
  results/metrics/route_yield_corrected.json
"""
from __future__ import annotations
import sys, json, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C

_spec = importlib.util.spec_from_file_location(
    "cert_corrected", ROOT / "scripts" / "45_certification_corrected.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

R_CUT, S_CUT = cc.R_CUT, cc.S_CUT


def main() -> None:
    single = cc.load_single()                     # supplies y, R_quarter, R_pooled
    cert = json.load(open(C.METRICS / "cert_corrected.json"))

    out = {"design": "corrected double-nested replay (45_certification_corrected.py, "
                     "arm 'corrected'); outcome = next-quarter UPM labour above the "
                     "system-specific p75 threshold"}

    for arm, rcol in (("corrected", "R_quarter"), ("single_nested", "R_pooled")):
        outer_S = cert[arm]["outer_S"]
        campuses = sorted(outer_S)
        frames = []
        for j in campuses:
            sub = single[single.university.astype(str) == j].copy()
            sub["S"] = sub.system.map(outer_S[j])
            sub["gated"] = sub["S"].notna()
            sub["S"] = sub["S"].fillna(0.0)       # no held-out evidence -> uncertified
            frames.append(sub[["university", "system", "y", rcol, "S", "gated"]]
                          .rename(columns={rcol: "R"}))
        u = pd.concat(frames, ignore_index=True)

        hiR, hiS = u.R >= R_CUT, u.S >= S_CUT
        pma, ver, low = u[hiR & hiS], u[hiR & ~hiS], u[~hiR]
        prec = u[hiR].groupby("university")["y"].mean().sort_values()

        # the same three rates restricted to the 16 gated strata, so the effect of
        # inheriting the ungated micro-strata is visible rather than assumed
        g = u[u.gated]
        ghiR, ghiS = g.R >= R_CUT, g.S >= S_CUT
        gpma, gver = g[ghiR & ghiS], g[ghiR & ~ghiS]

        out[arm] = {
            "risk_column": rcol,
            "n_routed_units": int(len(u)),
            "pma_n": int(len(pma)),
            "pma_outcome_rate": round(float(pma.y.mean()), 4),
            "verify_n": int(len(ver)),
            "verify_outcome_rate": round(float(ver.y.mean()), 4),
            "lowrisk_n": int(len(low)),
            "lowrisk_outcome_rate": round(float(low.y.mean()), 4),
            "pma_share_of_highrisk": round(len(pma) / (len(pma) + len(ver)), 4),
            "verify_share_of_highrisk": round(len(ver) / (len(pma) + len(ver)), 4),
            "highrisk_rate_ratio": round(float(pma.y.mean() / ver.y.mean()), 2),
            "louo_prec_top10_median": round(float(prec.median()), 4),
            "louo_prec_iqr": [round(float(prec.quantile(0.25)), 4),
                              round(float(prec.quantile(0.75)), 4)],
            "gated_strata_only": {
                "pma_n": int(len(gpma)),
                "pma_outcome_rate": round(float(gpma.y.mean()), 4),
                "verify_n": int(len(gver)),
                "verify_outcome_rate": round(float(gver.y.mean()), 4),
                "highrisk_rate_ratio": round(float(gpma.y.mean() / gver.y.mean()), 2),
            },
        }

    json.dump(out, open(C.METRICS / "route_yield_corrected.json", "w"), indent=2)
    print(json.dumps(out, indent=2))

    c = out["corrected"]
    assert c["pma_n"] == 49801, f"pma {c['pma_n']} != Table 8's 49801"
    assert c["verify_n"] == 1872, f"verify {c['verify_n']} != Table 8's 1872"
    print("\n[ok] route counts reproduce Table 8 exactly")


if __name__ == "__main__":
    main()
