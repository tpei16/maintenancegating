#!/usr/bin/env python
"""
Absolute burden thresholds and base rates.

Absolute burden thresholds. The p75/p90 targets are percentile-based and
operationally opaque; here we report the actual labour-hour thresholds per system so
"high-burden" (p75) and "extreme high-burden" (p90) are interpretable. Thresholds are
the system-specific percentile of POSITIVE next-quarter UPM labour-hours (the same
quantity the model targets).

Base-rate reconciliation. Several base rates appear in the paper with different
denominators; we list them in one place so apparent inconsistencies are explained.

Outputs -> results/tables/burden_thresholds.csv
           results/metrics/base_rate_reconciliation.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

MAJOR = ["HVAC", "Plumbing", "Electrical", "Interior Finishes", "Equipment",
         "Fire Protection", "Roofing", "Conveying", "Stairs"]


def main():
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    desc = p.drop_duplicates(C.COL_SYSTEM).set_index(C.COL_SYSTEM)["SystemDescription"].to_dict()

    # ---- A4: absolute thresholds (positive next-quarter UPM labour) ----
    pos = p[p["upm_labour_next"] > 0]
    rows = []
    for sysc, g in pos.groupby(C.COL_SYSTEM, observed=True):
        rows.append({"system_code": sysc, "system": desc.get(sysc, sysc),
                     "n_pos": int(len(g)),
                     "p75_hours": float(g["upm_labour_next"].quantile(0.75)),
                     "p90_hours": float(g["upm_labour_next"].quantile(0.90)),
                     "median_hours": float(g["upm_labour_next"].median())})
    thr = pd.DataFrame(rows).sort_values("n_pos", ascending=False)
    thr.to_csv(C.TABLES / "burden_thresholds.csv", index=False)
    major = thr[thr["system"].isin(MAJOR)].set_index("system").reindex(MAJOR).dropna(how="all").reset_index()

    # ---- A7: base-rate reconciliation ----
    occ = float(p["occurrence_next"].mean())
    def pctl_base(q):
        thr_s = pos.groupby(C.COL_SYSTEM, observed=True)["upm_labour_next"].quantile(q)
        t = p[C.COL_SYSTEM].map(thr_s)
        return float(((p["upm_labour_next"] > t) & (p["upm_labour_next"] > 0)).mean())
    recon = {
        "occurrence_full_panel": {"value": occ,
            "definition": "P(any UPM in t+1) over all known-system cell-quarters"},
        "highburden_p75_pooled": {"value": pctl_base(0.75),
            "definition": "P(next-qtr UPM labour > system p75) pooled over the full panel"},
        "highburden_p50_pooled": {"value": pctl_base(0.50), "definition": "as above, p50 threshold"},
        "highburden_p90_pooled": {"value": pctl_base(0.90), "definition": "as above, p90 threshold (extreme high-burden)"},
        "highburden_p75_LOUO_median": {"value": 0.0808,
            "definition": "median p75 base rate across the 9 held-out campuses (campuses differ; from decomposition_summary.json)"},
        "recent_activity_p75": {"value": 0.0957,
            "definition": "p75 base rate restricted to cells active in the prior 4 quarters (removes dormant negatives; from recent_activity_summary.json)"},
        "coescalation_4q_marginal": {"value": 0.1471,
            "definition": "P(a cell is high-burden in any of t+1..t+4); 4-quarter window inflates vs the single-quarter rate (from coescalation.json)"},
    }
    out = {"base_rates": recon,
           "median_threshold_p75_hours": float(thr["p75_hours"].median()),
           "median_threshold_p90_hours": float(thr["p90_hours"].median()),
           "major_systems": major.to_dict("records")}
    json.dump(out, open(C.METRICS / "base_rate_reconciliation.json", "w"), indent=2, default=float)

    print("[A4] absolute thresholds (labour-hours), major systems:")
    print(major[["system", "p75_hours", "p90_hours", "median_hours", "n_pos"]].round(1).to_string(index=False))
    print("\n[A7] base-rate reconciliation:")
    for k, v in recon.items():
        print(f"  {k:32s} {v['value']:.4f}  -- {v['definition']}")


if __name__ == "__main__":
    main()
