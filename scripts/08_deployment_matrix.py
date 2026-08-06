#!/usr/bin/env python
"""
Phase 6 — deployment-readiness matrix (output under contribution C3).

Crosses BURDEN (high/low reactive labour share per system) with SCREENING
(works/fails, by whether the primary severity-p75 LOUO lift clears the 2x bar)
to produce a practical action matrix. Built only because heterogeneity is
empirically supported (system lift ranges widely).

Outputs -> results/tables/deployment_matrix.csv + results/metrics/deployment_matrix.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

ACTION = {
    ("high", "works"): "Use record-based top-k inspection",
    ("high", "fails"): "Improve data / add inspection or condition data / consider targeted sensing",
    ("low", "works"): "Low-risk monitoring: simple rules or periodic review",
    ("low", "fails"): "Low priority: avoid over-investing in analytics",
}
INTERP = {
    ("high", "works"): "Records are useful for this stratum",
    ("high", "fails"): "Records alone are insufficient",
    ("low", "works"): "Low-risk monitoring",
    ("low", "fails"): "Low priority",
}


def main():
    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    het = pd.read_csv(C.METRICS / "heterogeneity_by_system.csv")

    # burden class: high if system's share of total UPM labour is above the median share
    burden = burden.rename(columns={C.COL_SYSTEM: "system", C.COL_SYSTEM_DESC: "system_desc"})
    med_share = burden["share_of_total_upm_labour"].median()
    burden["burden_class"] = np.where(burden["share_of_total_upm_labour"] >= med_share, "high", "low")

    m = het.merge(burden[["system", "system_desc", "share_of_total_upm_labour", "burden_class"]],
                  on="system", how="left")
    # screening class: "works" requires ROBUST stability — clears 2x in >=80% of
    # evaluable campuses with a campus-median lift >=2 (not merely a pooled lift).
    lift_basis = m["lift_median_across_campuses"].fillna(m["lift_top10"])
    m["screening_class"] = np.where(
        (lift_basis >= C.SUFFICIENCY_MIN_LIFT) &
        (m["frac_campuses_meet_2x"].fillna(0) >= 0.8), "works", "fails")
    m["cell"] = list(zip(m["burden_class"], m["screening_class"]))
    m["interpretation"] = m["cell"].map(INTERP)
    m["action"] = m["cell"].map(ACTION)
    m = m.drop(columns=["cell"]).sort_values(["burden_class", "lift_top10"], ascending=[True, False])
    m.to_csv(C.TABLES / "deployment_matrix.csv", index=False)

    # 2x2 counts
    counts = (m.groupby(["burden_class", "screening_class"], observed=True)["system"]
                .apply(lambda s: list(s)).to_dict())
    grid = {f"{b}_burden__{s}_screening": {"n": len(counts.get((b, s), [])),
                                           "systems": counts.get((b, s), []),
                                           "action": ACTION[(b, s)]}
            for b in ("high", "low") for s in ("works", "fails")}
    json.dump(grid, open(C.METRICS / "deployment_matrix.json", "w"), indent=2)
    print("[deploy] deployment-readiness matrix:\n", json.dumps(grid, indent=2), flush=True)
    print(m[["system", "system_desc", "burden_class", "lift_top10",
             "frac_campuses_meet_2x", "screening_class", "action"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
