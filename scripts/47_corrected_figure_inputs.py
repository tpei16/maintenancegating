#!/usr/bin/env python
"""
47_corrected_figure_inputs.py -- rewrite the two CSVs the certification figures
read, from the corrected pipeline.

Two changes beyond the corrected gate scores:

  * burden_share now carries each stratum's share of ALL reactive labor, taken
    from burden_by_system.csv.  The column previously written here was
    renormalised over the gated strata, so it summed to 0.9997 by construction,
    and that sum was then reported in the text as evidence that the gated strata
    carry essentially all reactive labor.  The statistic was circular.  Over all
    21 strata the 16 gated ones carry 98.97%.

  * the paper-side gate names are written alongside the code-side ones, so a
    reader starting from the repository can map them without the supplement.

Outputs -> results/tables/cepi_gates.csv, cepi_priority_map.csv (overwritten)
"""
from __future__ import annotations
import sys
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

# code-side name -> paper-side name (also carried in the gate module)
GATE_NAME_MAP = {
    "data_sufficiency":        "data sufficiency",
    "risk_concentration":      "risk concentration",
    "transfer_stability":      "transfer stability",
    "calibration_reliability": "risk-gradient reliability",
    "antecedent_signal":       "prior-record trace",
}
CORRECTED_TO_CODE = {
    "data_sufficiency": "data_sufficiency",
    "risk_concentration": "risk_concentration",
    "transfer_stability": "transfer_stability",
    "risk_gradient": "calibration_reliability",
    "prior_record_trace": "antecedent_signal",
}


def main() -> None:
    ev = pd.read_csv(C.TABLES / "gate_evidence_corrected.csv")
    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    true_share = dict(zip(burden.SystemCode, burden.share_of_total_upm_labour))

    g = pd.DataFrame({
        "system": ev.system,
        "system_desc": ev.system_desc.astype(str).str.strip(),
        "burden_share": [true_share.get(s, float("nan")) for s in ev.system],
    })
    for new, code in CORRECTED_TO_CODE.items():
        g[code] = ev[new].to_numpy()
    g["S"] = ev.S_pooled.to_numpy()
    g["G"] = 0.0
    g = g.sort_values("burden_share", ascending=False)
    g.to_csv(C.TABLES / "cepi_gates.csv", index=False)

    pm_old = pd.read_csv(C.TABLES / "cepi_priority_map.csv")
    keep = ["system", "n_units", "n_pos", "mean_R", "mean_CEPI",
            "frac_high_CEPI_0.45", "dominant_class"]
    pm = pm_old[[c for c in keep if c in pm_old.columns]].copy()
    pm = pm.merge(g[["system", "system_desc", "burden_share", "S", "G"]],
                  on="system", how="right")
    pm.to_csv(C.TABLES / "cepi_priority_map.csv", index=False)

    print("[fig-inputs] cepi_gates.csv and cepi_priority_map.csv rewritten")
    print(f"  gated strata: {len(g)}")
    print(f"  share of ALL reactive labor in gated strata: "
          f"{g.burden_share.sum()*100:.2f}%  (was reported as 99.97)")
    print(f"  certified (S >= 2/3): {(g.S >= 0.67).sum()}")
    print("\n  gate name mapping written for the released module:")
    for code, paper in GATE_NAME_MAP.items():
        print(f"    {code:24s} -> {paper}")


if __name__ == "__main__":
    main()
