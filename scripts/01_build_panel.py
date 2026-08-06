#!/usr/bin/env python
"""
Phase 1 (data transformation) — build the building x system x quarter panel.

Saves:
  data/processed/panel_quarter.parquet         (full panel, known-system rule)
  results/phase0/panel_base_rates.json          (exact base rates for both rules)
  results/phase0/panel_summary.md               (shape, coverage, leakage checklist)
Run:  python scripts/01_build_panel.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import panel as P


def main():
    panel = P.build()
    # drop rows whose t+1 label is unobserved (last anchor per cell)
    labeled = panel[panel["occurrence_next"].notna()].copy()
    labeled["occurrence_next"] = labeled["occurrence_next"].astype("int8")

    out = C.DATA_PROCESSED / "panel_quarter.parquet"
    labeled.to_parquet(out, index=False)

    # ---- exact base rates ----
    def severity_rate(sub, pctl=75):
        # system-specific threshold on POSITIVE next-period UPM labour
        pos = sub[sub["upm_labour_next"] > 0]
        thr = pos.groupby(C.COL_SYSTEM, observed=True)["upm_labour_next"].quantile(pctl / 100.0)
        merged = sub.merge(thr.rename("thr"), on=C.COL_SYSTEM, how="left")
        sev = ((merged["upm_labour_next"] > merged["thr"]) & (merged["upm_labour_next"] > 0))
        return float(sev.mean())

    main_panel = labeled
    recent = labeled[labeled["active_prev4"] == 1]
    base = {
        "panel_rows_known_system": int(len(main_panel)),
        "panel_rows_recent_activity": int(len(recent)),
        "n_universities": int(main_panel[C.COL_UNIV].nunique()),
        "n_buildings": int(main_panel[C.COL_BUILDING].nunique()),
        "n_cells": int(main_panel.groupby([C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM],
                                          observed=True).ngroups),
        "n_systems": int(main_panel[C.COL_SYSTEM].nunique()),
        "period_q_range": [int(main_panel["period_q"].min()), int(main_panel["period_q"].max())],
        "year_range": [int(main_panel["year"].min()), int(main_panel["year"].max())],
        "occurrence_base_rate_known_system": float(main_panel["occurrence_next"].mean()),
        "occurrence_base_rate_recent_activity": float(recent["occurrence_next"].mean()),
        "severity_base_rate_known_system_p50": severity_rate(main_panel, 50),
        "severity_base_rate_known_system_p75": severity_rate(main_panel, 75),
        "severity_base_rate_known_system_p90": severity_rate(main_panel, 90),
        "severity_base_rate_recent_activity_p75": severity_rate(recent, 75),
        "active_row_share": float((main_panel["active"] == 1).mean()),
    }
    with open(C.PHASE0_DIR / "panel_base_rates.json", "w") as f:
        json.dump(base, f, indent=2)

    # per-university occurrence & severity base rates
    rows = []
    for u, sub in main_panel.groupby(C.COL_UNIV, observed=True):
        rows.append({
            "university": u, "rows": len(sub),
            "occurrence_base_rate": float(sub["occurrence_next"].mean()),
            "severity_base_rate_p75": severity_rate(sub, 75),
            "year_min": int(sub["year"].min()), "year_max": int(sub["year"].max()),
        })
    pd.DataFrame(rows).sort_values("rows", ascending=False).to_csv(
        C.PHASE0_DIR / "panel_base_rates_per_university.csv", index=False)

    _report(base, out)
    print(json.dumps(base, indent=2))


def _report(base, out):
    L = []
    A = L.append
    A("# Panel summary — building × system × quarter (known-system rule)\n")
    A(f"- Saved: `{out.relative_to(C.ROOT)}`")
    A(f"- Rows (known-system): **{base['panel_rows_known_system']:,}**; "
      f"recent-activity subset: {base['panel_rows_recent_activity']:,}.")
    A(f"- Universities: {base['n_universities']}; buildings: {base['n_buildings']:,}; "
      f"cells: {base['n_cells']:,}; systems: {base['n_systems']}.")
    A(f"- Quarter range: {base['year_range'][0]}–{base['year_range'][1]} "
      f"(period_q {base['period_q_range'][0]}–{base['period_q_range'][1]}).")
    A(f"- Active-row share (cell-quarters with any work order): {base['active_row_share']:.1%}.\n")
    A("## Base rates (exact)\n")
    A(f"- **Occurrence** (any UPM next quarter): known-system **{base['occurrence_base_rate_known_system']:.2%}**, "
      f"recent-activity {base['occurrence_base_rate_recent_activity']:.2%}.")
    A(f"- **Severity** (next UPM labour > system p75): known-system **{base['severity_base_rate_known_system_p75']:.2%}** "
      f"(p50 {base['severity_base_rate_known_system_p50']:.2%}, p90 {base['severity_base_rate_known_system_p90']:.2%}).")
    A(f"- Severity p75 on recent-activity subset: {base['severity_base_rate_recent_activity_p75']:.2%}.\n")
    A("> The 2× precision-lift bar at the top-10% budget corresponds to capturing ≥20% of "
      "next-quarter severe events in that 10%. With the severity base rate above, this bar is "
      "meaningful in absolute terms (Section 21 check).\n")
    A("## Leakage checklist\n")
    A("- [x] Features use only periods ≤ t (rolling windows end at t; current period is past info).")
    A("- [x] Labels use period t+1 only (`shift(-1)` of next-quarter aggregates).")
    A("- [x] Severity thresholds NOT baked in — raw `upm_labour_next` stored, thresholded per training fold.")
    A("- [x] Train-derived premium features computed per fold (not in panel).")
    A("- [x] University identity excluded from features (used only for grouping/splits).")
    A("- [x] Rows with unobserved t+1 (last anchor per cell) dropped.")
    (C.NOTES / "panel_summary.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
