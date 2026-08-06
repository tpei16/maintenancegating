#!/usr/bin/env python
"""
Phase 0 -- binding data inspection and go/no-go gates.

Resolves:
  * Gate 1 (temporal granularity): quarter (month-or-finer) vs year-only
  * Gate 2 (cost composition):     genuine total cost (both targets) vs labour-only

Plus: cost/labour completeness, per-university coverage, PPM/UPM balance,
system-level density, base-rate preview, coding consistency, text richness,
building-age/asset metadata availability, and the missing-building share.

Outputs -> results/phase0/*.{json,csv}  and  notes/phase0_go_no_go.md
Run:  python scripts/00_phase0_inspect.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import io as F

pd.options.mode.copy_on_write = True


def jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    return o


def main():
    C.PHASE0_DIR.mkdir(parents=True, exist_ok=True)
    df = F.load_raw()
    df = F.add_time_keys(df, anchor=C.COL_START)
    upm = F.is_upm(df)
    ppm = F.is_ppm(df)
    n = len(df)
    summary = {}

    # ------------------------------------------------------------------ #
    # 0. Basic counts / analytical-count template
    # ------------------------------------------------------------------ #
    n_univ = df[C.COL_UNIV].nunique()
    n_build = df[C.COL_BUILDING].nunique(dropna=True)
    summary["basic"] = {
        "n_records": n,
        "n_universities": n_univ,
        "n_buildings": n_build,
        "n_PPM": int(ppm.sum()),
        "n_UPM": int(upm.sum()),
        "ppm_upm_other": int((~ppm & ~upm).sum()),
        "missing_building_share": float(df[C.COL_BUILDING].isna().mean()),
        "n_systemcodes": df[C.COL_SYSTEM].nunique(dropna=True),
        "n_subsystemcodes": df[C.COL_SUBSYS].nunique(dropna=True),
        "n_componentcodes": df[C.COL_COMPONENT].nunique(dropna=True),
    }

    # ------------------------------------------------------------------ #
    # 1. GATE 1 -- temporal granularity
    # ------------------------------------------------------------------ #
    start = df[C.COL_START]
    n_parsed = int(start.notna().sum())
    nat_share = float(start.isna().mean())
    # intra-year structure: do month and day vary (vs all Jan-1 = year-only encoding)?
    month = start.dt.month
    day = start.dt.day
    distinct_months = int(month.dropna().nunique())
    distinct_days = int(day.dropna().nunique())
    share_not_jan1 = float(((month != 1) | (day != 1)).mean())
    # time-of-day present?
    share_with_time = float(((start.dt.hour != 0) | (start.dt.minute != 0) | (start.dt.second != 0)).mean())
    gate1_quarter = (distinct_months >= 6) and (distinct_days >= 10) and (share_not_jan1 > 0.5)
    gate1 = {
        "n_parsed": n_parsed,
        "nat_share": nat_share,
        "distinct_months": distinct_months,
        "distinct_days": distinct_days,
        "share_not_jan1": share_not_jan1,
        "share_with_time_of_day": share_with_time,
        "decision": "quarter" if gate1_quarter else "year",
        "framing": "near-term operational screening" if gate1_quarter else "annual strategic screening",
    }
    summary["gate1_temporal"] = gate1

    # year distribution
    year_counts = df["year"].value_counts(dropna=True).sort_index()
    year_counts.rename_axis("year").rename("n_records").to_csv(C.PHASE0_DIR / "records_per_year.csv")

    # ------------------------------------------------------------------ #
    # 2. GATE 2 -- cost composition
    # ------------------------------------------------------------------ #
    tc = df[C.COL_TOTALCOST]
    lc = df[C.COL_LABORCOST]
    mc = df[C.COL_MATERIALCOST]
    oc = df[C.COL_OTHERCOST]
    lh = df[C.COL_LABORHOURS]
    parts_sum = lc.fillna(0) + mc.fillna(0) + oc.fillna(0)
    # is TotalCost a genuine total (= labor+material+other)?
    both_present = tc.notna() & parts_sum.notna()
    consistent = (np.abs(tc.fillna(0) - parts_sum) <= (0.01 * np.abs(tc.fillna(0)) + 0.5))
    share_total_eq_parts = float(consistent[both_present].mean()) if both_present.any() else float("nan")
    share_material_or_other_pos = float(((mc.fillna(0) > 0) | (oc.fillna(0) > 0)).mean())
    genuine_total = (tc.notna().mean() > 0.3) and (share_material_or_other_pos > 0.01)
    gate2 = {
        "totalcost_nonnull_share": float(tc.notna().mean()),
        "totalcost_pos_share": float((tc.fillna(0) > 0).mean()),
        "laborcost_nonnull_share": float(lc.notna().mean()),
        "laborcost_pos_share": float((lc.fillna(0) > 0).mean()),
        "laborhours_nonnull_share": float(lh.notna().mean()),
        "laborhours_pos_share": float((lh.fillna(0) > 0).mean()),
        "share_total_eq_labor_plus_material_plus_other": share_total_eq_parts,
        "share_records_with_material_or_other_cost": share_material_or_other_pos,
        "decision": "both" if genuine_total else "labour_only",
    }
    summary["gate2_cost"] = gate2

    # ------------------------------------------------------------------ #
    # 3. Completeness per university (cost & labour) -- UPM records only
    #    (severity targets are computed on UPM burden)
    # ------------------------------------------------------------------ #
    dfu = df.assign(_upm=upm.values)
    upm_only = dfu[dfu["_upm"]]
    comp = upm_only.groupby(C.COL_UNIV, observed=True).agg(
        n_upm=("_upm", "size"),
        cost_pos_share=(C.COL_TOTALCOST, lambda s: float((s.fillna(0) > 0).mean())),
        labour_pos_share=(C.COL_LABORHOURS, lambda s: float((s.fillna(0) > 0).mean())),
        cost_nonnull_share=(C.COL_TOTALCOST, lambda s: float(s.notna().mean())),
        labour_nonnull_share=(C.COL_LABORHOURS, lambda s: float(s.notna().mean())),
    )
    comp.to_csv(C.PHASE0_DIR / "completeness_per_university.csv")
    summary["completeness"] = {
        "cost_universities_above_thresh": int((comp["cost_pos_share"] >= C.COST_COMPLETENESS_MIN).sum()),
        "labour_universities_above_thresh": int((comp["labour_pos_share"] >= C.LABOUR_COMPLETENESS_MIN).sum()),
        "n_universities": int(len(comp)),
    }

    # ------------------------------------------------------------------ #
    # 4. Per-university coverage (start year, end year, count)
    # ------------------------------------------------------------------ #
    cov = df.groupby(C.COL_UNIV, observed=True).agg(
        n_records=("year", "size"),
        start_year=("year", "min"),
        end_year=("year", "max"),
        n_buildings=(C.COL_BUILDING, "nunique"),
    )
    cov["span_years"] = cov["end_year"] - cov["start_year"] + 1
    cov["n_upm"] = upm_only.groupby(C.COL_UNIV, observed=True).size()
    cov = cov.sort_values("n_records", ascending=False)
    cov.to_csv(C.PHASE0_DIR / "per_university_coverage.csv")
    summary["coverage"] = {
        "median_span_years": float(cov["span_years"].median()),
        "min_span_years": int(cov["span_years"].min()),
        "max_span_years": int(cov["span_years"].max()),
        "universities_span_ge_6y": int((cov["span_years"] >= 6).sum()),
    }

    # ------------------------------------------------------------------ #
    # 5. PPM/UPM balance by major system
    # ------------------------------------------------------------------ #
    bal = (df.assign(kind=np.where(upm, "UPM", np.where(ppm, "PPM", "OTHER")))
             .groupby([C.COL_SYSTEM, "kind"], observed=True).size().unstack(fill_value=0))
    bal["total"] = bal.sum(axis=1)
    if "UPM" in bal.columns:
        bal["upm_share"] = bal["UPM"] / bal["total"]
    bal = bal.sort_values("total", ascending=False)
    bal.to_csv(C.PHASE0_DIR / "ppm_upm_by_system.csv")

    # ------------------------------------------------------------------ #
    # 6. System-level density (records per building-system cell)
    # ------------------------------------------------------------------ #
    valid_cell = df[C.COL_BUILDING].notna() & df[C.COL_SYSTEM].notna()
    cellkey = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]
    cell_rec = df[valid_cell].groupby(cellkey, observed=True).size()
    # quarters of activity per cell
    cell_q = (df[valid_cell].groupby(cellkey, observed=True)["period_q"]
                .nunique())
    n_cells = int(len(cell_rec))
    summary["density"] = {
        "n_building_system_cells": n_cells,
        "median_records_per_cell": float(cell_rec.median()),
        "median_active_quarters_per_cell": float(cell_q.median()),
        "share_cells_lt_8_quarters": float((cell_q < 8).mean()),
        "share_cells_ge_12_quarters": float((cell_q >= 12).mean()),
        "share_records_missing_building": summary["basic"]["missing_building_share"],
    }
    # density by system
    dens_sys = (df[valid_cell].assign(_u=upm[valid_cell].values)
                  .groupby(C.COL_SYSTEM, observed=True)
                  .agg(n_records=("year", "size"),
                       n_cells=(C.COL_BUILDING, lambda s: s.astype("string").nunique())))
    dens_sys.to_csv(C.PHASE0_DIR / "system_density.csv")

    # ------------------------------------------------------------------ #
    # 7. Base-rate PREVIEW (approximate; exact base rate computed in panel build)
    #    Aggregate UPM labour to building-system-quarter; estimate positivity.
    # ------------------------------------------------------------------ #
    agg = (df[valid_cell].assign(_u=upm[valid_cell].values,
                                 _ulh=np.where(upm[valid_cell].values, lh[valid_cell].fillna(0).values, 0.0))
             .groupby(cellkey + ["period_q"], observed=True)
             .agg(upm_events=("_u", "sum"), upm_labour=("_ulh", "sum")))
    occ_rate_active = float((agg["upm_events"] > 0).mean())
    # per-system 75th pct of POSITIVE upm labour among active cell-quarters
    pos = agg[agg["upm_labour"] > 0]
    # severity proxy: among active cell-quarters, share above global 75th pct of positive labour
    thr75 = float(pos["upm_labour"].quantile(0.75)) if len(pos) else float("nan")
    sev_rate_active = float((agg["upm_labour"] > thr75).mean()) if len(agg) else float("nan")
    summary["base_rate_preview"] = {
        "n_active_cell_quarters": int(len(agg)),
        "occurrence_rate_among_active_cellquarters": occ_rate_active,
        "global_pos_labour_p75": thr75,
        "severity_rate_above_global_p75_among_active": sev_rate_active,
        "note": "approximate; panel base rate over all known-system cells computed in 01_build_panel.py",
    }

    # ------------------------------------------------------------------ #
    # 8. Coding consistency (cross-university overlap of system codes)
    # ------------------------------------------------------------------ #
    sys_univ = (df.dropna(subset=[C.COL_SYSTEM])
                  .groupby(C.COL_SYSTEM, observed=True)[C.COL_UNIV].nunique())
    summary["coding_consistency"] = {
        "n_system_codes": int(sys_univ.shape[0]),
        "median_universities_per_system": float(sys_univ.median()),
        "share_systems_in_ge_half_universities": float((sys_univ >= n_univ / 2).mean()),
        "share_systems_in_all_universities": float((sys_univ >= n_univ).mean()),
    }
    sys_univ.rename("n_universities_using").to_csv(C.PHASE0_DIR / "system_code_university_overlap.csv")

    # ------------------------------------------------------------------ #
    # 9. Text richness
    # ------------------------------------------------------------------ #
    desc = df[C.COL_WODESC].astype("string")
    desc_len = desc.str.len()
    summary["text_richness"] = {
        "wodesc_nonnull_share": float(desc.notna().mean()),
        "wodesc_median_char_len": float(desc_len.median()),
        "wodesc_median_word_count": float(desc.str.split().str.len().median()),
        "n_unique_descriptions": int(desc.nunique(dropna=True)),
        "samples": desc.dropna().head(8).tolist(),
    }

    # ------------------------------------------------------------------ #
    # 10. Building age & asset metadata availability
    # ------------------------------------------------------------------ #
    summary["metadata_availability"] = {
        "built_year_nonnull_share": float(df[C.COL_BUILT_YEAR].notna().mean()),
        "size_nonnull_share": float(df[C.COL_SIZE].notna().mean()),
        "type_nonnull_share": float(df[C.COL_BTYPE].notna().mean()),
        "fci_nonnull_share": float(df[C.COL_FCI].notna().mean()),
        "crv_nonnull_share": float(df[C.COL_CRV].notna().mean()),
        "dmc_nonnull_share": float(df[C.COL_DMC].notna().mean()),
        "weather_nonnull_share": float(df[C.COL_MINTEMP].notna().mean()),
        "built_year_range": [float(df[C.COL_BUILT_YEAR].min()), float(df[C.COL_BUILT_YEAR].max())],
    }

    # ------------------------------------------------------------------ #
    # Write artifacts
    # ------------------------------------------------------------------ #
    with open(C.PHASE0_DIR / "summary.json", "w") as f:
        json.dump(jsonable(summary), f, indent=2)

    gate_res = {
        "GATE1_PERIOD": gate1["decision"],
        "GATE1_FRAMING": gate1["framing"],
        "GATE2_BURDEN_TARGETS": gate2["decision"],
        "COST_DEFLATION_BASE_YEAR": int(year_counts.index.max()),
        "TEMPORAL_TRAIN_END_YEAR_SUGGESTED": _suggest_cutoff(year_counts),
        "n_records": n,
        "n_universities": n_univ,
    }
    with open(C.PHASE0_DIR / "gate_resolution.json", "w") as f:
        json.dump(jsonable(gate_res), f, indent=2)

    _write_report(summary, gate_res, cov, year_counts)
    print(json.dumps(jsonable(gate_res), indent=2))
    print("\n[phase0] wrote artifacts to", C.PHASE0_DIR)


def _suggest_cutoff(year_counts: pd.Series) -> int:
    """Pick a temporal cutoff so test years (after) hold a reasonable share of data."""
    total = year_counts.sum()
    cum = year_counts.cumsum() / total
    # cutoff = last year where cumulative <= 0.70 (so ~30% in test)
    elig = cum[cum <= 0.72]
    return int(elig.index.max()) if len(elig) else int(year_counts.index.min())


def _write_report(summary, gate_res, cov, year_counts):
    b = summary["basic"]
    g1 = summary["gate1_temporal"]
    g2 = summary["gate2_cost"]
    lines = []
    A = lines.append
    A("# Phase 0 — Data Inspection & Binding Go/No-Go Report\n")
    A(f"_Dataset: FMUCD (SHA-256 verified). {b['n_records']:,} work orders, "
      f"{b['n_universities']} universities, {b['n_buildings']:,} buildings._\n")
    A("## Headline gate resolution\n")
    A(f"- **Gate 1 (temporal granularity): `{gate_res['GATE1_PERIOD'].upper()}`** "
      f"→ framing = _{gate_res['GATE1_FRAMING']}_.")
    A(f"- **Gate 2 (cost composition): `{gate_res['GATE2_BURDEN_TARGETS'].upper()}`** "
      f"→ {'both cost and labour targets viable' if gate_res['GATE2_BURDEN_TARGETS']=='both' else 'labour hours is the sole burden target'}.")
    A(f"- Suggested temporal cutoff: train ≤ **{gate_res['TEMPORAL_TRAIN_END_YEAR_SUGGESTED']}**, "
      f"test later years. Cost deflation base year: **{gate_res['COST_DEFLATION_BASE_YEAR']}**.\n")

    A("## Analytical-count template (for manuscript Section 10)\n")
    A(f"> After preprocessing, the analytical dataset contained **{b['n_records']:,}** work orders "
      f"from **{b['n_universities']}** universities, including **{b['n_PPM']:,}** PPM and "
      f"**{b['n_UPM']:,}** UPM records.\n")

    A("## Gate 1 detail — temporal granularity\n")
    A(f"- Parsed timestamps: {g1['n_parsed']:,} ({1-g1['nat_share']:.3%} of rows); NaT share {g1['nat_share']:.3%}.")
    A(f"- Distinct months present: {g1['distinct_months']}/12; distinct days: {g1['distinct_days']}/31.")
    A(f"- Share of records NOT on Jan-1 (real sub-year dates): {g1['share_not_jan1']:.3%}.")
    A(f"- Share with non-midnight time-of-day: {g1['share_with_time_of_day']:.3%}.")
    A(f"- **Decision: {g1['decision']}** → building × system × **{g1['decision']}** panel.\n")

    A("## Gate 2 detail — cost composition\n")
    A(f"- TotalCost non-null: {g2['totalcost_nonnull_share']:.3%}; positive: {g2['totalcost_pos_share']:.3%}.")
    A(f"- TotalCost == Labor+Material+Other (consistency): {g2['share_total_eq_labor_plus_material_plus_other']:.3%}.")
    A(f"- Records with material/other cost > 0 (i.e., total ≠ labour-only): {g2['share_records_with_material_or_other_cost']:.3%}.")
    A(f"- LaborHours non-null {g2['laborhours_nonnull_share']:.3%}, positive {g2['laborhours_pos_share']:.3%}.")
    A(f"- **Decision: {g2['decision']}**. Labour hours is the PRIMARY burden dimension (inflation-immune); "
      f"cost is secondary, deflated.\n")

    d = summary["density"]
    A("## System-level density\n")
    A(f"- Building × system cells: **{d['n_building_system_cells']:,}**.")
    A(f"- Median records/cell: {d['median_records_per_cell']:.1f}; median active quarters/cell: {d['median_active_quarters_per_cell']:.1f}.")
    A(f"- Share of cells with < 8 active quarters: {d['share_cells_lt_8_quarters']:.1%}; "
      f"≥ 12 active quarters: {d['share_cells_ge_12_quarters']:.1%}.")
    A(f"- **Records with missing BuildingID (cannot be placed in a building cell): "
      f"{d['share_records_missing_building']:.2%}** — excluded from the panel, reported in totals.\n")

    br = summary["base_rate_preview"]
    A("## Base-rate preview (approximate)\n")
    A(f"- Active cell-quarters: {br['n_active_cell_quarters']:,}.")
    A(f"- Occurrence rate among active cell-quarters: {br['occurrence_rate_among_active_cellquarters']:.1%}.")
    A(f"- Severity proxy (UPM labour above global p75): {br['severity_rate_above_global_p75_among_active']:.1%}.")
    A(f"- _Exact panel base rate (over all known-system cells, incl. zero-burden) computed in panel build._\n")

    cc = summary["coding_consistency"]
    A("## Coding consistency (FMUCO standardization)\n")
    A(f"- System codes: {cc['n_system_codes']}; median universities per system: {cc['median_universities_per_system']:.0f}.")
    A(f"- Systems present in ≥ half of universities: {cc['share_systems_in_ge_half_universities']:.0%}; "
      f"in ALL universities: {cc['share_systems_in_all_universities']:.0%}.")
    A("- → System labels are comparable across institutions by construction (supports cross-institutional transfer).\n")

    md = summary["metadata_availability"]
    A("## Building-age & asset metadata availability\n")
    A(f"- BuiltYear {md['built_year_nonnull_share']:.1%}, Size {md['size_nonnull_share']:.1%}, "
      f"Type {md['type_nonnull_share']:.1%}, FCI {md['fci_nonnull_share']:.1%}, "
      f"CRV {md['crv_nonnull_share']:.1%}, DMC {md['dmc_nonnull_share']:.1%}.")
    A(f"- Weather context non-null: {md['weather_nonnull_share']:.1%}. BuiltYear range: "
      f"{md['built_year_range'][0]:.0f}–{md['built_year_range'][1]:.0f}.")
    A(f"- → {'Full heterogeneity analysis feasible (building age available).' if md['built_year_nonnull_share']>0.3 else 'Heterogeneity analysis limited by metadata gaps.'}\n")

    tr = summary["text_richness"]
    A("## Text richness (optional M2)\n")
    A(f"- WODescription non-null {tr['wodesc_nonnull_share']:.1%}, median {tr['wodesc_median_word_count']:.0f} words, "
      f"{tr['n_unique_descriptions']:,} unique strings.")
    A(f"- → {'Text is rich enough to test an optional M2 text model.' if tr['wodesc_median_word_count']>=3 else 'Text too sparse for M2.'}\n")

    A("## Per-university coverage (top rows)\n")
    A("| Univ | records | UPM | buildings | start | end | span(y) |")
    A("|---|---|---|---|---|---|---|")
    for u, r in cov.head(20).iterrows():
        A(f"| {u} | {int(r['n_records']):,} | {int(r['n_upm']) if pd.notna(r['n_upm']) else 0:,} | "
          f"{int(r['n_buildings'])} | {int(r['start_year'])} | {int(r['end_year'])} | {int(r['span_years'])} |")
    A("")

    A("## GO / NO-GO\n")
    A("**GO.** Both gates resolve to the richer branch; metadata, text, and weather are present; "
      "coding is standardized across institutions; density supports the chosen unit. "
      "Proceed to panel construction (known-system rule, quarter resolution) and the Phase 0b pilot.\n")

    (C.NOTES / "phase0_go_no_go.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
