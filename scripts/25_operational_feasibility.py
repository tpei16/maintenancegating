#!/usr/bin/env python
"""
T2.4 — Operational feasibility, preventive-to-corrective ratio, and FCI grounding.

Translates the top-10% screening budget into a concrete FM workload and connects
the study to standard FM condition metrics that exist in FMUCD:
  (a) Operational feasibility: cells scored per campus-quarter -> 10% inspections
      per quarter and per week.
  (b) Preventive-to-corrective (PPM:UPM) ratio overall and by system -- the
      maintenance-management KPI the FM literature benchmarks.
  (c) Facility Condition Index (FCI) grounding: coverage in FMUCD, distribution,
      and whether buildings carrying more reactive labour have worse (higher) FCI,
      i.e. whether the record-based burden signal aligns with the FM condition KPI.

Outputs -> results/metrics/operational_feasibility.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, io as F

WEEKS_PER_QUARTER = 13.0


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    out = {"topk_budget": C.TOPK_BUDGET}

    # (a) cells scored per campus-quarter (known-system panel rows that are present)
    cpq = panel.groupby([C.COL_UNIV, "period_q"], observed=True).size()
    med = float(cpq.median()); q1 = float(cpq.quantile(0.25)); q3 = float(cpq.quantile(0.75))
    # per-campus typical panel size (active cell-quarters per quarter)
    per_campus = panel.groupby(C.COL_UNIV, observed=True).apply(
        lambda g: g.groupby("period_q").size().median())
    out["cells_per_campus_quarter"] = {
        "median": med, "iqr": [q1, q3],
        "per_campus_median": {str(k): float(v) for k, v in per_campus.items()},
        "max_campus": float(per_campus.max()), "min_campus": float(per_campus.min()),
    }
    out["inspections_at_10pct"] = {
        "per_quarter_median": med * C.TOPK_BUDGET,
        "per_week_median": med * C.TOPK_BUDGET / WEEKS_PER_QUARTER,
        "per_quarter_largest_campus": float(per_campus.max()) * C.TOPK_BUDGET,
        "per_week_largest_campus": float(per_campus.max()) * C.TOPK_BUDGET / WEEKS_PER_QUARTER,
    }

    # (b) preventive-to-corrective ratio
    upm_n = float(panel["upm_count"].sum()); ppm_n = float(panel["ppm_count"].sum())
    upm_l = float(panel["upm_labour"].sum()); ppm_l = float(panel["ppm_labour"].sum())
    out["preventive_to_corrective"] = {
        "ppm_to_upm_count_ratio": ppm_n / upm_n if upm_n else float("nan"),
        "ppm_to_upm_labour_ratio": ppm_l / upm_l if upm_l else float("nan"),
        "upm_share_of_orders": upm_n / (upm_n + ppm_n) if (upm_n + ppm_n) else float("nan"),
    }
    bysys = panel.groupby(C.COL_SYSTEM, observed=True).agg(
        upm_n=("upm_count", "sum"), ppm_n=("ppm_count", "sum"),
        desc=("SystemDescription", "first"))
    bysys["ppm_to_upm"] = bysys["ppm_n"] / bysys["upm_n"].replace(0, np.nan)
    bysys = bysys.sort_values("upm_n", ascending=False)
    out["ppm_to_upm_by_top_systems"] = bysys.head(8)[["desc", "ppm_to_upm"]].reset_index().to_dict("records")

    # (c) FCI grounding from raw
    raw = F.load_raw(usecols=[C.COL_UNIV, C.COL_BUILDING, C.COL_FCI, C.COL_DMC,
                              C.COL_CRV, C.COL_LABORHOURS, C.COL_PPMUPM], verbose=True)
    raw = raw.dropna(subset=[C.COL_BUILDING])
    raw["bkey"] = raw[C.COL_UNIV].astype("string") + "/" + raw[C.COL_BUILDING].astype("string")
    fci_bldg = raw.groupby("bkey", observed=True)[C.COL_FCI].median()
    out["fci"] = {
        "record_coverage": float(raw[C.COL_FCI].notna().mean()),
        "building_coverage": float(fci_bldg.notna().mean()),
        "n_buildings_with_fci": int(fci_bldg.notna().sum()),
        "median_fci": float(fci_bldg.median()),
        "fci_iqr": [float(fci_bldg.quantile(0.25)), float(fci_bldg.quantile(0.75))],
    }
    # do high-reactive-burden buildings have worse (higher) FCI?
    raw["is_upm"] = F.is_upm(raw)
    burden = (raw[raw["is_upm"]].groupby("bkey", observed=True)[C.COL_LABORHOURS]
                .sum().rename("upm_labour_total"))
    j = pd.concat([fci_bldg.rename("fci"), burden], axis=1).dropna()
    if len(j) > 30:
        rho, p = spearmanr(j["fci"], j["upm_labour_total"])
        out["fci"]["spearman_fci_vs_upm_burden"] = float(rho)
        out["fci"]["spearman_p"] = float(p)
        out["fci"]["n_buildings_paired"] = int(len(j))

    json.dump(out, open(C.METRICS / "operational_feasibility.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
