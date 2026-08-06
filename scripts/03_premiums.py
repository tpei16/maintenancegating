#!/usr/bin/env python
"""
Phase 1 — reactive-maintenance premium analysis +
RQ1a descriptive burden across systems and institutions.

DESCRIPTIVE (full data): UPM-to-PPM burden ratios (labour & cost) by system,
component, and institution. This is the descriptive precursor; the model-feature
version of the premium is computed per training fold inside features.py.

Outputs -> results/tables/premiums_*.csv, results/metrics/burden_descriptive.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, io as F, panel as P


def main():
    cols = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM, C.COL_SYSTEM_DESC,
            C.COL_COMPONENT, C.COL_COMPONENT_DESC, C.COL_PPMUPM,
            C.COL_START, C.COL_LABORHOURS, C.COL_TOTALCOST]
    df = F.load_raw(usecols=cols)
    df = F.add_time_keys(df, anchor=C.COL_START)
    df[C.COL_SYSTEM] = df[C.COL_SYSTEM].astype("string").str.strip().str.upper()
    df["upm"] = F.is_upm(df).to_numpy()
    df["ppm"] = F.is_ppm(df).to_numpy()
    df["lab"] = df[C.COL_LABORHOURS].fillna(0.0)
    df["defl"] = P._cost_deflator(df["year"]).to_numpy()
    df["cost"] = df[C.COL_TOTALCOST].fillna(0.0) * df["defl"]
    df = df[df["upm"] | df["ppm"]]

    # ---- system-level burden + premium ----
    def premium_table(group_cols, min_each=50):
        g = df.groupby(group_cols, observed=True)
        rows = []
        for key, sub in g:
            u = sub[sub["upm"]]; p = sub[sub["ppm"]]
            if len(u) < min_each or len(p) < min_each:
                continue
            row = dict(zip(group_cols if isinstance(group_cols, list) else [group_cols],
                           key if isinstance(key, tuple) else (key,)))
            row.update({
                "n_upm": int(len(u)), "n_ppm": int(len(p)),
                "upm_lab_median": float(u["lab"].median()),
                "ppm_lab_median": float(p["lab"].median()),
                "upm_lab_mean": float(u["lab"].mean()),
                "ppm_lab_mean": float(p["lab"].mean()),
                "lab_premium_median": float(u["lab"].median() / p["lab"].median()) if p["lab"].median() > 0 else np.nan,
                "lab_premium_mean": float(u["lab"].mean() / p["lab"].mean()) if p["lab"].mean() > 0 else np.nan,
                "upm_cost_mean": float(u["cost"].mean()),
                "ppm_cost_mean": float(p["cost"].mean()),
                "cost_premium_mean": float(u["cost"].mean() / p["cost"].mean()) if p["cost"].mean() > 0 else np.nan,
            })
            rows.append(row)
        return pd.DataFrame(rows)

    sys_prem = premium_table([C.COL_SYSTEM, C.COL_SYSTEM_DESC]).sort_values("n_upm", ascending=False)
    sys_prem.to_csv(C.TABLES / "premiums_by_system.csv", index=False)

    comp_prem = premium_table([C.COL_SYSTEM, C.COL_COMPONENT, C.COL_COMPONENT_DESC], min_each=100)
    comp_prem = comp_prem.sort_values("lab_premium_mean", ascending=False)
    comp_prem.to_csv(C.TABLES / "premiums_by_component.csv", index=False)

    sysinst_prem = premium_table([C.COL_UNIV, C.COL_SYSTEM]).sort_values([C.COL_UNIV])
    sysinst_prem.to_csv(C.TABLES / "premiums_by_system_institution.csv", index=False)

    # ---- RQ1a: reactive burden across systems & institutions ----
    burden_sys = (df[df["upm"]].groupby([C.COL_SYSTEM, C.COL_SYSTEM_DESC], observed=True)
                  .agg(n_upm=("upm", "size"), total_upm_labour=("lab", "sum"),
                       mean_upm_labour=("lab", "mean"))
                  .reset_index().sort_values("total_upm_labour", ascending=False))
    burden_sys["share_of_total_upm_labour"] = burden_sys["total_upm_labour"] / burden_sys["total_upm_labour"].sum()
    burden_sys.to_csv(C.TABLES / "burden_by_system.csv", index=False)

    burden_inst = (df[df["upm"]].groupby(C.COL_UNIV, observed=True)
                   .agg(n_upm=("upm", "size"), total_upm_labour=("lab", "sum"),
                        mean_upm_labour=("lab", "mean"))
                   .reset_index().sort_values("total_upm_labour", ascending=False))
    burden_inst.to_csv(C.TABLES / "burden_by_institution.csv", index=False)

    summary = {
        "n_systems_with_premium": int(len(sys_prem)),
        "median_lab_premium_across_systems": float(sys_prem["lab_premium_median"].median()),
        "systems_with_lab_premium_gt_1": int((sys_prem["lab_premium_median"] > 1).sum()),
        "top5_systems_by_upm_labour": burden_sys.head(5)[[C.COL_SYSTEM_DESC, "share_of_total_upm_labour"]].to_dict("records"),
        "cost_premium_available": bool(sys_prem["cost_premium_mean"].notna().any()),
        "median_cost_premium_across_systems": float(sys_prem["cost_premium_mean"].median()),
    }
    json.dump(summary, open(C.METRICS / "burden_descriptive.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))
    print("\n[premiums] top systems by labour premium (median):")
    print(sys_prem[[C.COL_SYSTEM_DESC, "n_upm", "lab_premium_median", "cost_premium_mean"]]
          .head(12).to_string(index=False))


if __name__ == "__main__":
    main()
