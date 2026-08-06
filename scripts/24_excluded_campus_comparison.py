#!/usr/bin/env python
"""
T2.6 — Excluded-campus selection-bias check.

Three universities (8, 9, 12) report no BuildingID and are dropped from the
building-level panel (22.5% of records). A reviewer will ask whether these
campuses represent a different maintenance culture or data quality, which would
limit the transfer claim to "good-data" campuses. We compare the EXCLUDED three
against the INCLUDED nine on dimensions that do not require BuildingID:
  * UPM share of work orders (reactive intensity),
  * median UPM labour hours per work order,
  * the system-level burden distribution (UPM-count share by UNIFORMAT system),
    summarised by Spearman rank correlation and cosine similarity of the
    system-share vectors.

Outputs -> results/metrics/excluded_campus_comparison.json
           results/tables/excluded_vs_included_system_share.csv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, mannwhitneyu

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, io as F

EXCLUDED = set(C.UNIVERSITIES_NO_BUILDING)   # {"8","9","12"}


def main():
    df = F.load_raw(usecols=[C.COL_UNIV, C.COL_BUILDING, C.COL_PPMUPM,
                             C.COL_LABORHOURS, C.COL_SYSTEM, C.COL_SYSTEM_DESC],
                    verbose=True)
    df["univ"] = df[C.COL_UNIV].astype("string")
    df["is_upm"] = F.is_upm(df)
    df["is_ppm"] = F.is_ppm(df)
    df["grp"] = np.where(df["univ"].isin(EXCLUDED), "excluded", "included")

    out = {"excluded_universities": sorted(EXCLUDED)}
    for g, sub in df.groupby("grp"):
        n = len(sub)
        upm = int(sub["is_upm"].sum()); ppm = int(sub["is_ppm"].sum())
        lab = sub.loc[sub["is_upm"], C.COL_LABORHOURS].dropna()
        out[g] = {
            "n_records": n, "n_universities": int(sub["univ"].nunique()),
            "upm_count": upm, "ppm_count": ppm,
            "upm_share": float(upm / (upm + ppm)) if (upm + ppm) else float("nan"),
            "ppm_to_upm_ratio": float(ppm / upm) if upm else float("nan"),
            "median_upm_labour_hours": float(lab.median()) if len(lab) else float("nan"),
            "mean_upm_labour_hours": float(lab.mean()) if len(lab) else float("nan"),
            "labour_coverage": float(sub.loc[sub["is_upm"], C.COL_LABORHOURS].notna().mean()),
        }

    # system-level UPM burden distribution (count share by system)
    sysmap = (df.dropna(subset=[C.COL_SYSTEM]).drop_duplicates(C.COL_SYSTEM)
                .set_index(C.COL_SYSTEM)[C.COL_SYSTEM_DESC])
    def sys_share(sub):
        s = sub[sub["is_upm"]].groupby(C.COL_SYSTEM, observed=True).size()
        return (s / s.sum()).rename("share")
    inc = sys_share(df[df.grp == "included"]).rename("included_share")
    exc = sys_share(df[df.grp == "excluded"]).rename("excluded_share")
    comp = pd.concat([inc, exc], axis=1).fillna(0.0)
    comp["system_desc"] = comp.index.map(sysmap)
    comp = comp.sort_values("included_share", ascending=False)
    comp.to_csv(C.TABLES / "excluded_vs_included_system_share.csv")

    rho, pval = spearmanr(comp["included_share"], comp["excluded_share"])
    a, b = comp["included_share"].to_numpy(), comp["excluded_share"].to_numpy()
    cosine = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    # Mann-Whitney on UPM labour hours (excluded vs included)
    li = df.loc[(df.grp == "included") & df["is_upm"], C.COL_LABORHOURS].dropna()
    le = df.loc[(df.grp == "excluded") & df["is_upm"], C.COL_LABORHOURS].dropna()
    try:
        u_stat, u_p = mannwhitneyu(le.sample(min(len(le), 200000), random_state=42),
                                   li.sample(min(len(li), 200000), random_state=42),
                                   alternative="two-sided")
    except Exception:
        u_stat, u_p = float("nan"), float("nan")

    out["system_burden_spearman_rho"] = float(rho)
    out["system_burden_spearman_p"] = float(pval)
    out["system_burden_cosine_similarity"] = cosine
    out["labour_mannwhitney_p"] = float(u_p)
    out["top_systems_included_vs_excluded"] = comp.head(8)[
        ["system_desc", "included_share", "excluded_share"]].reset_index().to_dict("records")

    json.dump(out, open(C.METRICS / "excluded_campus_comparison.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("top_systems_included_vs_excluded",)}, indent=2))
    print("\nSystem-burden shares (top 8):")
    print(comp.head(8)[["system_desc", "included_share", "excluded_share"]].to_string())


if __name__ == "__main__":
    main()
