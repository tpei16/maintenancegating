#!/usr/bin/env python
"""
Analysis X (C1 headline) — cross-system co-escalation within buildings.

Question: when one building system has a SEVERE reactive event (UPM labour above
its system-specific p75), do OTHER systems in the same building show elevated
risk of a severe event within the next 1-4 quarters?

Design (descriptive, full 2002-2021, no train/test split):
  * Severe(cell, t)  := upm_labour(cell,t) > p75 of positive upm_labour for that
    system (threshold computed on full data).
  * sev_next4(cell,t):= cell severe in any of t+1..t+4 (window fully observed:
    period_q+4 <= building_last_q).
  * Cell-pair risk ratio = P(sibling cell (b,B) severe in t+1..4 | cell (b,A)
    severe at t, B!=A) / P(cell severe in t+1..4)  [population marginal].
  * Trigger->follower RR matrix RR(A->B) by system pair.
  * Building-level "any-other-system" RR.

Confound (stated, not resolved): shared building-level factors (age, occupancy,
envelope) may drive co-escalation; with 24% age coverage this is associational,
not causal. The population-marginal baseline is the between-building control.

Outputs -> results/metrics/coescalation.json, results/tables/coescalation_matrix.csv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

CELL = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]


def main():
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    p["bkey"] = p[C.COL_UNIV].astype("string") + "/" + p[C.COL_BUILDING].astype("string")
    # system-specific severity threshold (p75 of positive UPM labour, full data)
    thr = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.75)
    p["sev"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr).fillna(np.inf)).astype(int)

    # forward severe within the SAME cell over the next 4 quarters
    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    g = p.groupby(CELL, observed=True)["sev"]
    nxt = None
    for w in (1, 2, 3, 4):
        s = g.shift(-w)
        nxt = s if nxt is None else np.fmax(nxt, s)
    p["sev_next4"] = (pd.Series(nxt, index=p.index).fillna(0) > 0).astype(int)
    p["win_valid"] = (p["period_q"] + 4 <= p["building_last_q"]).astype(int)

    present = p[p["win_valid"] == 1].copy()      # cells with a fully-observed forward window
    present["both_row"] = ((present["sev"] == 1) & (present["sev_next4"] == 1)).astype(int)
    # building-quarters with >= 2 systems present (co-escalation only meaningful there)
    bq = present.groupby(["bkey", "period_q"], observed=True).agg(
        n_present=("sev", "size"), n_trig=("sev", "sum"),
        n_sevnext4=("sev_next4", "sum"), both=("both_row", "sum"),
    )
    bq2 = bq[bq["n_present"] >= 2]

    # ---- cell-pair conditional vs marginal ----
    cooccur_total = float((bq2["n_trig"] * bq2["n_sevnext4"] - bq2["both"]).sum())
    elig_total = float((bq2["n_trig"] * (bq2["n_present"] - 1)).sum())
    conditional = cooccur_total / elig_total if elig_total else float("nan")
    marginal = float(present["sev_next4"].mean())
    RR_cellpair = conditional / marginal if marginal else float("nan")

    # ---- building-clustered bootstrap CI for the cell-pair risk ratio ----
    bq2r = bq2.reset_index()
    bq2r["cooc"] = bq2r["n_trig"] * bq2r["n_sevnext4"] - bq2r["both"]
    bq2r["elig"] = bq2r["n_trig"] * (bq2r["n_present"] - 1)
    g_cooc = bq2r.groupby("bkey")["cooc"].sum()
    g_elig = bq2r.groupby("bkey")["elig"].sum()
    g_msum = present.groupby("bkey")["sev_next4"].sum()
    g_mcnt = present.groupby("bkey")["sev_next4"].count()
    bkeys = g_cooc.index.to_numpy()
    rng = np.random.default_rng(42)
    boots = np.empty(1000)
    for bi in range(1000):
        samp = rng.choice(bkeys, size=len(bkeys), replace=True)
        cond_b = g_cooc.reindex(samp).sum() / g_elig.reindex(samp).sum()
        marg_b = g_msum.reindex(samp).sum() / g_mcnt.reindex(samp).sum()
        boots[bi] = cond_b / marg_b if marg_b else np.nan
    rr_ci = [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))]

    # ---- building-level "any other system" ----
    trig_bq = bq2[bq2["n_trig"] >= 1]
    any_follower = ((trig_bq["n_trig"] * trig_bq["n_sevnext4"] - trig_bq["both"]) > 0)
    P_within_any = float(any_follower.mean())
    marginal_building = float((bq2["n_sevnext4"] >= 1).mean())
    RR_building = P_within_any / marginal_building if marginal_building else float("nan")

    # ---- pairwise RR matrix via co-present join ----
    T = present[present["sev"] == 1][["bkey", "period_q", C.COL_SYSTEM]].rename(columns={C.COL_SYSTEM: "A"})
    P_ = present[["bkey", "period_q", C.COL_SYSTEM, "sev_next4"]].rename(columns={C.COL_SYSTEM: "B"})
    j = T.merge(P_, on=["bkey", "period_q"])
    j = j[j["A"] != j["B"]]
    elig = j.groupby(["A", "B"], observed=True).size().rename("elig")
    cooc = j.groupby(["A", "B"], observed=True)["sev_next4"].sum().rename("cooc")
    marg_B = present.groupby(C.COL_SYSTEM, observed=True)["sev_next4"].mean()
    mat = pd.concat([elig, cooc], axis=1).reset_index()
    mat = mat[mat["elig"] >= 50]
    mat["P_B_given_A"] = mat["cooc"] / mat["elig"]
    mat["baseline_B"] = mat["B"].map(marg_B)
    mat["risk_ratio"] = mat["P_B_given_A"] / mat["baseline_B"]
    sysdesc = p.drop_duplicates(C.COL_SYSTEM).set_index(C.COL_SYSTEM)["SystemDescription"]
    mat["A_desc"] = mat["A"].map(sysdesc); mat["B_desc"] = mat["B"].map(sysdesc)
    mat = mat.sort_values("risk_ratio", ascending=False)
    mat.to_csv(C.TABLES / "coescalation_matrix.csv", index=False)

    out = {
        "n_severe_events_in_multisys_bq": int(bq2["n_trig"].sum()),
        "n_multisystem_building_quarters": int(len(bq2)),
        "mean_systems_per_building": float(p.groupby("bkey", observed=True)[C.COL_SYSTEM].nunique().mean()),
        "severity_marginal_next4": marginal,
        "cellpair_conditional_next4": conditional,
        "cellpair_risk_ratio": RR_cellpair,
        "cellpair_risk_ratio_ci95_building_clustered": rr_ci,
        "building_any_follower_prob": P_within_any,
        "building_marginal_prob": marginal_building,
        "building_risk_ratio": RR_building,
        "top_pairs": mat.head(12)[["A_desc", "B_desc", "elig", "P_B_given_A", "baseline_B", "risk_ratio"]].to_dict("records"),
        "median_pairwise_RR": float(mat["risk_ratio"].median()),
        "share_pairs_RR_gt_1": float((mat["risk_ratio"] > 1).mean()),
        "n_pairs": int(len(mat)),
    }
    json.dump(out, open(C.METRICS / "coescalation.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "top_pairs"}, indent=2))
    print("\nTop trigger->follower pairs (RR):")
    print(mat.head(12)[["A_desc", "B_desc", "elig", "P_B_given_A", "baseline_B", "risk_ratio"]].to_string(index=False))


if __name__ == "__main__":
    main()
