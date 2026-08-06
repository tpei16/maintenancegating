#!/usr/bin/env python
"""
Analysis X+ (C1 defence) — robustness of cross-system co-escalation.

This script stress-tests the headline co-escalation finding (cell-pair RR 2.76)
against the reviewer attack that it is merely a proxy for building age, building
size, or reporting intensity, and characterises its temporal structure. It reuses
the EXACT severity definition and cell-pair risk-ratio estimator of
scripts/16_coescalation.py, so the full-panel RR here reproduces the published value.

Checks (Improvement Plan T1.1, T2.1):
  (1) Building-AGE stratification (BuiltYear tertiles, ~24% coverage). Within each
      age stratum the RR is recomputed against the STRATUM'S OWN marginal, so an
      age-driven baseline cannot inflate it. If RR stays elevated in every stratum
      (incl. the newest), age is not the driver.
  (2) WITHIN-BUILDING control (Mantel-Haenszel RR, building strata). Each building
      is its own control: P(follower severe | a sibling severe at t) vs the SAME
      building's rate when no sibling is severe. This removes ALL time-invariant
      building confounds (age, occupancy, envelope) by construction.
  (3) LAG-DECAY profile: RR at exactly t+1, t+2, ... t+8 (vs the single-quarter
      marginal). A decaying profile is a temporal signal, not a static trait.
  (4) Building-SIZE and ACTIVITY normalisation: RR within tertiles of (a) number of
      systems present and (b) total UPM work-order volume per building. Persistence
      across strata rules out a mechanical reporting-intensity artefact.

Outputs -> results/metrics/coescalation_robustness.json
           results/tables/coescalation_lag_decay.csv
           results/tables/coescalation_strata.csv
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, io as F

CELL = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]
RNG_SEED = 42
N_BOOT = 1000


# --------------------------------------------------------------------------- #
# Severity + forward windows (identical construction to script 16)
# --------------------------------------------------------------------------- #
def prepare(panel: pd.DataFrame) -> pd.DataFrame:
    p = panel.copy()
    p["bkey"] = p[C.COL_UNIV].astype("string") + "/" + p[C.COL_BUILDING].astype("string")
    thr = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.75)
    p["sev"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr).fillna(np.inf)).astype(int)
    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    g = p.groupby(CELL, observed=True)["sev"]
    # pooled t+1..t+4 (matches script 16)
    nxt = None
    for w in (1, 2, 3, 4):
        s = g.shift(-w)
        nxt = s if nxt is None else np.fmax(nxt, s)
    p["sev_next4"] = (pd.Series(nxt, index=p.index).fillna(0) > 0).astype(int)
    # exact single-lag forward severities for the decay profile
    for L in range(1, 9):
        p[f"sev_lag{L}"] = (g.shift(-L).fillna(0) > 0).astype(int)
        p[f"valid_lag{L}"] = (p["period_q"] + L <= p["building_last_q"]).astype(int)
    p["win_valid"] = (p["period_q"] + 4 <= p["building_last_q"]).astype(int)
    return p


# --------------------------------------------------------------------------- #
# Cell-pair RR (script-16 estimator) on an arbitrary subset of buildings,
# with the marginal computed WITHIN the subset. Optional building-clustered CI.
# --------------------------------------------------------------------------- #
def cellpair_rr(present: pd.DataFrame, outcome="sev_next4", n_boot=0):
    present = present.copy()
    present["both_row"] = ((present["sev"] == 1) & (present[outcome] == 1)).astype(int)
    bq = present.groupby(["bkey", "period_q"], observed=True).agg(
        n_present=("sev", "size"), n_trig=("sev", "sum"),
        n_out=(outcome, "sum"), both=("both_row", "sum"),
    )
    bq2 = bq[bq["n_present"] >= 2]
    cooc = float((bq2["n_trig"] * bq2["n_out"] - bq2["both"]).sum())
    elig = float((bq2["n_trig"] * (bq2["n_present"] - 1)).sum())
    conditional = cooc / elig if elig else float("nan")
    marginal = float(present[outcome].mean())
    rr = conditional / marginal if marginal else float("nan")
    out = {"rr": rr, "conditional": conditional, "marginal": marginal,
           "n_trigger_events": int(bq2["n_trig"].sum()),
           "n_multisys_bq": int(len(bq2)), "n_buildings": int(present["bkey"].nunique())}
    if n_boot:
        bq2r = bq2.reset_index()
        bq2r["cooc"] = bq2r["n_trig"] * bq2r["n_out"] - bq2r["both"]
        bq2r["elig"] = bq2r["n_trig"] * (bq2r["n_present"] - 1)
        g_cooc = bq2r.groupby("bkey")["cooc"].sum()
        g_elig = bq2r.groupby("bkey")["elig"].sum()
        g_msum = present.groupby("bkey")[outcome].sum()
        g_mcnt = present.groupby("bkey")[outcome].count()
        bkeys = g_cooc.index.to_numpy()
        rng = np.random.default_rng(RNG_SEED)
        boots = np.empty(n_boot)
        for bi in range(n_boot):
            samp = rng.choice(bkeys, size=len(bkeys), replace=True)
            cb = g_cooc.reindex(samp).sum() / max(g_elig.reindex(samp).sum(), 1e-9)
            mb = g_msum.reindex(samp).sum() / max(g_mcnt.reindex(samp).sum(), 1e-9)
            boots[bi] = cb / mb if mb else np.nan
        out["ci95"] = [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))]
    return out


# --------------------------------------------------------------------------- #
# (2) Within-building Mantel-Haenszel RR — each building its own control
# --------------------------------------------------------------------------- #
def within_building_mh(present: pd.DataFrame, n_boot=N_BOOT):
    """Exposure = >=1 sibling system severe at t; outcome = this cell severe in
    t+1..t+4. RR pooled across building strata (Mantel-Haenszel), so every
    comparison is within a single building."""
    df = present[["bkey", "period_q", "sev", "sev_next4"]].copy()
    # siblings severe at t in the same building-quarter (exclude self)
    bq_trig = df.groupby(["bkey", "period_q"], observed=True)["sev"].transform("sum")
    bq_n = df.groupby(["bkey", "period_q"], observed=True)["sev"].transform("size")
    df = df[bq_n >= 2]                                   # cells with >=1 sibling
    sib_trig = (bq_trig - df["sev"]).loc[df.index]
    df["exposed"] = (sib_trig >= 1).astype(int)
    df["y"] = df["sev_next4"].astype(int)
    # per-building 2x2 cells
    grp = df.groupby("bkey", observed=True)
    agg = grp.apply(lambda g: pd.Series({
        "a": int(((g.exposed == 1) & (g.y == 1)).sum()),   # exposed, outcome
        "b": int(((g.exposed == 1) & (g.y == 0)).sum()),   # exposed, no outcome
        "c": int(((g.exposed == 0) & (g.y == 1)).sum()),   # unexposed, outcome
        "d": int(((g.exposed == 0) & (g.y == 0)).sum()),   # unexposed, no outcome
    })).reset_index()
    agg["N"] = agg[["a", "b", "c", "d"]].sum(axis=1)
    agg = agg[(agg["a"] + agg["b"] > 0) & (agg["c"] + agg["d"] > 0)]  # informative strata

    def mh_rr(a):
        num = (a["a"] * (a["c"] + a["d"]) / a["N"]).sum()
        den = (a["c"] * (a["a"] + a["b"]) / a["N"]).sum()
        return num / den if den else float("nan")

    point = mh_rr(agg)
    # pooled (ignoring building) for contrast
    A, B = agg["a"].sum(), agg["b"].sum()
    Cc, D = agg["c"].sum(), agg["d"].sum()
    pooled = ((A / (A + B)) / (Cc / (Cc + D))) if (A + B) and (Cc + D) else float("nan")
    # building-clustered bootstrap
    keys = agg["bkey"].to_numpy()
    rng = np.random.default_rng(RNG_SEED)
    arr = agg.set_index("bkey")
    boots = np.empty(n_boot)
    for bi in range(n_boot):
        samp = rng.choice(keys, size=len(keys), replace=True)
        s = arr.loc[samp]
        boots[bi] = mh_rr(s)
    ci = [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))]
    return {"mh_rr_within_building": float(point), "pooled_rr_contrast": float(pooled),
            "ci95": ci, "n_informative_buildings": int(len(agg)),
            "exposed_rate": float(A / (A + B)) if (A + B) else float("nan"),
            "unexposed_rate": float(Cc / (Cc + D)) if (Cc + D) else float("nan")}


# --------------------------------------------------------------------------- #
# (3) Lag-decay profile
# --------------------------------------------------------------------------- #
def lag_decay(p: pd.DataFrame):
    rows = []
    for L in range(1, 9):
        present = p[p[f"valid_lag{L}"] == 1].copy()
        r = cellpair_rr(present, outcome=f"sev_lag{L}", n_boot=0)
        rows.append({"lag_quarters": L, "rr": r["rr"], "conditional": r["conditional"],
                     "marginal": r["marginal"], "n_buildings": r["n_buildings"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Building-level attributes for stratification
# --------------------------------------------------------------------------- #
def building_attributes(panel: pd.DataFrame) -> pd.DataFrame:
    bk = panel.assign(bkey=panel[C.COL_UNIV].astype("string") + "/" +
                      panel[C.COL_BUILDING].astype("string"))
    attr = bk.groupby("bkey", observed=True).agg(
        n_systems=(C.COL_SYSTEM, "nunique"),
        upm_volume=("upm_count", "sum"),
    ).reset_index()
    # building age (median BuiltYear per building from raw, like script 13)
    raw = F.load_raw(usecols=[C.COL_UNIV, C.COL_BUILDING, C.COL_BUILT_YEAR], verbose=False)
    raw = raw.dropna(subset=[C.COL_BUILDING])
    age = (raw.groupby([C.COL_UNIV, C.COL_BUILDING], observed=True)[C.COL_BUILT_YEAR]
              .median().reset_index())
    age["bkey"] = age[C.COL_UNIV].astype("string") + "/" + age[C.COL_BUILDING].astype("string")
    attr = attr.merge(age[["bkey", C.COL_BUILT_YEAR]].rename(columns={C.COL_BUILT_YEAR: "built_year"}),
                      on="bkey", how="left")
    attr["building_age_2021"] = 2021 - attr["built_year"]
    return attr


def stratified_rr(p, present, attr, col, labels_fn, label_name, n_boot=600):
    """RR within strata defined by a building attribute (marginal computed within stratum)."""
    pres = present.merge(attr[["bkey", col]], on="bkey", how="left")
    rows = []
    strata = labels_fn(attr[col].dropna())
    for name, (lo, hi) in strata.items():
        sub = pres[(pres[col] >= lo) & (pres[col] < hi)]
        if sub["bkey"].nunique() < 30 or sub["sev"].sum() < 50:
            continue
        r = cellpair_rr(sub, outcome="sev_next4", n_boot=n_boot)
        rows.append({"dimension": label_name, "stratum": name,
                     "range": f"[{lo:.0f}, {hi:.0f})", "rr": r["rr"],
                     "conditional": r["conditional"], "marginal": r["marginal"],
                     "ci_lo": r.get("ci95", [np.nan, np.nan])[0],
                     "ci_hi": r.get("ci95", [np.nan, np.nan])[1],
                     "n_buildings": r["n_buildings"], "n_trigger_events": r["n_trigger_events"]})
    return rows


def tertile_bounds(s: pd.Series):
    q1, q2 = s.quantile(1/3), s.quantile(2/3)
    return {"low (oldest/smallest)": (-np.inf, q1),
            "mid": (q1, q2),
            "high (newest/largest)": (q2, np.inf)}


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    p = prepare(panel)
    present = p[p["win_valid"] == 1].copy()

    out = {}
    # full-panel reproduction (sanity vs published 2.755)
    full = cellpair_rr(present, "sev_next4", n_boot=N_BOOT)
    out["full_panel_cellpair_rr"] = full
    print(f"[check] full-panel cell-pair RR = {full['rr']:.3f} "
          f"(published 2.755), CI {full.get('ci95')}", flush=True)

    # (2) within-building MH
    out["within_building"] = within_building_mh(present)
    print(f"[within-building] MH RR = {out['within_building']['mh_rr_within_building']:.3f} "
          f"CI {out['within_building']['ci95']} "
          f"(exposed {out['within_building']['exposed_rate']:.3f} vs "
          f"unexposed {out['within_building']['unexposed_rate']:.3f})", flush=True)

    # (3) lag decay
    decay = lag_decay(p)
    decay.to_csv(C.TABLES / "coescalation_lag_decay.csv", index=False)
    out["lag_decay"] = decay.to_dict("records")
    print("[lag-decay]\n" + decay.to_string(index=False), flush=True)

    # (1)+(4) stratified RRs
    attr = building_attributes(panel)
    out["age_coverage_buildings"] = float(attr["built_year"].notna().mean())
    strata_rows = []
    strata_rows += stratified_rr(p, present, attr, "built_year", tertile_bounds, "building_age")
    strata_rows += stratified_rr(p, present, attr, "n_systems", tertile_bounds, "n_systems")
    strata_rows += stratified_rr(p, present, attr, "upm_volume", tertile_bounds, "upm_volume")
    strata = pd.DataFrame(strata_rows)
    strata.to_csv(C.TABLES / "coescalation_strata.csv", index=False)
    out["strata"] = strata_rows
    print("[strata]\n" + strata.to_string(index=False), flush=True)

    # also report the built-year tertile cut points and implied ages for the text
    by = attr["built_year"].dropna()
    out["built_year_tertile_cuts"] = [float(by.quantile(1/3)), float(by.quantile(2/3))]

    json.dump(out, open(C.METRICS / "coescalation_robustness.json", "w"), indent=2, default=float)
    print("\n[done] -> results/metrics/coescalation_robustness.json", flush=True)


if __name__ == "__main__":
    main()
