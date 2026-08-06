#!/usr/bin/env python
"""
40_outer_s.py -- nested outer-campus derivation of the sufficiency score S (WP1),
with the revised antecedent gate (WP2: no-antecedent share only) and the renamed
risk-gradient gate (WP3).

Design:
  For each outer campus j: every gate quantity is recomputed using ONLY the other
  campuses' held-out predictions and histories (development corpus). S_s^(-j) is
  frozen, then every scored unit of campus j is routed with R_ij (already
  leakage-controlled under LOUO) and S_s^(-j). Gate definitions mirror
  07_decomposition_heterogeneity.py (eligibility: system n>=200 & n_pos>=20;
  per-campus evaluable: n>=100 & n_pos>=10) and 18_trajectory.py (p90 events with
  >=4 prior quarters; zero-prior = no work orders in the prior 4 quarters).

Revised gate 5 (prior-record trace): pass if no-antecedent share < 0.10,
  caution 0.10-0.25, fail >= 0.25; <30 events -> caution.
Gate 4 renamed risk-gradient reliability (same spec: monotone score-band outcome
  rates, top-band enrichment).

Outputs:
  results/metrics/outer_s.json      per-system S by fold, median/range, folds sufficient
  results/tables/outer_s_by_system.csv
  results/metrics/outer_routes.json outer-fold route counts + queue composition
  results/tables/gate_evidence.csv  raw auditable gate quantities (pooled + outer-fold stability)
"""
from __future__ import annotations
import sys, json, glob
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import metrics as MET

PASS, CAUTION, FAIL = 1.0, 0.5, 0.0
GATES = ["data_sufficiency", "risk_concentration", "transfer_stability",
         "risk_gradient", "prior_record_trace"]
HARD = ["data_sufficiency", "transfer_stability", "risk_gradient"]
CELL = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]
R_CUT, S_CUT = 0.90, 0.67


def sufficiency(sc):
    S = float(np.mean([sc[g] for g in GATES]))
    nhf = sum(1 for g in HARD if sc[g] == FAIL)
    if nhf >= 2:
        return 0.0
    if nhf >= 1:
        return min(S, 0.50)
    return S


def load_preds():
    files = sorted((C.DATA_PROCESSED / "pred_louo").glob("*.parquet"))
    preds = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    prim = preds[preds.cfg_id == "severity_labour|p75|M1|gbdt"].copy()
    prim["R"] = prim.groupby("held_out_university")["score"].rank(pct=True, method="average")
    return prim


def prior_trace_table():
    """Per-campus, per-system: p90 events with >=4q history and their zero-prior share
    (exact replication of 18_trajectory.py, retaining campus identity)."""
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    thr90 = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.90)
    p["p90"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr90).fillna(np.inf)).astype(int)
    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    g = p.groupby(CELL, observed=True)["upm_count"]
    prior_recent = g.shift(1).rolling(2, min_periods=2).sum()
    prior_older = g.shift(3).rolling(2, min_periods=2).sum()
    p["prior_total"] = prior_recent + prior_older
    p["has_history"] = (p["period_q"] - 4 >= p["cell_first_q"]).astype(int)
    ev = p[(p["p90"] == 1) & (p["has_history"] == 1)].copy()
    ev["zero_prior"] = (ev["prior_total"] == 0).astype(int)
    t = (ev.groupby([C.COL_UNIV, C.COL_SYSTEM], observed=True)
         .agg(n_events=("zero_prior", "size"), n_zero=("zero_prior", "sum")).reset_index())
    t.columns = ["campus", "system", "n_events", "n_zero"]
    t["campus"] = t["campus"].astype(str)
    return t


def gates_for(dev, trace_dev):
    """Compute all five gates per system from a development corpus of predictions."""
    out = {}
    for sysc, g in dev.groupby("system", observed=True):
        if g["y"].sum() < 20 or len(g) < 200:
            continue  # ineligible -> not gated (fails data sufficiency by default)
        # per-campus lifts (transfer gate)
        lifts = []
        for u, gg in g.groupby("held_out_university", observed=True):
            if gg["y"].sum() >= 10 and len(gg) >= 100:
                lifts.append(MET.lift_at_topk(gg["y"].to_numpy(), gg["score"].to_numpy(), C.TOPK_BUDGET))
        nc, npos = len(lifts), int(g["y"].sum())
        g1 = PASS if (nc >= 6 and npos >= 500) else (FAIL if (nc < 3 or npos < 100) else CAUTION)
        pooled_lift = MET.lift_at_topk(g["y"].to_numpy(), g["score"].to_numpy(), C.TOPK_BUDGET)
        g2 = PASS if pooled_lift >= 2.0 else (CAUTION if pooled_lift >= 1.5 else FAIL)
        frac2x = float(np.mean([l >= 2 for l in lifts])) if lifts else np.nan
        g3 = PASS if frac2x >= 0.8 else (CAUTION if frac2x >= 0.6 else FAIL)
        # risk-gradient gate on within-campus percentile ranks
        d = g.dropna(subset=["R"])
        if len(d) < 200 or d["y"].sum() < 30:
            g4, grad = CAUTION, {}
        else:
            q = pd.qcut(d["R"], 5, labels=False, duplicates="drop")
            obs = d.groupby(q)["y"].mean()
            base = d["y"].mean()
            top = float(obs.iloc[-1] / base) if base > 0 else np.nan
            rho, _ = spearmanr(obs.index.values, obs.values)
            g4 = PASS if (top >= 2.0 and rho >= 0.9) else (CAUTION if (top >= 1.5 and rho >= 0.5) else FAIL)
            grad = {"top_band": round(top, 2), "rho": round(float(rho), 2)}
        # prior-record trace gate (revised)
        tr = trace_dev[trace_dev.system == sysc]
        ne, nz = int(tr.n_events.sum()), int(tr.n_zero.sum())
        if ne < 30:
            g5, zp = CAUTION, np.nan
        else:
            zp = nz / ne
            g5 = PASS if zp < 0.10 else (CAUTION if zp < 0.25 else FAIL)
        sc = {"data_sufficiency": g1, "risk_concentration": g2, "transfer_stability": g3,
              "risk_gradient": g4, "prior_record_trace": g5}
        out[sysc] = {"scores": sc, "S": sufficiency(sc), "n_campuses": nc, "n_pos": npos,
                     "pooled_lift": round(float(pooled_lift), 2), "frac_2x": None if np.isnan(frac2x) else round(frac2x, 3),
                     "zero_prior_share": None if (isinstance(zp, float) and np.isnan(zp)) else round(float(zp), 3),
                     **grad}
    return out


def main():
    prim = load_preds()
    trace = prior_trace_table()
    campuses = sorted(prim["held_out_university"].astype(str).unique())
    name = dict(pd.read_csv(C.TABLES / "burden_by_system.csv")[["SystemCode", "SystemDescription"]].values)
    name = {k: str(v).strip() for k, v in name.items()}

    # ---------- pooled gates (deployment map; revised gate spec) ----------
    pooled = gates_for(prim, trace)

    # ---------- outer-fold derivation ----------
    outer = {}   # fold j -> {system: {...}}
    for j in campuses:
        dev = prim[prim.held_out_university.astype(str) != j]
        tdev = trace[trace.campus != j]
        outer[j] = gates_for(dev, tdev)

    systems = sorted(pooled.keys())
    rows = []
    for s in systems:
        Ss = [outer[j][s]["S"] for j in campuses if s in outer[j]]
        rows.append({
            "system": s, "system_desc": name.get(s, s),
            "S_pooled": round(pooled[s]["S"], 2),
            "S_median_outer": round(float(np.median(Ss)), 2) if Ss else np.nan,
            "S_min": round(min(Ss), 2) if Ss else np.nan,
            "S_max": round(max(Ss), 2) if Ss else np.nan,
            "folds_gated": len(Ss),
            "folds_sufficient": int(sum(x >= S_CUT for x in Ss)),
        })
    tab = pd.DataFrame(rows)

    # ---------- route campus j with frozen S^(−j) ----------
    routes = {"pma": 0, "verify": 0, "continue": 0, "watch": 0}
    qrows = []
    for j in campuses:
        sub = prim[prim.held_out_university.astype(str) == j]
        Smap = {s: v["S"] for s, v in outer[j].items()}
        sub = sub[sub.system.isin(Smap)].copy()
        sub["S"] = sub.system.map(Smap)
        hiR, hiS = sub.R >= R_CUT, sub.S >= S_CUT
        routes["pma"] += int((hiR & hiS).sum()); routes["verify"] += int((hiR & ~hiS).sum())
        routes["continue"] += int((~hiR & hiS).sum()); routes["watch"] += int((~hiR & ~hiS).sum())
        q = sub[hiR & ~hiS]
        if len(q):
            qq = q.groupby("system").size().reset_index(name="n"); qq["campus"] = j
            qrows.append(qq)
    n_routed = sum(routes.values())
    queue = (pd.concat(qrows).groupby("system")["n"].sum().sort_values(ascending=False)
             if qrows else pd.Series(dtype=int))

    # ---------- gate evidence table (auditable raw quantities, pooled) ----------
    ev = []
    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    share = dict(zip(burden.SystemCode, burden.share_of_total_upm_labour))
    for s in systems:
        p = pooled[s]
        r = tab[tab.system == s].iloc[0]
        ev.append({
            "system": s, "system_desc": name.get(s, s),
            "burden_share": round(float(share.get(s, np.nan)), 4),
            "campuses_evaluable": p["n_campuses"], "n_pos": p["n_pos"],
            "pooled_lift_top10": p["pooled_lift"], "frac_campuses_2x": p["frac_2x"],
            "top_band_enrichment": p.get("top_band"), "band_spearman": p.get("rho"),
            "zero_prior_share": p["zero_prior_share"],
            **{g: p["scores"][g] for g in GATES},
            "S_pooled": round(p["S"], 2),
            "S_median_outer": r["S_median_outer"], "S_range": f"{r['S_min']}-{r['S_max']}",
            "folds_sufficient": f"{r['folds_sufficient']}/{r['folds_gated']}",
        })
    evtab = pd.DataFrame(ev).sort_values("burden_share", ascending=False)
    evtab.to_csv(C.TABLES / "gate_evidence.csv", index=False)
    tab.to_csv(C.TABLES / "outer_s_by_system.csv", index=False)
    json.dump({"outer_S": {j: {s: v["S"] for s, v in outer[j].items()} for j in campuses}},
              open(C.METRICS / "outer_s.json", "w"), indent=2)
    json.dump({"design": "nested outer-campus: S derived per fold from the other campuses only, frozen, then routes campus j",
               "gate5": "prior-record trace (zero-prior share): pass<0.10, caution 0.10-0.25, fail>=0.25",
               "n_routed_units": int(n_routed), "route_counts": routes,
               "route_shares": {k: round(v / n_routed, 4) for k, v in routes.items()},
               "verification_queue_by_system": {s: int(n) for s, n in queue.items()},
               }, open(C.METRICS / "outer_routes.json", "w"), indent=2)

    pd.set_option("display.width", 220, "display.max_columns", 40)
    print("=== POOLED S (revised gates) vs OUTER-FOLD ===")
    print(tab.to_string(index=False))
    print("\n=== OUTER-FOLD ROUTES ===", routes, " n=", n_routed)
    print("\nverification queue by system:", dict(queue))
    print("\n=== GATE EVIDENCE (pooled) ===")
    print(evtab[["system", "burden_share", "campuses_evaluable", "n_pos", "pooled_lift_top10",
                 "frac_campuses_2x", "top_band_enrichment", "zero_prior_share", "S_pooled",
                 "folds_sufficient"]].to_string(index=False))


if __name__ == "__main__":
    main()
