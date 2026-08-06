#!/usr/bin/env python
"""
45_certification_corrected.py -- certification without target-context leakage,
with the risk percentile taken at decision time and the transfer gate repaired.

Certification here differs from a single-nested implementation
(scripts/40_outer_s.py) in three ways. Each is switched independently, so that
its contribution is measured rather than assumed:

  (1) NESTING.  Published Stage 2 certified target c from the held-out
      predictions of the other campuses, but those come from f^(-d), trained on
      C\\{d} which contains c.  Corrected: read campus d through f^(-{c,d}), so
      no quantity entering S^(-c) has seen c.

  (2) RANKING.  Published R ranked scores within campus over the whole window,
      so a decision in 2017 used the 2017-2019 distribution.  Corrected: rank
      within (campus, quarter), which is what is available on the day.

  (3) TRANSFER GATE.  The share of contexts clearing two-fold was computed over
      however many contexts existed; with one context it returned 1.0 and scored
      pass.  Corrected: cap the gate at caution whenever the evaluable-context
      count falls below the data-sufficiency pass level, because a share over
      fewer contexts than that is not an estimate of transfer.

Gate definitions, eligibility filters and the aggregation rule are otherwise
copied from 40_outer_s.py unchanged.

Outputs -> results/metrics/cert_corrected.json     all four construction arms
           results/tables/gate_evidence_corrected.csv
           results/tables/outer_s_corrected.csv
           results/metrics/leakage_delta.json      the measured bias (item 2)
"""
from __future__ import annotations
import sys, json, itertools
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
MIN_CTX_FOR_TRANSFER = 6      # the data-sufficiency pass level


# --------------------------------------------------------------------------
# aggregation (unchanged from the published rule)
# --------------------------------------------------------------------------
def sufficiency(sc: dict) -> float:
    S = float(np.mean([sc[g] for g in GATES]))
    nhf = sum(1 for g in HARD if sc[g] == FAIL)
    if nhf >= 2:
        return 0.0
    if nhf >= 1:
        return min(S, 0.50)
    return S


# --------------------------------------------------------------------------
# prediction loading
# --------------------------------------------------------------------------
def load_single() -> pd.DataFrame:
    """f^(-c) scoring c.  Supplies R, exactly as the published screen does."""
    files = sorted((C.DATA_PROCESSED / "pred_single").glob("*.parquet"))
    assert len(files) == 9, f"expected 9 single fits, found {len(files)}"
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    # (2) decision-time ranking, and the published pooled ranking for contrast
    d["R_quarter"] = d.groupby(["university", "period_q"])["score"].rank(pct=True,
                                                                        method="average")
    d["R_pooled"] = d.groupby("university")["score"].rank(pct=True, method="average")
    return d


def load_pairs() -> dict[frozenset, pd.DataFrame]:
    files = sorted((C.DATA_PROCESSED / "pred_pair").glob("*.parquet"))
    assert len(files) == 36, f"expected 36 pair fits, found {len(files)}"
    out = {}
    for f in files:
        a, b = f.stem.split("__")
        out[frozenset((a, b))] = pd.read_parquet(f)
    return out


def _add_ranks(d: pd.DataFrame) -> pd.DataFrame:
    """Within-context percentile ranks, on whichever scores the frame carries.
    The risk-gradient gate reads these, so they must come from the same models
    as the rest of the development corpus."""
    d = d.copy()
    d["R_pooled"] = d.groupby("university")["score"].rank(pct=True, method="average")
    d["R_quarter"] = d.groupby(["university", "period_q"])["score"].rank(pct=True,
                                                                        method="average")
    return d


def dev_corpus(target: str, campuses: list[str], single: pd.DataFrame,
               pairs: dict, nesting: str) -> pd.DataFrame:
    """Predictions on every certifying context, under the chosen nesting."""
    if nesting == "single":          # published, leaky
        return _add_ranks(single[single.university != target])
    frames = []                      # corrected: campus d read through f^(-{c,d})
    for d in campuses:
        if d == target:
            continue
        pf = pairs[frozenset((target, d))]
        frames.append(pf[pf.university == d])
    return _add_ranks(pd.concat(frames, ignore_index=True))


# --------------------------------------------------------------------------
# gates
# --------------------------------------------------------------------------
def prior_trace_table() -> pd.DataFrame:
    """Per-campus, per-system p90 events and their zero-prior share.
    Computed from recorded histories, not from any model, so it carries no
    dependence on which campuses were used for training."""
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    thr90 = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.90)
    p["p90"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr90).fillna(np.inf)).astype(int)
    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    g = p.groupby(CELL, observed=True)["upm_count"]
    p["prior_total"] = g.shift(1).rolling(2, min_periods=2).sum() + \
                       g.shift(3).rolling(2, min_periods=2).sum()
    p["has_history"] = (p["period_q"] - 4 >= p["cell_first_q"]).astype(int)
    ev = p[(p["p90"] == 1) & (p["has_history"] == 1)].copy()
    ev["zero_prior"] = (ev["prior_total"] == 0).astype(int)
    t = (ev.groupby([C.COL_UNIV, C.COL_SYSTEM], observed=True)
           .agg(n_events=("zero_prior", "size"), n_zero=("zero_prior", "sum")).reset_index())
    t.columns = ["campus", "system", "n_events", "n_zero"]
    t["campus"] = t["campus"].astype(str)
    return t


def gates_for(dev: pd.DataFrame, trace_dev: pd.DataFrame,
              rcol: str, fix_transfer: bool) -> dict:
    out = {}
    for sysc, g in dev.groupby("system", observed=True):
        if g["y"].sum() < 20 or len(g) < 200:
            continue
        lifts = []
        for _, gg in g.groupby("university", observed=True):
            if gg["y"].sum() >= 10 and len(gg) >= 100:
                lifts.append(MET.lift_at_topk(gg["y"].to_numpy(), gg["score"].to_numpy(),
                                              C.TOPK_BUDGET))
        nc, npos = len(lifts), int(g["y"].sum())
        g1 = PASS if (nc >= 6 and npos >= 500) else (FAIL if (nc < 3 or npos < 100) else CAUTION)
        pooled_lift = MET.lift_at_topk(g["y"].to_numpy(), g["score"].to_numpy(), C.TOPK_BUDGET)
        g2 = PASS if pooled_lift >= 2.0 else (CAUTION if pooled_lift >= 1.5 else FAIL)

        frac2x = float(np.mean([l >= 2 for l in lifts])) if lifts else np.nan
        g3 = PASS if frac2x >= 0.8 else (CAUTION if frac2x >= 0.6 else FAIL)
        if np.isnan(frac2x):
            g3 = CAUTION
        # (3) a share over too few contexts is not a transfer estimate
        if fix_transfer and nc < MIN_CTX_FOR_TRANSFER:
            g3 = min(g3, CAUTION)

        d = g.dropna(subset=[rcol])
        if len(d) < 200 or d["y"].sum() < 30:
            g4, grad = CAUTION, {}
        else:
            q = pd.qcut(d[rcol], 5, labels=False, duplicates="drop")
            obs = d.groupby(q)["y"].mean(); base = d["y"].mean()
            top = float(obs.iloc[-1] / base) if base > 0 else np.nan
            rho, _ = spearmanr(obs.index.values, obs.values)
            g4 = PASS if (top >= 2.0 and rho >= 0.9) else (CAUTION if (top >= 1.5 and rho >= 0.5) else FAIL)
            grad = {"top_band": round(top, 2), "rho": round(float(rho), 2)}

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
                     "pooled_lift": round(float(pooled_lift), 2),
                     "frac_2x": None if np.isnan(frac2x) else round(frac2x, 3),
                     "zero_prior_share": None if (isinstance(zp, float) and np.isnan(zp)) else round(float(zp), 3),
                     **grad}
    return out


# --------------------------------------------------------------------------
# routing
# --------------------------------------------------------------------------
def route(single: pd.DataFrame, outer: dict, campuses: list[str], rcol: str) -> dict:
    """Route every scored unit.  Units in strata with no held-out gate evidence
    fail data sufficiency by default and therefore route as uncertified, which
    is what a portfolio applying the method would inherit."""
    routes = {"pma": 0, "verify": 0, "continue": 0, "watch": 0}
    qrows, gated, ungated = [], 0, 0
    for j in campuses:
        sub = single[single.university == j].copy()
        Smap = {s: v["S"] for s, v in outer[j].items()}
        sub["S"] = sub.system.map(Smap)
        ungated += int(sub["S"].isna().sum()); gated += int(sub["S"].notna().sum())
        sub["S"] = sub["S"].fillna(0.0)          # no evidence -> not certified
        hiR, hiS = sub[rcol] >= R_CUT, sub.S >= S_CUT
        routes["pma"] += int((hiR & hiS).sum()); routes["verify"] += int((hiR & ~hiS).sum())
        routes["continue"] += int((~hiR & hiS).sum()); routes["watch"] += int((~hiR & ~hiS).sum())
        q = sub[hiR & ~hiS]
        if len(q):
            qq = q.groupby("system").size().reset_index(name="n"); qq["campus"] = j
            qrows.append(qq)
    n = sum(routes.values())
    queue = (pd.concat(qrows).groupby("system")["n"].sum().sort_values(ascending=False)
             if qrows else pd.Series(dtype=int))
    hi = routes["pma"] + routes["verify"]
    return {"route_counts": routes, "n_routed": n,
            "route_shares": {k: round(v / n, 4) for k, v in routes.items()},
            "n_high_risk": hi,
            "share_high_risk_cleared": round(routes["pma"] / hi, 4) if hi else None,
            "units_in_gated_strata": gated, "units_in_ungated_strata": ungated,
            "queue_by_system": {s: int(v) for s, v in queue.items()}}


# --------------------------------------------------------------------------
def build(nesting: str, rcol: str, fix_transfer: bool,
          single: pd.DataFrame, pairs: dict, trace: pd.DataFrame,
          campuses: list[str]) -> dict:
    outer = {}
    for j in campuses:
        dev = dev_corpus(j, campuses, single, pairs, nesting)
        outer[j] = gates_for(dev, trace[trace.campus != j], rcol, fix_transfer)
    pooled = gates_for(single, trace, rcol, fix_transfer)
    r = route(single, outer, campuses, rcol)
    return {"pooled": pooled, "outer": outer, "routing": r}


def main() -> None:
    single, pairs, trace = load_single(), load_pairs(), prior_trace_table()
    campuses = sorted(single.university.unique())

    arms = {
        # single-nested construction, reproduced here for the delta
        "single_nested":  dict(nesting="single", rcol="R_pooled",  fix_transfer=False),
        # each change switched on alone, so its contribution is attributable
        "nesting_only":  dict(nesting="double", rcol="R_pooled",  fix_transfer=False),
        "ranking_only":  dict(nesting="single", rcol="R_quarter", fix_transfer=False),
        "transfer_only": dict(nesting="single", rcol="R_pooled",  fix_transfer=True),
        # all three: the corrected pipeline the paper will report
        "corrected":  dict(nesting="double", rcol="R_quarter", fix_transfer=True),
    }
    res = {}
    for name, kw in arms.items():
        print(f"[cert] building arm: {name} ({kw})", flush=True)
        res[name] = build(single=single, pairs=pairs, trace=trace,
                          campuses=campuses, **kw)

    # ---------------- the measured leakage / ranking / gate deltas -------------
    def certmap(arm):
        return {j: {s: v["S"] for s, v in res[arm]["outer"][j].items()} for j in campuses}

    def compare(a, b):
        A, B = certmap(a), certmap(b)
        keys = sorted({(j, s) for j in campuses for s in set(A[j]) | set(B[j])})
        dS, flips = [], []
        for j, s in keys:
            sa, sb = A[j].get(s), B[j].get(s)
            if sa is None or sb is None:
                continue
            dS.append(sb - sa)
            ca, cb = sa >= S_CUT, sb >= S_CUT
            if ca != cb:
                flips.append({"campus": j, "system": s, "S_from": sa, "S_to": sb,
                              "certified_from": ca, "certified_to": cb})
        ra, rb = res[a]["routing"], res[b]["routing"]
        return {
            "n_stratum_fold_pairs": len(dS),
            "max_abs_delta_S": round(float(np.max(np.abs(dS))), 4) if dS else 0.0,
            "mean_abs_delta_S": round(float(np.mean(np.abs(dS))), 5) if dS else 0.0,
            "n_certification_flips": len(flips), "flips": flips,
            "route_counts_from": ra["route_counts"], "route_counts_to": rb["route_counts"],
            "delta_route_counts": {k: rb["route_counts"][k] - ra["route_counts"][k]
                                   for k in ra["route_counts"]},
            "share_high_risk_cleared_from": ra["share_high_risk_cleared"],
            "share_high_risk_cleared_to": rb["share_high_risk_cleared"],
        }

    deltas = {
        "leakage_only__single_nested_vs_nesting_only": compare("single_nested", "nesting_only"),
        "ranking_only__single_nested_vs_ranking_only": compare("single_nested", "ranking_only"),
        "transfer_gate__single_nested_vs_transfer_only": compare("single_nested", "transfer_only"),
        "all_three__single_nested_vs_corrected": compare("single_nested", "corrected"),
    }
    json.dump(deltas, open(C.METRICS / "leakage_delta.json", "w"), indent=2, default=str)

    # ---------------- publishable tables for the corrected arm ---------------
    cor = res["corrected"]
    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    share = dict(zip(burden.SystemCode, burden.share_of_total_upm_labour))
    name = {k: str(v).strip() for k, v in
            burden[["SystemCode", "SystemDescription"]].values}

    rows = []
    for s in sorted(cor["pooled"]):
        p = cor["pooled"][s]
        Ss = [cor["outer"][j][s]["S"] for j in campuses if s in cor["outer"][j]]
        rows.append({
            "system": s, "system_desc": name.get(s, s),
            "burden_share": round(float(share.get(s, np.nan)), 4),
            "campuses_evaluable": p["n_campuses"], "n_pos": p["n_pos"],
            "pooled_lift_top10": p["pooled_lift"], "frac_campuses_2x": p["frac_2x"],
            "top_band_enrichment": p.get("top_band"), "band_spearman": p.get("rho"),
            "zero_prior_share": p["zero_prior_share"],
            **{g: p["scores"][g] for g in GATES},
            "S_pooled": round(p["S"], 2),
            "S_median_outer": round(float(np.median(Ss)), 2) if Ss else np.nan,
            "S_min": round(min(Ss), 2) if Ss else np.nan,
            "S_max": round(max(Ss), 2) if Ss else np.nan,
            "folds_certified": int(sum(x >= S_CUT for x in Ss)), "folds_gated": len(Ss),
        })
    ev = pd.DataFrame(rows).sort_values("burden_share", ascending=False)
    ev.to_csv(C.TABLES / "gate_evidence_corrected.csv", index=False)
    ev[["system", "system_desc", "S_pooled", "S_median_outer", "S_min", "S_max",
        "folds_certified", "folds_gated"]].to_csv(C.TABLES / "outer_s_corrected.csv", index=False)

    json.dump({k: {"pooled": {s: {"S": v["S"], "scores": v["scores"],
                                  "n_campuses": v["n_campuses"], "n_pos": v["n_pos"],
                                  "pooled_lift": v["pooled_lift"], "frac_2x": v["frac_2x"],
                                  "zero_prior_share": v["zero_prior_share"]}
                              for s, v in res[k]["pooled"].items()},
                   "routing": res[k]["routing"],
                   "outer_S": {j: {s: v["S"] for s, v in res[k]["outer"][j].items()}
                               for j in campuses}}
               for k in arms}, open(C.METRICS / "cert_corrected.json", "w"), indent=2)

    # ---------------- console summary ----------------
    pd.set_option("display.width", 240, "display.max_columns", 40)
    print("\n=== CORRECTED gate evidence ===")
    print(ev[["system_desc", "burden_share", "campuses_evaluable", "n_pos",
              "pooled_lift_top10", "frac_campuses_2x", "zero_prior_share",
              "S_pooled", "folds_certified", "folds_gated"]].to_string(index=False))
    print("\n=== ROUTING by arm ===")
    for k in arms:
        r = res[k]["routing"]
        print(f"  {k:14s} pma={r['route_counts']['pma']:6d} verify={r['route_counts']['verify']:5d} "
              f"cleared={r['share_high_risk_cleared']} n={r['n_routed']}")
    print("\n=== MEASURED DELTAS vs single-nested ===")
    for k, v in deltas.items():
        print(f"  {k}: flips={v['n_certification_flips']}, max|dS|={v['max_abs_delta_S']}, "
              f"droutes={v['delta_route_counts']}")


if __name__ == "__main__":
    main()
