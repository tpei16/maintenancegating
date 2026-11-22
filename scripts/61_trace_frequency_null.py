#!/usr/bin/env python
"""
61_trace_frequency_null.py -- is the no-antecedent share of a building system a
property of its extreme events, or of how rarely the system is serviced?

The prior-record-trace gate of 45_certification_corrected.py scores, per
stratum, the share of extreme events (cell-quarters whose reactive labour
exceeds the system's 90th percentile of positive reactive labour) that follow
four quarters with no reactive work order on the cell.  A system that is
serviced rarely will show a high share even if extreme events are no less
announced than ordinary reactive quarters.  This script sets the observed share
beside references that hold service frequency fixed, and checks whether the
share depends on where the record starts.

Per stratum with at least 30 extreme events on the pooled panel:

  observed        the no-antecedent share of the extreme events, exactly the
                  quantity trace_for(None) of script 45 aggregates (asserted).
  with PM         the same events, counting preventive work orders (ppm_count)
                  as prior activity as well.
  reference A     among cell-quarters of the stratum with at least one reactive
                  work order and four quarters of cell history, the share with
                  no reactive work order in the preceding four quarters,
                  computed per campus and weighted to the campus mix of the
                  extreme events.  The ratio observed / reference A near one
                  means extreme events are preceded by reactive work no more
                  often than ordinary reactive quarters are.
  reference B     the same over all cell-quarters with four quarters of
                  history, active or not, with the same weighting.
  record start    the observed share after dropping events in the first 4, 8
                  and 12 quarters of each campus's record (the campus's first
                  panel quarter), with event counts and the gate verdict.  Four
                  quarters of cell history already exclude the first four
                  quarters of every campus, so the 4-quarter row equals the
                  observed share by construction.  A second version measures
                  the start from each cell's own first quarter.
  frequency       mean reactive work orders per cell-quarter in the stratum.

Intervals are building-clustered percentile bootstraps: buildings are resampled
with replacement within each campus (the set of campuses is fixed), and every
quantity, including the campus weights, is recomputed on the resampled
buildings.  Gate verdicts use the thresholds of script 45 (pass < 0.10,
caution < 0.25, fail otherwise; caution below 30 events).

Outputs -> results/metrics/trace_frequency_null.json
           results/tables/trace_frequency_null.csv
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"

import sys, json, time, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C

_spec = importlib.util.spec_from_file_location(
    "cert_corrected", ROOT / "scripts" / "45_certification_corrected.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

R_CUT, S_CUT = cc.R_CUT, cc.S_CUT
CELL = cc.CELL
MIN_EVENTS = 30
RECORD_START_Q = (4, 8, 12)
N_BOOT = C.N_BOOTSTRAP


# --------------------------------------------------------------------------
# reference check: the main pipeline this analysis is read against
# --------------------------------------------------------------------------
def check_reference() -> dict:
    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    res = cc.build(nesting="double", rcol="R_quarter", fix_transfer=True,
                   single=single, pairs=pairs, campuses=campuses)
    ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]
    got = {j: {s: v["S"] for s, v in res["outer"][j].items()} for j in campuses}
    assert sum(len(v) for v in got.values()) == 141, "expected 141 stratum-fold certificates"
    assert got == ref["outer_S"], "fold certificates differ from cert_corrected.json"
    n_cert = sum(v["S"] >= S_CUT for v in res["pooled"].values())
    assert len(res["pooled"]) == 16 and n_cert == 12, (len(res["pooled"]), n_cert)
    assert res["routing"]["route_counts"] == ref["routing"]["route_counts"], \
        (res["routing"]["route_counts"], ref["routing"]["route_counts"])
    return res


# --------------------------------------------------------------------------
# event-level trace frame (the construction of script 45's prior_trace_table,
# kept at the level of the cell-quarter so that buildings can be resampled)
# --------------------------------------------------------------------------
def trace_frame(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    thr90 = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.90)
    p["p90"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr90).astype(float).fillna(np.inf)).astype(int)
    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    g = p.groupby(CELL, observed=True)["upm_count"]
    p["prior_total"] = g.shift(1).rolling(2, min_periods=2).sum() + \
                       g.shift(3).rolling(2, min_periods=2).sum()
    p["any_count"] = p["upm_count"] + p["ppm_count"]
    ga = p.groupby(CELL, observed=True)["any_count"]
    p["prior_any"] = ga.shift(1).rolling(2, min_periods=2).sum() + \
                     ga.shift(3).rolling(2, min_periods=2).sum()
    p["has_history"] = (p["period_q"] - 4 >= p["cell_first_q"]).astype(int)
    p["zero_prior"] = (p["prior_total"] == 0).astype(int)
    p["zero_prior_any"] = (p["prior_any"] == 0).astype(int)
    p["is_event"] = ((p["p90"] == 1) & (p["has_history"] == 1)).astype(int)
    p["campus"] = p[C.COL_UNIV].astype(str)
    p["system"] = p[C.COL_SYSTEM].astype(str)
    p["bld"] = p["campus"] + "/" + p[C.COL_BUILDING].astype(str)
    return p


def verdict(share: float, n_events: int) -> str:
    if n_events < MIN_EVENTS or share is None or np.isnan(share):
        return "caution"
    return "pass" if share < 0.10 else ("caution" if share < 0.25 else "fail")


def _q(v, lo=0.025, hi=0.975):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return [None, None]
    return [round(float(np.quantile(v, lo)), 4), round(float(np.quantile(v, hi)), 4)]


def stratum_block(f: pd.DataFrame, rng: np.random.Generator,
                  camp_start: dict) -> dict:
    """All quantities for one stratum's rows `f`, with the clustered bootstrap."""
    ev = f[f.is_event == 1]
    hist = f[f.has_history == 1]
    act = hist[hist.upm_count >= 1]

    # per-building sums; buildings carry their campus
    keys = ["campus", "bld"]
    agg = pd.DataFrame({
        "E": ev.groupby(keys).size(),
        "Z": ev.groupby(keys)["zero_prior"].sum(),
        "A": act.groupby(keys).size(),
        "AZ": act.groupby(keys)["zero_prior"].sum(),
        "H": hist.groupby(keys).size(),
        "HZ": hist.groupby(keys)["zero_prior"].sum(),
    })
    allb = f.groupby(keys).size().index           # every building of the stratum
    agg = agg.reindex(allb).fillna(0.0).reset_index()

    def point(a: pd.DataFrame) -> dict:
        by = a.groupby("campus")[["E", "Z", "A", "AZ", "H", "HZ"]].sum()
        E = by.E.sum()
        w = by.E / E
        refA = float((w * (by.AZ / by.A.where(by.A > 0))).fillna(0).sum())
        refB = float((w * (by.HZ / by.H.where(by.H > 0))).fillna(0).sum())
        obs = float(by.Z.sum() / E)
        return {"obs": obs, "refA": refA, "refB": refB}

    pt = point(agg)

    # building-clustered bootstrap, buildings resampled within each campus
    cols = ["E", "Z", "A", "AZ", "H", "HZ"]
    camp_tot = {k: np.zeros((N_BOOT, len(cols))) for k in sorted(agg.campus.unique())}
    for camp, a in agg.groupby("campus"):
        M = a[cols].to_numpy(float)                       # n_b x 6
        nb = len(a)
        W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=N_BOOT)   # B x n_b
        camp_tot[camp] = W @ M                            # B x 6
    T = np.stack([camp_tot[k] for k in sorted(camp_tot)], axis=1)   # B x n_campus x 6
    E, Z, A, AZ, H, HZ = (T[:, :, i] for i in range(6))
    Etot = E.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        w = E / Etot[:, None]
        obs_b = Z.sum(1) / Etot
        refA_b = np.nansum(w * np.where(A > 0, AZ / A, 0.0), axis=1)
        refB_b = np.nansum(w * np.where(H > 0, HZ / H, 0.0), axis=1)
        ratioA_b = obs_b / refA_b
        ratioB_b = obs_b / refB_b
    ratioA_b[~np.isfinite(ratioA_b)] = np.nan
    ratioB_b[~np.isfinite(ratioB_b)] = np.nan

    # the active reference with the extreme events themselves left out
    act_ne = act[act.is_event == 0]
    by_ne = act_ne.groupby("campus").agg(n=("zero_prior", "size"), z=("zero_prior", "sum"))
    w_ev = ev.groupby("campus").size() / len(ev)
    refA_ne = float((w_ev * (by_ne.z / by_ne.n).reindex(w_ev.index)).fillna(0).sum())

    # record start, from the campus's first panel quarter and from the cell's own
    rs_campus, rs_cell = {}, {}
    for q in RECORD_START_Q:
        keep = ev[ev.period_q >= ev.campus.map(camp_start) + q]
        n, s = int(len(keep)), (float(keep.zero_prior.mean()) if len(keep) else float("nan"))
        rs_campus[str(q)] = {"n_events": n, "share": None if np.isnan(s) else round(s, 4),
                             "verdict": verdict(s, n)}
        keep = ev[ev.period_q - ev.cell_first_q >= q]
        n, s = int(len(keep)), (float(keep.zero_prior.mean()) if len(keep) else float("nan"))
        rs_cell[str(q)] = {"n_events": n, "share": None if np.isnan(s) else round(s, 4),
                           "verdict": verdict(s, n)}

    return {
        "n_events": int(len(ev)), "n_zero": int(ev.zero_prior.sum()),
        "share": round(pt["obs"], 4), "share_ci": _q(obs_b),
        "verdict": verdict(pt["obs"], len(ev)),
        "share_counting_pm": round(float(ev.zero_prior_any.mean()), 4),
        "verdict_counting_pm": verdict(float(ev.zero_prior_any.mean()), len(ev)),
        "reference_active": round(pt["refA"], 4), "reference_active_ci": _q(refA_b),
        "ratio_active": round(pt["obs"] / pt["refA"], 3) if pt["refA"] > 0 else None,
        "ratio_active_ci": _q(ratioA_b),
        "reference_active_excluding_events": round(refA_ne, 4),
        "reference_all": round(pt["refB"], 4), "reference_all_ci": _q(refB_b),
        "ratio_all": round(pt["obs"] / pt["refB"], 3) if pt["refB"] > 0 else None,
        "ratio_all_ci": _q(ratioB_b),
        "n_active_history_rows": int(len(act)), "n_history_rows": int(len(hist)),
        "record_start_campus": rs_campus, "record_start_cell": rs_cell,
        "mean_reactive_wo_per_cell_quarter": round(float(f.upm_count.mean()), 4),
        "n_cell_quarters": int(len(f)),
        "n_buildings": int(len(agg)),
        "campus_mix_of_events": {k: round(float(v), 4) for k, v in w_ev.items()},
    }


def main() -> None:
    t0 = time.time()
    res = check_reference()
    pooled_ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]["pooled"]
    print("[61] reference pipeline reproduced (141 fold certificates, 12/16 pooled, routes)",
          flush=True)

    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    p = trace_frame(panel)

    # ---- reproduction: the event table must equal trace_for(None) exactly ----
    ev = p[p.is_event == 1]
    mine = (ev.groupby(["campus", "system"]).agg(n_events=("zero_prior", "size"),
                                                  n_zero=("zero_prior", "sum"))
              .reset_index().sort_values(["campus", "system"]).reset_index(drop=True))
    ref = cc.trace_for(None).sort_values(["campus", "system"]).reset_index(drop=True)
    ref["system"] = ref["system"].astype(str)
    assert len(mine) == len(ref), (len(mine), len(ref))
    assert (mine.campus.values == ref.campus.values).all()
    assert (mine.system.values == ref.system.values).all()
    assert (mine.n_events.values == ref.n_events.values).all()
    assert (mine.n_zero.values == ref.n_zero.values).all()
    print(f"[61] event table equals trace_for(None): {len(ref)} campus-system rows, "
          f"{int(ref.n_events.sum())} events", flush=True)

    camp_start = panel.groupby(panel[C.COL_UNIV].astype(str))["period_q"].min().to_dict()
    names = (panel[[C.COL_SYSTEM, C.COL_SYSTEM_DESC]].astype(str).drop_duplicates(C.COL_SYSTEM)
             .set_index(C.COL_SYSTEM)[C.COL_SYSTEM_DESC].str.strip().to_dict())

    rng = np.random.default_rng(C.RANDOM_SEED)
    out_strata, rows = {}, []
    n_ev_sys = ev.groupby("system").size()
    for sysc in sorted(n_ev_sys.index):
        if n_ev_sys[sysc] < MIN_EVENTS:
            continue
        f = p[p.system == sysc]
        b = stratum_block(f, rng, camp_start)
        b["system_desc"] = names.get(sysc, sysc)
        gated = sysc in res["pooled"]
        b["in_pooled_map"] = gated
        if gated:
            # the pooled gate must read the same share
            want = pooled_ref[sysc]["zero_prior_share"]
            assert want == round(b["n_zero"] / b["n_events"], 3), (sysc, want, b["share"])
            sc = res["pooled"][sysc]["scores"]["prior_record_trace"]
            assert {1.0: "pass", 0.5: "caution", 0.0: "fail"}[sc] == b["verdict"], sysc
            b["pooled_S"] = res["pooled"][sysc]["S"]
            b["pooled_certified"] = bool(res["pooled"][sysc]["S"] >= S_CUT)
        out_strata[sysc] = b
        rows.append({
            "system": sysc, "system_desc": b["system_desc"], "in_pooled_map": gated,
            "pooled_certified": b.get("pooled_certified"),
            "n_events": b["n_events"], "share": b["share"],
            "share_lo": b["share_ci"][0], "share_hi": b["share_ci"][1],
            "verdict": b["verdict"],
            "share_counting_pm": b["share_counting_pm"],
            "reference_active": b["reference_active"],
            "ratio_active": b["ratio_active"],
            "ratio_active_lo": b["ratio_active_ci"][0], "ratio_active_hi": b["ratio_active_ci"][1],
            "reference_active_excluding_events": b["reference_active_excluding_events"],
            "reference_all": b["reference_all"], "ratio_all": b["ratio_all"],
            **{f"share_drop_campus_q{q}": b["record_start_campus"][str(q)]["share"]
               for q in RECORD_START_Q},
            **{f"n_drop_campus_q{q}": b["record_start_campus"][str(q)]["n_events"]
               for q in RECORD_START_Q},
            **{f"verdict_drop_campus_q{q}": b["record_start_campus"][str(q)]["verdict"]
               for q in RECORD_START_Q},
            **{f"share_drop_cell_q{q}": b["record_start_cell"][str(q)]["share"]
               for q in RECORD_START_Q},
            **{f"n_drop_cell_q{q}": b["record_start_cell"][str(q)]["n_events"]
               for q in RECORD_START_Q},
            "mean_reactive_wo_per_cell_quarter": b["mean_reactive_wo_per_cell_quarter"],
        })
        print(f"  {sysc}: n={b['n_events']:5d} share={b['share']:.3f} "
              f"refA={b['reference_active']:.3f} ratio={b['ratio_active']} "
              f"{b['ratio_active_ci']} mean_wo={b['mean_reactive_wo_per_cell_quarter']:.3f}",
              flush=True)

    tab = pd.DataFrame(rows)
    rho, pval = spearmanr(tab["share"], tab["mean_reactive_wo_per_cell_quarter"])
    g = tab[tab.in_pooled_map]
    rho_g, pval_g = spearmanr(g["share"], g["mean_reactive_wo_per_cell_quarter"])

    # verdict changes under each variant, among strata of the pooled map
    def changes(col):
        return [{"system": r.system, "observed": r.verdict, "variant": getattr(r, col)}
                for r in g.itertuples() if getattr(r, col) != r.verdict]
    pm_verd = {s: out_strata[s]["verdict_counting_pm"] for s in g.system}

    out = {
        "design": ("Per stratum with at least 30 extreme events on the pooled panel, the "
                   "no-antecedent share of extreme events (script 45's prior-record-trace "
                   "quantity) beside frequency-matched references: the same no-reactive-work "
                   "share among reactive cell-quarters with four quarters of history, and among "
                   "all cell-quarters with history, both weighted to the campus mix of the "
                   "events; intervals resample buildings within campuses."),
        "definitions": {
            "extreme_event": ("cell-quarter with reactive labour above the system's 90th "
                              "percentile of positive reactive labour (pooled panel), with "
                              "four quarters of cell history"),
            "no_antecedent": "no reactive work order on the cell in the preceding four quarters",
            "no_antecedent_counting_pm": ("no reactive and no preventive work order in the "
                                          "preceding four quarters"),
            "reference_active": ("share with no reactive work order in the preceding four "
                                 "quarters among cell-quarters of the stratum with at least one "
                                 "reactive work order and four quarters of history (extreme "
                                 "events included), per campus, weighted by the campus shares "
                                 "of the extreme events"),
            "reference_active_excluding_events": "reference_active with the extreme events left out",
            "reference_all": "the same share over all cell-quarters with four quarters of history",
            "record_start_campus": ("events dropped when period_q < first panel quarter of the "
                                    "campus + q; the 4-quarter version removes nothing because "
                                    "events already need four quarters of cell history"),
            "record_start_cell": "events dropped when period_q < the cell's first quarter + q",
            "verdict": "pass < 0.10, caution < 0.25, fail otherwise; caution below 30 events",
            "bootstrap": (f"{N_BOOT} replicates; buildings resampled with replacement within "
                          "each campus; campus weights recomputed per replicate; percentile "
                          "95% intervals"),
            "mean_reactive_wo_per_cell_quarter": "mean upm_count over every cell-quarter of the stratum",
        },
        "campus_first_quarter": {k: int(v) for k, v in sorted(camp_start.items())},
        "n_strata": int(len(tab)),
        "pooled_map_strata_below_30_events": {
            s: int(n_ev_sys.get(s, 0)) for s in sorted(res["pooled"]) if s not in out_strata},
        "spearman_share_vs_frequency": {
            "all_strata": {"rho": round(float(rho), 3), "p": float(pval), "n": int(len(tab))},
            "pooled_map_strata": {"rho": round(float(rho_g), 3), "p": float(pval_g),
                                  "n": int(len(g))},
        },
        "verdict_changes_pooled_map": {
            "counting_pm": [{"system": s, "observed": out_strata[s]["verdict"], "variant": v}
                            for s, v in pm_verd.items() if v != out_strata[s]["verdict"]],
            **{f"drop_campus_first_{q}q": changes(f"verdict_drop_campus_q{q}")
               for q in RECORD_START_Q},
        },
        "strata": out_strata,
        "runtime_seconds": None,
    }
    out["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(C.METRICS / "trace_frequency_null.json", "w"), indent=2, default=float)
    tab.to_csv(C.TABLES / "trace_frequency_null.csv", index=False)

    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\n", tab[["system", "in_pooled_map", "n_events", "share", "verdict",
                     "share_counting_pm", "reference_active", "ratio_active",
                     "ratio_active_lo", "ratio_active_hi", "reference_all",
                     "share_drop_campus_q12", "verdict_drop_campus_q12",
                     "mean_reactive_wo_per_cell_quarter"]].to_string(index=False))
    print(f"\nSpearman(share, reactive WO per cell-quarter) = {rho:.3f} (p={pval:.3g}, "
          f"n={len(tab)}); pooled-map strata {rho_g:.3f} (p={pval_g:.3g}, n={len(g)})")
    print(f"[61] done in {out['runtime_seconds']} s")


if __name__ == "__main__":
    main()
