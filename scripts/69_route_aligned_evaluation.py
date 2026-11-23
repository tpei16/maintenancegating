#!/usr/bin/env python
"""
69_route_aligned_evaluation.py -- the verdicts read against the units the
routing rule actually flags, the scale on which the pooled gate quantities are
computed, and the resolution of ties at a top-decile boundary.

Two gate quantities read the top decile of one building system's units ranked
by the raw screening score: the pooled top-decile lift of the risk-
concentration gate (units of the system pooled over the peer campuses, each
peer scored by its own model) and the per-campus lifts behind transfer
stability.  Routing flags a unit when its risk percentile R, the percentile of
the score within the campus and quarter over all building systems, is at least
0.90.  This script measures how far the two readings agree.

(a) ROUTE-ALIGNED EVALUATION ON THE HELD-OUT CAMPUS.  For each of the 141
    campus-system pairs (c, k) of the held-out maps, on the units of system k
    on campus c scored by f^(-c) (the model that excluded c, exactly the
    scores the stored routing reads): the units, the units flagged by
    R >= 0.90, the share flagged, the outcome rate among flagged units
    (precision of the flags), among the system's unflagged units and among
    all its units on that campus (base rate), and the route-aligned lift
    (precision over base rate).  Each carries a 95% percentile interval from
    1,000 bootstrap resamples of the campus's buildings that hold units of
    the system, with R and the flags held at their stored values.  A pair is
    route-evaluable with at least 30 flagged units, at least 10 high-burden
    quarters and at least 100 units of the system on the campus; the levels
    were fixed before any outcome was read.  The flagged counts summed by
    verdict, together with the flags in systems without a verdict, must
    reproduce the stored route counts and outcome rates (asserted).

(b) SCALE OF THE POOLED GATE.  The pooled top-decile lift and the risk-
    gradient quantities (top-band ratio and band correlation) are recomputed
    with a percentile in place of the raw score whenever units of different
    campuses are pooled: (i) the percentile of the score within campus,
    quarter and building system; (ii) the campus-wide risk percentile R.  The
    risk-gradient gate already reads R, so under (ii) only the pooled lift
    changes.  The per-campus lifts, data sufficiency and prior-record trace
    keep their stored values.  The two recomputed gate scores replace the
    stored ones and S is aggregated by sufficiency() of script 45.

(c) TIES AT A TOP-DECILE BOUNDARY.  lift_at_topk keeps the first ceil(0.1 n)
    units of a stable sort, so units tied at the boundary score are taken in
    the stored order of the records.  Per pair and gate quantity the tied
    units are counted; the verdicts are then recomputed (1) with the rows of
    every map in random order, 200 repetitions with fixed seeds, which breaks
    every tie at random and leaves every other gate computation unchanged;
    (2) with fractional inclusion of the tied units, the expected precision
    under random tie-breaking; (3) at the two extremes, tied units taken with
    the non-high-burden ones first and with the high-burden ones first.  S is
    non-decreasing in every lift, so a verdict equal at both extremes cannot
    change under any resolution of the ties.  The same is done for the prior-
    year count rule as the risk score (script 48's rule_arm: the units and
    high-burden labels of the screening model's maps, in the same order, with
    the count as the score), whose discrete scores tie widely.

INVARIANT (the runner aborts on a mismatch).  With raw scores and ties taken
in stored order the recomputation equals the stored pooled map
(cert_corrected.json, gate_evidence_corrected.csv), the 141 stored verdicts
of the held-out maps (fold_gate_verdicts.csv) and the stored route counts; the
count-rule maps equal learner_invariance.json and learner_agreement.json.

Nothing is fitted.  CPU only.

Outputs -> results/metrics/route_aligned_evaluation.json
           results/tables/route_aligned_evaluation.csv     (one row per pair)
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FMSCREEN_DEVICE", "cpu")
os.environ.setdefault("FMSCREEN_THREADS", "2")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys, json, time, argparse, importlib.util
from contextlib import contextmanager
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import metrics as MET


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load("cert_corrected", "45_certification_corrected.py")
sr = _load("fold_verdicts", "56_fold_verdicts_and_simple_rule.py")
li = _load("learner_invariance", "48_learner_invariance.py")
assert cc.MET is MET, "the gate functions must read the same metrics module"

GATES, HARD = cc.GATES, cc.HARD
FAIL, CAUTION, PASS = cc.FAIL, cc.CAUTION, cc.PASS
R_CUT, S_CUT = cc.R_CUT, cc.S_CUT
RCOL = "R_quarter"
K = C.TOPK_BUDGET
SEED = C.RANDOM_SEED
CTX_MIN_POS, CTX_MIN_ROWS = sr.CTX_MIN_POS, sr.CTX_MIN_ROWS
# route-evaluable pair: fixed before any outcome was read
ROUTE_MIN_FLAGGED, ROUTE_MIN_POS, ROUTE_MIN_UNITS = 30, 10, 100
N_BOOT_ROUTE = 1000
N_TIE_REPS = 200
LEVEL = {FAIL: "fail", CAUTION: "caution", PASS: "pass"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _same(a, b) -> bool:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (isinstance(b, float) and np.isnan(b))
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return False
    return abs(float(a) - float(b)) < 1e-9


def rnd(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


def spearman_or_none(x, y) -> dict | None:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3:
        return None
    return sr.spearman(x, y)


def pct_interval(v: np.ndarray) -> list | None:
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return None
    return [round(float(np.quantile(v, 0.025)), 4), round(float(np.quantile(v, 0.975)), 4)]


# --------------------------------------------------------------------------
# lifts under a stated resolution of ties at the top-decile boundary
# --------------------------------------------------------------------------
def boundary(y: np.ndarray, s: np.ndarray, k: float = K) -> dict:
    """The units around the top-decile boundary of one ranking.  A tie
    straddles the boundary when more units share the boundary score than the
    top decile has room for after the units scoring above it."""
    s = np.asarray(s, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    n = len(s)
    m = max(1, int(np.ceil(k * n)))
    v = -np.partition(-s, m - 1)[m - 1]
    gt, eq = s > v, s == v
    n_gt, n_eq = int(gt.sum()), int(eq.sum())
    p_eq = float(y[eq].sum())
    return {"n": n, "m": m, "v": float(v), "n_gt": n_gt, "n_eq": n_eq,
            "slots": m - n_gt, "pos_gt": float(y[gt].sum()), "pos_eq": p_eq,
            "neg_eq": n_eq - p_eq, "straddles": bool(n_gt + n_eq > m),
            "n_tied_excluded": n_gt + n_eq - m}


def tie_lift(y, scores, k: float, mode: str) -> float:
    y = np.asarray(y, dtype=np.float64)
    base = y.mean()
    if base <= 0:
        return float("nan")
    b = boundary(y, scores, k)
    t = b["slots"]
    if mode == "fractional":
        inc = t * b["pos_eq"] / b["n_eq"]
    elif mode == "low":
        inc = max(0.0, t - b["neg_eq"])
    elif mode == "high":
        inc = min(float(t), b["pos_eq"])
    else:
        raise ValueError(mode)
    return float((b["pos_gt"] + inc) / b["m"] / base)


@contextmanager
def tie_rule(mode: str | None):
    """Swap the top-decile lift read by gates_for for a tie-aware version.
    mode None keeps the stored rule (stable order of the records)."""
    orig = MET.lift_at_topk
    if mode is not None:
        MET.lift_at_topk = lambda y, s, k: tie_lift(y, s, k, mode)
    try:
        yield
    finally:
        MET.lift_at_topk = orig


# --------------------------------------------------------------------------
# maps
# --------------------------------------------------------------------------
def gbdt_frames(single: pd.DataFrame, pairs: dict, campuses: list[str]):
    devs = {j: cc.dev_corpus(j, campuses, single, pairs, "double") for j in campuses}
    for j, d in devs.items():
        # verdicts from peers only: no unit of the target enters its map, and
        # every peer d is read through the model that excluded both j and d
        assert not (d.university == j).any(), f"target {j} in its own map"
        assert all(set(h.split("+")) == {j, u} for h, u in
                   d[["held_out", "university"]].drop_duplicates().itertuples(index=False)), j
    return devs, single


def rule_frames(single: pd.DataFrame, pairs: dict, campuses: list[str]):
    """The count-rule maps: the screening model's maps with the count as the
    risk score (labels asserted equal in rule_arm of script 48)."""
    rs, rp, rule_name = li.rule_arm(single, pairs)
    devs = {j: cc.dev_corpus(j, campuses, rs, rp, "double") for j in campuses}
    return devs, rs, rule_name


def gate_maps(devs: dict, pooled_frame: pd.DataFrame, campuses: list[str],
              rcol: str = RCOL) -> tuple[dict, dict]:
    outer = {j: cc.gates_for(devs[j], cc.trace_for(j), rcol, True) for j in campuses}
    pooled = cc.gates_for(pooled_frame, cc.trace_for(None), rcol, True)
    return outer, pooled


def verdict_vector(outer: dict, pooled: dict, keys: list, pkeys: list):
    S = np.array([outer[j][s]["S"] if s in outer[j] else np.nan for j, s in keys])
    P = np.array([pooled[s]["S"] if s in pooled else np.nan for s in pkeys])
    return S, P


# --------------------------------------------------------------------------
# reference checks
# --------------------------------------------------------------------------
def check_reference(outer: dict, pooled: dict, routing: dict, campuses: list) -> dict:
    ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]
    assert set(pooled) == set(ref["pooled"]), "pooled map systems differ"
    for s, v in pooled.items():
        r = ref["pooled"][s]
        assert v["scores"] == r["scores"] and abs(v["S"] - r["S"]) < 1e-9, f"pooled {s}"
        for q in ("n_campuses", "n_pos", "pooled_lift", "frac_2x", "zero_prior_share"):
            assert _same(v[q], r[q]), f"pooled {s} {q}: {v[q]} != {r[q]}"
    ge = pd.read_csv(C.TABLES / "gate_evidence_corrected.csv")
    for row in ge.itertuples():
        v = pooled[row.system]
        assert _same(v.get("top_band"), row.top_band_enrichment) and \
            _same(v.get("rho"), row.band_spearman), f"pooled {row.system} risk gradient"
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    assert len(fv) == sum(len(outer[c]) for c in campuses) == 141, "held-out verdict count"
    for row in fv.itertuples():
        v = outer[row.campus][row.system]
        for g in GATES:
            assert v["scores"][g] == getattr(row, g), f"held-out {row.campus}/{row.system} {g}"
        assert abs(round(v["S"], 2) - row.S) < 1e-9 and \
            bool(v["S"] >= S_CUT) == bool(row.certified), f"held-out {row.campus}/{row.system} S"
        for q in ("n_campuses", "n_pos", "pooled_lift", "frac_2x", "top_band", "rho",
                  "zero_prior_share"):
            assert _same(v.get(q), getattr(row, q)), f"held-out {row.campus}/{row.system} {q}"
    for j in campuses:
        for s, v in outer[j].items():
            assert abs(v["S"] - ref["outer_S"][j][s]) < 1e-9, f"S {j}/{s}"
    assert routing["route_counts"] == ref["routing"]["route_counts"], "route counts"
    return {"pooled_map_supported": f"{sum(v['S'] >= S_CUT for v in pooled.values())}/{len(pooled)}",
            "held_out_verdicts": int(len(fv)), "held_out_supported": int(fv.certified.sum()),
            "route_counts": routing["route_counts"],
            "matches_cert_corrected_json": True, "matches_gate_evidence_corrected_csv": True,
            "matches_fold_gate_verdicts_csv": True}


def check_rule_reference(outer: dict, pooled: dict, rule_name: str) -> dict:
    inv = json.load(open(C.METRICS / "learner_invariance.json"))
    agr = json.load(open(C.METRICS / "learner_agreement.json"))
    assert inv["rule_used"] == rule_name, rule_name
    assert set(pooled) == set(inv["pooled_S"]["rule"]), "rule pooled systems"
    for s, v in pooled.items():
        assert abs(v["S"] - inv["pooled_S"]["rule"][s]) < 1e-9, f"rule pooled {s}"
        assert _same(v["pooled_lift"], inv["pooled_lift"]["rule"][s]), f"rule pooled lift {s}"
    n_f = sum(len(m) for m in outer.values())
    n_sup = sum(v["S"] >= S_CUT for m in outer.values() for v in m.values())
    assert n_f == agr["n_stratum_folds"]["rule"] and n_sup == agr["certified_counts_folds"]["rule"], \
        f"rule folds {n_sup}/{n_f}"
    n_pool = sum(v["S"] >= S_CUT for v in pooled.values())
    assert n_pool == agr["certified_counts_pooled"]["rule"], "rule pooled count"
    return {"rule": rule_name, "pooled_map_supported": f"{n_pool}/{len(pooled)}",
            "held_out_supported": f"{n_sup}/{n_f}",
            "matches_learner_invariance_json": True, "matches_learner_agreement_json": True}


# --------------------------------------------------------------------------
# (a) route-aligned evaluation
# --------------------------------------------------------------------------
def route_aligned(single: pd.DataFrame, outer: dict, campuses: list[str],
                  fv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    assert (single.held_out == single.university).all(), "held-out units must be scored by f^(-c)"
    u = single.copy()
    u["flag"] = (u[RCOL] >= R_CUT).astype(np.int8)
    rows = []
    for i, r in enumerate(fv.itertuples()):
        j, s = r.campus, r.system
        g = u[(u.university == j) & (u.system == s)]
        n, nf = len(g), int(g.flag.sum())
        y = g.y.to_numpy(np.float64); f = g.flag.to_numpy(bool)
        rec = {"campus": j, "system": s, "system_desc": r.system_desc,
               "S": r.S, "supported": bool(r.certified),
               "heldout_evaluable": bool(r.heldout_evaluable),
               "heldout_lift": r.heldout_lift,
               "n_units": n, "n_buildings": int(g.building.nunique()),
               "n_pos": int(y.sum()), "n_flagged": nf, "n_flagged_pos": int(y[f].sum())}
        base = y.mean() if n else np.nan
        prec = y[f].mean() if nf else np.nan
        unfl = y[~f].mean() if n - nf else np.nan
        rec.update({"share_flagged": nf / n if n else np.nan, "base_rate": base,
                    "precision_flags": prec, "rate_unflagged": unfl,
                    "route_lift": prec / base if (nf and base > 0) else np.nan})
        # overlap of the flags with the system's own top decile on the campus
        if n:
            top = MET._topk_mask(g.score.to_numpy(), K)
            rec["n_system_top_decile"] = int(top.sum())
            rec["n_flagged_in_system_top_decile"] = int((top & f).sum())
        else:
            rec["n_system_top_decile"] = 0; rec["n_flagged_in_system_top_decile"] = 0
        # bootstrap over the campus's buildings that hold units of the system
        ci = {}
        if n and rec["n_buildings"] >= 2:
            agg = (pd.DataFrame({"b": g.building.to_numpy(), "n": 1, "f": f.astype(int),
                                 "y": y, "fy": (y * f)})
                   .groupby("b", sort=True)[["n", "f", "y", "fy"]].sum().to_numpy(np.float64))
            nb = len(agg)
            rng = np.random.default_rng([SEED, i])
            W = rng.multinomial(nb, np.full(nb, 1.0 / nb), size=N_BOOT_ROUTE).astype(np.float64)
            tot = W @ agg                       # columns: units, flagged, positives, flagged positives
            with np.errstate(divide="ignore", invalid="ignore"):
                b_share = tot[:, 1] / tot[:, 0]
                b_base = tot[:, 2] / tot[:, 0]
                b_prec = np.where(tot[:, 1] > 0, tot[:, 3] / tot[:, 1], np.nan)
                b_unfl = np.where(tot[:, 0] - tot[:, 1] > 0,
                                  (tot[:, 2] - tot[:, 3]) / (tot[:, 0] - tot[:, 1]), np.nan)
                b_lift = np.where(b_base > 0, b_prec / b_base, np.nan)
            for name, v in (("share_flagged", b_share), ("base_rate", b_base),
                            ("precision_flags", b_prec), ("rate_unflagged", b_unfl),
                            ("route_lift", b_lift)):
                iv = pct_interval(v)
                ci[name] = iv
                rec[f"{name}_lo"], rec[f"{name}_hi"] = (iv if iv else (np.nan, np.nan))
            rec["route_lift_boot_undefined"] = int(np.isnan(b_lift).sum())
        else:
            for name in ("share_flagged", "base_rate", "precision_flags", "rate_unflagged",
                         "route_lift"):
                rec[f"{name}_lo"], rec[f"{name}_hi"] = np.nan, np.nan
            rec["route_lift_boot_undefined"] = None
        rec["route_evaluable"] = bool(nf >= ROUTE_MIN_FLAGGED and rec["n_pos"] >= ROUTE_MIN_POS
                                      and n >= ROUTE_MIN_UNITS)
        rec["route_lift_ge_2"] = (bool(rec["route_lift"] >= 2.0)
                                  if rec["route_evaluable"] else None)
        rows.append(rec)
    ra = pd.DataFrame(rows)

    # flags in building systems that carry no verdict on the campus
    ung = []
    for j in campuses:
        sub = u[(u.university == j) & (~u.system.isin(list(outer[j])))]
        ung.append(sub)
    ung = pd.concat(ung)
    ung_f = ung[ung.flag == 1]

    # ---- reproduction of the stored routing ----
    ry = json.load(open(C.METRICS / "route_yield_corrected.json"))["corrected"]
    rc = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]["routing"]["route_counts"]
    sup, nsup = ra[ra.supported], ra[~ra.supported]
    n_task = int(sup.n_flagged.sum()); n_ver_g = int(nsup.n_flagged.sum())
    n_ver = n_ver_g + len(ung_f)
    rate_task = sup.n_flagged_pos.sum() / n_task
    rate_ver = (nsup.n_flagged_pos.sum() + ung_f.y.sum()) / n_ver
    assert n_task == rc["pma"] == ry["pma_n"] == 50968, f"decision tasks {n_task}"
    assert n_ver == rc["verify"] == ry["verify_n"] == 705, f"verification tasks {n_ver}"
    assert n_ver_g == ry["gated_strata_only"]["verify_n"], f"gated verification {n_ver_g}"
    assert round(rate_task, 4) == ry["pma_outcome_rate"], f"task rate {rate_task}"
    assert round(rate_ver, 4) == ry["verify_outcome_rate"], f"verification rate {rate_ver}"
    assert round(nsup.n_flagged_pos.sum() / n_ver_g, 4) == ry["gated_strata_only"]["verify_outcome_rate"]
    assert int(ra.n_units.sum()) + len(ung) == len(single) == ry["n_routed_units"]
    repro = {"decision_tasks": n_task, "decision_task_outcome_rate": round(float(rate_task), 4),
             "verification_tasks": n_ver, "verification_outcome_rate": round(float(rate_ver), 4),
             "verification_tasks_in_pairs_with_verdict": n_ver_g,
             "verification_tasks_in_systems_without_verdict": int(len(ung_f)),
             "matches_route_yield_corrected_json": True, "matches_cert_corrected_routing": True}

    # ---- summaries ----
    ev = ra[ra.route_evaluable]

    def group(d: pd.DataFrame) -> dict:
        k = int(d.route_lift_ge_2.sum()) if len(d) else 0
        return {"n_pairs": int(len(d)),
                "flagged_units": int(d.n_flagged.sum()),
                "pooled_precision": rnd(d.n_flagged_pos.sum() / d.n_flagged.sum()) if len(d) else None,
                "median_precision": rnd(d.precision_flags.median()) if len(d) else None,
                "median_route_lift": rnd(d.route_lift.median(), 3) if len(d) else None,
                "n_route_lift_ge_2": k,
                "share_route_lift_ge_2": rnd(k / len(d)) if len(d) else None,
                "share_route_lift_ge_2_wilson95": sr.wilson(k, len(d)),
                "median_share_flagged": rnd(d.share_flagged.median()) if len(d) else None,
                "pairs": [f"{a}/{b}" for a, b in zip(d.campus, d.system)] if len(d) <= 20 else None}

    def all_pairs(d: pd.DataFrame) -> dict:
        fl = d[d.n_flagged > 0]
        return {"n_pairs": int(len(d)), "n_pairs_with_flags": int(len(fl)),
                "flagged_units": int(d.n_flagged.sum()),
                "pooled_precision": rnd(d.n_flagged_pos.sum() / d.n_flagged.sum())
                if d.n_flagged.sum() else None,
                "pooled_base_rate_of_these_units": rnd(d.n_pos.sum() / d.n_units.sum())}

    both = ev[ev.heldout_evaluable]
    summary = {
        "n_pairs": int(len(ra)),
        "n_route_evaluable": int(len(ev)),
        "n_route_evaluable_supported": int(ev.supported.sum()),
        "n_route_evaluable_not_supported": int((~ev.supported).sum()),
        "n_route_evaluable_and_heldout_evaluable": int(len(both)),
        "route_evaluable": {"supported": group(ev[ev.supported]),
                            "not_supported": group(ev[~ev.supported])},
        "all_141_pairs": {"supported": all_pairs(ra[ra.supported]),
                          "not_supported": all_pairs(ra[~ra.supported])},
        "spearman_route_lift_vs_heldout_top_decile_lift": spearman_or_none(both.route_lift,
                                                                          both.heldout_lift),
        "spearman_route_lift_vs_S": spearman_or_none(ev.route_lift, ev.S),
        "heldout_top_decile_lift_on_route_evaluable_pairs": {
            "supported": sr.lift_summary(both[both.supported].heldout_lift.tolist()),
            "not_supported": sr.lift_summary(both[~both.supported].heldout_lift.tolist())},
        "route_lift_and_top_decile_lift_agree_on_two_fold": {
            "n": int(len(both)),
            "both_ge_2": int(((both.route_lift >= 2) & (both.heldout_lift >= 2)).sum()),
            "route_only_ge_2": int(((both.route_lift >= 2) & (both.heldout_lift < 2)).sum()),
            "top_decile_only_ge_2": int(((both.route_lift < 2) & (both.heldout_lift >= 2)).sum()),
            "neither": int(((both.route_lift < 2) & (both.heldout_lift < 2)).sum())},
        "not_route_evaluable": {
            "n": int((~ra.route_evaluable).sum()),
            "supported": int((~ra.route_evaluable & ra.supported).sum()),
            "not_supported": int((~ra.route_evaluable & ~ra.supported).sum()),
            "with_zero_flags": int((ra.n_flagged == 0).sum()),
            "supported_with_zero_flags": int(((ra.n_flagged == 0) & ra.supported).sum()),
            "supported_with_1_to_29_flags": int(((ra.n_flagged > 0) &
                                                 (ra.n_flagged < ROUTE_MIN_FLAGGED) &
                                                 ra.supported).sum()),
            "supported_with_at_least_30_flags_but_too_few_units_or_positives": int(
                ((ra.n_flagged >= ROUTE_MIN_FLAGGED) & ~ra.route_evaluable & ra.supported).sum()),
            "with_no_units": int((ra.n_units == 0).sum())},
        "overlap_with_system_top_decile": {
            "flagged_units": int(ra.n_flagged.sum()),
            "flagged_units_in_system_top_decile": int(ra.n_flagged_in_system_top_decile.sum()),
            "system_top_decile_units": int(ra.n_system_top_decile.sum()),
            "share_of_flags_in_system_top_decile": rnd(ra.n_flagged_in_system_top_decile.sum()
                                                       / ra.n_flagged.sum()),
            "share_of_system_top_decile_flagged": rnd(ra.n_flagged_in_system_top_decile.sum()
                                                      / ra.n_system_top_decile.sum())},
    }

    # how the campus-wide rule spreads its flags over building systems
    by_sys = (u[u.university.isin(campuses)].assign(fy=lambda d: d.y * d.flag)
              .groupby("system").agg(units=("y", "size"), flagged=("flag", "sum"),
                                     pos=("y", "sum"), flagged_pos=("fy", "sum")))
    by_sys["share_flagged"] = by_sys.flagged / by_sys.units
    by_sys["base_rate"] = by_sys.pos / by_sys.units
    by_sys["precision_flags"] = by_sys.flagged_pos / by_sys.flagged.replace(0, np.nan)
    by_sys["share_of_all_flags"] = by_sys.flagged / by_sys.flagged.sum()
    by_sys["share_of_all_units"] = by_sys.units / by_sys.units.sum()
    spread = ra[ra.n_units >= ROUTE_MIN_UNITS].groupby("system").share_flagged.agg(
        ["min", "median", "max", "size"])
    per_system = {s: {"units": int(r.units), "flagged": int(r.flagged),
                      "share_flagged": rnd(r.share_flagged), "base_rate": rnd(r.base_rate),
                      "precision_flags": rnd(r.precision_flags),
                      "share_of_all_flags": rnd(r.share_of_all_flags),
                      "share_of_all_units": rnd(r.share_of_all_units),
                      "share_flagged_across_campuses": (
                          {"min": rnd(spread.loc[s, "min"]), "median": rnd(spread.loc[s, "median"]),
                           "max": rnd(spread.loc[s, "max"]), "n_campuses": int(spread.loc[s, "size"])}
                          if s in spread.index else None)}
                  for s, r in by_sys.sort_values("share_flagged", ascending=False).iterrows()}
    gated = ra[ra.n_units >= ROUTE_MIN_UNITS]
    summary["per_system_share_flagged"] = per_system
    summary["share_flagged_spread"] = {
        "overall_share_flagged": rnd(u.flag.mean()),
        "pairs_with_at_least_100_units": int(len(gated)),
        "pair_share_flagged_min": rnd(gated.share_flagged.min()),
        "pair_share_flagged_median": rnd(gated.share_flagged.median()),
        "pair_share_flagged_max": rnd(gated.share_flagged.max()),
        "pairs_share_flagged_below_0.02": int((gated.share_flagged < 0.02).sum()),
        "pairs_share_flagged_above_0.20": int((gated.share_flagged > 0.20).sum()),
        "spearman_pair_share_flagged_vs_base_rate": spearman_or_none(gated.share_flagged,
                                                                     gated.base_rate),
        "system_share_flagged_min": rnd(by_sys.share_flagged.min()),
        "system_share_flagged_max": rnd(by_sys.share_flagged.max()),
    }
    return ra, {"reproduction": repro, "summary": summary}


# --------------------------------------------------------------------------
# (b) percentile scale for the pooled quantities
# --------------------------------------------------------------------------
def with_system_percentile(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["R_system"] = d.groupby(["university", "period_q", "system"], observed=True)["score"] \
        .rank(pct=True, method="average")
    return d


def scale_variant(devs: dict, pooled_frame: pd.DataFrame, campuses: list[str],
                  outer0: dict, pooled0: dict, col: str) -> tuple[dict, dict]:
    """Gates with `col` in place of the raw score for the pooled top-decile lift
    and the risk-gradient bands; all other gates as stored."""
    def one(frame, trace, base):
        f = frame.copy()
        f["score"] = f[col]
        var = cc.gates_for(f, trace, col, True)
        out = {}
        for s, v0 in base.items():
            v = var[s]
            assert v["n_campuses"] == v0["n_campuses"] and v["n_pos"] == v0["n_pos"]
            sc = dict(v0["scores"])
            sc["risk_concentration"] = v["scores"]["risk_concentration"]
            sc["risk_gradient"] = v["scores"]["risk_gradient"]
            out[s] = {"scores": sc, "S": cc.sufficiency(sc), "pooled_lift": v["pooled_lift"],
                      "top_band": v.get("top_band"), "rho": v.get("rho")}
        return out
    outer = {j: one(devs[j], cc.trace_for(j), outer0[j]) for j in campuses}
    pooled = one(pooled_frame, cc.trace_for(None), pooled0)
    return outer, pooled


def compare_maps(base_outer, base_pooled, var_outer, var_pooled, campuses) -> dict:
    def diff(keyed):
        changes, gate_changes, dS, dlift = [], {g: 0 for g in GATES}, [], []
        transitions = {g: {} for g in ("risk_concentration", "risk_gradient")}
        below2 = [0, 0]
        for key, v0, v1 in keyed:
            dS.append(v1["S"] - v0["S"])
            dlift.append(v1["pooled_lift"] - v0["pooled_lift"])
            below2[0] += int(v0["pooled_lift"] < 2.0); below2[1] += int(v1["pooled_lift"] < 2.0)
            for g in GATES:
                gate_changes[g] += int(v0["scores"][g] != v1["scores"][g])
            for g in transitions:
                if v0["scores"][g] != v1["scores"][g]:
                    t = f"{LEVEL[v0['scores'][g]]}->{LEVEL[v1['scores'][g]]}"
                    transitions[g][t] = transitions[g].get(t, 0) + 1
            if (v0["S"] >= S_CUT) != (v1["S"] >= S_CUT):
                changes.append({"pair": key, "S_stored": round(v0["S"], 2),
                                "S_variant": round(v1["S"], 2),
                                "supported_stored": bool(v0["S"] >= S_CUT),
                                "supported_variant": bool(v1["S"] >= S_CUT),
                                "gates_changed": {g: [LEVEL[v0["scores"][g]], LEVEL[v1["scores"][g]]]
                                                  for g in GATES if v0["scores"][g] != v1["scores"][g]},
                                "pooled_lift": [v0["pooled_lift"], v1["pooled_lift"]],
                                "top_band": [v0.get("top_band"), v1.get("top_band")],
                                "rho": [v0.get("rho"), v1.get("rho")]})
        dS, dlift = np.array(dS), np.array(dlift)
        return {"n": len(dS), "n_verdicts_changed": len(changes), "verdicts_changed": changes,
                "n_gate_score_changes": gate_changes,
                "gate_score_transitions": transitions,
                "n_pooled_lift_below_2_stored": below2[0],
                "n_pooled_lift_below_2_variant": below2[1],
                "n_verdicts_changed_with_risk_concentration_change": int(sum(
                    "risk_concentration" in c["gates_changed"] for c in changes)),
                "n_S_changed": int((np.abs(dS) > 1e-9).sum()),
                "max_abs_delta_S": rnd(np.abs(dS).max(), 3),
                "pooled_lift_delta_median": rnd(np.median(dlift), 3),
                "pooled_lift_delta_min": rnd(dlift.min(), 3),
                "pooled_lift_delta_max": rnd(dlift.max(), 3)}
    held = [(f"{j}/{s}", base_outer[j][s], var_outer[j][s]) for j in campuses
            for s in sorted(base_outer[j])]
    pool = [(s, base_pooled[s], var_pooled[s]) for s in sorted(base_pooled)]
    return {"held_out_maps": diff(held), "pooled_map": diff(pool)}


# --------------------------------------------------------------------------
# (c) ties
# --------------------------------------------------------------------------
def tie_table(devs: dict, campuses: list[str], outer: dict, heldout: pd.DataFrame | None) -> dict:
    """Per pair: the tie at the boundary of the pooled top decile, of each
    evaluable peer campus's top decile and of the held-out campus's top decile."""
    out = {}
    for j in campuses:
        d = devs[j]
        for s in outer[j]:
            g = d[d.system == s]
            pb = boundary(g.y.to_numpy(), g.score.to_numpy())
            peers = []
            for _, gg in g.groupby("university", observed=True):
                if gg.y.sum() >= CTX_MIN_POS and len(gg) >= CTX_MIN_ROWS:
                    peers.append(boundary(gg.y.to_numpy(), gg.score.to_numpy()))
            rec = {"pooled_lift_n_at_boundary_score": pb["n_eq"],
                   "pooled_lift_tie_straddles": pb["straddles"],
                   "pooled_lift_tied_units_excluded": pb["n_tied_excluded"],
                   "peer_lifts_with_straddling_tie": int(sum(b["straddles"] for b in peers)),
                   "peer_lifts_evaluable": len(peers),
                   "peer_lifts_max_n_at_boundary_score": int(max((b["n_eq"] for b in peers if b["straddles"]),
                                                                 default=0)),
                   "peer_lifts_tied_units_excluded": int(sum(b["n_tied_excluded"] for b in peers))}
            y = g.y.to_numpy()
            rec["pooled_lift_low"] = tie_lift(y, g.score.to_numpy(), K, "low")
            rec["pooled_lift_high"] = tie_lift(y, g.score.to_numpy(), K, "high")
            if heldout is not None:
                h = heldout[(heldout.university == j) & (heldout.system == s)]
                if len(h) >= CTX_MIN_ROWS and h.y.sum() >= CTX_MIN_POS:
                    hb = boundary(h.y.to_numpy(), h.score.to_numpy())
                    rec.update({"heldout_lift_tie_straddles": hb["straddles"],
                                "heldout_lift_n_at_boundary_score": hb["n_eq"],
                                "heldout_lift_low": tie_lift(h.y.to_numpy(), h.score.to_numpy(), K, "low"),
                                "heldout_lift_high": tie_lift(h.y.to_numpy(), h.score.to_numpy(), K, "high"),
                                "heldout_lift_fractional": tie_lift(h.y.to_numpy(), h.score.to_numpy(),
                                                                    K, "fractional")})
            out[(j, s)] = rec
    return out


def pooled_tie_table(pooled_frame: pd.DataFrame, pooled: dict) -> dict:
    out = {}
    for s in pooled:
        g = pooled_frame[pooled_frame.system == s]
        pb = boundary(g.y.to_numpy(), g.score.to_numpy())
        peers = [boundary(gg.y.to_numpy(), gg.score.to_numpy())
                 for _, gg in g.groupby("university", observed=True)
                 if gg.y.sum() >= CTX_MIN_POS and len(gg) >= CTX_MIN_ROWS]
        out[s] = {"pooled_lift_n_at_boundary_score": pb["n_eq"],
                  "pooled_lift_tie_straddles": pb["straddles"],
                  "peer_lifts_with_straddling_tie": int(sum(b["straddles"] for b in peers)),
                  "peer_lifts_evaluable": len(peers)}
    return out


def tie_experiments(devs: dict, pooled_frame: pd.DataFrame, campuses: list[str],
                    keys: list, pkeys: list, label: str) -> dict:
    """Verdicts under random tie-breaking, fractional inclusion and the two
    extreme resolutions."""
    base_outer, base_pooled = gate_maps(devs, pooled_frame, campuses)
    S0, P0 = verdict_vector(base_outer, base_pooled, keys, pkeys)
    res = {"stored": (S0, P0)}
    for mode in ("fractional", "low", "high"):
        with tie_rule(mode):
            o, p = gate_maps(devs, pooled_frame, campuses)
        res[mode] = verdict_vector(o, p, keys, pkeys)
    # S is non-decreasing in every lift, so the extremes bracket every resolution
    for a, b in ((res["low"][0], S0), (S0, res["high"][0]),
                 (res["low"][1], P0), (P0, res["high"][1])):
        ok = np.isnan(a) | (a <= b + 1e-9)
        assert ok.all(), f"{label}: extreme resolutions do not bracket the stored S"
    t0 = time.time()
    RS = np.full((N_TIE_REPS, len(keys)), np.nan)
    RP = np.full((N_TIE_REPS, len(pkeys)), np.nan)
    frames = {**{j: devs[j] for j in campuses}, None: pooled_frame}
    for r in range(N_TIE_REPS):
        rng = np.random.default_rng([SEED, r])
        shuffled = {key: f.iloc[rng.permutation(len(f))] for key, f in frames.items()}
        o, p = gate_maps({j: shuffled[j] for j in campuses}, shuffled[None], campuses)
        RS[r], RP[r] = verdict_vector(o, p, keys, pkeys)
        if r == 0:
            print(f"[69] {label}: one random tie-breaking repetition in {time.time() - t0:.1f} s",
                  flush=True)
    res["random"] = (RS, RP)
    print(f"[69] {label}: {N_TIE_REPS} repetitions in {time.time() - t0:.1f} s", flush=True)
    return res


def summarize_ties(res: dict, keys: list, pkeys: list, ties: dict, pties: dict) -> dict:
    S0, P0 = res["stored"]
    RS, RP = res["random"]

    def block(S0, R, Sf, Sl, Sh, names):
        sup0 = S0 >= S_CUT
        valid = ~np.isnan(S0)
        share = np.nanmean(R >= S_CUT, axis=0)
        ever = (np.nanmin(R, axis=0) < S_CUT) & (np.nanmax(R, axis=0) >= S_CUT)
        S_changed = np.nanmax(np.abs(R - S0[None, :]), axis=0) > 1e-9
        frac_change = ((Sf >= S_CUT) != sup0) & valid
        ext_change = ((Sl >= S_CUT) != (Sh >= S_CUT)) & valid
        return {
            "n": int(valid.sum()),
            "n_supported_stored": int(sup0[valid].sum()),
            "random_tie_breaking": {
                "n_repetitions": int(R.shape[0]),
                "n_verdicts_ever_changed": int(ever[valid].sum()),
                "verdicts_ever_changed": [
                    {"pair": names[i], "supported_stored": bool(sup0[i]),
                     "share_of_repetitions_supported": rnd(share[i], 3),
                     "S_stored": rnd(S0[i], 2), "S_min": rnd(np.nanmin(R[:, i]), 2),
                     "S_max": rnd(np.nanmax(R[:, i]), 2)}
                    for i in np.where(ever & valid)[0]],
                "n_pairs_S_ever_changed": int(S_changed[valid].sum()),
                "n_supported_per_repetition_min": int(np.nanmin((R >= S_CUT).sum(axis=1))),
                "n_supported_per_repetition_max": int(np.nanmax((R >= S_CUT).sum(axis=1)))},
            "fractional_inclusion": {
                "n_verdicts_changed": int(frac_change.sum()),
                "verdicts_changed": [names[i] for i in np.where(frac_change)[0]],
                "n_supported": int((Sf[valid] >= S_CUT).sum())},
            "extreme_resolutions": {
                "n_supported_low": int((Sl[valid] >= S_CUT).sum()),
                "n_supported_high": int((Sh[valid] >= S_CUT).sum()),
                "n_verdicts_that_can_change": int(ext_change.sum()),
                "verdicts_that_can_change": [names[i] for i in np.where(ext_change)[0]]},
        }
    names = [f"{j}/{s}" for j, s in keys]
    held = block(S0, RS, res["fractional"][0], res["low"][0], res["high"][0], names)
    pool = block(P0, RP, res["fractional"][1], res["low"][1], res["high"][1], list(pkeys))
    tv = [ties[k] for k in keys if k in ties]
    held["boundary_ties"] = {
        "pairs_with_straddling_tie_pooled_lift": int(sum(t["pooled_lift_tie_straddles"] for t in tv)),
        "max_units_at_boundary_score_pooled_lift": int(max(t["pooled_lift_n_at_boundary_score"]
                                                           for t in tv)),
        "pairs_with_any_straddling_tie_peer_lift": int(sum(t["peer_lifts_with_straddling_tie"] > 0
                                                           for t in tv)),
        "peer_lifts_with_straddling_tie": int(sum(t["peer_lifts_with_straddling_tie"] for t in tv)),
        "peer_lifts_evaluated": int(sum(t["peer_lifts_evaluable"] for t in tv)),
        "max_units_at_boundary_score_peer_lift": int(max(t["peer_lifts_max_n_at_boundary_score"]
                                                         for t in tv)),
        "max_abs_pooled_lift_range": rnd(max(t["pooled_lift_high"] - t["pooled_lift_low"]
                                             for t in tv), 4)}
    hv = [t for t in tv if "heldout_lift_tie_straddles" in t]
    if hv:
        held["boundary_ties"]["heldout_lift_pairs_evaluated"] = len(hv)
        held["boundary_ties"]["heldout_lift_with_straddling_tie"] = int(
            sum(t["heldout_lift_tie_straddles"] for t in hv))
        held["boundary_ties"]["heldout_two_fold_status_can_change"] = int(
            sum((t["heldout_lift_low"] >= 2) != (t["heldout_lift_high"] >= 2) for t in hv))
    pv = list(pties.values())
    pool["boundary_ties"] = {
        "systems_with_straddling_tie_pooled_lift": int(sum(t["pooled_lift_tie_straddles"] for t in pv)),
        "peer_lifts_with_straddling_tie": int(sum(t["peer_lifts_with_straddling_tie"] for t in pv)),
        "peer_lifts_evaluated": int(sum(t["peer_lifts_evaluable"] for t in pv))}
    return {"held_out_maps": held, "pooled_map": pool}


# --------------------------------------------------------------------------
def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if np.isnan(o) else float(o)
    return o


def main() -> None:
    global N_TIE_REPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=N_TIE_REPS)
    args = ap.parse_args()
    N_TIE_REPS = args.reps
    t0 = time.time()
    out_json = C.METRICS / "route_aligned_evaluation.json"
    out_csv = C.TABLES / "route_aligned_evaluation.csv"
    for f in (out_json, out_csv):
        assert not f.exists(), f"{f.name} exists; this script does not overwrite stored results"

    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    devs, pooled_frame = gbdt_frames(single, pairs, campuses)
    outer, pooled = gate_maps(devs, pooled_frame, campuses)
    routing = cc.route(single, outer, campuses, RCOL)
    ref = check_reference(outer, pooled, routing, campuses)
    print(f"[69] stored verdicts reproduced: {ref}", flush=True)

    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    keys = list(zip(fv.campus, fv.system))
    pkeys = sorted(pooled)

    # ---------------- (a) ----------------
    ra, route_out = route_aligned(single, outer, campuses, fv)
    print(f"[69] (a) route reproduction: {route_out['reproduction']}", flush=True)

    # ---------------- (b) ----------------
    devs_p = {j: with_system_percentile(d) for j, d in devs.items()}
    pooled_p = with_system_percentile(pooled_frame)
    scale = {}
    var_maps = {}
    for name, col in (("system_percentile", "R_system"), ("campus_percentile_R", RCOL)):
        vo, vp = scale_variant(devs_p, pooled_p, campuses, outer, pooled, col)
        var_maps[name] = (vo, vp)
        scale[name] = compare_maps(outer, pooled, vo, vp, campuses)
        print(f"[69] (b) {name}: held-out changes "
              f"{scale[name]['held_out_maps']['n_verdicts_changed']}, pooled changes "
              f"{scale[name]['pooled_map']['n_verdicts_changed']}", flush=True)
    # with the campus-wide R, the risk-gradient gate is the stored one
    assert scale["campus_percentile_R"]["held_out_maps"]["n_gate_score_changes"]["risk_gradient"] == 0
    assert scale["campus_percentile_R"]["pooled_map"]["n_gate_score_changes"]["risk_gradient"] == 0

    # ---------------- (c) ----------------
    ties = tie_table(devs, campuses, outer, single)
    pties = pooled_tie_table(pooled_frame, pooled)
    tie_res = tie_experiments(devs, pooled_frame, campuses, keys, pkeys, "screening model")
    assert np.allclose(tie_res["stored"][0], [outer[j][s]["S"] for j, s in keys])
    tie_sum = summarize_ties(tie_res, keys, pkeys, ties, pties)

    rdevs, rpooled, rule_name = rule_frames(single, pairs, campuses)
    r_outer, r_pooled = gate_maps(rdevs, rpooled, campuses)
    rule_ref = check_rule_reference(r_outer, r_pooled, rule_name)
    print(f"[69] count-rule maps reproduced: {rule_ref}", flush=True)
    rkeys = [(j, s) for j in campuses for s in sorted(r_outer[j])]
    rpkeys = sorted(r_pooled)
    rties = tie_table(rdevs, campuses, r_outer, None)
    rpties = pooled_tie_table(rpooled, r_pooled)
    rtie_res = tie_experiments(rdevs, rpooled, campuses, rkeys, rpkeys, "prior-year count rule")
    rtie_sum = summarize_ties(rtie_res, rkeys, rpkeys, rties, rpties)

    # ---------------- table, one row per pair ----------------
    S_rand = tie_res["random"][0]
    rS = dict(zip(rkeys, range(len(rkeys))))
    extra = []
    for i, (j, s) in enumerate(keys):
        t = ties[(j, s)]
        e = {}
        for name, (vo, _) in var_maps.items():
            v = vo[j][s]
            e[f"pooled_lift_{name}"] = v["pooled_lift"]
            e[f"top_band_{name}"] = v.get("top_band")
            e[f"rho_{name}"] = v.get("rho")
            e[f"risk_concentration_{name}"] = v["scores"]["risk_concentration"]
            e[f"risk_gradient_{name}"] = v["scores"]["risk_gradient"]
            e[f"S_{name}"] = round(v["S"], 2)
            e[f"supported_{name}"] = bool(v["S"] >= S_CUT)
        e.update({k: (rnd(v, 4) if isinstance(v, float) else v) for k, v in t.items()})
        e["share_supported_random_ties"] = rnd(float(np.mean(S_rand[:, i] >= S_CUT)), 3)
        e["supported_fractional_ties"] = bool(tie_res["fractional"][0][i] >= S_CUT)
        e["supported_low_ties"] = bool(tie_res["low"][0][i] >= S_CUT)
        e["supported_high_ties"] = bool(tie_res["high"][0][i] >= S_CUT)
        if (j, s) in rS:
            ri = rS[(j, s)]
            rt = rties[(j, s)]
            e.update({"count_rule_S": round(rtie_res["stored"][0][ri], 2),
                      "count_rule_supported": bool(rtie_res["stored"][0][ri] >= S_CUT),
                      "count_rule_share_supported_random_ties":
                          rnd(float(np.mean(rtie_res["random"][0][:, ri] >= S_CUT)), 3),
                      "count_rule_supported_fractional_ties": bool(rtie_res["fractional"][0][ri] >= S_CUT),
                      "count_rule_supported_low_ties": bool(rtie_res["low"][0][ri] >= S_CUT),
                      "count_rule_supported_high_ties": bool(rtie_res["high"][0][ri] >= S_CUT),
                      "count_rule_pooled_lift_n_at_boundary_score": rt["pooled_lift_n_at_boundary_score"],
                      "count_rule_pooled_lift_tied_units_excluded": rt["pooled_lift_tied_units_excluded"],
                      "count_rule_peer_lifts_with_straddling_tie": rt["peer_lifts_with_straddling_tie"]})
        extra.append(e)
    table = pd.concat([ra.reset_index(drop=True), pd.DataFrame(extra)], axis=1)
    for c in table.columns:
        if table[c].dtype == float:
            table[c] = table[c].round(4)
    table.to_csv(out_csv, index=False)

    out = {
        "design": ("The 141 verdicts of the held-out maps (double nesting, risk percentile within "
                   "campus and quarter, capped transfer gate) recomputed with the functions of "
                   "45_certification_corrected.py and read against the units that the routing rule "
                   "flags on the held-out campus; the pooled gate quantities recomputed on a "
                   "percentile scale; and the verdicts under other resolutions of ties at a "
                   "top-decile boundary, for the screening model and for the prior-year count rule."),
        "definitions": {
            "flag": "unit with risk percentile R >= 0.90, R the percentile of the f^(-c) score within "
                    "campus and quarter over all building systems (R_quarter), as in route() of script 45",
            "precision_flags": "share of flagged units of the system on the held-out campus that have "
                               "a high-burden next quarter",
            "base_rate": "share of all units of the system on the held-out campus with a high-burden "
                         "next quarter",
            "route_lift": "precision_flags / base_rate",
            "route_evaluable": (f"at least {ROUTE_MIN_FLAGGED} flagged units, at least {ROUTE_MIN_POS} "
                                f"high-burden quarters and at least {ROUTE_MIN_UNITS} units of the "
                                "system on the held-out campus; fixed before any outcome was read"),
            "bootstrap": (f"{N_BOOT_ROUTE} resamples of the buildings of the campus that hold units of "
                          "the system, with replacement, R and flags held at their stored values; "
                          "95% percentile interval; seed [42, pair index]"),
            "heldout_lift": "top-decile lift of f^(-c) within the system on campus c (raw score, ties "
                            "in stored order), as in fold_gate_verdicts.csv",
            "system_percentile": "percentile of the score within campus, quarter and building system",
            "campus_percentile_R": "the campus-wide risk percentile R",
            "scale_variant": ("the pooled top-decile lift and the risk-gradient bands read the stated "
                              "percentile instead of the raw score; per-campus lifts, data sufficiency "
                              "and prior-record trace as stored; S by sufficiency() of script 45"),
            "tie_straddles": ("more units share the boundary score of the top decile than the decile has "
                              "room for after the units scoring above it, so the stored record order "
                              "decides which of them enter"),
            "random_tie_breaking": (f"rows of every map put in random order before the gates are "
                                    f"computed, {N_TIE_REPS} repetitions, seeds [42, r]"),
            "fractional_inclusion": "tied units at the boundary enter the top decile with equal "
                                    "fractional weight (expected precision under random tie-breaking)",
            "extreme_resolutions": ("tied units taken non-high-burden first (low) and high-burden first "
                                    "(high); S is non-decreasing in every lift, so a verdict equal at both "
                                    "extremes is the same under any resolution of the ties"),
            "count_rule": f"{rule_name}: prior-year reactive work-order count as the risk score, maps "
                          "built as in scripts 48 and 59 on the units and high-burden labels of the "
                          "screening model's maps",
        },
        "reference_check": ref,
        "count_rule_reference_check": rule_ref,
        "route_aligned": route_out,
        "pooled_gate_scale": scale,
        "ties": {"screening_model": tie_sum, "count_rule": rtie_sum,
                 "count_rule_pairs_without_screening_model_verdict":
                     [f"{j}/{s}" for j, s in rkeys if (j, s) not in set(keys)]},
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(jsonable(out), open(out_json, "w"), indent=2)

    s = route_out["summary"]
    print(f"\n=== (a) route-evaluable pairs: {s['n_route_evaluable']} "
          f"({s['n_route_evaluable_supported']} supported, "
          f"{s['n_route_evaluable_not_supported']} not supported)")
    for k, v in s["route_evaluable"].items():
        print(f"  {k:14s} {({x: y for x, y in v.items() if x != 'pairs'})}")
    print(f"  Spearman route lift vs held-out lift: {s['spearman_route_lift_vs_heldout_top_decile_lift']}")
    print(f"  Spearman route lift vs S: {s['spearman_route_lift_vs_S']}")
    print(f"  overlap: {s['overlap_with_system_top_decile']}")
    print(f"  spread: {s['share_flagged_spread']}")
    print("\n=== (b) percentile scale ===")
    for k, v in scale.items():
        print(f"  {k}: held-out {v['held_out_maps']['n_verdicts_changed']} "
              f"{[c['pair'] for c in v['held_out_maps']['verdicts_changed']]}, "
              f"pooled {v['pooled_map']['n_verdicts_changed']} "
              f"{[c['pair'] for c in v['pooled_map']['verdicts_changed']]}, "
              f"gate changes {v['held_out_maps']['n_gate_score_changes']}")
    print("\n=== (c) ties ===")
    for lab, ts in (("model", tie_sum), ("count rule", rtie_sum)):
        h = ts["held_out_maps"]
        print(f"  {lab}: random ever changed {h['random_tie_breaking']['n_verdicts_ever_changed']}, "
              f"fractional changed {h['fractional_inclusion']['n_verdicts_changed']}, "
              f"can change {h['extreme_resolutions']['n_verdicts_that_can_change']}; "
              f"ties {h['boundary_ties']}")
        print(f"    pooled: {ts['pooled_map']['random_tie_breaking']['n_verdicts_ever_changed']} "
              f"{ts['pooled_map']['extreme_resolutions']}")
    print(f"\n[69] done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
