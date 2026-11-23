#!/usr/bin/env python
"""
73_graded_transfer_loss.py -- the gates and simpler rules that read the scores,
on the real records, when the ranking of one building system is weakened step
by step or reversed on a share of the peer campuses.

The permuted-score test (66_deranged_ranking_transfer.py) removes the ranking
information of one building system completely on k of its evaluable campuses.
This script grades that loss and compares the evidence gating with rules that
also read the scores, under the same perturbations.

CASES.  The cases of script 66: each building system supported on a map and
evaluable on at least six of its campuses.  On the pooled map (nine campuses,
each read through f^(-d)) these are the nine stable building systems; on the
held-out maps (the eight peers of target campus c, each read through
f^(-{c,d})) they are 84 campus-system pairs.

AFFECTED CAMPUSES.  For a share q in {1/4, 1/2, 3/4, 1} of the n evaluable
campuses of a case, k = ceil(q n) campuses are drawn at random; the draw is
made exactly as in script 66, from numpy.random.default_rng([RANDOM_SEED, map
index, system index, k, draw]), so the same campuses are affected under every
perturbation of a given draw.  Thirty draws per cell.

PERTURBATIONS, applied on each affected campus to the units of the system
within every quarter.  Each one reassigns the scores of the system's units in a
campus and quarter among themselves, so the set of scores of every
campus-quarter is unchanged and the within-campus-quarter percentile R travels
with its score (checked once per map and perturbation against a full
recomputation of the ranks).
  partial permutation, a in {0.25, 0.5, 0.75, 1}: in every quarter a randomly
      chosen fraction a of the system's units (round(a m) of m) have their
      scores permuted at random among themselves; the others keep theirs.  At
      a = 1 this is the permutation of script 66, drawn with the same generator,
      so its draws equal the stored draws of script 66 (asserted draw by draw).
  rank mixing, w in {0.25, 0.5, 0.75}: in every quarter the units are ordered
      by (1 - w) x (rank of the true score) + w x (rank of a random
      permutation), and the quarter's scores are handed out in that order (the
      unit first in the order receives the lowest score).  w = 0 keeps the
      ranking and w = 1 is a full permutation.
  reversal: in every quarter the order of the system's scores is inverted,
      the unit with the highest score receiving the lowest, and so on.
The labels, the counts, the prior-record trace and every other building system
are not touched.

RULES, each recomputed in every draw.  A rule withholds when:
  evidence gating                 S < 2/3 (gates_for of script 45 at the main
                                  setting; the withholding gate is named as in
                                  script 66)
  transfer only, fail             the transfer-stability gate scores fail
                                  (share of evaluable campuses at two-fold
                                  below 0.6)
  transfer only, fail or caution  the transfer-stability gate does not pass
                                  (share below 0.8)
  pooled lift, raw score          pooled top-decile lift below 2 (the quantity
                                  of the risk-concentration gate)
  pooled lift, system percentile  the same with the score replaced by its
                                  percentile within campus, quarter and
                                  building system (script 69)
  pooled lift, campus-wide R      the same with the score replaced by R
                                  (script 69)
  partially pooled rule           predictive probability of two-fold lift on a
                                  new campus below 0.8 (script 67)
  count-based rule                script 56; it reads counts only and cannot
                                  change (asserted in every draw).
The pooled lifts are unrounded here; the stored values are rounded to two
decimals, and the unperturbed check compares at that rounding.

PARTIALLY POOLED RULE IN A DRAW.  Each evaluable campus contributes its log
top-decile lift and a sampling variance.  A campus that is not affected keeps
its stored variance, the 1,000-replicate building bootstrap of script 67 with
script 67's seeds (reproduced, and P asserted equal to the stored value for
every case).  An affected campus gets the variance of its perturbed log lift
from a reduced building bootstrap of B = 200 replicates (boot_log_lift of
script 67, same continuity correction).  The reduced bootstrap is validated on
the unperturbed data: P is recomputed for the 141 pairs and the 16 systems of
the pooled map with every campus's variance from 200 replicates and compared
with the stored P and verdicts.

INVARIANTS (the runner aborts on a mismatch).
  - The full pipeline reproduces the stored pooled map and the 141 verdicts of
    the held-out maps (check_reference of script 66).
  - With no perturbation every rule reproduces its stored verdict and quantity:
    S and the gate scores (cert_corrected.json, fold_gate_verdicts.csv), the
    pooled lifts on the raw and the two percentile scales
    (route_aligned_evaluation.csv; for the pooled map, scale_variant of
    script 69), P (partial_pooling_rule.csv and .json) and the count-based rule.
  - Partial permutation at a = 1: every draw equals the stored draw of script
    66 with the same seed (gate scores, S, verdict, withholding gate, pooled
    lift, share at two-fold).
  - The count-based rule never changes.

Seeds.  Campus draw and the permutation at a = 1: [RANDOM_SEED, map index,
system index, k, draw]; the other perturbations add [perturbation code,
100 x strength]; the reduced bootstrap adds a further 7.  RANDOM_SEED = 42.

Usage:  python scripts/73_graded_transfer_loss.py [--draws 30] [--workers 5]
        [--maps pooled|all]

Outputs -> results/metrics/graded_transfer_loss.json
           results/tables/graded_transfer_loss.csv   (one row per case and cell)
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FMSCREEN_DEVICE", "cpu")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "FMSCREEN_THREADS"):
    os.environ.setdefault(_v, "1")

import sys, json, time, math, argparse, importlib.util
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import metrics as MET


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


dr = _load("deranged_ranking_transfer", "66_deranged_ranking_transfer.py")
pp = _load("partial_pooling_rule", "67_partial_pooling_rule.py")
va = _load("route_aligned_evaluation", "69_route_aligned_evaluation.py")
cc, sr = dr.cc, dr.sr
assert va.MET is MET and dr.MET is MET and pp.MET is MET

GATES, HARD = cc.GATES, cc.HARD
FAIL, CAUTION, PASS = cc.FAIL, cc.CAUTION, cc.PASS
S_CUT = cc.S_CUT
RCOL = dr.RCOL
K = C.TOPK_BUDGET
SEED = C.RANDOM_SEED
LIFT_LEVEL = 2.0
P0 = pp.P0_PRINCIPAL
B_REDUCED = 200
SHARES = (0.25, 0.5, 0.75, 1.0)
PERTURBATIONS = {"partial_permutation": (0.25, 0.5, 0.75, 1.0),
                 "rank_mixing": (0.25, 0.5, 0.75),
                 "reversal": (1.0,)}
PCODE = {"none": 0, "partial_permutation": 1, "rank_mixing": 2, "reversal": 3}
RULES = ["evidence_gating", "transfer_only_fail", "transfer_only_fail_or_caution",
         "pooled_lift_raw", "pooled_lift_system_percentile", "pooled_lift_campus_R",
         "partially_pooled", "count_based"]
SIMPLER = [r for r in RULES if r not in ("evidence_gating", "count_based")]
REASONS = dr.REASONS
LEVEL = {FAIL: "fail", CAUTION: "caution", PASS: "pass"}
THRESHOLD_SHARE = 0.9

OUT_JSON = C.METRICS / "graded_transfer_loss.json"
OUT_CSV = C.TABLES / "graded_transfer_loss.csv"


def k_for(q: float, n: int) -> int:
    """Number of affected campuses for share q of n evaluable campuses."""
    return int(math.ceil(q * n - 1e-9))


# --------------------------------------------------------------------------
# one case: a building system on one map
# --------------------------------------------------------------------------
@dataclass
class Case:
    cell: object             # the Cell of script 66
    idx: dict                # campus -> row positions in the system's frame (frame order)
    bld: dict                # campus -> building codes of those rows (as in script 67)
    gcode: np.ndarray        # campus-quarter group of every row
    own0: dict               # campus -> unperturbed own lift
    pe0: dict                # campus -> (log lift, variance) from the stored bootstrap
    P0: float | None
    lift_sys0: float
    lift_R0: float
    lift_raw0: float


CASES: dict[tuple, Case] = {}


def point_log_lift(y: np.ndarray, sc: np.ndarray) -> float:
    """The log lift that script 67 enters for one campus: the uncorrected lift,
    continuity-corrected only where no high-burden unit reaches the top decile."""
    lift = float(MET.lift_at_topk(y, sc, K))
    top = MET._topk_mask(sc, K)
    M, a = int(top.sum()), int(y[top].sum())
    if a == 0:
        return float(np.log((a + pp.CC_HALF) / M) - np.log((y.sum() + pp.CC_HALF) / len(y)))
    return float(np.log(lift))


def pp_seed_table(campuses: list[str], systems_all: list[str], seed: int) -> dict:
    """The per-(corpus, campus, system) bootstrap seeds of script 67's collect()."""
    corpora = ["pooled"] + campuses
    keys = [(x, d, s) for x in corpora for d in campuses for s in systems_all]
    arr = np.random.default_rng(seed).integers(0, 2**63 - 1, size=len(keys))
    return dict(zip(keys, (int(s) for s in arr)))


def build_case(kind, target, mi, system, si, mapframe, trace, base, corpus, seeds) -> Case:
    cell = dr.build_cell(kind, target, mi, system, si, mapframe, trace, base)
    g = mapframe[mapframe.system == system]
    bnames = g["building"].astype(str).to_numpy()
    assert np.array_equal(g["university"].astype(str).to_numpy(), cell.univ)
    idx, bld, own0 = {}, {}, {}
    for d in cell.evaluable:
        ii = np.flatnonzero(cell.univ == d)
        idx[d] = ii
        bld[d] = pd.factorize(bnames[ii])[0]
        own0[d] = float(MET.lift_at_topk(cell.y[ii], cell.score[ii], K))
    gcode = (pd.DataFrame({"u": cell.univ, "q": cell.frame["period_q"].to_numpy()})
             .groupby(["u", "q"], sort=False).ngroup().to_numpy())
    # script 67's peer evidence with its own seeds
    ev = pp.peer_evidence(mapframe, system,
                          lambda d, s_, x=corpus: np.random.default_rng(seeds[(x, d, s_)]))
    assert [e["campus"] for e in ev] == cell.evaluable, (kind, target, system)
    pe0 = {e["campus"]: (e["log_lift"], e["var_log_lift"]) for e in ev}
    for e in ev:
        assert abs(e["lift"] - own0[e["campus"]]) < 1e-12
        assert abs(point_log_lift(cell.y[idx[e["campus"]]], cell.score[idx[e["campus"]]])
                   - e["log_lift"]) < 1e-12
    yv = np.array([pe0[d][0] for d in cell.evaluable]); vv = np.array([pe0[d][1] for d in cell.evaluable])
    P = pp.random_effects(yv, vv, "REML")["P_lift_ge_2"]
    lift_sys = float(MET.lift_at_topk(cell.y, system_percentile(cell.score, gcode), K))
    lift_R = float(MET.lift_at_topk(cell.y, cell.R, K))
    lift_raw = float(MET.lift_at_topk(cell.y, cell.score, K))
    return Case(cell, idx, bld, gcode, own0, pe0, P, lift_sys, lift_R, lift_raw)


def system_percentile(score: np.ndarray, gcode: np.ndarray) -> np.ndarray:
    """Percentile of the score within campus and quarter among the system's
    units, as with_system_percentile of script 69 computes it."""
    return pd.Series(score).groupby(gcode).rank(pct=True, method="average").to_numpy()


# --------------------------------------------------------------------------
# perturbations
# --------------------------------------------------------------------------
def _groups(qs: np.ndarray):
    """Start index and size of each run of equal quarters (qs sorted)."""
    brk = np.flatnonzero(np.diff(qs)) + 1
    starts = np.concatenate([[0], brk])
    sizes = np.diff(np.concatenate([starts, [len(qs)]]))
    return starts, sizes


def perturb(case: Case, picked: list, pert: str, strength: float, rng: np.random.Generator):
    cell = case.cell
    score, R = cell.score.copy(), cell.R.copy()
    for d in picked:
        pos, qs = cell.rows[d]                       # positions sorted by quarter (stable)
        starts, sizes = _groups(qs)
        size_of = np.repeat(sizes, sizes)
        start_of = np.repeat(starts, sizes)
        if pert == "partial_permutation":
            u = rng.random(len(pos))
            o = np.lexsort((u, qs))                  # random order within each quarter
            rank = np.arange(len(pos)) - start_of    # position within the quarter after o
            n_sel = np.floor(strength * size_of + 0.5).astype(int)
            sel = o[rank < n_sel]                    # chosen units (indices into pos)
            if len(sel) == 0:
                continue
            v = rng.random(len(sel))
            src = sel[np.lexsort((v, qs[sel]))]      # same quarters, random order
            dst = sel[np.lexsort((np.arange(len(sel)), qs[sel]))]
            score[pos[dst]] = cell.score[pos[src]]
            R[pos[dst]] = cell.R[pos[src]]
        elif pert == "rank_mixing":
            sc = cell.score[pos]
            o = np.lexsort((sc, qs))                 # ascending score within quarter
            r_true = np.empty(len(pos)); r_true[o] = np.arange(len(pos)) - start_of
            o2 = np.lexsort((rng.random(len(pos)), qs))
            r_rand = np.empty(len(pos)); r_rand[o2] = np.arange(len(pos)) - start_of
            key = (1.0 - strength) * r_true + strength * r_rand
            o3 = np.lexsort((rng.random(len(pos)), key, qs))   # new order within quarter
            score[pos[o3]] = cell.score[pos[o]]      # j-th in the new order gets the j-th lowest score
            R[pos[o3]] = cell.R[pos[o]]
        elif pert == "reversal":
            sc = cell.score[pos]
            o = np.lexsort((sc, qs))                 # ascending score within quarter
            rank = np.arange(len(pos)) - start_of
            partner = start_of + (size_of - 1 - rank)
            score[pos[o]] = cell.score[pos[o[partner]]]
            R[pos[o]] = cell.R[pos[o[partner]]]
        else:
            raise ValueError(pert)
    return score, R


def perturbed(case: Case, pert: str, strength: float, k: int, j: int):
    cell = case.cell
    rng = np.random.default_rng([SEED, cell.map_i, cell.sys_i, k, j])
    picked = dr.pick(cell, k, rng)
    if pert == "none":
        return picked, cell.score.copy(), cell.R.copy()
    if pert == "partial_permutation" and strength == 1.0:
        score, R = dr.deranged_scores(cell, picked, rng)      # the draw of script 66
        return picked, score, R
    rng2 = np.random.default_rng([SEED, cell.map_i, cell.sys_i, k, j, PCODE[pert],
                                  int(round(100 * strength))])
    score, R = perturb(case, picked, pert, strength, rng2)
    return picked, score, R


# --------------------------------------------------------------------------
# one draw
# --------------------------------------------------------------------------
def partially_pooled_P(case: Case, score: np.ndarray, picked: list, rng) -> float | None:
    cell = case.cell
    pk = set(picked)
    yv, vv = [], []
    for d in cell.evaluable:
        if d in pk:
            ii = case.idx[d]
            y, sc = cell.y[ii], score[ii]
            yv.append(point_log_lift(y, sc))
            v = pp.boot_log_lift(y, sc, case.bld[d], rng, B_REDUCED, n_check=0)["var_log_lift"]
            assert v > 0, (cell.target, cell.system, d)
            vv.append(v)
        else:
            yv.append(case.pe0[d][0]); vv.append(case.pe0[d][1])
    return pp.random_effects(np.array(yv), np.array(vv), "REML")["P_lift_ge_2"]


def draw(key: tuple, pert: str, strength: float, q: float, k: int, j: int) -> dict:
    case = CASES[key]
    cell = case.cell
    picked, score, R = perturbed(case, pert, strength, k, j)
    frame = cell.frame.assign(score=score, **{RCOL: R})
    v = cc.gates_for(frame, cell.trace, RCOL, True)[cell.system]
    sc, S = v["scores"], v["S"]
    rule = sr.simple_rule(v["n_campuses"], v["n_pos"], cell.zp, **sr.PRESPECIFIED)
    assert rule == cell.count_rule0 and v["n_campuses"] == cell.base["n_campuses"] \
        and v["n_pos"] == cell.base["n_pos"], f"count-based rule changed: {key} {pert} k={k} draw={j}"
    lift_raw = float(MET.lift_at_topk(cell.y, score, K))
    lift_sys = float(MET.lift_at_topk(cell.y, system_percentile(score, case.gcode), K))
    lift_R = float(MET.lift_at_topk(cell.y, R, K))
    if pert == "none":
        P = case.P0
    else:
        rng_b = np.random.default_rng([SEED, cell.map_i, cell.sys_i, k, j, PCODE[pert],
                                       int(round(100 * strength)), 7])
        P = partially_pooled_P(case, score, picked, rng_b)
    own = {d: float(MET.lift_at_topk(cell.y[case.idx[d]], score[case.idx[d]], K)) for d in picked}
    pos_aff = int(sum(cell.y[case.idx[d]].sum() for d in picked))
    wb = dr.withheld_by(sc, S)
    withhold = {
        "evidence_gating": bool(S < S_CUT),
        "transfer_only_fail": bool(sc["transfer_stability"] == FAIL),
        "transfer_only_fail_or_caution": bool(sc["transfer_stability"] < PASS),
        "pooled_lift_raw": bool(lift_raw < LIFT_LEVEL),
        "pooled_lift_system_percentile": bool(lift_sys < LIFT_LEVEL),
        "pooled_lift_campus_R": bool(lift_R < LIFT_LEVEL),
        "partially_pooled": bool(P is None or P < P0),
        "count_based": bool(not rule),
    }
    return {
        "map": cell.map_kind, "target_campus": cell.target, "system": cell.system,
        "n_evaluable": len(cell.evaluable), "perturbation": pert, "strength": strength,
        "q": q, "k": k, "share_affected": round(k / len(cell.evaluable), 4), "draw": j,
        "affected_campuses": ";".join(picked),
        "share_pos_affected": round(pos_aff / v["n_pos"], 4),
        "mean_own_lift_affected": (float(np.mean(list(own.values()))) if own else None),
        "mean_own_lift_affected_unperturbed": (float(np.mean([case.own0[d] for d in picked]))
                                               if picked else None),
        "min_own_lift_affected": (float(min(own.values())) if own else None),
        "pooled_lift": v["pooled_lift"], "frac_2x": v["frac_2x"],
        "top_band": v.get("top_band"), "rho": v.get("rho"),
        "pooled_lift_raw": lift_raw, "pooled_lift_system_percentile": lift_sys,
        "pooled_lift_campus_R": lift_R, "P": P,
        **{g: sc[g] for g in GATES},
        "S": round(float(S), 4), "withheld_by": wb or "none",
        **{f"withhold_{r}": withhold[r] for r in RULES},
    }


def _limit_threads() -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _task(args):
    return draw(*args)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_transport(case: Case, mapframe: pd.DataFrame, pert: str, strength: float) -> None:
    """The percentile travels with its score under the perturbation: equal to a
    recomputation of the ranks of the whole map, and the gates on the system's
    rows equal those on the full map."""
    cell = case.cell
    k = max(1, len(cell.evaluable) // 2)
    _, score, R = perturbed(case, pert, strength, k, 0)
    full = mapframe.copy()
    m = (full.system == cell.system).to_numpy()
    s_ = full["score"].to_numpy().copy(); s_[m] = score
    full["score"] = s_
    full = cc._add_ranks(full)
    assert np.array_equal(full.loc[m, RCOL].to_numpy(), R), f"percentile does not travel ({pert})"
    a = cc.gates_for(full, cell.trace, RCOL, True)[cell.system]
    b = cc.gates_for(cell.frame.assign(score=score, **{RCOL: R}), cell.trace, RCOL, True)[cell.system]
    assert a == b, f"gates differ between the full map and the system's rows ({pert})"
    # the scores of every campus-quarter are a rearrangement of the stored ones
    f0 = cell.frame.assign(s=cell.score).sort_values(["university", "period_q", "s"])
    f1 = cell.frame.assign(s=score).sort_values(["university", "period_q", "s"])
    assert np.array_equal(f0.s.to_numpy(), f1.s.to_numpy())


def reduced_bootstrap_validation(maps: list, systems: list, stored_ho: pd.DataFrame,
                                 stored_pool: dict) -> dict:
    """P with every campus's variance from B_REDUCED replicates, against the
    stored P of script 67 (1,000 replicates), on the unperturbed data."""
    rows = []
    for mi, (kind, tgt, frame, trace, res) in enumerate(maps):
        for s in sorted(res):
            g = frame[frame.system == s]
            yv, vv = [], []
            for ci, (d, gg) in enumerate(g.groupby("university", observed=True, sort=True)):
                y, sc = gg["y"].to_numpy(), gg["score"].to_numpy()
                if not (y.sum() >= dr.CTX_MIN_POS and len(y) >= dr.CTX_MIN_ROWS):
                    continue
                bld = pd.factorize(gg["building"].to_numpy())[0]
                rng = np.random.default_rng([SEED, B_REDUCED, mi, systems.index(s), ci])
                yv.append(point_log_lift(y, sc))
                vv.append(pp.boot_log_lift(y, sc, bld, rng, B_REDUCED, n_check=0)["var_log_lift"])
            P = pp.random_effects(np.array(yv), np.array(vv), "REML")["P_lift_ge_2"]
            if kind == "pooled":
                Pst, acc = stored_pool[s]["P"], stored_pool[s]["accepted_p0.8"]
            else:
                r = stored_ho[(stored_ho.campus == tgt) & (stored_ho.system == s)].iloc[0]
                Pst = None if pd.isna(r.P_lift_ge_2) else float(r.P_lift_ge_2)
                acc = bool(r["accepted_p0.8"])
            rows.append({"map": kind, "target": tgt, "system": s, "k": len(yv),
                         "P_reduced": P, "P_stored": Pst,
                         "accepted_reduced": bool(P is not None and P >= P0), "accepted_stored": acc})
    d = pd.DataFrame(rows)
    w = d[d.P_stored.notna() & d.P_reduced.notna()]
    dP = (w.P_reduced - w.P_stored).abs()
    ch = d[d.accepted_reduced != d.accepted_stored]
    keys = set(CASES)
    sel = pd.Series([(a, b, c) in keys for a, b, c in zip(d["map"], d["target"], d["system"])],
                    index=d.index)
    ws = w[sel[w.index]]
    return {"B_reduced": B_REDUCED, "B_stored": pp.B_BOOT,
            "n_estimates_compared": int(len(w)),
            "max_abs_delta_P": round(float(dP.max()), 4), "median_abs_delta_P": round(float(dP.median()), 5),
            "p90_abs_delta_P": round(float(dP.quantile(0.9)), 4),
            "n_verdicts_compared": int(len(d)), "n_verdicts_changed_at_0.8": int(len(ch)),
            "verdicts_changed": [{"map": r.map, "target": r.target, "system": r.system,
                                  "P_reduced": round(float(r.P_reduced), 4) if r.P_reduced is not None else None,
                                  "P_stored": r.P_stored} for r in ch.itertuples()],
            "cases_of_this_analysis": {
                "n": int(sel.sum()),
                "max_abs_delta_P": round(float((ws.P_reduced - ws.P_stored).abs().max()), 4),
                "n_verdicts_changed_at_0.8": int((d[sel].accepted_reduced != d[sel].accepted_stored).sum())}}


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------
def summary(d: pd.DataFrame) -> dict:
    n = len(d)
    out = {"n_draws": int(n)}
    for r in RULES:
        out[f"share_withheld_{r}"] = round(float(d[f"withhold_{r}"].mean()), 4)
    g = d["withhold_evidence_gating"].to_numpy(bool)
    for r in SIMPLER:
        x = d[f"withhold_{r}"].to_numpy(bool)
        out[f"share_gating_only_vs_{r}"] = round(float((g & ~x).mean()), 4)
        out[f"share_{r}_only_vs_gating"] = round(float((~g & x).mean()), 4)
    for gt in ("risk_concentration", "transfer_stability", "risk_gradient"):
        out[f"share_fail_{gt}"] = round(float((d[gt] == FAIL).mean()), 4)
    out["share_transfer_stability_caution"] = round(float((d.transfer_stability == CAUTION).mean()), 4)
    for rs in REASONS:
        out[f"share_withheld_by_{rs}"] = round(float((d.withheld_by == rs).mean()), 4)
    a = d.mean_own_lift_affected.dropna()
    out["mean_own_lift_affected"] = round(float(a.mean()), 3) if len(a) else None
    out["median_own_lift_affected"] = round(float(a.median()), 3) if len(a) else None
    b = d.mean_own_lift_affected_unperturbed.dropna()
    out["mean_own_lift_affected_unperturbed"] = round(float(b.mean()), 3) if len(b) else None
    for c in ("pooled_lift_raw", "pooled_lift_system_percentile", "pooled_lift_campus_R", "P", "frac_2x"):
        x = d[c].dropna()
        out[f"median_{c}"] = round(float(x.median()), 4) if len(x) else None
    out["mean_S"] = round(float(d.S.mean()), 4)
    return out


def cell_tables(dd: pd.DataFrame, case_cols: list) -> tuple[pd.DataFrame, dict]:
    """Per case and cell (rows for the CSV) and pooled over cases (JSON)."""
    rows = []
    for key, g in dd.groupby(case_cols + ["perturbation", "strength", "q"], sort=True):
        s = summary(g)
        rec = dict(zip(case_cols + ["perturbation", "strength", "q"], key))
        rec.update({"k": int(g.k.iloc[0]), "n_evaluable": int(g.n_evaluable.iloc[0]),
                    "share_affected": float(g.share_affected.iloc[0]), **s})
        rows.append(rec)
    pooled = {}
    for (p, st, q), g in dd.groupby(["perturbation", "strength", "q"], sort=True):
        s = summary(g)
        s["mean_share_affected"] = round(float(g.share_affected.mean()), 4)
        pooled.setdefault(p, {}).setdefault(f"{st:g}", {})[f"{q:g}"] = s
    return pd.DataFrame(rows), pooled


def thresholds(pooled: dict) -> dict:
    """Per rule: the smallest share q reaching 90% withheld for each strength,
    the smallest strength for each q, and the least damaging cell (highest mean
    own lift of the affected campuses) at which the rule withholds in at least
    90% of draws."""
    out = {}
    for r in RULES:
        blk = {"smallest_q_by_strength": {}, "smallest_strength_by_q": {}}
        cells = []
        for p, by_s in pooled.items():
            if p == "none":
                continue
            for st, by_q in by_s.items():
                qs = sorted(by_q, key=float)
                hit = [q for q in qs if by_q[q][f"share_withheld_{r}"] >= THRESHOLD_SHARE]
                blk["smallest_q_by_strength"][f"{p}:{st}"] = (
                    {"q": float(hit[0]), "holds_for_every_larger_q":
                     all(by_q[q][f"share_withheld_{r}"] >= THRESHOLD_SHARE for q in qs if float(q) >= float(hit[0]))}
                    if hit else None)
                for q in qs:
                    cells.append((p, float(st), float(q), by_q[q][f"share_withheld_{r}"],
                                  by_q[q]["mean_own_lift_affected"]))
        for p in PERTURBATIONS:
            if len(PERTURBATIONS[p]) < 2:
                continue
            for q in SHARES:
                c = sorted([x for x in cells if x[0] == p and x[2] == q], key=lambda x: x[1])
                hit = [x for x in c if x[3] >= THRESHOLD_SHARE]
                blk["smallest_strength_by_q"][f"{p}:{q:g}"] = hit[0][1] if hit else None
        reach = [x for x in cells if x[3] >= THRESHOLD_SHARE]
        miss = [x for x in cells if x[3] < THRESHOLD_SHARE]
        if reach:
            best = max(reach, key=lambda x: x[4])
            blk["least_damaging_cell_reaching_90"] = {"perturbation": best[0], "strength": best[1],
                                                      "q": best[2], "share_withheld": best[3],
                                                      "mean_own_lift_affected": best[4]}
        else:
            blk["least_damaging_cell_reaching_90"] = None
        if miss:
            worst = min(miss, key=lambda x: x[4])
            blk["most_damaging_cell_below_90"] = {"perturbation": worst[0], "strength": worst[1],
                                                  "q": worst[2], "share_withheld": worst[3],
                                                  "mean_own_lift_affected": worst[4]}
        else:
            blk["most_damaging_cell_below_90"] = None
        # at each share q: the highest mean own lift of the affected campuses
        # among the cells in which the rule withholds in at least 90% of draws
        blk["highest_affected_lift_reaching_90_by_q"] = {
            f"{q:g}": (max((x for x in reach if x[2] == q), key=lambda x: x[4])[4]
                       if any(x[2] == q for x in reach) else None) for q in SHARES}
        blk["n_cells_reaching_90"] = len(reach)
        blk["n_cells"] = len(cells)
        out[r] = blk
    return out


def discordant_cells(pooled: dict, margin: float = 0.5) -> dict:
    """Cells in which the gating withholds and a simpler rule does not (shares
    differing by at least `margin`), and the reverse."""
    out = {}
    for r in SIMPLER:
        a, b = [], []
        for p, by_s in pooled.items():
            for st, by_q in by_s.items():
                for q, s in by_q.items():
                    gsh, rsh = s["share_withheld_evidence_gating"], s[f"share_withheld_{r}"]
                    rec = {"perturbation": p, "strength": float(st), "q": float(q),
                           "gating": gsh, "rule": rsh,
                           "mean_own_lift_affected": s["mean_own_lift_affected"]}
                    if gsh - rsh >= margin:
                        a.append(rec)
                    elif rsh - gsh >= margin:
                        b.append(rec)
        out[r] = {"gating_withholds_rule_does_not": a, "rule_withholds_gating_does_not": b}
    return out


def jsonable(o):
    return dr.jsonable(o)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--maps", choices=("pooled", "all"), default="all")
    ap.add_argument("--out-dir", default=None,
                    help="write both outputs to this folder instead of results/")
    args = ap.parse_args()
    assert 1 <= args.workers <= 6
    out_json, out_csv = OUT_JSON, OUT_CSV
    if args.out_dir:
        out_json, out_csv = Path(args.out_dir) / OUT_JSON.name, Path(args.out_dir) / OUT_CSV.name
    for p in (out_json, out_csv):
        assert not p.exists(), f"{p.name} exists; this script writes new files only"
    t0 = time.time()

    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    systems = sorted(single.system.unique())
    outer = {c: cc.gates_for(cc.dev_corpus(c, campuses, single, pairs, "double"),
                             cc.trace_for(c), RCOL, True) for c in campuses}
    pooled = cc.gates_for(single, cc.trace_for(None), RCOL, True)
    ref = dr.check_reference(outer, pooled, campuses)
    print(f"[73] stored verdicts reproduced: {ref}", flush=True)

    stored_ho = pd.read_csv(C.TABLES / "partial_pooling_rule.csv", dtype={"campus": str})
    stored_pool = json.load(open(C.METRICS / "partial_pooling_rule.json"))["pooled_map"]
    ra = pd.read_csv(C.TABLES / "route_aligned_evaluation.csv", dtype={"campus": str})
    seeds = pp_seed_table(campuses, systems, SEED)

    maps = [("pooled", "all", single, cc.trace_for(None), pooled)]
    for c in campuses:
        maps.append(("held_out", c, cc.dev_corpus(c, campuses, single, pairs, "double"),
                     cc.trace_for(c), outer[c]))
    # pooled map on the percentile scales, through scale_variant of script 69
    pool_sys = va.scale_variant({}, va.with_system_percentile(single), [], {}, pooled, "R_system")[1]
    pool_R = va.scale_variant({}, va.with_system_percentile(single), [], {}, pooled, RCOL)[1]

    selection = {}
    for mi, (kind, tgt, frame, trace, res) in enumerate(maps):
        if args.maps == "pooled" and kind != "pooled":
            continue
        chosen = [s for s in sorted(res) if res[s]["S"] >= S_CUT and res[s]["n_campuses"] >= dr.MIN_EVALUABLE]
        selection[tgt] = chosen
        corpus = "pooled" if kind == "pooled" else tgt
        for s in chosen:
            case = build_case(kind, tgt, mi, s, systems.index(s), frame, trace, res[s], corpus, seeds)
            # unperturbed: every rule reproduces its stored quantity
            if kind == "pooled":
                Pst = stored_pool[s]["P"]
                ls, lR = pool_sys[s]["pooled_lift"], pool_R[s]["pooled_lift"]
            else:
                r = stored_ho[(stored_ho.campus == tgt) & (stored_ho.system == s)].iloc[0]
                Pst = float(r.P_lift_ge_2)
                rr = ra[(ra.campus == tgt) & (ra.system == s)].iloc[0]
                ls, lR = float(rr.pooled_lift_system_percentile), float(rr.pooled_lift_campus_percentile_R)
            assert round(case.P0, 4) == Pst, (kind, tgt, s, case.P0, Pst)
            assert abs(round(case.lift_sys0, 2) - ls) < 1e-9, (kind, tgt, s, case.lift_sys0, ls)
            assert abs(round(case.lift_R0, 2) - lR) < 1e-9, (kind, tgt, s, case.lift_R0, lR)
            assert abs(round(case.lift_raw0, 2) - res[s]["pooled_lift"]) < 1e-9
            CASES[(kind, tgt, s)] = case
        if chosen:
            for p, sts in PERTURBATIONS.items():
                check_transport(CASES[(kind, tgt, chosen[0])], frame, p, sts[0])
    n_cases = {"pooled": len(selection.get("all", [])),
               "held_out": sum(len(v) for t, v in selection.items() if t != "all")}
    print(f"[73] cases {n_cases}; stored P, pooled lifts and rank transport checked "
          f"({time.time() - t0:.0f} s)", flush=True)

    val = reduced_bootstrap_validation(maps if args.maps == "all" else maps[:1], systems, stored_ho,
                                       stored_pool)
    print(f"[73] reduced bootstrap against stored P: {val}", flush=True)
    del maps

    # ---------------- draws ----------------
    tasks = []
    for key, case in CASES.items():
        n = len(case.cell.evaluable)
        tasks.append((key, "none", 0.0, 0.0, 0, 0))
        for p, sts in PERTURBATIONS.items():
            for st in sts:
                for q in SHARES:
                    tasks += [(key, p, st, q, k_for(q, n), j) for j in range(args.draws)]
    tb = time.time()
    recs = []
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_limit_threads) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=16), 1):
            recs.append(r)
            if i % max(1, len(tasks) // 20) == 0:
                print(f"[73] {i}/{len(tasks)} draws ({time.time() - tb:.0f} s)", flush=True)
    draw_secs = time.time() - tb
    dd = (pd.DataFrame(recs)
          .sort_values(["map", "target_campus", "system", "perturbation", "strength", "q", "draw"])
          .reset_index(drop=True))

    # ---------------- invariants ----------------
    z = dd[dd.perturbation == "none"]
    assert len(z) == len(CASES)
    unpert = {}
    for r in z.itertuples():
        case = CASES[(r.map, r.target_campus, r.system)]
        b = case.cell.base
        for g in GATES:
            assert getattr(r, g) == b["scores"][g], (r.map, r.target_campus, r.system, g)
        assert abs(r.S - round(b["S"], 4)) < 1e-9 and not r.withhold_evidence_gating
        assert r.withhold_transfer_only_fail == (b["scores"]["transfer_stability"] == FAIL)
        assert r.withhold_transfer_only_fail_or_caution == (b["scores"]["transfer_stability"] < PASS)
        assert r.withhold_pooled_lift_raw == (r.pooled_lift_raw < LIFT_LEVEL)
        assert abs(r.P - case.P0) < 1e-12 and r.withhold_count_based == (not case.cell.count_rule0)
    for rl in RULES:
        unpert[rl] = {"n_cases": int(len(z)), "n_withheld": int(z[f"withhold_{rl}"].sum()),
                      "withheld_cases": [f"{a}/{b}/{c}" for a, b, c in
                                         z.loc[z[f"withhold_{rl}"], ["map", "target_campus", "system"]]
                                         .itertuples(index=False)]}
    assert (dd.groupby(["map", "target_campus", "system"]).withhold_count_based.nunique() == 1).all()

    # partial permutation at a = 1 equals script 66 draw by draw
    d66 = pd.read_csv(C.TABLES / "deranged_ranking_transfer.csv", dtype={"target_campus": str})
    a1 = dd[(dd.perturbation == "partial_permutation") & (dd.strength == 1.0)]
    m = a1.merge(d66, left_on=["map", "target_campus", "system", "k", "draw"],
                 right_on=["map", "target_campus", "system", "k_deranged", "draw"],
                 how="left", suffixes=("", "_66"), validate="one_to_one")
    assert m.S_66.notna().all(), "a draw at a = 1 has no counterpart in script 66"
    for c in GATES + ["S", "pooled_lift", "frac_2x", "withheld_by"]:
        x, y = m[c], m[f"{c}_66"]
        same = (x == y) | (x.isna() & y.isna())
        if x.dtype.kind == "f":
            same = same | ((x - y).abs() < 1e-9)
        assert same.all(), f"a = 1 differs from script 66 in {c}"
    assert (m.affected_campuses == m.deranged_campuses.fillna("")).all()
    # shares beside the 50 stored draws of script 66 at the same k
    a1_cmp = {}
    for kind in sorted(a1["map"].unique()):
        rows = []
        for (t, s, k), g in a1[a1["map"] == kind].groupby(["target_campus", "system", "k"]):
            g66 = d66[(d66["map"] == kind) & (d66.target_campus == t) & (d66.system == s) & (d66.k_deranged == k)]
            rows.append({"q": float(g.q.iloc[0]), "n": len(g), "w": int(g.withhold_evidence_gating.sum()),
                         "n66": len(g66), "w66": int((~g66.supported).sum())})
        rr = pd.DataFrame(rows)
        a1_cmp[kind] = {}
        for q, g in rr.groupby("q"):
            p30, p50 = g.w.sum() / g.n.sum(), g.w66.sum() / g.n66.sum()
            a1_cmp[kind][f"{q:g}"] = {"n_draws_here": int(g.n.sum()), "share_withheld_here": round(float(p30), 4),
                                      "n_draws_script_66": int(g.n66.sum()),
                                      "share_withheld_script_66": round(float(p50), 4),
                                      "binomial_se_here": round(float(np.sqrt(p30 * (1 - p30) / g.n.sum())), 4)}
    print(f"[73] invariants hold: unperturbed rules reproduce stored verdicts; a = 1 equals script 66 "
          f"in {len(m)} draws; count-based rule constant ({draw_secs:.0f} s for {len(tasks)} draws)",
          flush=True)

    # ---------------- summaries ----------------
    pert = dd[dd.perturbation != "none"]
    blocks, tables = {}, []
    for kind in ("pooled", "held_out"):
        dk = pert[pert["map"] == kind]
        if dk.empty:
            continue
        tab, pooled_cells = cell_tables(dk, ["map", "target_campus", "system"])
        tables.append(tab)
        blk = {"cells": pooled_cells, "thresholds_90": thresholds(pooled_cells),
               "discordant_cells": discordant_cells(pooled_cells)}
        per_case = {}
        for (t, s), g in tab.groupby(["target_campus", "system"]):
            pc = {}
            for (p, st), gg in g.groupby(["perturbation", "strength"]):
                gg = gg.sort_values("q")
                pc[f"{p}:{st:g}"] = {r: (float(gg.q[gg[f"share_withheld_{r}"] >= THRESHOLD_SHARE].iloc[0])
                                         if (gg[f"share_withheld_{r}"] >= THRESHOLD_SHARE).any() else None)
                                     for r in RULES if r != "count_based"}
            per_case[f"{t}/{s}"] = pc
        blk["smallest_q_reaching_90_per_case"] = per_case
        if kind == "held_out":
            st_ = dk[~dk.system.isin(sr.PROVISIONAL)]
            _, pc2 = cell_tables(st_, ["map", "target_campus", "system"])
            blk["stable_systems_only"] = {"n_cases": int(st_.groupby(["target_campus", "system"]).ngroups),
                                          "cells": pc2, "thresholds_90": thresholds(pc2)}
        blocks[kind] = blk
    z_rows = z.assign(perturbation="none")
    zt = z_rows[["map", "target_campus", "system", "n_evaluable", "S", "P", "pooled_lift_raw",
                 "pooled_lift_system_percentile", "pooled_lift_campus_R", "frac_2x"]
                + GATES + [f"withhold_{r}" for r in RULES]].copy()
    zt.insert(3, "perturbation", "none")
    zt[[f"withhold_{r}" for r in RULES]] = zt[[f"withhold_{r}" for r in RULES]].astype(float)
    table = pd.concat(tables + [zt.rename(columns={f"withhold_{r}": f"share_withheld_{r}" for r in RULES})],
                      ignore_index=True, sort=False)
    for c in ("k", "n_evaluable", "n_draws"):
        table[c] = table[c].astype("Int64")

    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    name = {k: str(v).strip() for k, v in burden[["SystemCode", "SystemDescription"]].values}
    total = time.time() - t0
    out = {
        "design": ("For each building system supported on a map and evaluable on at least six of its "
                   "campuses (the cases of 66_deranged_ranking_transfer.py), k = ceil(q n) of its n "
                   "evaluable campuses are affected (q = 1/4, 1/2, 3/4, 1; 30 draws per cell) by a partial "
                   "permutation of the system's scores within every quarter (fraction a of the units), a "
                   "mixing of the true and a random rank (weight w on the random rank), or a reversal of "
                   "the order; the five gates, S and the verdict, two transfer-only rules, three pooled-lift "
                   "rules, the partially pooled rule and the count-based rule are recomputed."),
        "definitions": {
            "q": "share of the evaluable campuses affected; k = ceil(q n), share_affected = k / n",
            "partial_permutation": ("in every quarter of an affected campus, round(a m) of the m units of the "
                                    "system, chosen at random, have their scores permuted at random among "
                                    "themselves; a = 1 is the permutation of script 66"),
            "rank_mixing": ("in every quarter of an affected campus the units are ordered by (1 - w) x rank of "
                            "the true score + w x rank of a random permutation and receive the quarter's scores "
                            "in that order; w is the weight on the random rank"),
            "reversal": "in every quarter of an affected campus the order of the system's scores is inverted",
            "mean_own_lift_affected": ("mean over the affected campuses of the top-decile lift of the "
                                       "perturbed scores on the campus's own units of the system; "
                                       "mean_own_lift_affected_unperturbed is the same with the stored scores"),
            "withhold": {
                "evidence_gating": "S < 2/3",
                "transfer_only_fail": "transfer-stability gate at fail (share of evaluable campuses at two-fold < 0.6)",
                "transfer_only_fail_or_caution": "transfer-stability gate not at pass (share < 0.8)",
                "pooled_lift_raw": "pooled top-decile lift of the raw scores below 2",
                "pooled_lift_system_percentile": ("pooled top-decile lift below 2 with the score replaced by its "
                                                  "percentile within campus, quarter and building system"),
                "pooled_lift_campus_R": "pooled top-decile lift below 2 with the score replaced by R",
                "partially_pooled": ("predictive probability of two-fold lift on a new campus below 0.8 "
                                     "(REML random-effects model of script 67)"),
                "count_based": "the count-based rule of script 56 does not support the system",
            },
            "withheld_by": ("first prerequisite gate scoring fail, in the order data sufficiency, transfer "
                            "stability, risk-gradient reliability, or mean_below_support_level"),
            "share_gating_only_vs_<rule>": "share of draws withheld by the gating and not by the rule",
            "share_<rule>_only_vs_gating": "share of draws withheld by the rule and not by the gating",
            "thresholds_90": ("per rule, the smallest q at which the rule withholds in at least 90% of the draws "
                              "pooled over cases, for each perturbation and strength; the smallest strength "
                              "for each q; and the cell reaching 90% with the highest mean own lift of the "
                              "affected campuses"),
            "discordant_cells": "cells whose withheld shares of the gating and the rule differ by at least 0.5",
            "partially_pooled_variance": (f"campuses not affected keep the stored {pp.B_BOOT}-replicate "
                                          f"building-bootstrap variance of script 67; affected campuses use "
                                          f"{B_REDUCED} replicates of the same bootstrap"),
            "cells_pooled_over_cases": "every draw of every case in the cell counts once",
        },
        "reference_check": ref,
        "checks": [
            "the full pipeline reproduces cert_corrected.json, gate_evidence_corrected.csv and "
            "fold_gate_verdicts.csv",
            "with no perturbation every rule reproduces its stored verdict: gate scores and S, P of script 67 "
            "(4 decimals), the raw and percentile pooled lifts of script 69 (2 decimals), the count-based rule",
            "the percentile travels with its score under each perturbation (checked on each map)",
            "partial permutation at a = 1 equals the stored draws of script 66 with the same seeds",
            "the count-based rule gives its unperturbed verdict in every draw",
        ],
        "unperturbed": unpert,
        "a1_against_script_66": {"n_draws_identical": int(len(m)), "shares_by_q": a1_cmp},
        "reduced_bootstrap_validation": val,
        "cases": {"n": n_cases, "selection": selection,
                  "system_desc": {s: name.get(s, s) for s in systems}},
        "draws_per_cell": args.draws,
        "pooled_map": blocks.get("pooled"),
        "held_out_maps": blocks.get("held_out"),
        "runtime": {"total_seconds": round(total, 1), "draw_seconds": round(draw_secs, 1),
                    "n_draws": len(tasks), "workers": args.workers, "threads_per_worker": 1},
    }
    json.dump(jsonable(out), open(out_json, "w"), indent=2)
    table.to_csv(out_csv, index=False)

    # ---------------- console ----------------
    for kind, blk in blocks.items():
        print(f"\n=== {kind}: share withheld, pooled over cases ===")
        for p, by_s in blk["cells"].items():
            for st, by_q in by_s.items():
                for q, s in by_q.items():
                    print(f"  {p:20s} {st:>5s} q={q:>4s} lift_aff={s['mean_own_lift_affected']} "
                          + " ".join(f"{r[:12]}={s[f'share_withheld_{r}']:.2f}" for r in RULES[:-1]))
    print(f"\n[73] done in {total:.0f} s")


if __name__ == "__main__":
    main()
