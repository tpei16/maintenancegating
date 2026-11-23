#!/usr/bin/env python
"""
67_partial_pooling_rule.py -- a threshold on a partially pooled (random-effects)
estimate of the ranking's top-decile lift, set beside the evidence gating.

For each campus-system pair (target campus c, building system k) that receives a
verdict, the evidence is read from the peer campuses only, each peer d through
the model f^(-{c,d}) that excluded both c and d: the double-nested corpus that
45_certification_corrected.py passes to the gates, asserted row by row to hold
no row of c and no prediction from a model trained on c.

  (1) Per evaluable peer campus d (>= 10 high-burden quarters and >= 100 units of
      system k, the evaluability rule of the gates): the top-decile lift L_d and
      the sampling variance v_d of log L_d.  v_d is the variance over B = 1000
      bootstrap replicates that resample, with replacement, the buildings holding
      system k on campus d.  Buildings are the resampling unit because the
      quarters of one building system of one building form a serially correlated
      series; resampling units as if independent would understate the variance.
      Each replicate's log lift carries a continuity correction of 0.5 on the two
      counts it is built from (high-burden units in the top decile, high-burden
      units in total), so that a replicate with no high-burden unit in the top
      decile keeps a finite value.  The point lift is the uncorrected lift the
      gates read; it is corrected only where it is zero.
  (2) A random-effects model on the log lifts of the k evaluable peers,
          log L_d = theta_d + e_d,  e_d ~ N(0, v_d),  theta_d ~ N(mu, tau^2),
      with tau^2 estimated by restricted maximum likelihood (REML) and mu by
      inverse-variance weighting with weights 1 / (v_d + tau^2).
  (3) The predictive probability P that the true lift on a new campus is at
      least 2, from the predictive distribution of theta_new
      (Higgins, Thompson and Spiegelhalter 2009): for k >= 3,
          (theta_new - mu_hat) / sqrt(tau^2 + SE(mu_hat)^2) ~ t with k - 2 df;
      for k = 2 the normal form, flagged; for k < 2 no estimate.
  (4) Accepted by the partially pooled rule when P >= p0.  The principal level
      p0 = 0.8 was fixed before any agreement was computed; 0.5 and 0.9 are
      sensitivity levels, and the DerSimonian-Laird estimator of tau^2 at
      p0 = 0.8 is a further sensitivity.  A pair with fewer than two evaluable
      peers is not accepted.  A last variant, added after the principal results
      were seen, gives no verdict to pairs with two evaluable peers, for which
      the t-based predictive distribution does not exist.

The same construction on the nine campuses, each read through f^(-d), gives the
partially pooled verdict for the 16 systems of the pooled map.

Reported: agreement with the evidence gating (count, share, Cohen's kappa, and a
95% interval for kappa that resamples building systems, the clusters of the
pairs) over the 141 pairs, the 16 systems of the pooled map, the 36 pairs of the
four provisional systems, the 81 pairs of the nine systems supported in every
held-out map and the 24 pairs of the three systems supported in none; every
disagreeing pair; and the held-out check: among the pairs evaluable on the
target campus, the top-decile lift of f^(-c) on system k in campus c, for pairs
accepted and not accepted by each rule.

Checks asserted before anything is reported: the main pipeline reproduces
cert_corrected.json and fold_gate_verdicts.csv; every per-pair and pooled-map
count of evaluable peers and share of peers at two-fold equals the stored
n_campuses and frac_2x; the bootstrap lift equals fmscreen.metrics.lift_at_topk
on the materialized identity and resampled frames; and the REML and
DerSimonian-Laird estimators reproduce the published estimates for the BCG
vaccine trials (Colditz et al. 1994).  The whole bootstrap is repeated with an
independent seed, and the change in P and in the verdicts is reported as the
Monte Carlo error of the rule.

Nothing is fitted; the stored predictions are read as they are.

Outputs -> results/metrics/partial_pooling_rule.json
           results/tables/partial_pooling_rule.csv      (one row per pair)
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FMSCREEN_DEVICE", "cpu")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys, json, time, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import metrics as MET


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fv = _load("fold_verdicts", "56_fold_verdicts_and_simple_rule.py")
cc = fv.cc                                  # the gate functions of script 45, as script 56 loads them

S_CUT = cc.S_CUT
CTX_MIN_POS, CTX_MIN_ROWS = fv.CTX_MIN_POS, fv.CTX_MIN_ROWS   # evaluability of a campus
PROVISIONAL = fv.PROVISIONAL
B_BOOT = C.N_BOOTSTRAP                     # 1000 resamples per campus and system
N_BOOT_KAPPA = C.N_BOOTSTRAP
CC_HALF = 0.5                              # continuity correction of the replicate counts
LOG2 = float(np.log(2.0))
P0_PRINCIPAL = 0.8
P0_LEVELS = (0.5, 0.8, 0.9)


# --------------------------------------------------------------------------
# random-effects estimators
# --------------------------------------------------------------------------
def _neg_restricted_loglik(t: float, y: np.ndarray, v: np.ndarray) -> float:
    w = 1.0 / (v + t)
    mu = float(np.sum(w * y) / np.sum(w))
    return 0.5 * float(np.sum(np.log(v + t)) + np.log(np.sum(w)) + np.sum(w * (y - mu) ** 2))


def tau2_reml(y: np.ndarray, v: np.ndarray) -> float:
    """REML estimate of the between-campus variance, constrained to tau^2 >= 0:
    a log-spaced grid locates the optimum, a bounded scalar search refines it."""
    if len(y) < 2:
        return 0.0
    ub = max(10.0 * float(np.var(y, ddof=1)), 10.0 * float(v.max()), 1.0)
    grid = np.concatenate([[0.0], np.geomspace(1e-10, ub, 800)])
    vals = np.array([_neg_restricted_loglik(t, y, v) for t in grid])
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(_neg_restricted_loglik, bounds=(lo, hi), args=(y, v),
                          method="bounded", options={"xatol": 1e-12})
    t = float(res.x) if res.fun < vals[i] else float(grid[i])
    if _neg_restricted_loglik(0.0, y, v) <= _neg_restricted_loglik(t, y, v):
        t = 0.0
    return t


def tau2_dl(y: np.ndarray, v: np.ndarray) -> float:
    """DerSimonian-Laird moment estimate of the between-campus variance."""
    if len(y) < 2:
        return 0.0
    w = 1.0 / v
    mu0 = float(np.sum(w * y) / np.sum(w))
    q = float(np.sum(w * (y - mu0) ** 2))
    denom = float(np.sum(w) - np.sum(w ** 2) / np.sum(w))
    return max(0.0, (q - (len(y) - 1)) / denom)


def random_effects(y: np.ndarray, v: np.ndarray, estimator: str = "REML") -> dict:
    """Pooled mean, tau, and the predictive probability that the true lift on a
    new campus is at least 2."""
    k = len(y)
    if k < 2:
        return {"k": k, "form": "none", "mu": None, "se_mu": None, "tau2": None, "tau": None,
                "i2": None, "pred_sd": None, "df": None, "P_lift_ge_2": None,
                "pred_lift_95": [None, None]}
    t2 = tau2_reml(y, v) if estimator == "REML" else tau2_dl(y, v)
    w = 1.0 / (v + t2)
    mu = float(np.sum(w * y) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    sd = float(np.sqrt(t2 + se ** 2))
    w0 = 1.0 / v
    q = float(np.sum(w0 * (y - np.sum(w0 * y) / np.sum(w0)) ** 2))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    z = (LOG2 - mu) / sd
    if k >= 3:
        df, form = k - 2, "t"
        p = float(stats.t.sf(z, df))
        h = float(stats.t.ppf(0.975, df)) * sd
    else:
        df, form = None, "normal_k2"
        p = float(stats.norm.sf(z))
        h = float(stats.norm.ppf(0.975)) * sd
    return {"k": k, "form": form, "mu": mu, "se_mu": se, "tau2": t2, "tau": float(np.sqrt(t2)),
            "i2": i2, "pred_sd": sd, "df": df, "P_lift_ge_2": p,
            "pred_lift_95": [float(np.exp(mu - h)), float(np.exp(mu + h))]}


def check_estimators() -> dict:
    """BCG vaccine trials, log risk ratios (Colditz et al. 1994): the published
    estimates are tau^2 = 0.3132 and mu = -0.7145 under REML, tau^2 = 0.3088 and
    mu = -0.7141 under DerSimonian-Laird."""
    tpos = np.array([4, 6, 3, 62, 33, 180, 8, 505, 29, 17, 186, 5, 27], float)
    tneg = np.array([119, 300, 228, 13536, 5036, 1361, 2537, 87886, 7470, 1699, 50448, 2493, 16886], float)
    cpos = np.array([11, 29, 11, 248, 47, 372, 10, 499, 45, 65, 141, 3, 29], float)
    cneg = np.array([128, 274, 209, 12619, 5761, 1079, 619, 87892, 7232, 1600, 27197, 2338, 17825], float)
    y = np.log((tpos / (tpos + tneg)) / (cpos / (cpos + cneg)))
    v = 1 / tpos - 1 / (tpos + tneg) + 1 / cpos - 1 / (cpos + cneg)
    r, d = random_effects(y, v, "REML"), random_effects(y, v, "DL")
    assert abs(r["tau2"] - 0.3132) < 5e-4 and abs(r["mu"] + 0.7145) < 5e-4, r
    assert abs(d["tau2"] - 0.3088) < 5e-4 and abs(d["mu"] + 0.7141) < 5e-4, d
    return {"data": "BCG vaccine trials, 13 log risk ratios (Colditz et al. 1994)",
            "REML": {"tau2": round(r["tau2"], 4), "mu": round(r["mu"], 4), "published": [0.3132, -0.7145]},
            "DL": {"tau2": round(d["tau2"], 4), "mu": round(d["mu"], 4), "published": [0.3088, -0.7141]}}


# --------------------------------------------------------------------------
# per-campus lift and its bootstrap variance
# --------------------------------------------------------------------------
def boot_log_lift(y: np.ndarray, score: np.ndarray, bld: np.ndarray,
                  rng: np.random.Generator, B: int, n_check: int = 2) -> dict:
    """Variance of the log top-decile lift over B building resamples.

    The rows are sorted once by descending score (stable), so that a replicate
    is a vector of building multiplicities: the top decile of the replicate is
    read off the cumulative multiplicities, which equals lift_at_topk on the
    materialized replicate (asserted for the first n_check replicates)."""
    o = np.argsort(-score, kind="stable")
    ys = y[o].astype(np.int64)
    bs = bld[o]
    nb = int(bld.max()) + 1
    logs = np.empty(B)
    zero_top = zero_pos = 0
    for r in range(B):
        m = np.bincount(rng.integers(0, nb, nb), minlength=nb)
        w = m[bs]
        n = int(w.sum())
        M = max(1, int(np.ceil(C.TOPK_BUDGET * n)))
        cw = np.cumsum(w)
        j = int(np.searchsorted(cw, M, side="left"))
        prev = int(cw[j - 1]) if j > 0 else 0
        a = int(np.dot(ys[:j], w[:j])) + int(ys[j]) * (M - prev)
        P = int(np.dot(ys, w))
        zero_top += int(a == 0)
        zero_pos += int(P == 0)
        logs[r] = np.log((a + CC_HALF) / M) - np.log((P + CC_HALF) / n)
        if r < n_check and P > 0:
            idx = np.repeat(np.arange(len(y)), m[bld])
            ref = MET.lift_at_topk(y[idx], score[idx], C.TOPK_BUDGET)
            fast = (a / M) / (P / n)
            assert abs(fast - ref) <= 1e-9 * max(1.0, abs(ref)), (fast, ref)
    return {"var_log_lift": float(np.var(logs, ddof=1)), "n_buildings": nb,
            "n_replicates_zero_top": zero_top, "n_replicates_zero_pos": zero_pos}


def peer_evidence(frame: pd.DataFrame, system: str, rng_for) -> list[dict]:
    """Per evaluable campus of `frame`: the point lift the gates read, the log
    lift, and its bootstrap variance."""
    g = frame[frame.system == system]
    out = []
    for d, gg in g.groupby("university", observed=True, sort=True):
        y, sc = gg["y"].to_numpy(), gg["score"].to_numpy()
        if not (y.sum() >= CTX_MIN_POS and len(y) >= CTX_MIN_ROWS):
            continue
        lift = float(MET.lift_at_topk(y, sc, C.TOPK_BUDGET))
        top = MET._topk_mask(sc, C.TOPK_BUDGET)
        M, a = int(top.sum()), int(y[top].sum())
        zero_point = a == 0
        log_lift = (np.log((a + CC_HALF) / M) - np.log((y.sum() + CC_HALF) / len(y))
                    if zero_point else float(np.log(lift)))
        bld = pd.factorize(gg["building"].to_numpy())[0]
        bt = boot_log_lift(y, sc, bld, rng_for(str(d), system), B_BOOT)
        # identity resample: the fast computation equals the point lift
        ident = boot_identity(y, sc)
        assert abs(ident - lift) <= 1e-9 * max(1.0, lift), (d, system, ident, lift)
        # the row-level binomial approximation, which ignores the clustering of
        # quarters within buildings, for comparison with the building bootstrap
        P_, n_ = int(y.sum()), int(len(y))
        sd_binom = (float(np.sqrt(1 / a - 1 / M + 1 / P_ - 1 / n_)) if a > 0 else None)
        out.append({"campus": str(d), "rows": n_, "pos": P_, "top_rows": M, "top_pos": a,
                    "lift": lift, "log_lift": float(log_lift), "zero_point_lift": bool(zero_point),
                    "sd_log_lift_binomial": sd_binom, **bt})
    return out


def boot_identity(y: np.ndarray, score: np.ndarray) -> float:
    o = np.argsort(-score, kind="stable")
    ys = y[o].astype(np.int64)
    n = len(y)
    M = max(1, int(np.ceil(C.TOPK_BUDGET * n)))
    return (int(ys[:M].sum()) / M) / (int(ys.sum()) / n)


# --------------------------------------------------------------------------
# agreement summaries
# --------------------------------------------------------------------------
def agreement_block(keys: list, gate: dict, rule: dict) -> dict:
    a = np.array([bool(gate[k]) for k in keys])
    b = np.array([bool(rule[k]) for k in keys])
    out = fv.agreement(a, b)
    if a.all() or not a.any():
        out["kappa"] = None          # uninformative when one verdict is constant over the subset
    out.update({"n_supported_by_gating": int(a.sum()), "n_accepted_by_rule": int(b.sum()),
                "n_gating_yes_rule_no": int((a & ~b).sum()),
                "n_gating_no_rule_yes": int((~a & b).sum())})
    return out


def kappa_interval(keys: list, gate: dict, rule: dict, rng: np.random.Generator) -> dict:
    """Percentile interval of kappa, resampling building systems with replacement
    (the pairs of one system share most of their peers)."""
    systems = sorted({k[1] for k in keys})
    by = {s: [k for k in keys if k[1] == s] for s in systems}
    vals, undefined = [], 0
    for row in rng.integers(0, len(systems), size=(N_BOOT_KAPPA, len(systems))):
        ks = [k for i in row for k in by[systems[i]]]
        kap = fv.cohen_kappa([gate[k] for k in ks], [rule[k] for k in ks])
        if kap is None:
            undefined += 1
        else:
            vals.append(kap)
    return {"n_boot": N_BOOT_KAPPA, "n_clusters": len(systems), "n_undefined": undefined,
            "ci95": ([round(float(np.percentile(vals, 2.5)), 4),
                      round(float(np.percentile(vals, 97.5)), 4)] if vals else None)}


def r4(x):
    return None if x is None else round(float(x), 4)


# --------------------------------------------------------------------------
def collect(single: pd.DataFrame, pairs: dict, campuses: list[str], outer: dict,
            pooled: dict, stored: pd.DataFrame, seed: int, t0: float) -> tuple:
    """Peer evidence and random-effects fits for the 141 pairs and the pooled
    map, with the bootstrap seeds drawn from `seed`; asserts the exclusion of
    the target and the invariant on n_campuses and frac_2x."""
    # one seed per (corpus, campus, system), drawn in a fixed order
    corpora = ["pooled"] + campuses
    systems_all = sorted(single.system.unique())
    seed_keys = [(x, d, s) for x in corpora for d in campuses for s in systems_all]
    seed_arr = np.random.default_rng(seed).integers(0, 2**63 - 1, size=len(seed_keys))
    SEEDS = dict(zip(seed_keys, (int(s) for s in seed_arr)))

    # ---------------- per-pair peer evidence and random-effects fits ----------------
    rows, peer_rows = [], []
    exclusion = {"n_rows_checked": 0, "n_corpora": 0}
    for c in campuses:
        dev = cc.dev_corpus(c, campuses, single, pairs, "double")
        assert (dev.university.astype(str) != c).all(), f"target {c} in its own corpus"
        assert (dev["n_held_out"] == 2).all(), f"single-nested prediction in the corpus of {c}"
        combos = dev.groupby([dev["held_out"].astype(str), dev["university"].astype(str)]).size()
        for (h, u), _n in combos.items():
            assert set(h.split("+")) == {c, u}, \
                f"rows of campus {u} in the corpus of {c} scored by the model holding out {h}"
        exclusion["n_rows_checked"] += int(len(dev)); exclusion["n_corpora"] += 1
        for s in sorted(outer[c]):
            ev = peer_evidence(dev, s, lambda d, sys_, c=c: np.random.default_rng(SEEDS[(c, d, sys_)]))
            g = outer[c][s]
            st = stored[(stored.campus == c) & (stored.system == s)].iloc[0]
            # invariant: the per-campus lifts are those behind the stored frac_2x
            assert len(ev) == g["n_campuses"] == int(st.n_campuses), (c, s, len(ev))
            f2 = round(float(np.mean([e["lift"] >= 2 for e in ev])), 3) if ev else None
            assert f2 == g["frac_2x"], (c, s, f2, g["frac_2x"])
            assert (f2 is None and pd.isna(st.frac_2x)) or abs(f2 - st.frac_2x) < 1e-9, (c, s)
            yv = np.array([e["log_lift"] for e in ev]); vv = np.array([e["var_log_lift"] for e in ev])
            assert (vv > 0).all(), (c, s, vv)
            reml, dl = random_effects(yv, vv, "REML"), random_effects(yv, vv, "DL")
            for e in ev:
                peer_rows.append({"target": c, "system": s, **e})
            rows.append({"campus": c, "system": s, "ev": ev, "reml": reml, "dl": dl,
                         "frac_2x": f2, "st": st})
        print(f"[67] target {c}: {len(outer[c])} pairs ({time.time() - t0:.0f} s)", flush=True)
    assert len(rows) == 141

    # ---------------- pooled map: nine campuses, each through f^(-d) ----------------
    assert (single["n_held_out"] == 1).all()
    assert (single["held_out"].astype(str) == single["university"].astype(str)).all(), \
        "a pooled-map prediction comes from a model trained on the campus it scores"
    pooled_rows = []
    for s in sorted(pooled):
        ev = peer_evidence(single, s, lambda d, sys_: np.random.default_rng(SEEDS[("pooled", d, sys_)]))
        g = pooled[s]
        assert len(ev) == g["n_campuses"], (s, len(ev))
        f2 = round(float(np.mean([e["lift"] >= 2 for e in ev])), 3) if ev else None
        assert f2 == g["frac_2x"], (s, f2, g["frac_2x"])
        yv = np.array([e["log_lift"] for e in ev]); vv = np.array([e["var_log_lift"] for e in ev])
        pooled_rows.append({"system": s, "ev": ev, "reml": random_effects(yv, vv, "REML"),
                            "dl": random_effects(yv, vv, "DL"), "S": g["S"],
                            "supported": bool(g["S"] >= S_CUT), "frac_2x": f2,
                            "n_campuses": g["n_campuses"]})
    print(f"[67] pooled map done ({time.time() - t0:.0f} s)", flush=True)

    return rows, pooled_rows, peer_rows, exclusion


def main() -> None:
    t0 = time.time()
    est_check = check_estimators()
    print(f"[67] estimators reproduce the BCG estimates: {est_check}", flush=True)

    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    outer, pooled, routing = fv.final_pipeline(single, pairs, campuses)
    ref_check = fv.check_reference(outer, pooled, routing)

    stored = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    keys = [(r.campus, r.system) for r in stored.itertuples()]
    assert sorted(keys) == sorted((j, s) for j in campuses for s in outer[j]), "pair set differs"
    for r in stored.itertuples():
        v = outer[r.campus][r.system]
        assert abs(round(v["S"], 2) - r.S) < 1e-9 and bool(v["S"] >= S_CUT) == bool(r.certified), \
            (r.campus, r.system)
    ref_check["matches_fold_gate_verdicts_csv"] = True
    print(f"[67] stored verdicts reproduced: {ref_check}", flush=True)

    rows, pooled_rows, peer_rows, exclusion = collect(single, pairs, campuses, outer, pooled,
                                                      stored, C.RANDOM_SEED, t0)
    # Monte Carlo check: the whole bootstrap repeated with an independent seed
    rows_mc, pooled_mc, _, _ = collect(single, pairs, campuses, outer, pooled,
                                       stored, C.RANDOM_SEED + 1, t0)

    # ---------------- verdicts ----------------
    def accepted(re: dict, p0: float) -> bool:
        return re["P_lift_ge_2"] is not None and re["P_lift_ge_2"] >= p0

    gate = {(r["campus"], r["system"]): bool(r["st"].certified) for r in rows}
    count_rule = {(r["campus"], r["system"]): bool(r["st"].simple_rule_prespecified) for r in rows}
    rules = {f"reml_p{p0}": {(r["campus"], r["system"]): accepted(r["reml"], p0) for r in rows}
             for p0 in P0_LEVELS}
    rules[f"dl_p{P0_PRINCIPAL}"] = {(r["campus"], r["system"]): accepted(r["dl"], P0_PRINCIPAL)
                                    for r in rows}
    # a pair with two evaluable peers has no t-based predictive distribution; this
    # variant gives it no verdict instead of the normal form
    rules[f"reml_p{P0_PRINCIPAL}_min3peers"] = {
        (r["campus"], r["system"]): accepted(r["reml"], P0_PRINCIPAL) and r["reml"]["k"] >= 3
        for r in rows}
    principal = f"reml_p{P0_PRINCIPAL}"
    pool_gate = {p["system"]: p["supported"] for p in pooled_rows}
    pool_rules = {f"reml_p{p0}": {p["system"]: accepted(p["reml"], p0) for p in pooled_rows}
                  for p0 in P0_LEVELS}
    pool_rules[f"dl_p{P0_PRINCIPAL}"] = {p["system"]: accepted(p["dl"], P0_PRINCIPAL)
                                         for p in pooled_rows}
    pool_rules[f"reml_p{P0_PRINCIPAL}_min3peers"] = {
        p["system"]: accepted(p["reml"], P0_PRINCIPAL) and p["reml"]["k"] >= 3 for p in pooled_rows}
    variants = {
        "reml_p0.8": "principal: REML, p0 = 0.8, normal form at two peers",
        "reml_p0.5": "sensitivity: p0 = 0.5",
        "reml_p0.9": "sensitivity: p0 = 0.9",
        "dl_p0.8": "sensitivity: DerSimonian-Laird estimate of tau^2, p0 = 0.8",
        "reml_p0.8_min3peers": ("sensitivity added after the principal results were seen: as the "
                                "principal rule, but no verdict for pairs with two evaluable peers"),
    }
    assert set(variants) == set(rules) == set(pool_rules)

    keys = [(r["campus"], r["system"]) for r in rows]
    stable = sorted({s for s in {k[1] for k in keys}
                     if all(gate[k] for k in keys if k[1] == s)})
    never = sorted({s for s in {k[1] for k in keys}
                    if not any(gate[k] for k in keys if k[1] == s)})
    prov = sorted(set(k[1] for k in keys) - set(stable) - set(never))
    assert prov == sorted(PROVISIONAL), prov
    subsets = {"all_141": keys,
               "provisional_36": [k for k in keys if k[1] in PROVISIONAL],
               "supported_in_every_map": [k for k in keys if k[1] in stable],
               "supported_in_no_map": [k for k in keys if k[1] in never]}
    assert len(subsets["provisional_36"]) == 36

    rng_k = np.random.default_rng(C.RANDOM_SEED)
    agreement = {}
    for name, rule in list(rules.items()) + [("count_based_rule", count_rule)]:
        blk = {}
        for sub, ks in subsets.items():
            blk[sub] = agreement_block(ks, gate, rule)
            if sub in ("all_141", "provisional_36"):
                blk[sub]["kappa_ci95_resampling_systems"] = kappa_interval(ks, gate, rule, rng_k)
        if name in pool_rules:
            blk["pooled_map_16"] = agreement_block(sorted(pool_gate), pool_gate, pool_rules[name])
        agreement[name] = blk

    def disagreements(rule: dict) -> list:
        out = []
        for r in rows:
            k = (r["campus"], r["system"])
            if rule[k] == gate[k]:
                continue
            st = r["st"]; re = r["reml"]
            out.append({"campus": r["campus"], "system": r["system"],
                        "gating_supported": gate[k], "pooled_rule_accepted": rule[k],
                        "P": r4(re["P_lift_ge_2"]), "tau": r4(re["tau"]), "k": re["k"],
                        "form": re["form"], "pooled_mean_lift": r4(np.exp(re["mu"])) if re["mu"] is not None else None,
                        "S": float(st.S), "frac_2x": r["frac_2x"],
                        "transfer_stability": float(st.transfer_stability),
                        "risk_gradient": float(st.risk_gradient),
                        "heldout_evaluable": bool(st.heldout_evaluable),
                        "heldout_lift": None if pd.isna(st.heldout_lift) else float(st.heldout_lift)})
        return out

    # ---------------- per building system ----------------
    per_system = {}
    for sysc in sorted({k[1] for k in keys}):
        rs = [r for r in rows if r["system"] == sysc]
        ps = [r["reml"]["P_lift_ge_2"] for r in rs if r["reml"]["P_lift_ge_2"] is not None]
        ts = [r["reml"]["tau"] for r in rs if r["reml"]["tau"] is not None]
        ks_ = [r["reml"]["k"] for r in rs]
        per_system[sysc] = {
            "system_desc": None, "n_pairs": len(rs),
            "group": ("provisional" if sysc in PROVISIONAL else
                      "supported_in_every_map" if sysc in stable else "supported_in_no_map"),
            "n_supported_by_gating": int(sum(gate[(r["campus"], sysc)] for r in rs)),
            "n_accepted_count_based_rule": int(sum(count_rule[(r["campus"], sysc)] for r in rs)),
            **{f"n_accepted_{n}": int(sum(rl[(r["campus"], sysc)] for r in rs)) for n, rl in rules.items()},
            "k_range": [int(min(ks_)), int(max(ks_))],
            "P_median": r4(np.median(ps)) if ps else None,
            "P_range": [r4(min(ps)), r4(max(ps))] if ps else None,
            "tau_median": r4(np.median(ts)) if ts else None,
            "n_pairs_tau_zero": int(sum(t < 1e-6 for t in ts)),
        }

    # ---------------- held-out check ----------------
    ho = [r for r in rows if bool(r["st"].heldout_evaluable)]
    hk = [(r["campus"], r["system"]) for r in ho]
    hl = {(r["campus"], r["system"]): float(r["st"].heldout_lift) for r in ho}
    assert len(ho) == 95

    def heldout_block(flag: dict) -> dict:
        return {"accepted": fv.lift_summary([hl[k] for k in hk if flag[k]]),
                "not_accepted": fv.lift_summary([hl[k] for k in hk if not flag[k]])}

    heldout = {"n_evaluable": len(ho),
               "evidence_gating": heldout_block(gate),
               "count_based_rule": heldout_block(count_rule),
               "accept_every_pair": heldout_block({k: True for k in hk}),
               **{f"partially_pooled_{n}": heldout_block(rl) for n, rl in rules.items()}}
    # the paper's figures for the gating and the count-based rule
    g_, c_ = heldout["evidence_gating"], heldout["count_based_rule"]
    assert (g_["accepted"]["n"], g_["accepted"]["n_lift_ge_2"]) == (77, 74), g_
    assert (g_["not_accepted"]["n"], g_["not_accepted"]["n_lift_ge_2"]) == (18, 14), g_
    assert (c_["accepted"]["n"], c_["accepted"]["n_lift_ge_2"]) == (83, 78), c_

    def disagreement_heldout(rule: dict) -> dict:
        out = {}
        for lab, want in (("gating_no_rule_yes", (False, True)), ("gating_yes_rule_no", (True, False))):
            ks_ = [k for k in keys if (gate[k], rule[k]) == want]
            ke = [k for k in ks_ if k in hl]
            out[lab] = {"n": len(ks_), "n_heldout_evaluable": len(ke),
                        "n_heldout_lift_ge_2": int(sum(hl[k] >= 2.0 for k in ke)),
                        "heldout_lifts": {f"{k[0]}/{k[1]}": hl[k] for k in ke}}
        return out

    heldout["disagreeing_pairs"] = {n: disagreement_heldout(rl) for n, rl in rules.items()}
    heldout["disagreeing_pairs"]["count_based_rule"] = disagreement_heldout(count_rule)

    P_ho = {k: r["reml"]["P_lift_ge_2"] for k, r in zip(hk, ho)}
    with_est = [k for k in hk if P_ho[k] is not None]
    S_ho = {k: float(r["st"].S) for k, r in zip(hk, ho)}
    rank = {
        "spearman_P_vs_heldout_lift_pairs_with_estimate":
            fv.spearman([P_ho[k] for k in with_est], [hl[k] for k in with_est]),
        "spearman_S_vs_heldout_lift_same_pairs":
            fv.spearman([S_ho[k] for k in with_est], [hl[k] for k in with_est]),
        "spearman_P_vs_heldout_lift_all_95_P0_if_no_estimate":
            fv.spearman([P_ho[k] if P_ho[k] is not None else 0.0 for k in hk], [hl[k] for k in hk]),
        "spearman_S_vs_heldout_lift_all_95":
            fv.spearman([S_ho[k] for k in hk], [hl[k] for k in hk]),
        "n_heldout_pairs_without_estimate": int(len(hk) - len(with_est)),
        "spearman_P_vs_S_141_pairs_with_estimate":
            fv.spearman([r["reml"]["P_lift_ge_2"] for r in rows if r["reml"]["P_lift_ge_2"] is not None],
                        [float(r["st"].S) for r in rows if r["reml"]["P_lift_ge_2"] is not None]),
    }
    assert abs(rank["spearman_S_vs_heldout_lift_all_95"]["rho"] - 0.2628) < 1e-4

    # ---------------- Monte Carlo check of the bootstrap ----------------
    def mc_summary() -> dict:
        out = {"seed_principal": C.RANDOM_SEED, "seed_check": C.RANDOM_SEED + 1}
        pa = [r["reml"]["P_lift_ge_2"] for r in rows]
        pb = [r["reml"]["P_lift_ge_2"] for r in rows_mc]
        d = [abs(a - b) for a, b in zip(pa, pb) if a is not None]
        out["max_abs_delta_P_141"] = r4(max(d)); out["median_abs_delta_P_141"] = r4(np.median(d))
        for p0 in P0_LEVELS:
            ch = [(r["campus"], r["system"], r4(a), r4(b)) for r, a, b in zip(rows, pa, pb)
                  if a is not None and (a >= p0) != (b >= p0)]
            out[f"verdict_changes_141_p{p0}"] = {"n": len(ch), "pairs": [list(x) for x in ch]}
        qa = [p["reml"]["P_lift_ge_2"] for p in pooled_rows]
        qb = [p["reml"]["P_lift_ge_2"] for p in pooled_mc]
        out["max_abs_delta_P_pooled_map"] = r4(max(abs(a - b) for a, b in zip(qa, qb) if a is not None))
        out[f"verdict_changes_pooled_map_p{P0_PRINCIPAL}"] = [
            p["system"] for p, a, b in zip(pooled_rows, qa, qb)
            if a is not None and (a >= P0_PRINCIPAL) != (b >= P0_PRINCIPAL)]
        return out

    monte_carlo = mc_summary()

    # ---------------- distribution of k and of the fits ----------------
    kdist = pd.Series([r["reml"]["k"] for r in rows]).value_counts().sort_index()
    forms = pd.Series([r["reml"]["form"] for r in rows]).value_counts()
    tau_vals = np.array([r["reml"]["tau"] for r in rows if r["reml"]["tau"] is not None])
    peers = pd.DataFrame(peer_rows)
    fits = {
        "k_distribution": {str(int(k)): int(v) for k, v in kdist.items()},
        "prediction_form_counts": {str(k): int(v) for k, v in forms.items()},
        "n_pairs_no_estimate_k_lt_2": int((kdist[kdist.index < 2]).sum()),
        "n_pairs_normal_form_k_eq_2": int(kdist.get(2, 0)),
        "tau_reml": {"median": r4(np.median(tau_vals)), "min": r4(tau_vals.min()),
                     "max": r4(tau_vals.max()), "share_zero": r4(np.mean(tau_vals < 1e-6))},
        "P_by_group": {
            grp: {"median": r4(np.median(v)), "min": r4(np.min(v)), "max": r4(np.max(v)), "n": len(v)}
            for grp, ks in subsets.items()
            for v in [[r["reml"]["P_lift_ge_2"] for r in rows
                       if (r["campus"], r["system"]) in set(ks) and r["reml"]["P_lift_ge_2"] is not None]]
            if v},
        "peer_evidence": {
            "n_campus_system_estimates": int(len(peers)),
            "buildings_per_estimate": {"median": int(peers.n_buildings.median()),
                                       "min": int(peers.n_buildings.min())},
            "sd_log_lift": {"median": r4(np.sqrt(peers.var_log_lift).median()),
                            "min": r4(np.sqrt(peers.var_log_lift).min()),
                            "max": r4(np.sqrt(peers.var_log_lift).max())},
            "ratio_bootstrap_sd_to_binomial_sd": {
                "median": r4((np.sqrt(peers.var_log_lift) / peers.sd_log_lift_binomial).median()),
                "p10": r4((np.sqrt(peers.var_log_lift) / peers.sd_log_lift_binomial).quantile(0.10)),
                "p90": r4((np.sqrt(peers.var_log_lift) / peers.sd_log_lift_binomial).quantile(0.90))},
            "n_point_lifts_zero": int(peers.zero_point_lift.sum()),
            "n_estimates_with_a_zero_top_replicate": int((peers.n_replicates_zero_top > 0).sum()),
            "max_share_zero_top_replicates": r4((peers.n_replicates_zero_top / B_BOOT).max()),
            "n_replicates_zero_pos_total": int(peers.n_replicates_zero_pos.sum()),
        },
    }

    # ---------------- table ----------------
    name = {k: str(v).strip() for k, v in
            pd.read_csv(C.TABLES / "burden_by_system.csv")[["SystemCode", "SystemDescription"]].values}
    for sysc in per_system:
        per_system[sysc]["system_desc"] = name.get(sysc, sysc)
    csv_rows = []
    for r in rows:
        st, re, dl = r["st"], r["reml"], r["dl"]
        k = (r["campus"], r["system"])
        csv_rows.append({
            "campus": r["campus"], "system": r["system"], "system_desc": name.get(r["system"], r["system"]),
            "provisional": r["system"] in PROVISIONAL,
            "k_evaluable_peers": re["k"], "n_campuses_stored": int(st.n_campuses),
            "frac_2x_recomputed": r["frac_2x"], "frac_2x_stored": None if pd.isna(st.frac_2x) else float(st.frac_2x),
            "peer_lifts": ";".join(f"{e['campus']}:{e['lift']:.3f}" for e in r["ev"]),
            "peer_sd_log_lift": ";".join(f"{e['campus']}:{np.sqrt(e['var_log_lift']):.3f}" for e in r["ev"]),
            "prediction_form": re["form"], "df": re["df"],
            "mu_log_lift": r4(re["mu"]), "pooled_mean_lift": r4(np.exp(re["mu"])) if re["mu"] is not None else None,
            "se_mu": r4(re["se_mu"]), "tau": r4(re["tau"]), "i2": r4(re["i2"]), "pred_sd": r4(re["pred_sd"]),
            "pred_lift_95_lo": r4(re["pred_lift_95"][0]), "pred_lift_95_hi": r4(re["pred_lift_95"][1]),
            "P_lift_ge_2": r4(re["P_lift_ge_2"]),
            "accepted_p0.5": rules["reml_p0.5"][k], "accepted_p0.8": rules["reml_p0.8"][k],
            "accepted_p0.9": rules["reml_p0.9"][k],
            "tau_dl": r4(dl["tau"]), "P_lift_ge_2_dl": r4(dl["P_lift_ge_2"]),
            "accepted_p0.8_dl": rules[f"dl_p{P0_PRINCIPAL}"][k],
            "accepted_p0.8_min3peers": rules[f"reml_p{P0_PRINCIPAL}_min3peers"][k],
            "S": float(st.S), "supported": bool(st.certified),
            "count_based_rule": bool(st.simple_rule_prespecified),
            "heldout_evaluable": bool(st.heldout_evaluable),
            "heldout_lift": None if pd.isna(st.heldout_lift) else float(st.heldout_lift),
        })
    tab = pd.DataFrame(csv_rows)

    pooled_map = {p["system"]: {
        "system_desc": name.get(p["system"], p["system"]), "k": p["reml"]["k"],
        "form": p["reml"]["form"], "pooled_mean_lift": r4(np.exp(p["reml"]["mu"])) if p["reml"]["mu"] is not None else None,
        "tau": r4(p["reml"]["tau"]), "P": r4(p["reml"]["P_lift_ge_2"]),
        "pred_lift_95": [r4(x) for x in p["reml"]["pred_lift_95"]],
        "accepted_p0.8": pool_rules[principal][p["system"]],
        "accepted_p0.5": pool_rules["reml_p0.5"][p["system"]],
        "accepted_p0.9": pool_rules["reml_p0.9"][p["system"]],
        "P_dl": r4(p["dl"]["P_lift_ge_2"]),
        "S": p["S"], "supported": p["supported"], "frac_2x": p["frac_2x"],
        "peer_lifts": {e["campus"]: round(e["lift"], 3) for e in p["ev"]}}
        for p in pooled_rows}

    out = {
        "design": ("For each campus-system pair, the top-decile lift of the ranking on each "
                   "evaluable peer campus, read through the model that excluded the target and "
                   "the peer, with a building-level bootstrap variance of its logarithm; a "
                   "random-effects model on the log lifts (REML); the predictive probability P "
                   "that the true lift on a new campus is at least 2; accepted when P >= p0. "
                   "Compared with the evidence gating over the 141 pairs and the 16 systems of "
                   "the pooled map, and checked on the held-out campus."),
        "definitions": {
            "peer_lift": ("top-decile lift of f^(-{c,d}) on the units of system k on peer campus d, "
                          "for peers with >= 10 high-burden quarters and >= 100 units of the system"),
            "variance": (f"variance of the log lift over {B_BOOT} replicates resampling with "
                         "replacement the buildings that hold system k on campus d; replicate counts "
                         f"carry a continuity correction of {CC_HALF}"),
            "random_effects": ("log L_d ~ N(theta_d, v_d), theta_d ~ N(mu, tau^2); tau^2 by REML, "
                               "mu by inverse-variance weights 1/(v_d + tau^2)"),
            "P": ("P(theta_new >= log 2), theta_new ~ mu_hat + sqrt(tau^2 + SE(mu_hat)^2) * t_(k-2) "
                  "for k >= 3; standard normal in place of t for k = 2; no estimate for k < 2"),
            "accepted": "P >= p0; not accepted when fewer than two peers are evaluable",
            "principal_level": ("p0 = 0.8, fixed as a convention before agreement with the gating "
                                "was computed; 0.5 and 0.9 reported as sensitivity"),
            "heldout": ("top-decile lift of f^(-c) on system k in campus c, for the 95 pairs "
                        "evaluable there (>= 10 high-burden quarters, >= 100 units)"),
            "exchangeability": ("the rule treats the target campus as exchangeable with its peers: "
                                "its true log lift is a draw from the same normal distribution"),
        },
        "reference_check": ref_check,
        "invariant": {"per_pair_n_campuses_and_frac_2x_reproduced": True, "n_pairs": len(rows),
                      "pooled_map_n_campuses_and_frac_2x_reproduced": True,
                      "n_pooled_systems": len(pooled_rows),
                      "bootstrap_lift_equals_lift_at_topk": True},
        "exclusion_check": {**exclusion,
                            "statement": ("every row of the corpus for target c is a peer row "
                                          "scored by a model whose held-out set contains c and "
                                          "that peer")},
        "estimator_check": est_check,
        "monte_carlo_check": monte_carlo,
        "bootstrap": {"B": B_BOOT, "unit": "buildings holding the system on the campus",
                      "seeds": ("one seed per (corpus, campus, system), drawn in a fixed order from "
                                "numpy.random.default_rng(RANDOM_SEED)"),
                      "continuity_correction": CC_HALF},
        "fits": fits,
        "rule_variants": variants,
        "principal_rule": principal,
        "agreement": agreement,
        "per_system": per_system,
        "disagreements": {n: disagreements(rl) for n, rl in rules.items()},
        "heldout_check": heldout,
        "rank_correlation": rank,
        "pooled_map": pooled_map,
        "runtime_seconds": None,
    }
    out["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(fv.jsonable(out), open(C.METRICS / "partial_pooling_rule.json", "w"), indent=2)
    tab.to_csv(C.TABLES / "partial_pooling_rule.csv", index=False)

    # ---------------- console ----------------
    pd.set_option("display.width", 250, "display.max_columns", 40)
    for n, blk in agreement.items():
        print(f"  {n:22s} " + " | ".join(
            f"{s}: {v['n_agree']}/{v['n']} k={v['kappa']}" for s, v in blk.items()))
    print("\n  disagreements (principal):")
    for d in out["disagreements"][principal]:
        print(f"    {d}")
    print("\n  held-out check:")
    for n, v in heldout.items():
        if isinstance(v, dict) and "accepted" in v:
            print(f"    {n:30s} acc {v['accepted']} | not {v['not_accepted']}")
    print(f"\n  rank: {rank}")
    print(f"  fits: {fits}")
    print(f"  Monte Carlo check: {monte_carlo}")
    print(f"\n[67] done in {out['runtime_seconds']} s")


if __name__ == "__main__":
    main()
