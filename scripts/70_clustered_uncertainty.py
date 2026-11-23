#!/usr/bin/env python
"""
70_clustered_uncertainty.py -- uncertainty of the held-out comparisons when
campus-system pairs that share a campus or a building system are not treated
as independent.

The held-out check compares, over the campus-system pairs evaluable on the
held-out campus, the share of supported and of not supported pairs whose
top-decile lift reaches 2, and correlates the evidence score S with the lift.
Pairs of one campus share its records and pairs of one building system share
its peer evidence, so an interval over pairs treats as independent what is
not.  This script reports, from the stored tables and without any model:

  (i)   the difference between the share of supported and of not supported
        pairs reaching two-fold lift, with 95% percentile intervals from
          - a two-way cluster bootstrap: campuses and building systems are
            resampled independently with replacement, each over the clusters
            present among the pairs analysed, and a pair (c, k) enters a
            replicate with multiplicity a_c x b_k, the number of times its
            campus and its system were drawn (the pigeonhole bootstrap);
          - a one-way cluster bootstrap over campuses (multiplicity a_c) and
            one over building systems (multiplicity b_k);
          - the pair-level bootstrap (pairs resampled as independent) and the
            Newcombe hybrid score interval built from the two Wilson
            intervals that the paper quotes;
        5,000 replicates each.  A replicate in which one of the two groups
        receives no pair leaves the difference undefined; it is counted and
        left out of the interval;
  (ii)  a permutation p-value for the difference that permutes the verdicts
        of whole building systems: under a random permutation pi of the
        building systems analysed, pair (c, k) takes the verdict (and S) of
        pair (c, pi(k)) in the stored verdict table, so each system's verdict
        profile over the campuses moves as one block and the campus pattern of
        the verdicts is kept.  A pair whose source (c, pi(k)) has no verdict is
        left out of that permutation.  10,000 permutations; two-sided p from
        |difference|, with the pair-level permutation and Fisher's exact test
        beside it;
  (iii) the Spearman correlation of S with the held-out lift, with the same
        intervals (the naive one by the Fisher z transformation with the
        Bonett-Wright standard error) and the system-block permutation p;
  (iv)  the same quantities for the count-based rule (verdict of
        56_fold_verdicts_and_simple_rule.py) and for the forward test (pairs
        supported from peers against pairs not supported, everything dated
        2016 or earlier, lift in 2017 to 2019, local_history_baseline.csv);
        the rank correlation of the binary verdict with lift is given for all
        three so that the count-based rule, which has no score, can be read
        next to the others;
  (v)   the difference in median lift between the two groups, same intervals;
  (vi)  the same for the route-aligned evaluation of
        69_route_aligned_evaluation.py: the share of route-evaluable pairs
        whose flags reach two-fold lift, and the correlation of the
        route-aligned lift with the system-level top-decile lift and with S.

The stored counts (74 of 77 and 14 of 18; 78 of 83 and 10 of 12; 54 of 61 and
24 of 27) and the Spearman coefficient of 0.26 are reproduced and asserted.

Output -> results/metrics/clustered_uncertainty.json
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import sys, json, time, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr, fisher_exact

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sr = _load("fold_verdicts", "56_fold_verdicts_and_simple_rule.py")

SEED = C.RANDOM_SEED
N_BOOT = 5000
N_PERM = 10000
TWO_FOLD = 2.0
Z = 1.959963984540054
SCHEMES = ("two_way_cluster", "campus_cluster", "system_cluster", "pair_level")


def rnd(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


# --------------------------------------------------------------------------
# statistics on a weighted set of pairs (weights are integer multiplicities)
# --------------------------------------------------------------------------
def share_diff(w: np.ndarray, g: np.ndarray, o: np.ndarray) -> float:
    ws, wn = (w * g).sum(), (w * ~g).sum()
    if ws == 0 or wn == 0:
        return np.nan
    return float((w * g * o).sum() / ws - (w * ~g * o).sum() / wn)


def median_diff(w: np.ndarray, g: np.ndarray, lift: np.ndarray) -> float:
    a, b = np.repeat(lift[g], w[g]), np.repeat(lift[~g], w[~g])
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(np.median(a) - np.median(b))


def wspearman(w: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    xx, yy = np.repeat(x, w), np.repeat(y, w)
    if len(xx) < 3 or np.ptp(xx) == 0 or np.ptp(yy) == 0:
        return np.nan
    return float(spearmanr(xx, yy)[0])


def weights(scheme: str, ci: np.ndarray, si: np.ndarray, nC: int, nK: int,
            rng: np.random.Generator, B: int) -> np.ndarray:
    n = len(ci)
    if scheme == "pair_level":
        return rng.multinomial(n, np.full(n, 1.0 / n), size=B)
    A = rng.multinomial(nC, np.full(nC, 1.0 / nC), size=B)
    Bk = rng.multinomial(nK, np.full(nK, 1.0 / nK), size=B)
    if scheme == "two_way_cluster":
        return A[:, ci] * Bk[:, si]
    if scheme == "campus_cluster":
        return A[:, ci]
    if scheme == "system_cluster":
        return Bk[:, si]
    raise ValueError(scheme)


def interval(v: np.ndarray) -> dict:
    ok = v[~np.isnan(v)]
    return {"ci95": [rnd(np.quantile(ok, 0.025)), rnd(np.quantile(ok, 0.975))] if len(ok) else None,
            "n_defined": int(len(ok)), "n_undefined": int(np.isnan(v).sum()),
            "share_of_defined_at_or_below_0": rnd(float(np.mean(ok <= 0)) if len(ok) else np.nan)}


def newcombe(k1: int, n1: int, k2: int, n2: int) -> list:
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = sr.wilson(k1, n1)
    l2, u2 = sr.wilson(k2, n2)
    d = p1 - p2
    return [rnd(d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)),
            rnd(d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))]


def fisher_z(r: float, n: int) -> list:
    z = np.arctanh(r)
    se = np.sqrt((1 + r * r / 2) / (n - 3))
    return [rnd(np.tanh(z - Z * se)), rnd(np.tanh(z + Z * se))]


# --------------------------------------------------------------------------
def analyse(d: pd.DataFrame, lookup: dict, label: str, seed_key: int,
            has_score: bool) -> dict:
    """d: one row per pair with campus, system, group (bool), lift, score.
    lookup: (campus, system) -> (group, score) for every pair with a verdict,
    the table the system-block permutation draws from."""
    d = d.reset_index(drop=True)
    g = d.group.to_numpy(bool)
    lift = d.lift.to_numpy(float)
    o = (lift >= TWO_FOLD)
    score = d.score.to_numpy(float) if has_score else None
    camps, ci = np.unique(d.campus.to_numpy(), return_inverse=True)
    syss, si = np.unique(d.system.to_numpy(), return_inverse=True)
    n = len(d)
    ones = np.ones(n, dtype=int)

    k1, n1, k2, n2 = int(o[g].sum()), int(g.sum()), int(o[~g].sum()), int((~g).sum())
    obs = {"diff_share": share_diff(ones, g, o), "diff_median": median_diff(ones, g, lift),
           "spearman_verdict": wspearman(ones, g.astype(float), lift)}
    if has_score:
        obs["spearman_S"] = wspearman(ones, score, lift)

    # ---------------- bootstrap ----------------
    boot = {}
    for b_i, scheme in enumerate(SCHEMES):
        rng = np.random.default_rng([SEED, seed_key, b_i])
        W = weights(scheme, ci, si, len(camps), len(syss), rng, N_BOOT)
        vals = {k: np.empty(N_BOOT) for k in obs}
        for r in range(N_BOOT):
            w = W[r]
            vals["diff_share"][r] = share_diff(w, g, o)
            vals["diff_median"][r] = median_diff(w, g, lift)
            vals["spearman_verdict"][r] = wspearman(w, g.astype(float), lift)
            if has_score:
                vals["spearman_S"][r] = wspearman(w, score, lift)
        boot[scheme] = {k: interval(v) for k, v in vals.items()}
        boot[scheme]["mean_pairs_per_replicate"] = rnd(W.sum(axis=1).mean(), 1)

    # ---------------- permutation of whole building systems ----------------
    rng = np.random.default_rng([SEED, seed_key, 99])
    perm = {k: np.full(N_PERM, np.nan) for k in obs}
    n_used = np.zeros(N_PERM, dtype=int)
    keys_c = d.campus.to_numpy(); keys_s = d.system.to_numpy()
    for r in range(N_PERM):
        pi = dict(zip(syss, rng.permutation(syss)))
        src = [lookup.get((c, pi[s])) for c, s in zip(keys_c, keys_s)]
        use = np.array([x is not None for x in src])
        n_used[r] = int(use.sum())
        gg = np.array([x[0] for x in src if x is not None], dtype=bool)
        ll, oo = lift[use], o[use]
        w = np.ones(len(ll), dtype=int)
        perm["diff_share"][r] = share_diff(w, gg, oo)
        perm["diff_median"][r] = median_diff(w, gg, ll)
        perm["spearman_verdict"][r] = wspearman(w, gg.astype(float), ll)
        if has_score:
            ss = np.array([x[1] for x in src if x is not None], dtype=float)
            perm["spearman_S"][r] = wspearman(w, ss, ll)
    # the identity permutation reproduces the observed statistics
    ident = [lookup[(c, s)] for c, s in zip(keys_c, keys_s)]
    assert np.array_equal(np.array([x[0] for x in ident], dtype=bool), g), label
    if has_score:
        assert np.allclose(np.array([x[1] for x in ident], dtype=float), score), label

    def pvals(k):
        v = perm[k]; ok = v[~np.isnan(v)]
        t = obs[k]
        return {"p_two_sided": rnd((1 + np.sum(np.abs(ok) >= abs(t) - 1e-12)) / (1 + len(ok)), 4),
                "p_one_sided_greater": rnd((1 + np.sum(ok >= t - 1e-12)) / (1 + len(ok)), 4),
                "n_permutations_defined": int(len(ok)),
                "n_permutations_undefined": int(np.isnan(v).sum())}

    # pair-level permutation of the verdicts, for comparison
    rng = np.random.default_rng([SEED, seed_key, 98])
    pl = np.empty(N_PERM)
    for r in range(N_PERM):
        pl[r] = share_diff(ones, rng.permutation(g), o)
    p_pair = (1 + np.sum(np.abs(pl) >= abs(obs["diff_share"]) - 1e-12)) / (1 + N_PERM)

    out = {
        "n_pairs": n, "n_supported": n1, "n_not_supported": n2,
        "n_campuses": int(len(camps)), "n_building_systems": int(len(syss)),
        "campuses": [str(c) for c in camps], "building_systems": [str(s) for s in syss],
        "supported": {"n_lift_ge_2": k1, "n": n1, "share": rnd(k1 / n1),
                      "wilson95_quoted": sr.wilson(k1, n1),
                      "median_lift": rnd(np.median(lift[g]), 3)},
        "not_supported": {"n_lift_ge_2": k2, "n": n2, "share": rnd(k2 / n2),
                          "wilson95_quoted": sr.wilson(k2, n2),
                          "median_lift": rnd(np.median(lift[~g]), 3)},
        "difference_in_share_two_fold": {
            "point": rnd(obs["diff_share"]),
            "naive_newcombe95": newcombe(k1, n1, k2, n2),
            **{f"{s}_bootstrap": boot[s]["diff_share"] for s in SCHEMES}},
        "permutation_difference_in_share": {
            "system_blocks": pvals("diff_share"),
            "mean_pairs_per_permutation": rnd(n_used.mean(), 1),
            "min_pairs_per_permutation": int(n_used.min()),
            "pair_level_permutation_p_two_sided": rnd(p_pair),
            "fisher_exact_p_two_sided": rnd(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])},
        "difference_in_median_lift": {
            "point": rnd(obs["diff_median"], 3),
            **{f"{s}_bootstrap": boot[s]["diff_median"] for s in SCHEMES},
            "permutation_system_blocks": pvals("diff_median")},
        "spearman_binary_verdict_vs_lift": {
            "rho": rnd(obs["spearman_verdict"]),
            **{f"{s}_bootstrap": boot[s]["spearman_verdict"] for s in SCHEMES},
            "permutation_system_blocks": pvals("spearman_verdict")},
        "mean_pairs_per_replicate": {s: boot[s]["mean_pairs_per_replicate"] for s in SCHEMES},
    }
    if has_score:
        r_, p_ = spearmanr(score, lift)
        out["spearman_S_vs_lift"] = {
            "rho": rnd(obs["spearman_S"]), "p_naive": float(f"{p_:.4g}"),
            "naive_fisher_z95": fisher_z(obs["spearman_S"], n),
            **{f"{s}_bootstrap": boot[s]["spearman_S"] for s in SCHEMES},
            "permutation_system_blocks": pvals("spearman_S")}
    return out


def correlation_only(d: pd.DataFrame, x: str, y: str, seed_key: int) -> dict:
    d = d.reset_index(drop=True)
    xv, yv = d[x].to_numpy(float), d[y].to_numpy(float)
    camps, ci = np.unique(d.campus.to_numpy(), return_inverse=True)
    syss, si = np.unique(d.system.to_numpy(), return_inverse=True)
    r_, p_ = spearmanr(xv, yv)
    out = {"n_pairs": int(len(d)), "rho": rnd(r_), "p_naive": float(f"{p_:.4g}"),
           "naive_fisher_z95": fisher_z(r_, len(d))}
    for b_i, scheme in enumerate(SCHEMES):
        rng = np.random.default_rng([SEED, seed_key, b_i])
        W = weights(scheme, ci, si, len(camps), len(syss), rng, N_BOOT)
        out[f"{scheme}_bootstrap"] = interval(np.array([wspearman(W[r], xv, yv)
                                                        for r in range(N_BOOT)]))
    return out


# --------------------------------------------------------------------------
def main() -> None:
    t0 = time.time()
    out_json = C.METRICS / "clustered_uncertainty.json"
    assert not out_json.exists(), f"{out_json.name} exists; this script does not overwrite stored results"

    # ---------------- inputs and their reference numbers ----------------
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    ref = json.load(open(C.METRICS / "fold_verdicts_and_simple_rule.json"))["heldout_performance"]
    e = fv[fv.heldout_evaluable].copy()
    assert len(e) == ref["n_evaluable"] == 95
    cert = ref["by_decision_rule"]["certificate"]
    rule = ref["by_decision_rule"]["simple_rule_prespecified"]
    for grp, flag in (("certified", True), ("withheld", False)):
        x = e[e.certified == flag]
        assert len(x) == cert[grp]["n"] and int((x.heldout_lift >= 2).sum()) == cert[grp]["n_lift_ge_2"]
        x = e[e.simple_rule_prespecified == flag]
        assert len(x) == rule[grp]["n"] and int((x.heldout_lift >= 2).sum()) == rule[grp]["n_lift_ge_2"]
    r_, p_ = spearmanr(e.S, e.heldout_lift)
    assert round(r_, 4) == ref["spearman_S_vs_heldout_lift"]["rho"] == 0.2628
    assert abs(p_ - ref["spearman_S_vs_heldout_lift"]["p_value"]) < 1e-5

    lh = pd.read_csv(C.TABLES / "local_history_baseline.csv", dtype={"campus": str})
    lref = json.load(open(C.METRICS / "local_history_baseline.json"))["summary"]["by_arm_outcome_model"]["peer"]
    fe = lh[lh.evaluable & lh.verdict_peer].copy()
    for grp, flag in (("certified", True), ("withheld_with_verdict", False)):
        x = fe[fe.cert_peer == flag]
        assert len(x) == lref[grp]["n"] and int(x.holds_model.sum()) == lref[grp]["n_hold"], grp
        assert ((x.lift_eval_model >= 2) == x.holds_model).all()
    n_nov = int((lh.evaluable & ~lh.verdict_peer).sum())
    assert n_nov == lref["no_verdict"]["n"]

    # ---------------- (i)-(v) ----------------
    def frame(df, gcol, lcol, scol=None):
        return pd.DataFrame({"campus": df.campus.to_numpy(), "system": df.system.to_numpy(),
                             "group": df[gcol].astype(bool).to_numpy(),
                             "lift": df[lcol].to_numpy(float),
                             "score": df[scol].to_numpy(float) if scol else np.nan})

    look_gate = {(c, s): (bool(v), float(S)) for c, s, v, S in
                 zip(fv.campus, fv.system, fv.certified, fv.S)}
    look_rule = {(c, s): (bool(v), np.nan) for c, s, v in
                 zip(fv.campus, fv.system, fv.simple_rule_prespecified)}
    lv = lh[lh.verdict_peer]
    look_fwd = {(c, s): (bool(v), float(S)) for c, s, v, S in
                zip(lv.campus, lv.system, lv.cert_peer, lv.S_peer)}
    # sensitivity: a pair without a peer verdict counts as not supported
    look_fwd_all = {(c, s): (bool(v) and bool(h), float(S) if h else np.nan) for c, s, v, h, S in
                    zip(lh.campus, lh.system, lh.cert_peer, lh.verdict_peer, lh.S_peer)}

    # every source pair of the system-block permutation has a verdict on these pairs
    for c in e.campus.unique():
        assert all((c, k) in look_gate for k in e.system.unique()), "permutation source missing"
    tests = {}
    tests["held_out_check"] = analyse(frame(e, "certified", "heldout_lift", "S"), look_gate,
                                      "held_out_check", 1, True)
    print(f"[70] held-out check done ({time.time() - t0:.0f} s)", flush=True)
    tests["count_based_rule"] = analyse(frame(e, "simple_rule_prespecified", "heldout_lift"),
                                        look_rule, "count_based_rule", 2, False)
    print(f"[70] count-based rule done ({time.time() - t0:.0f} s)", flush=True)
    tests["forward_test"] = analyse(frame(fe, "cert_peer", "lift_eval_model", "S_peer"), look_fwd,
                                    "forward_test", 3, True)
    print(f"[70] forward test done ({time.time() - t0:.0f} s)", flush=True)
    fa = lh[lh.evaluable].copy()
    fa["sup"] = fa.cert_peer & fa.verdict_peer
    tests["forward_test_no_verdict_as_not_supported"] = analyse(
        frame(fa, "sup", "lift_eval_model"), look_fwd_all,
        "forward_test_no_verdict_as_not_supported", 4, False)
    print(f"[70] forward test, no verdict as not supported, done ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- (vi) route-aligned evaluation ----------------
    ra_path = C.TABLES / "route_aligned_evaluation.csv"
    route = None
    if ra_path.exists():
        ra = pd.read_csv(ra_path, dtype={"campus": str})
        rev = ra[ra.route_evaluable].copy()
        route = {"share_route_lift_ge_2": analyse(frame(rev, "supported", "route_lift", "S"),
                                                  look_gate, "route_aligned", 5, True)}
        both = rev[rev.heldout_evaluable]
        route["spearman_route_lift_vs_heldout_top_decile_lift"] = correlation_only(
            both, "route_lift", "heldout_lift", 6)
        print(f"[70] route-aligned done ({time.time() - t0:.0f} s)", flush=True)

    out = {
        "design": ("Uncertainty of the held-out comparisons with campus-system pairs clustered by "
                   "campus and by building system: two-way (pigeonhole) and one-way cluster "
                   "bootstraps, a permutation of the verdicts of whole building systems, and the "
                   "pair-level intervals beside them; from the stored tables, no model."),
        "definitions": {
            "two_way_cluster": ("campuses and building systems present among the analysed pairs "
                                "resampled independently with replacement; pair (c, k) enters with "
                                "multiplicity a_c x b_k"),
            "campus_cluster": "campuses resampled with replacement; pair (c, k) enters a_c times",
            "system_cluster": "building systems resampled with replacement; pair (c, k) enters b_k times",
            "pair_level": "pairs resampled with replacement as if independent",
            "bootstrap": f"{N_BOOT} replicates per scheme, 95% percentile interval, seeds [42, test, scheme]",
            "undefined_replicate": ("a replicate in which the supported or the not supported group "
                                    "receives no pair, or a correlation has a constant argument; "
                                    "counted and left out of the interval"),
            "system_block_permutation": (f"{N_PERM} random permutations pi of the building systems "
                                         "analysed; pair (c, k) takes the verdict and S of pair "
                                         "(c, pi(k)) in the stored verdict table; a pair whose source "
                                         "has no verdict is left out of that permutation; two-sided p "
                                         "= (1 + #|T*| >= |T|) / (1 + permutations)"),
            "two_fold": "lift of at least 2",
            "held_out_check": "95 pairs evaluable on the held-out campus, fold_gate_verdicts.csv",
            "count_based_rule": "simple_rule_prespecified of fold_gate_verdicts.csv on the same 95 pairs",
            "forward_test": ("pairs evaluable in 2017 to 2019 with a peer verdict from records dated "
                             "2016 or earlier, local_history_baseline.csv (cert_peer, S_peer, "
                             "lift_eval_model)"),
            "naive_newcombe95": "Newcombe hybrid score interval from the two Wilson intervals",
            "naive_fisher_z95": "Fisher z interval with the Bonett-Wright standard error",
        },
        "reference_check": {"held_out_check": "74/77 and 14/18, Spearman 0.2628 (p 0.0101)",
                            "count_based_rule": "78/83 and 10/12",
                            "forward_test": "54/61 and 24/27, 3 pairs without a peer verdict",
                            "matches_stored_tables": True},
        "tests": tests,
        "route_aligned": route,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(sr.jsonable(out), open(out_json, "w"), indent=2)

    for k, v in tests.items():
        d = v["difference_in_share_two_fold"]
        print(f"\n=== {k}: {v['supported']['n_lift_ge_2']}/{v['n_supported']} vs "
              f"{v['not_supported']['n_lift_ge_2']}/{v['n_not_supported']}")
        print(f"  diff {d['point']}  newcombe {d['naive_newcombe95']}  "
              + "  ".join(f"{s}: {d[f'{s}_bootstrap']['ci95']} (undef {d[f'{s}_bootstrap']['n_undefined']})"
                          for s in SCHEMES))
        print(f"  permutation {v['permutation_difference_in_share']}")
        m = v["difference_in_median_lift"]
        print(f"  median diff {m['point']}  " + "  ".join(f"{s}: {m[f'{s}_bootstrap']['ci95']}" for s in SCHEMES))
        if "spearman_S_vs_lift" in v:
            q = v["spearman_S_vs_lift"]
            print(f"  spearman S {q['rho']} p {q['p_naive']} fisher {q['naive_fisher_z95']}  "
                  + "  ".join(f"{s}: {q[f'{s}_bootstrap']['ci95']}" for s in SCHEMES)
                  + f"  perm {q['permutation_system_blocks']['p_two_sided']}")
    if route:
        v = route["share_route_lift_ge_2"]
        print(f"\n=== route-aligned: diff {v['difference_in_share_two_fold']['point']} "
              f"two-way {v['difference_in_share_two_fold']['two_way_cluster_bootstrap']['ci95']}")
        print(f"  {route['spearman_route_lift_vs_heldout_top_decile_lift']}")
    print(f"\n[70] done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
