#!/usr/bin/env python
"""
57_nesting_size_control.py -- does double nesting change the certificates
because the target leaves the training set, or only because each model is
trained on one campus fewer?

When certifying target campus c, a certifying campus d can be read three ways,
with everything else at the main setting (risk percentile within campus and
quarter, capped transfer gate, the prior-record-trace table that excludes c):

  single   d scored by f^(-d), trained on the eight other campuses, c included.
  matched  d scored by f^(-{d,e}), e a third campus outside {c, d}.  The model
           excludes two campuses, as the double-nested model does, but its
           training set contains c.  The rows of d come from the pair fit {d, e}.
  double   d scored by f^(-{c,d}), the construction the study reports.

matched differs from single in training-set size only, and double differs from
matched only in whether the target is in the training set.  Two versions of
matched are computed:
  (i)  random: e drawn uniformly from the seven campuses outside {c, d}, afresh
       for every ordered (c, d), 500 draws; reported as the mean and the 2.5 to
       97.5 percentile range over draws;
  (ii) deterministic: e is the campus outside {c, d} whose panel row count is
       closest to that of c, so that the matched model loses about as many
       training rows as the double-nested one.

For each contrast (single to matched, matched to double, single to double), over
the stratum-fold certificates: certified flags that differ, mean and maximum
absolute change in S, the strata in which flags differ, and the change in the
route counts (maintenance-decision task, verification).  Also the mean number
of training rows behind each construction.  No model is fitted: every fit is
read from data/processed/pred_single/ and data/processed/pred_pair/.

Output -> results/metrics/nesting_size_control.json
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys, json, time, importlib.util
import multiprocessing as mp
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C

_spec = importlib.util.spec_from_file_location(
    "cert_corrected", ROOT / "scripts" / "45_certification_corrected.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

R_CUT, S_CUT = cc.R_CUT, cc.S_CUT
RCOL = "R_quarter"
N_DRAWS = int(os.environ.get("N_DRAWS", "500"))   # fewer only for a quick check
N_WORKERS = 2
ROUTES = ["pma", "verify", "continue", "watch"]

# shared with the forked workers
_SINGLE: pd.DataFrame | None = None
_PAIRS: dict | None = None
_CAMPUSES: list[str] = []
_ORDERED: list[tuple[str, str]] = []
_DRAWS: np.ndarray | None = None


# --------------------------------------------------------------------------
# reference numbers
# --------------------------------------------------------------------------
def check_reference(outer: dict, pooled: dict, routing: dict) -> dict:
    ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]
    n = sum(len(v) for v in outer.values())
    assert n == 141, f"{n} stratum-fold certificates, expected 141"
    assert set(outer) == set(ref["outer_S"]), "campus set differs from the reference"
    for j, m in outer.items():
        assert set(m) == set(ref["outer_S"][j]), f"strata differ for campus {j}"
        for s, v in m.items():
            assert abs(v["S"] - ref["outer_S"][j][s]) < 1e-9, \
                f"S^(-{j})_{s} = {v['S']} != reference {ref['outer_S'][j][s]}"
    n_cert = sum(v["S"] >= S_CUT for v in pooled.values())
    assert len(pooled) == 16 and n_cert == 12, f"pooled map certifies {n_cert}/{len(pooled)}"
    for s, v in pooled.items():
        assert abs(v["S"] - ref["pooled"][s]["S"]) < 1e-9 and v["scores"] == ref["pooled"][s]["scores"]
    assert routing["route_counts"] == ref["routing"]["route_counts"], \
        f"route counts {routing['route_counts']} != reference {ref['routing']['route_counts']}"
    return {"n_stratum_folds": n, "pooled_certified": f"{n_cert}/{len(pooled)}",
            "route_counts": routing["route_counts"], "matches_cert_corrected_json": True}


# --------------------------------------------------------------------------
# the three constructions
# --------------------------------------------------------------------------
def matched_corpus(target: str, campuses: list[str], pairs: dict,
                   partner: dict[str, str]) -> pd.DataFrame:
    """Campus d read through f^(-{d, partner[d]}); partner[d] is outside {target, d}
    except in the self-check below, where partner[d] = target gives double nesting."""
    frames = []
    for d in campuses:
        if d == target:
            continue
        pf = pairs[frozenset((d, partner[d]))]
        frames.append(pf[pf.university == d])
    return cc._add_ranks(pd.concat(frames, ignore_index=True))


def certify(corpus_for, campuses: list[str], single: pd.DataFrame) -> dict:
    outer = {c: cc.gates_for(corpus_for(c), cc.trace_for(c), RCOL, True) for c in campuses}
    return {"outer": outer, "routing": cc.route(single, outer, campuses, RCOL)}


def smap(res: dict) -> dict:
    return {(j, s): v["S"] for j, m in res["outer"].items() for s, v in m.items()}


def compare(a: dict, b: dict, ra: dict, rb: dict) -> dict:
    """Contrast construction a (from) with construction b (to)."""
    keys = sorted(set(a) & set(b))
    dS = np.array([b[k] - a[k] for k in keys])
    flips = [{"campus": k[0], "system": k[1], "S_from": a[k], "S_to": b[k]}
             for k in keys if (a[k] >= S_CUT) != (b[k] >= S_CUT)]
    return {
        "n_compared": len(keys),
        "n_gated_only_in_from": len(set(a) - set(b)),
        "n_gated_only_in_to": len(set(b) - set(a)),
        "n_flags_differ": len(flips),
        "n_certified_to_uncertified": sum(f["S_from"] >= S_CUT for f in flips),
        "n_uncertified_to_certified": sum(f["S_to"] >= S_CUT for f in flips),
        "mean_abs_delta_S": round(float(np.mean(np.abs(dS))), 5) if len(dS) else 0.0,
        "max_abs_delta_S": round(float(np.max(np.abs(dS))), 4) if len(dS) else 0.0,
        "strata_with_flags_differing": sorted({f["system"] for f in flips}),
        "flips": flips,
        "route_counts_from": ra["route_counts"], "route_counts_to": rb["route_counts"],
        "delta_route_counts": {k: rb["route_counts"][k] - ra["route_counts"][k] for k in ROUTES},
    }


def partner_from_draw(target: str, b: int) -> dict[str, str]:
    row = _DRAWS[b]
    return {d: row[i] for i, (c, d) in enumerate(_ORDERED) if c == target}


def _limit_threads() -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _one_draw(b: int) -> dict:
    """Matched construction under random draw b (runs in a forked worker)."""
    res = certify(lambda c: matched_corpus(c, _CAMPUSES, _PAIRS, partner_from_draw(c, b)),
                  _CAMPUSES, _SINGLE)
    return {"b": b, "S": smap(res), "routes": res["routing"]["route_counts"]}


def summarise(values) -> dict:
    v = np.asarray(values, dtype=float)
    return {"mean": round(float(v.mean()), 4),
            "p2.5": round(float(np.percentile(v, 2.5)), 4),
            "p97.5": round(float(np.percentile(v, 97.5)), 4),
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}


# --------------------------------------------------------------------------
def main() -> None:
    global _SINGLE, _PAIRS, _CAMPUSES, _ORDERED, _DRAWS
    t0 = time.time()
    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    _SINGLE, _PAIRS, _CAMPUSES = single, pairs, campuses
    for j in [None] + campuses:                 # fill the cache before forking
        cc.trace_for(j)

    # ---------------- single and double ----------------
    res_single = certify(lambda c: cc.dev_corpus(c, campuses, single, pairs, "single"),
                         campuses, single)
    res_double = certify(lambda c: cc.dev_corpus(c, campuses, single, pairs, "double"),
                         campuses, single)
    pooled = cc.gates_for(single, cc.trace_for(None), RCOL, True)
    ref_check = check_reference(res_double["outer"], pooled, res_double["routing"])
    print(f"[57] reference reproduced: {ref_check}", flush=True)

    # self-check of the matched assembly: partner = target is double nesting
    res_self = certify(lambda c: matched_corpus(c, campuses, pairs, {d: c for d in campuses}),
                       campuses, single)
    assert res_self["outer"] == res_double["outer"], "matched assembly does not reduce to double"
    assert res_self["routing"] == res_double["routing"]

    # ---------------- training rows ----------------
    rows = single.groupby("university").size().to_dict()
    N = int(sum(rows.values()))
    _ORDERED = [(c, d) for c in campuses for d in campuses if d != c]

    def closest(c: str, d: str) -> str:
        cand = [e for e in campuses if e not in (c, d)]
        return min(cand, key=lambda e: (abs(rows[e] - rows[c]), campuses.index(e)))

    det_partner = {(c, d): closest(c, d) for c, d in _ORDERED}
    train_rows = {
        "definition": "panel rows of the campuses in the training set, mean over the 72 ordered (c, d)",
        "total_panel_rows": N,
        "single": round(float(np.mean([N - rows[d] for c, d in _ORDERED])), 1),
        "matched_deterministic": round(float(np.mean([N - rows[d] - rows[det_partner[(c, d)]]
                                                      for c, d in _ORDERED])), 1),
        "double": round(float(np.mean([N - rows[c] - rows[d] for c, d in _ORDERED])), 1),
    }

    # ---------------- (ii) deterministic matched ----------------
    res_det = certify(lambda c: matched_corpus(c, campuses, pairs,
                                               {d: det_partner[(c, d)] for d in campuses if d != c}),
                      campuses, single)
    S_single, S_double, S_det = smap(res_single), smap(res_double), smap(res_det)
    deterministic = {
        "partner_rule": ("e = the campus outside {c, d} whose panel row count is closest to that "
                         "of c; ties to the lower campus code"),
        "partner": {f"{c}->{d}": det_partner[(c, d)] for c, d in _ORDERED},
        "single_to_matched": compare(S_single, S_det, res_single["routing"], res_det["routing"]),
        "matched_to_double": compare(S_det, S_double, res_det["routing"], res_double["routing"]),
        "single_to_double": compare(S_single, S_double, res_single["routing"], res_double["routing"]),
    }
    print(f"[57] deterministic matched done ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- (i) random matched, 500 draws ----------------
    rng = np.random.default_rng(C.RANDOM_SEED)
    others = [[e for e in campuses if e not in (c, d)] for c, d in _ORDERED]
    idx = rng.integers(0, 7, size=(N_DRAWS, len(_ORDERED)))
    _DRAWS = np.array([[others[i][idx[b, i]] for i in range(len(_ORDERED))]
                       for b in range(N_DRAWS)], dtype=object)
    ctx = mp.get_context("fork")
    with ctx.Pool(N_WORKERS, initializer=_limit_threads) as pool:
        draws = sorted(pool.map(_one_draw, range(N_DRAWS), chunksize=10), key=lambda r: r["b"])
    print(f"[57] {N_DRAWS} random draws done ({time.time() - t0:.0f} s)", flush=True)

    per_draw = {"single_to_matched": [], "matched_to_double": []}
    flips_by_stratum = {k: {} for k in per_draw}
    train_rand, cert_rand, route_rand = [], [], {k: [] for k in ROUTES}
    for r in draws:
        b, Sm = r["b"], r["S"]
        rm = {"route_counts": r["routes"]}
        for name, (a, bb, ra, rb) in {
            "single_to_matched": (S_single, Sm, res_single["routing"], rm),
            "matched_to_double": (Sm, S_double, rm, res_double["routing"]),
        }.items():
            cmp = compare(a, bb, ra, rb)
            per_draw[name].append(cmp)
            for s in {f["system"] for f in cmp["flips"]}:
                flips_by_stratum[name].setdefault(s, []).append(b)
        train_rand.append(np.mean([N - rows[d] - rows[_DRAWS[b][i]]
                                   for i, (c, d) in enumerate(_ORDERED)]))
        cert_rand.append(sum(v >= S_CUT for v in Sm.values()))
        for k in ROUTES:
            route_rand[k].append(r["routes"][k])
    train_rows["matched_random"] = summarise(train_rand)

    def summary_over_draws(name: str) -> dict:
        L = per_draw[name]
        per_stratum = {}
        for s in sorted(flips_by_stratum[name]):
            counts = [sum(f["system"] == s for f in c["flips"]) for c in L]
            per_stratum[s] = {"share_of_draws_with_a_flag_differing": round(
                                  len(set(flips_by_stratum[name][s])) / len(L), 4),
                              "mean_flags_differing_per_draw": round(float(np.mean(counts)), 4)}
        return {
            "n_compared": summarise([c["n_compared"] for c in L]),
            "n_flags_differ": summarise([c["n_flags_differ"] for c in L]),
            "n_certified_to_uncertified": summarise([c["n_certified_to_uncertified"] for c in L]),
            "n_uncertified_to_certified": summarise([c["n_uncertified_to_certified"] for c in L]),
            "mean_abs_delta_S": summarise([c["mean_abs_delta_S"] for c in L]),
            "max_abs_delta_S": summarise([c["max_abs_delta_S"] for c in L]),
            "delta_route_counts": {k: summarise([c["delta_route_counts"][k] for c in L])
                                   for k in ROUTES},
            "strata_with_flags_differing": per_stratum,
            "share_of_draws_with_no_flag_differing": round(
                float(np.mean([c["n_flags_differ"] == 0 for c in L])), 4),
        }

    random_v = {
        "n_draws": N_DRAWS,
        "draw_rule": "e uniform over the seven campuses outside {c, d}, drawn afresh for each ordered (c, d)",
        "single_to_matched": summary_over_draws("single_to_matched"),
        "matched_to_double": summary_over_draws("matched_to_double"),
        "single_to_double": {k: v for k, v in deterministic["single_to_double"].items()},
        "n_certified_matched": summarise(cert_rand),
        "route_counts_matched": {k: summarise(v) for k, v in route_rand.items()},
    }

    constructions = {}
    for name, res in (("single", res_single), ("matched_deterministic", res_det),
                      ("double", res_double)):
        S = smap(res)
        constructions[name] = {"n_stratum_folds": len(S),
                               "n_certified": int(sum(v >= S_CUT for v in S.values())),
                               "route_counts": res["routing"]["route_counts"]}

    out = {
        "design": ("Each certifying campus d is read through f^(-d) (single), through f^(-{d,e}) "
                   "with a third campus e outside {c, d} (matched: two campuses excluded, target "
                   "in training), or through f^(-{c,d}) (double, the study's construction); "
                   "everything else is the main pipeline of 45_certification_corrected.py, and "
                   "the three are compared over the stratum-fold certificates and the route counts."),
        "reference_check": ref_check,
        "self_check": "matched assembly with e = c reproduces the double-nested certificates exactly",
        "training_rows": train_rows,
        "constructions": constructions,
        "deterministic_matched": deterministic,
        "random_matched": random_v,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(out, open(C.METRICS / "nesting_size_control.json", "w"), indent=2)

    # ---------------- console ----------------
    print("\n=== training rows ===")
    print(f"  {train_rows}")
    print("\n=== constructions ===")
    for k, v in constructions.items():
        print(f"  {k:22s} {v}")
    print("\n=== deterministic matched ===")
    for k in ("single_to_matched", "matched_to_double", "single_to_double"):
        v = deterministic[k]
        print(f"  {k:18s} flags {v['n_flags_differ']} of {v['n_compared']}, mean|dS| "
              f"{v['mean_abs_delta_S']}, max|dS| {v['max_abs_delta_S']}, strata "
              f"{v['strata_with_flags_differing']}, droutes {v['delta_route_counts']}")
    print("\n=== random matched (500 draws) ===")
    for k in ("single_to_matched", "matched_to_double"):
        v = random_v[k]
        print(f"  {k:18s} flags {v['n_flags_differ']}, mean|dS| {v['mean_abs_delta_S']['mean']}, "
              f"dverify {v['delta_route_counts']['verify']}, strata {v['strata_with_flags_differing']}")
    print(f"\n[57] done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
