#!/usr/bin/env python
"""
62_queue_ordering.py -- in which order should the verification queue be worked?

The verification queue holds the high-risk units (R >= 0.90, R the
within-campus-quarter percentile of the deployed screen) of stratum-folds whose
certificate falls below S = 0.67, under the pipeline of
45_certification_corrected.py (double nesting, decision-time ranking, capped
transfer gate).  A stratum with no fold certificate counts as S = 0, as in
script 45's route().  The queue size reproduces the verify count of
cert_corrected.json (asserted).

Within each campus and quarter the queue is ordered four ways:

  gap           R * (1 - S), the study's evidence-gap index;
  risk          R alone;
  consequence   R times the stratum's mean next-quarter reactive labour hours
                per high-burden quarter.  For campus j the high-burden label and
                the mean are both taken on the other campuses' rows, with the
                label threshold of the single-campus fold of j (the system's
                75th percentile of positive next-quarter reactive labour on the
                other campuses), so no quantity in the weight reads campus j;
  random        the mean of 200 uniform shuffles.

For each ordering the script reports the share of next-quarter high-burden
outcomes (the label y of the screen) and the share of next-quarter reactive
labour hours (upm_labour_next) found in the first 25%, 50% and 75% of each
campus-quarter queue, pooled over campus-quarters, and the stratum composition
of the first 25%.  Queues are short, so positions are counted fractionally:
the unit at rank r of n occupies the interval [r/n, (r+1)/n) and is counted in
the first fraction f with the share of that interval below f.  Under a random
order every unit then has expected weight f, and tied keys share their
positions equally.  An ordering by the realized labour is added as the ceiling
any ordering could reach.  The same is reported for conveying (D10), both
inside the full queue and as a queue of its own.

Outputs -> results/metrics/queue_ordering.json
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import features as FE

_spec = importlib.util.spec_from_file_location(
    "cert_corrected", ROOT / "scripts" / "45_certification_corrected.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

R_CUT, S_CUT = cc.R_CUT, cc.S_CUT
FRACS = (0.25, 0.50, 0.75)
N_SHUFFLE = 200
CONVEYING = "D10"
ORDERINGS = ("gap", "risk", "consequence")


def check_reference(single: pd.DataFrame, pairs: dict, campuses: list[str]) -> dict:
    res = cc.build(nesting="double", rcol="R_quarter", fix_transfer=True,
                   single=single, pairs=pairs, campuses=campuses)
    ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]
    got = {j: {s: v["S"] for s, v in res["outer"][j].items()} for j in campuses}
    assert sum(len(v) for v in got.values()) == 141, "expected 141 stratum-fold certificates"
    assert got == ref["outer_S"], "fold certificates differ from cert_corrected.json"
    n_cert = sum(v["S"] >= S_CUT for v in res["pooled"].values())
    assert len(res["pooled"]) == 16 and n_cert == 12, (len(res["pooled"]), n_cert)
    assert res["routing"]["route_counts"] == ref["routing"]["route_counts"]
    return res, ref


# --------------------------------------------------------------------------
# fractional position weights inside each campus-quarter queue
# --------------------------------------------------------------------------
def inclusion(gid: np.ndarray, key: np.ndarray, f: float) -> np.ndarray:
    """Weight of each unit inside the first fraction f of its group's queue,
    ordered by `key` descending; tied keys within a group share positions."""
    order = np.lexsort((-key, gid))
    gs, ks = gid[order], key[order]
    n_g = np.bincount(gid)
    start = np.r_[0, np.cumsum(n_g)[:-1]]
    r = np.arange(len(gid)) - start[gs]
    inc = np.clip(f * n_g[gs] - r, 0.0, 1.0)
    new = np.r_[True, (gs[1:] != gs[:-1]) | (ks[1:] != ks[:-1])]
    blk = np.cumsum(new) - 1
    inc = (np.bincount(blk, inc) / np.bincount(blk))[blk]
    out = np.empty_like(inc)
    out[order] = inc
    return out


def summarise(q: pd.DataFrame, gid: np.ndarray, key: np.ndarray,
              strata: list[str], subset: np.ndarray | None = None) -> dict:
    """Shares of outcomes and labour in the first 25/50/75%, and the composition
    of the first 25%.  `subset` restricts the outcome sums to some units while
    positions stay those of the full queue."""
    y = q["y"].to_numpy(float)
    lab = q["upm_labour_next"].to_numpy(float)
    m = np.ones(len(q), bool) if subset is None else subset
    out = {"share_high_burden": {}, "share_labour_hours": {}}
    for f in FRACS:
        w = inclusion(gid, key, f)
        out["share_high_burden"][str(f)] = float((w * y)[m].sum() / y[m].sum())
        out["share_labour_hours"][str(f)] = float((w * lab)[m].sum() / lab[m].sum())
        if f == 0.25:
            comp = pd.Series(w[m]).groupby(q["system"].to_numpy()[m]).sum() / w[m].sum()
            out["composition_first_25pct"] = {s: float(comp.get(s, 0.0)) for s in strata}
    return out


def rounded(d):
    if isinstance(d, dict):
        return {k: rounded(v) for k, v in d.items()}
    if isinstance(d, float):
        return round(d, 4)
    return d


def run_orderings(q: pd.DataFrame, rng: np.random.Generator,
                  subset: np.ndarray | None = None) -> dict:
    gid = q.groupby(["university", "period_q"], sort=True).ngroup().to_numpy(np.int64)
    strata = sorted(q.system.unique()) if subset is None else sorted(q.system[subset].unique())
    keys = {"gap": (q.R * (1.0 - q.S)).to_numpy(float),
            "risk": q.R.to_numpy(float),
            "consequence": (q.R * q.w_labour).to_numpy(float)}
    res = {k: summarise(q, gid, v, strata, subset) for k, v in keys.items()}
    shuf = [summarise(q, gid, rng.random(len(q)), strata, subset) for _ in range(N_SHUFFLE)]
    rnd = {"share_high_burden": {}, "share_labour_hours": {}}
    for f in FRACS:
        for m in ("share_high_burden", "share_labour_hours"):
            v = np.array([s[m][str(f)] for s in shuf])
            rnd[m][str(f)] = float(v.mean())
            rnd[m + "_range_2.5_97.5"] = rnd.get(m + "_range_2.5_97.5", {})
            rnd[m + "_range_2.5_97.5"][str(f)] = [float(np.quantile(v, .025)),
                                                 float(np.quantile(v, .975))]
    rnd["composition_first_25pct"] = {
        s: float(np.mean([x["composition_first_25pct"][s] for x in shuf])) for s in strata}
    res["random"] = rnd
    res["ceiling_realized_labour"] = summarise(
        q, gid, q["upm_labour_next"].to_numpy(float), strata, subset)
    return rounded(res)


def main() -> None:
    t0 = time.time()
    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    res, ref = check_reference(single, pairs, campuses)
    print("[62] reference pipeline reproduced (141 fold certificates, 12/16 pooled, routes)",
          flush=True)

    # ---- the queue, routed exactly as in script 45 ----
    frames = []
    for j in campuses:
        sub = single[single.university == j].copy()
        sub["S"] = sub.system.map({s: v["S"] for s, v in res["outer"][j].items()})
        sub["gated"] = sub["S"].notna()
        sub["S"] = sub["S"].fillna(0.0)
        frames.append(sub)
    u = pd.concat(frames, ignore_index=True)
    q = u[(u.R_quarter >= R_CUT) & (u.S < S_CUT)].rename(columns={"R_quarter": "R"}).copy()
    assert len(q) == ref["routing"]["route_counts"]["verify"], (len(q), ref["routing"]["route_counts"])
    assert q.groupby("system").size().to_dict() == ref["routing"]["queue_by_system"]
    print(f"[62] queue reproduced: {len(q)} units", flush=True)

    # ---- next-quarter labour from the panel, keyed on the cell-quarter ----
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    panel["university"] = panel[C.COL_UNIV].astype(str)
    panel["building"] = panel["university"] + "/" + panel[C.COL_BUILDING].astype(str)
    panel["system"] = panel[C.COL_SYSTEM].astype(str)
    keys = ["university", "building", "system", "period_q"]
    n0 = len(q)
    q = q.merge(panel[keys + ["upm_labour_next"]], on=keys, how="left", validate="one_to_one")
    assert len(q) == n0 and q.upm_labour_next.notna().all(), "queue rows not found in the panel"

    # ---- consequence weight from the other campuses, with fold-j labels ----
    univ = panel["university"].to_numpy()
    wmap, fallback = {}, []
    for j in campuses:
        other = panel[univ != j]
        thr, gthr = FE.severity_thresholds(other, target="labour", pctl=75)
        # the labels of the single-campus fold must be the ones in the prediction file
        mine = panel[univ == j]
        y_re = FE.apply_severity(mine, thr, gthr, target="labour")
        sj = single[single.university == j]
        chk = mine[keys].assign(y_re=y_re).merge(sj[keys + ["y"]], on=keys,
                                                 validate="one_to_one")
        assert len(chk) == len(sj) == len(mine) and (chk.y_re == chk.y).all(), \
            f"labels of campus {j} do not match its fold thresholds"
        lab = FE.apply_severity(other, thr, gthr, target="labour") == 1
        hb = other[lab]
        mean_by = hb.groupby("system")["upm_labour_next"].mean()
        glob = float(hb["upm_labour_next"].mean())
        for s in q.loc[q.university == j, "system"].unique():
            v = mean_by.get(s, np.nan)
            if not np.isfinite(v):
                fallback.append({"campus": j, "system": s}); v = glob
            wmap[(j, s)] = float(v)
    q["w_labour"] = [wmap[(a, b)] for a, b in zip(q.university, q.system)]
    print("[62] labels match the fold thresholds for every campus", flush=True)

    rng = np.random.default_rng(C.RANDOM_SEED)
    q = q.sort_values(["university", "period_q", "system", "building"]).reset_index(drop=True)
    full = run_orderings(q, rng)
    conv_in_full = run_orderings(q, rng, subset=(q.system == CONVEYING).to_numpy())
    qc = q[q.system == CONVEYING].reset_index(drop=True)
    conv_alone = run_orderings(qc, rng)

    # ---- queue description ----
    gsize = q.groupby(["university", "period_q"]).size()
    gsize_c = qc.groupby(["university", "period_q"]).size()
    desc = {
        "n_units": int(len(q)),
        "n_campus_quarters": int(len(gsize)),
        "units_per_campus_quarter": {"median": float(gsize.median()), "mean": round(float(gsize.mean()), 2),
                                     "max": int(gsize.max()),
                                     "n_single_unit_queues": int((gsize == 1).sum())},
        "share_units_in_single_unit_queues": round(float((gsize == 1).sum() / len(q)), 4),
        "n_high_burden": int(q.y.sum()), "high_burden_rate": round(float(q.y.mean()), 4),
        "labour_hours_next_quarter": round(float(q.upm_labour_next.sum()), 1),
        "composition": {s: int(n) for s, n in q.system.value_counts().items()},
        "S_values": {str(k): int(v) for k, v in q.S.round(3).value_counts().sort_index().items()},
        "units_in_ungated_strata": int((~q.gated).sum()),
        "by_campus": {j: int(n) for j, n in q.university.value_counts().sort_index().items()},
        "consequence_weight_hours": {f"{a}|{b}": round(v, 2) for (a, b), v in sorted(wmap.items())},
        "consequence_weight_fallback_to_all_strata_mean": fallback,
        "conveying": {"n_units": int(len(qc)), "n_campus_quarters": int(len(gsize_c)),
                      "n_high_burden": int(qc.y.sum()),
                      "labour_hours_next_quarter": round(float(qc.upm_labour_next.sum()), 1),
                      "n_single_unit_queues": int((gsize_c == 1).sum())},
    }
    same = all(conv_alone[o] == conv_alone["risk"] for o in ORDERINGS)

    out = {
        "design": ("The verification queue of the main pipeline (high-risk units of uncertified "
                   "stratum-folds) ordered within each campus-quarter by the evidence gap R(1-S), "
                   "by R, by R times the stratum's mean next-quarter labour per high-burden quarter "
                   "on the other campuses, and at random; reported as the share of next-quarter "
                   "high-burden outcomes and labour hours in the first 25, 50 and 75% of each "
                   "campus-quarter queue, pooled, with fractional positions."),
        "definitions": {
            "queue": "R_quarter >= 0.90 and fold certificate S < 0.67 (missing certificate = 0)",
            "positions": ("unit at rank r of n covers [r/n, (r+1)/n); weight in the first f is the "
                          "covered share below f; tied keys share positions; random order gives "
                          "expected weight f to every unit"),
            "consequence_weight": ("mean upm_labour_next over the other campuses' rows of the "
                                   "stratum labelled high burden with the fold-j thresholds; the "
                                   "all-strata mean where the stratum has none"),
            "random": f"mean of {N_SHUFFLE} uniform shuffles, with the 2.5-97.5 percentile range",
            "ceiling_realized_labour": ("ordering by the realized next-quarter labour; not a "
                                        "decision rule, the most any ordering could place early"),
            "conveying_within_full_queue": ("positions from the full queue, outcome sums over "
                                            "conveying units only"),
            "conveying_queue_alone": ("conveying units only, ordered among themselves; S and the "
                                      "weight are constant within a campus, so gap, risk and "
                                      "consequence give one order"),
        },
        "queue": desc,
        "orderings": full,
        "conveying_within_full_queue": conv_in_full,
        "conveying_queue_alone": conv_alone,
        "conveying_alone_orderings_identical": bool(same),
        "runtime_seconds": None,
    }
    out["runtime_seconds"] = round(time.time() - t0, 1)
    json.dump(out, open(C.METRICS / "queue_ordering.json", "w"), indent=2)

    print(f"\nqueue {desc['n_units']} units over {desc['n_campus_quarters']} campus-quarters, "
          f"median {desc['units_per_campus_quarter']['median']} per queue, "
          f"{desc['n_high_burden']} high-burden outcomes")
    for name, blk in (("full queue", full), ("conveying in full queue", conv_in_full),
                      ("conveying alone", conv_alone)):
        print(f"\n=== {name} ===")
        for o in ORDERINGS + ("random", "ceiling_realized_labour"):
            hb, lb = blk[o]["share_high_burden"], blk[o]["share_labour_hours"]
            print(f"  {o:24s} high-burden " + " ".join(f"{hb[str(f)]:.3f}" for f in FRACS)
                  + "   labour " + " ".join(f"{lb[str(f)]:.3f}" for f in FRACS))
    print("\ncomposition of the first 25% (full queue):")
    for o in ORDERINGS + ("random",):
        c = full[o]["composition_first_25pct"]
        print(f"  {o:12s} " + " ".join(f"{s}={c[s]:.3f}" for s in sorted(c, key=lambda s: -c[s])[:6]))
    print(f"[62] done in {out['runtime_seconds']} s")


if __name__ == "__main__":
    main()
