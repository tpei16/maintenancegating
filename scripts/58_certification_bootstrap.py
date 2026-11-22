#!/usr/bin/env python
"""
58_certification_bootstrap.py -- how often would each stratum be certified if
the record had been drawn again?

The certificate S is a function of finite samples: the buildings each campus
happens to hold and the campuses the study happens to have.  Two resampling
schemes measure that sampling variability, for the pooled map and for every
stratum-fold certificate of the main pipeline.

  W (within contexts)   in every campus, resample its buildings with
                        replacement; the set of campuses is fixed.
  B (between and within) resample the campuses with replacement, each draw
                        counted as a separate context, then resample the
                        buildings within each drawn campus.  For the pooled map
                        the nine campuses are resampled; for the certificate
                        governing campus c its eight certifying campuses are.

In each replicate the prediction frame and the prior-record-trace table are
rebuilt from the resampled buildings, a building drawn twice contributing all
its rows twice; the percentile ranks are recomputed on the rebuilt frame with
_add_ranks, and the five gates and S are recomputed for every stratum with
gates_for of 45_certification_corrected.py.  Nothing is refitted: the risk
scores, the high-burden labels and the p90 threshold that defines an extreme
event are held at their fitted values, so the bootstrap measures the
variability of the evidence given the fitted screens.

The prior-record-trace table is needed at event level, keeping the building
identifier.  event_trace_table() repeats the construction of prior_trace_table
in script 45 line for line and keeps the building; aggregated without
resampling it must reproduce trace_for() exactly, which is asserted for the
pooled table and for each of the nine fold tables.  A first replicate with the
identity resample must reproduce the point certificates exactly (asserted for
both schemes).

Reported per stratum and scheme: the probability that S >= 0.67 (a stratum
left ungated in a replicate counts as not certified), the distribution of S,
and the share of replicates in which each gate scores fail, caution and pass.
Beside them: each gate quantity's margin to its nearest threshold at the point
estimate, and the fold frequency (folds certified out of folds gated).

Usage:  python scripts/58_certification_bootstrap.py [--B 1000] [--workers 4]

Outputs -> results/metrics/certification_bootstrap.json
           results/tables/certification_bootstrap.csv     (one row per stratum)
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import sys, json, time, argparse, importlib.util
import multiprocessing as mp
from dataclasses import dataclass
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

GATES, HARD = cc.GATES, cc.HARD
R_CUT, S_CUT = cc.R_CUT, cc.S_CUT
RCOL = "R_quarter"
CELL = cc.CELL
LEVEL = {cc.FAIL: "fail", cc.CAUTION: "caution", cc.PASS: "pass"}
# the thresholds each gate quantity is compared with in gates_for
THRESHOLDS = {"n_campuses": [3, 6], "n_pos": [100, 500], "pooled_lift": [1.5, 2.0],
              "frac_2x": [0.6, 0.8], "top_band": [1.5, 2.0], "rho": [0.5, 0.9],
              "zero_prior_share": [0.10, 0.25], "n_trace_events": [30]}


# --------------------------------------------------------------------------
# compact row sources: the rows of one campus scored by one model
# --------------------------------------------------------------------------
@dataclass
class Src:
    sys: np.ndarray      # system code (index into SYSTEMS)
    q: np.ndarray        # period_q
    y: np.ndarray
    score: np.ndarray
    b: np.ndarray        # building code within the campus

    @property
    def n(self) -> int:
        return len(self.y)


@dataclass
class Ev:
    sys: np.ndarray
    b: np.ndarray
    zero: np.ndarray     # 1 if the extreme event had no reactive work order in the four prior quarters


# state shared with the forked workers
CAMPUSES: list[str] = []
SYSTEMS: list[str] = []
SYS_CODE: dict[str, int] = {}
NB: dict[str, int] = {}                   # buildings per campus
SRC_SINGLE: dict[str, Src] = {}           # campus d scored by f^(-d): the pooled map
SRC_PAIR: dict[tuple, Src] = {}           # (c, d): campus d scored by f^(-{c,d})
EVENTS: dict = {}                         # exclusion (None or c) -> campus -> Ev


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
# event-level prior-record-trace table
# --------------------------------------------------------------------------
def event_trace_table(panel: pd.DataFrame, exclude: str | None = None) -> pd.DataFrame:
    """The construction of prior_trace_table (script 45), one row per extreme
    event, with the building kept."""
    p = panel
    if exclude is not None:
        p = p[p[C.COL_UNIV].astype(str) != str(exclude)].copy()
    else:
        p = p.copy()
    thr90 = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.90)
    p["p90"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr90).astype(float).fillna(np.inf)).astype(int)
    p = p.sort_values(CELL + ["period_q"]).reset_index(drop=True)
    g = p.groupby(CELL, observed=True)["upm_count"]
    p["prior_total"] = g.shift(1).rolling(2, min_periods=2).sum() + \
                       g.shift(3).rolling(2, min_periods=2).sum()
    p["has_history"] = (p["period_q"] - 4 >= p["cell_first_q"]).astype(int)
    ev = p[(p["p90"] == 1) & (p["has_history"] == 1)].copy()
    ev["zero_prior"] = (ev["prior_total"] == 0).astype(int)
    out = pd.DataFrame({
        "campus": ev[C.COL_UNIV].astype(str).to_numpy(),
        "building": (ev[C.COL_UNIV].astype(str) + "/" + ev[C.COL_BUILDING].astype(str)).to_numpy(),
        "system": ev[C.COL_SYSTEM].astype(str).to_numpy(),
        "zero_prior": ev["zero_prior"].to_numpy(),
    })
    return out


def check_event_table(ev: pd.DataFrame, exclude: str | None) -> None:
    agg = (ev.groupby(["campus", "system"]).agg(n_events=("zero_prior", "size"),
                                                n_zero=("zero_prior", "sum"))
             .reset_index().sort_values(["campus", "system"]).reset_index(drop=True))
    ref = (cc.trace_for(exclude)[["campus", "system", "n_events", "n_zero"]]
             .astype({"campus": str, "system": str})
             .sort_values(["campus", "system"]).reset_index(drop=True))
    assert len(agg) == len(ref), f"trace table rows differ (exclude={exclude})"
    assert (agg.campus.to_numpy() == ref.campus.to_numpy()).all()
    assert (agg.system.to_numpy() == ref.system.to_numpy()).all()
    assert (agg.n_events.to_numpy() == ref.n_events.to_numpy()).all(), f"n_events (exclude={exclude})"
    assert (agg.n_zero.to_numpy() == ref.n_zero.to_numpy()).all(), f"n_zero (exclude={exclude})"


# --------------------------------------------------------------------------
# rebuilding a resampled frame and trace table
# --------------------------------------------------------------------------
def make_src(frame: pd.DataFrame, campus: str, bidx: dict) -> Src:
    b = frame["building"].map(bidx).to_numpy()
    assert not pd.isna(b).any(), f"building outside the campus index ({campus})"
    return Src(sys=frame["system"].map(SYS_CODE).to_numpy().astype(np.int16),
               q=frame["period_q"].to_numpy(), y=frame["y"].to_numpy(),
               score=frame["score"].to_numpy(), b=b.astype(np.int32))


def frame_of(parts: list) -> pd.DataFrame:
    """parts: (Src, building multiplicities or None, context label), in corpus order."""
    labels = [lab for _, _, lab in parts]
    sys_, q_, y_, s_, u_ = [], [], [], [], []
    for i, (src, m, _) in enumerate(parts):
        if m is None:
            idx = np.arange(src.n)
        else:
            idx = np.repeat(np.arange(src.n), m[src.b])
        sys_.append(src.sys[idx]); q_.append(src.q[idx]); y_.append(src.y[idx])
        s_.append(src.score[idx]); u_.append(np.full(len(idx), i, dtype=np.int16))
    d = pd.DataFrame({
        "university": pd.Categorical.from_codes(np.concatenate(u_), categories=labels),
        "system": pd.Categorical.from_codes(np.concatenate(sys_), categories=SYSTEMS),
        "period_q": np.concatenate(q_), "y": np.concatenate(y_), "score": np.concatenate(s_),
    })
    return cc._add_ranks(d)


def trace_of(exclude: str | None, parts: list) -> pd.DataFrame:
    """parts: (campus, building multiplicities or None, context label)."""
    ns = len(SYSTEMS)
    rows = []
    for d, m, lab in parts:
        e = EVENTS[exclude][d]
        w = np.ones(len(e.b)) if m is None else m[e.b].astype(float)
        ne = np.bincount(e.sys, weights=w, minlength=ns)
        nz = np.bincount(e.sys, weights=w * e.zero, minlength=ns)
        for k in np.flatnonzero(ne > 0):
            rows.append((lab, SYSTEMS[k], int(round(ne[k])), int(round(nz[k]))))
    return pd.DataFrame(rows, columns=["campus", "system", "n_events", "n_zero"])


def gates(frame: pd.DataFrame, trace: pd.DataFrame) -> dict:
    return cc.gates_for(frame, trace, RCOL, True)


def replicate(scheme: str, rng: np.random.Generator | None) -> list[dict]:
    """Gate results for the pooled map and the nine fold certificates of one
    replicate; rng=None is the identity resample."""
    def mult(d: str):
        if rng is None:
            return None
        n = NB[d]
        return np.bincount(rng.integers(0, n, n), minlength=n)

    res = []
    if scheme == "W":
        m = {d: mult(d) for d in CAMPUSES}
        res.append(gates(frame_of([(SRC_SINGLE[d], m[d], d) for d in CAMPUSES]),
                         trace_of(None, [(d, m[d], d) for d in CAMPUSES])))
        for c in CAMPUSES:
            peers = [d for d in CAMPUSES if d != c]
            res.append(gates(frame_of([(SRC_PAIR[(c, d)], m[d], d) for d in peers]),
                             trace_of(c, [(d, m[d], d) for d in peers])))
    elif scheme == "B":
        draws = CAMPUSES if rng is None else [CAMPUSES[k] for k in rng.integers(0, 9, 9)]
        parts = [(d, mult(d), f"{d}#{k}") for k, d in enumerate(draws)]
        res.append(gates(frame_of([(SRC_SINGLE[d], m, lab) for d, m, lab in parts]),
                         trace_of(None, parts)))
        for c in CAMPUSES:
            peers = [d for d in CAMPUSES if d != c]
            draws = peers if rng is None else [peers[k] for k in rng.integers(0, 8, 8)]
            parts = [(d, mult(d), f"{d}#{k}") for k, d in enumerate(draws)]
            res.append(gates(frame_of([(SRC_PAIR[(c, d)], m, lab) for d, m, lab in parts]),
                             trace_of(c, parts)))
    else:
        raise ValueError(scheme)
    return res


def _limit_threads() -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)


def _task(args) -> tuple:
    scheme, b, seed = args
    res = replicate(scheme, np.random.default_rng(seed))
    S = np.full((len(res), len(SYSTEMS)), np.nan)
    G = np.full((len(res), len(SYSTEMS), len(GATES)), np.nan)
    for i, r in enumerate(res):
        for s, v in r.items():
            k = SYS_CODE[s]
            S[i, k] = v["S"]
            G[i, k, :] = [v["scores"][g] for g in GATES]
    return scheme, b, S, G


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------
def margin(value, thresholds: list) -> dict | None:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    t = min(thresholds, key=lambda x: (abs(value - x), x))
    return {"value": value, "nearest_threshold": t, "margin": round(float(value - t), 4)}


def stratum_summary(S: np.ndarray, G: np.ndarray) -> dict:
    """S: (B,), G: (B, 5) for one certificate across replicates."""
    gated = ~np.isnan(S)
    cert = np.where(gated, S >= S_CUT, False)
    dist = {}
    for v in np.round(S[gated], 1):
        dist[f"{v:.1f}"] = dist.get(f"{v:.1f}", 0) + 1
    B = len(S)
    dist = {k: round(v / B, 4) for k, v in sorted(dist.items())}
    if (~gated).any():
        dist["not_gated"] = round(float((~gated).mean()), 4)
    per_gate = {}
    for gi, g in enumerate(GATES):
        x = G[gated, gi]
        per_gate[g] = ({LEVEL[l]: round(float(np.mean(x == l)), 4)
                        for l in (cc.FAIL, cc.CAUTION, cc.PASS)} if len(x) else None)
    return {"p_certified": round(float(cert.mean()), 4), "share_gated": round(float(gated.mean()), 4),
            "S_distribution": dist, "gate_shares_among_gated": per_gate}


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


# --------------------------------------------------------------------------
def main() -> None:
    global CAMPUSES, SYSTEMS, SYS_CODE
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    assert 1 <= args.workers <= 4
    t0 = time.time()

    single, pairs = cc.load_single(), cc.load_pairs()
    CAMPUSES = sorted(single.university.unique())
    SYSTEMS = sorted(single.system.unique())
    SYS_CODE = {s: i for i, s in enumerate(SYSTEMS)}

    # ---------------- point certificates and the reference ----------------
    outer = {j: cc.gates_for(cc.dev_corpus(j, CAMPUSES, single, pairs, "double"),
                             cc.trace_for(j), RCOL, True) for j in CAMPUSES}
    pooled = cc.gates_for(single, cc.trace_for(None), RCOL, True)
    routing = cc.route(single, outer, CAMPUSES, RCOL)
    ref_check = check_reference(outer, pooled, routing)
    print(f"[58] reference reproduced: {ref_check}", flush=True)

    # ---------------- compact sources ----------------
    bidx = {}
    for d in CAMPUSES:
        blds = sorted(single.loc[single.university == d, "building"].unique())
        bidx[d] = {b: i for i, b in enumerate(blds)}
        NB[d] = len(blds)
        SRC_SINGLE[d] = make_src(single[single.university == d], d, bidx[d])
    for c in CAMPUSES:
        for d in CAMPUSES:
            if d != c:
                pf = pairs[frozenset((c, d))]
                SRC_PAIR[(c, d)] = make_src(pf[pf.university == d], d, bidx[d])

    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    for x in [None] + CAMPUSES:
        ev = event_trace_table(panel, exclude=x)
        check_event_table(ev, x)                      # reproduces trace_for(x) exactly
        EVENTS[x] = {}
        for d in CAMPUSES:
            if d == x:
                continue
            e = ev[ev.campus == d]
            b = e["building"].map(bidx[d]).to_numpy()
            assert not pd.isna(b).any()
            EVENTS[x][d] = Ev(sys=e["system"].map(SYS_CODE).to_numpy().astype(np.int64),
                              b=b.astype(np.int64), zero=e["zero_prior"].to_numpy().astype(float))
    del panel
    print(f"[58] event-level trace tables reproduce trace_for() for all 10 tables "
          f"({time.time() - t0:.0f} s)", flush=True)

    # ---------------- replicate 0: the identity resample ----------------
    point = [pooled] + [outer[c] for c in CAMPUSES]
    for scheme in ("W", "B"):
        ident = replicate(scheme, None)
        for i, (a, b) in enumerate(zip(ident, point)):
            assert a == b, f"identity resample, scheme {scheme}, certificate {i} differs"
    print(f"[58] identity resample reproduces the pooled and the 9 fold certificates "
          f"under both schemes ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- bootstrap ----------------
    B = args.B
    seeds = np.random.default_rng(C.RANDOM_SEED).integers(0, 2**63 - 1, size=(B, 2))
    tasks = [("W", b, int(seeds[b, 0])) for b in range(B)] + \
            [("B", b, int(seeds[b, 1])) for b in range(B)]
    nc = 1 + len(CAMPUSES)
    S_all = {k: np.full((B, nc, len(SYSTEMS)), np.nan) for k in ("W", "B")}
    G_all = {k: np.full((B, nc, len(SYSTEMS), len(GATES)), np.nan) for k in ("W", "B")}
    tb = time.time()
    ctx = mp.get_context("fork")
    done = 0
    with ctx.Pool(args.workers, initializer=_limit_threads) as pool:
        for scheme, b, S, G in pool.imap_unordered(_task, tasks, chunksize=4):
            S_all[scheme][b], G_all[scheme][b] = S, G
            done += 1
            if done % max(1, len(tasks) // 20) == 0:
                print(f"[58] {done}/{len(tasks)} replicates ({time.time() - tb:.0f} s)", flush=True)
    boot_secs = time.time() - tb
    for k in ("W", "B"):
        assert not np.isnan(S_all[k]).all(axis=(1, 2)).any(), f"empty replicate under {k}"

    # ---------------- summaries ----------------
    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    name = {k: str(v).strip() for k, v in burden[["SystemCode", "SystemDescription"]].values}
    trace0 = cc.trace_for(None)

    def n_events(trace: pd.DataFrame, s: str) -> int:
        return int(trace[trace.system == s].n_events.sum())

    def margins(v: dict, trace: pd.DataFrame, s: str) -> dict:
        q = {"n_campuses": v["n_campuses"], "n_pos": v["n_pos"], "pooled_lift": v["pooled_lift"],
             "frac_2x": v["frac_2x"], "top_band": v.get("top_band"), "rho": v.get("rho"),
             "zero_prior_share": v["zero_prior_share"], "n_trace_events": n_events(trace, s)}
        return {k: margin(q[k], THRESHOLDS[k]) for k in THRESHOLDS}

    strata = {}
    csv_rows = []
    for s in sorted(pooled):
        k = SYS_CODE[s]
        p = pooled[s]
        folds = [c for c in CAMPUSES if s in outer[c]]
        n_fc = sum(outer[c][s]["S"] >= S_CUT for c in folds)
        entry = {"system_desc": name.get(s, s), "S_pooled": p["S"],
                 "certified_pooled": bool(p["S"] >= S_CUT),
                 "scores_pooled": {g: LEVEL[p["scores"][g]] for g in GATES},
                 "margins_pooled": margins(p, trace0, s),
                 "folds_certified": int(n_fc), "folds_gated": len(folds),
                 "fold_frequency": round(n_fc / len(folds), 4) if folds else None}
        for sch in ("W", "B"):
            pool_sum = stratum_summary(S_all[sch][:, 0, k], G_all[sch][:, 0, k, :])
            fold_p = {}
            for i, c in enumerate(CAMPUSES, start=1):
                Sx = S_all[sch][:, i, k]
                if s in outer[c] or (~np.isnan(Sx)).any():
                    fold_p[c] = {"S_point": outer[c][s]["S"] if s in outer[c] else None,
                                 "p_certified": round(float(np.nanmean(np.where(
                                     np.isnan(Sx), 0.0, Sx >= S_CUT))), 4),
                                 "share_gated": round(float((~np.isnan(Sx)).mean()), 4)}
            pf_gated = [fold_p[c]["p_certified"] for c in folds]
            Gf = G_all[sch][:, 1:, k, :].reshape(-1, len(GATES))
            Sf = S_all[sch][:, 1:, k].reshape(-1)
            Gf = Gf[~np.isnan(Sf)]
            entry[sch] = {
                "pooled": pool_sum,
                "folds": fold_p,
                "fold_mean_p_certified": round(float(np.mean(pf_gated)), 4) if pf_gated else None,
                "fold_expected_certified": round(float(np.sum(pf_gated)), 2) if pf_gated else None,
                "fold_gate_shares_among_gated": {
                    g: {LEVEL[l]: round(float(np.mean(Gf[:, gi] == l)), 4)
                        for l in (cc.FAIL, cc.CAUTION, cc.PASS)} if len(Gf) else None
                    for gi, g in enumerate(GATES)},
            }
        strata[s] = entry
        mp_ = entry["margins_pooled"]
        row = {"system": s, "system_desc": entry["system_desc"], "S_pooled": p["S"],
               "certified_pooled": entry["certified_pooled"],
               "p_certified_W": entry["W"]["pooled"]["p_certified"],
               "p_certified_B": entry["B"]["pooled"]["p_certified"],
               "share_gated_W": entry["W"]["pooled"]["share_gated"],
               "share_gated_B": entry["B"]["pooled"]["share_gated"],
               "folds_certified": entry["folds_certified"], "folds_gated": entry["folds_gated"],
               "fold_frequency": entry["fold_frequency"],
               "fold_mean_p_certified_W": entry["W"]["fold_mean_p_certified"],
               "fold_mean_p_certified_B": entry["B"]["fold_mean_p_certified"]}
        for q in THRESHOLDS:
            row[f"{q}"] = mp_[q]["value"] if mp_[q] else None
            row[f"{q}_margin"] = mp_[q]["margin"] if mp_[q] else None
        csv_rows.append(row)

    # strata gated in some replicate but not on the point pooled map
    others = {}
    for sch in ("W", "B"):
        for s in SYSTEMS:
            if s in pooled:
                continue
            Sx = S_all[sch][:, 0, SYS_CODE[s]]
            if (~np.isnan(Sx)).any():
                others.setdefault(s, {})[sch] = stratum_summary(Sx, G_all[sch][:, 0, SYS_CODE[s], :])

    n_cert = {}
    for sch in ("W", "B"):
        cnt = np.nansum(np.where(np.isnan(S_all[sch][:, 0, :]), 0, S_all[sch][:, 0, :] >= S_CUT), axis=1)
        n_cert[sch] = {"point": 12, "mean": round(float(cnt.mean()), 3),
                       "p2.5": float(np.percentile(cnt, 2.5)), "p97.5": float(np.percentile(cnt, 97.5)),
                       "distribution": {str(int(v)): round(float(np.mean(cnt == v)), 4)
                                        for v in np.unique(cnt)}}

    fold_margins = []
    for c in CAMPUSES:
        for s in sorted(outer[c]):
            fold_margins.append({"campus": c, "system": s, "S": outer[c][s]["S"],
                                 **{q: (m["margin"] if m else None)
                                    for q, m in margins(outer[c][s], cc.trace_for(c), s).items()}})

    total = time.time() - t0
    out = {
        "design": ("Building-level bootstrap of the certificate: scheme W resamples buildings with "
                   "replacement within every campus, scheme B resamples campuses (each draw a separate "
                   "context) and then buildings within each drawn campus; each replicate rebuilds the "
                   "prediction frame and the event-level prior-record-trace table and recomputes the "
                   "five gates and S with the functions of 45_certification_corrected.py, holding the "
                   "fitted scores, labels and extreme-event thresholds fixed."),
        "reference_check": ref_check,
        "checks": ["event-level trace tables aggregate to trace_for(None) and trace_for(c) for all nine c",
                   "the identity resample reproduces the pooled map and all 141 stratum-fold "
                   "certificates under both schemes"],
        "B": B,
        "B_note": ("B = 1000, the design value" if B == 1000 else
                   f"B = {B}, fewer than the design value of 1000"),
        "seeds": ("per-replicate seeds drawn from numpy.random.default_rng(RANDOM_SEED) as a (B, 2) "
                  "array, column 0 for scheme W and column 1 for scheme B"),
        "scheme_B_folds": ("for the certificate governing campus c, its eight certifying campuses are "
                           "resampled with replacement (eight draws) and read through f^(-{c,d})"),
        "not_certified_if_ungated": ("a stratum that fails the eligibility filter (20 positives, 200 "
                                     "rows) in a replicate counts as not certified"),
        "n_pooled_certified": n_cert,
        "strata": strata,
        "other_strata_gated_in_some_replicates": others,
        "fold_margins": fold_margins,
        "runtime": {"total_seconds": round(total, 1), "bootstrap_seconds": round(boot_secs, 1),
                    "workers": args.workers, "threads_per_worker": 2,
                    "seconds_per_replicate_per_worker": round(boot_secs * args.workers / (2 * B), 2)},
    }
    json.dump(jsonable(out), open(C.METRICS / "certification_bootstrap.json", "w"), indent=2)
    pd.DataFrame(csv_rows).to_csv(C.TABLES / "certification_bootstrap.csv", index=False)

    # ---------------- console ----------------
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(pd.DataFrame(csv_rows)[["system", "S_pooled", "p_certified_W", "p_certified_B",
                                  "folds_certified", "folds_gated", "fold_mean_p_certified_W",
                                  "fold_mean_p_certified_B"]].to_string(index=False))
    print(f"  pooled certified count: {n_cert}")
    print(f"\n[58] B = {B}, done in {total:.0f} s (bootstrap {boot_secs:.0f} s)")


if __name__ == "__main__":
    main()
