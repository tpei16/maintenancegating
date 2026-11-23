#!/usr/bin/env python
"""
68_direct_local_validation.py -- direct validation of the peer-trained ranking
on a campus's own outcomes before 2017, against the verdicts drawn from peers
and from local history.

The forward test of scripts/60_local_history_baseline.py compares a verdict
drawn from the peer campuses with one drawn from the campus's own history, all
inputs dated 2016 or earlier, against the top-decile lift the screen realizes on
the campus in 2017 to 2019.  Its local-history verdict applies the five gates
with calendar years in the place of campuses, so a record with fewer than three
evaluable years receives no local verdict by construction.  This script adds the
plain form of local validation: the top-decile lift of the same peer-trained
ranking on the campus's own pre-2017 outcomes, read directly.

SCORES.  f_h^(-c) of script 60, trained on the other campuses' rows anchored in
2016 or earlier and scoring every row of campus c
(data/processed/pred_single_pre2017/{c}.parquet).  Nothing is refitted.  The
runner asserts from script 60's fit records that campus c is not in the
training set of f_h^(-c) and that no training row is anchored after 2016.

LABELS.  A unit at quarter t is high-burden when its reactive labour in quarter
t+1 exceeds the system-specific 75th percentile of positive next-quarter
reactive labour, fitted on the training rows of f_h^(-c), that is, on the other
campuses' rows anchored in 2016 or earlier (script 60's convention).  The
threshold therefore uses no outcome of campus c and no outcome after 2017 Q1;
the stored labels are recomputed from the panel with this threshold and
asserted equal.

WINDOWS (anchor quarters of campus c, labels one quarter ahead):
  all     every quarter anchored in 2016 or earlier
  last8   the eight quarters 2015 Q1 to 2016 Q4, the record of a portfolio
          whose work-order history begins in 2015
A history row anchored in 2016 Q4 carries the label of 2017 Q1; the evaluation
window of the outcome starts with the label of 2017 Q2, so no labelled quarter
is shared.

OUTCOME-WINDOW SWEEP.  The direct local validation of the 91 evaluable pairs is
repeated with the window restricted to the last W quarters before 2017,
W in {1, 2, 3, 4, 6, 8, 12, 16} (a campus with fewer quarters uses what it
has); only the outcome record is shortened, the features keep their look-back.
For each W: coverage, verdict counts and later hit rates, the median width of
the bootstrap interval, the rank correlation of the local lift with the later
lift and its difference from that of the peer evidence score on the pairs where
both exist (pairs resampled), and, for the pairs without a local evaluation,
the later hit rates of the peer-supported pairs, of the pairs not supported
from peers and of acting on every pair.  The windows "all" and "last8" of the
pair table are reproduced inside the sweep (asserted).

DIRECT LOCAL VERDICT per (campus c, building system s, window w):
  evaluable   >= 10 high-burden units and >= 100 units of s on c in w
              (the evaluability rule of the paper)
  lift        top-decile lift of f_h^(-c) within (c, s, w), the ranking taken
              over the window's units
  interval    95% percentile interval of the lift, buildings of c resampled
              with replacement (1000 resamples, seed 42, metrics.bootstrap_ci)
  point       validated when evaluable and lift >= 2
  margin      validated when evaluable, lift >= 2 and the lower end of the
              interval >= 2
  otherwise   not validated (evaluable) or no local evaluation (not evaluable)

OUTCOME.  The top-decile lift in 2017 to 2019 and the evaluable set of
results/tables/local_history_baseline.csv; the 91 evaluable pairs and their
lifts are recomputed from the stored predictions and asserted equal.  A pair
holds when that lift is at least 2.

REFERENCE.  Before any new quantity is computed the runner reproduces the
verdicts of the held-out maps (script 60's check_reference against
cert_corrected.json, and results/tables/fold_gate_verdicts.csv against the
same), and recomputes the peer and local-history evidence scores of the
forward test from the stored pre-2017 predictions with script 45's gates,
asserting equality with the stored table.

Outputs -> results/metrics/direct_local_validation.json
           results/tables/direct_local_validation.csv   (one row per campus-system pair)
           results/tables/direct_local_validation_by_window.csv   (one row per outcome window)
"""
from __future__ import annotations
import os

# CPU only, with the thread caps set before numpy or fmscreen is imported
N_THREADS = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FMSCREEN_DEVICE"] = "cpu"
os.environ["FMSCREEN_THREADS"] = N_THREADS
_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS")
for _v in _THREAD_VARS:
    os.environ[_v] = N_THREADS

import sys, json, time, itertools, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import features as FE
from fmscreen import metrics as MET


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load("cert_corrected", "45_certification_corrected.py")
lh = _load("local_history_baseline", "60_local_history_baseline.py")
# script 60 sets its own thread caps on import; restore the caps of this run
os.environ["FMSCREEN_THREADS"] = N_THREADS
for _v in _THREAD_VARS:
    os.environ[_v] = N_THREADS

HIST_END, EVAL_YEARS = lh.HIST_END, lh.EVAL_YEARS
MIN_POS, MIN_ROWS = lh.MIN_POS, lh.MIN_ROWS
LAST8_FIRST_Q = 4 * HIST_END - 4          # 2015 Q1 in period_q units
LAST8_LAST_Q = 4 * HIST_END + 3           # 2016 Q4
WINDOWS = {"all": None, "last8": (LAST8_FIRST_Q, LAST8_LAST_Q)}
W_LIST = (1, 2, 3, 4, 6, 8, 12, 16)   # outcome windows of the last W quarters before 2017
LIFT_LEVEL = 2.0
N_BOOT = 1000
N_BOOT_RHO = 2000
STATUSES = ("validated", "not_validated", "no_local_evaluation")
LH_STATUSES = ("supported", "not_supported_with_verdict", "no_verdict")
LABEL = {"validated": "validated", "not_validated": "not validated",
         "no_local_evaluation": "no local evaluation"}


# --------------------------------------------------------------------------
# data and invariants
# --------------------------------------------------------------------------
def _panel() -> pd.DataFrame:
    p = lh._panel()
    p["university"] = p[C.COL_UNIV].astype(str)
    p["building"] = p["university"] + "/" + p[C.COL_BUILDING].astype(str)
    p["system"] = p[C.COL_SYSTEM].astype(str)
    return p


def check_training_sets(fit_records: list[dict], campuses: list[str]) -> list[dict]:
    """f_h^(-c) never saw campus c, and no training row is anchored after 2016."""
    by = {r["held_out"]: r for r in fit_records}
    out = []
    for c in campuses:
        r = by[c]
        assert r["scope"] == "all" and "file" in r, f"no stored single fit for campus {c}"
        assert c not in r["train_campuses"], f"campus {c} in the training set of its own model"
        assert r["train_max_year"] <= HIST_END, f"training row after {HIST_END} for campus {c}"
        out.append({"campus": c, "train_campuses": r["train_campuses"],
                    "train_max_year": r["train_max_year"], "n_train": r["n_train"]})
    return out


def check_labels(panel: pd.DataFrame, single: dict, campuses: list[str]) -> dict:
    """Recompute the stored high-burden labels of every row of campus c with the
    system-specific p75 threshold fitted on the other campuses' rows anchored in
    2016 or earlier; return the thresholds per campus."""
    year = (panel["period_q"] // 4).to_numpy()
    thr_out = {}
    for c in campuses:
        train = panel[(panel.university != c) & (year <= HIST_END)]
        assert int((train.period_q // 4).max()) <= HIST_END and (train.university != c).all()
        thr, gthr = FE.severity_thresholds(train, target="labour", pctl=lh.PCTL)
        own = panel[panel.university == c]
        lab = own[["university", "building", "system", "period_q"]].assign(
            y_rebuilt=FE.apply_severity(own, thr, gthr, "labour"))
        m = single[c].merge(lab, on=["university", "building", "system", "period_q"],
                            how="left", validate="one_to_one")
        assert len(m) == len(single[c]) and m.y_rebuilt.notna().all(), f"rows of {c} not in panel"
        assert (m.y.to_numpy() == m.y_rebuilt.to_numpy()).all(), \
            f"stored labels of campus {c} differ from the peer pre-2017 threshold"
        thr_out[c] = {"by_system": {str(k): round(float(v), 4) for k, v in thr.items()},
                      "global": round(float(gthr), 4)}
    return thr_out


def check_reference(panel: pd.DataFrame) -> None:
    """The verdicts of the held-out maps are reproduced before anything new."""
    lh.check_reference(panel)
    ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]["outer_S"]
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv")
    assert len(fv) == 141
    for r in fv.itertuples():
        assert abs(ref[str(r.campus)][r.system] - r.S) < 1e-9, (r.campus, r.system)
        assert bool(r.certified) == (r.S >= cc.S_CUT)


def reproduce_forward_verdicts(panel: pd.DataFrame, single: dict, campuses: list[str],
                               stored: pd.DataFrame) -> None:
    """Peer and local-history evidence scores of the forward test, recomputed
    from the stored pre-2017 predictions, equal the stored table."""
    year = (panel["period_q"] // 4).to_numpy()
    univ = panel["university"].to_numpy()
    empty = pd.DataFrame({k: pd.Series(dtype=t) for k, t in
                          [("held_out", str), ("n_held_out", int), ("university", str),
                           ("building", str), ("system", str), ("period_q", int),
                           ("y", np.int8), ("score", np.float32)]})
    pair_pred = {}
    for a, b in itertools.combinations(campuses, 2):
        f = lh.PAIR_DIR / f"{a}__{b}.parquet"
        pair_pred[frozenset((a, b))] = pd.read_parquet(f) if f.exists() else empty

    def as_years(d):
        d = d.copy()
        d["university"] = (d["period_q"] // 4).astype(str)   # years take the place of contexts
        return d

    n_checked = 0
    for c in campuses:
        tr_peer = lh.trace_table(panel[(univ != c) & (year <= HIST_END)])
        tr_local = lh.trace_table(panel[(univ == c) & (year <= HIST_END)])
        dev = cc.dev_corpus(c, campuses, None, pair_pred, "double")
        assert (dev.university != c).all() and int((dev.period_q // 4).max()) <= HIST_END
        peer = cc.gates_for(dev, tr_peer, "R_quarter", True)
        own = single[c][(single[c].period_q // 4) <= HIST_END]
        local = cc.gates_for(as_years(cc._add_ranks(own)), tr_local, "R_quarter", True)
        for r in stored[stored.campus == c].itertuples():
            for arm, g in (("peer", peer), ("local", local)):
                s_new = g[r.system]["S"] if r.system in g else None
                s_old = getattr(r, f"S_{arm}")
                s_old = None if pd.isna(s_old) else float(s_old)
                assert (s_new is None) == (s_old is None), (c, r.system, arm, s_new, s_old)
                assert s_new is None or abs(s_new - s_old) < 1e-9, (c, r.system, arm, s_new, s_old)
            n_checked += 1
    assert n_checked == len(stored)


# --------------------------------------------------------------------------
# direct local lift
# --------------------------------------------------------------------------
def last_quarters(W: int) -> tuple[int, int]:
    """Anchor-quarter bounds of the last W quarters before 2017."""
    return (LAST8_LAST_Q + 1 - W, LAST8_LAST_Q)


def window_rows(d: pd.DataFrame, w: str | tuple | None) -> pd.DataFrame:
    """Rows of d anchored in 2016 or earlier, restricted to a named window or to
    explicit (first, last) anchor-quarter bounds."""
    h = d[(d.period_q // 4) <= HIST_END]
    bounds = WINDOWS[w] if isinstance(w, str) else w
    if bounds is not None:
        lo, hi = bounds
        h = h[(h.period_q >= lo) & (h.period_q <= hi)]
    assert h.empty or int(h.period_q.max()) <= LAST8_LAST_Q, "row anchored after 2016"
    return h


def local_lift(h: pd.DataFrame) -> dict:
    y, sc = h.y.to_numpy(), h.score.to_numpy()
    out = {"n_rows": int(len(h)), "n_pos": int(y.sum()),
           "n_buildings": int(h.building.nunique()), "n_quarters": int(h.period_q.nunique())}
    ev = bool(out["n_pos"] >= MIN_POS and out["n_rows"] >= MIN_ROWS)
    out["evaluable"] = ev
    if not ev:
        out.update({"lift": None, "lift_ci_lo": None, "lift_ci_hi": None, "n_boot": 0})
        return out
    ci = MET.bootstrap_ci(y, sc, h.building.to_numpy(), metric="lift_top10",
                          k=C.TOPK_BUDGET, n_boot=N_BOOT, seed=C.RANDOM_SEED)
    lift = float(MET.lift_at_topk(y, sc, C.TOPK_BUDGET))
    assert abs(lift - ci["point"]) < 1e-12
    out.update({"lift": round(lift, 3),
                "lift_ci_lo": None if np.isnan(ci["ci_lo"]) else round(ci["ci_lo"], 3),
                "lift_ci_hi": None if np.isnan(ci["ci_hi"]) else round(ci["ci_hi"], 3),
                "n_boot": int(ci["n_boot"])})
    return out


def verdicts(r: dict) -> tuple[str, str]:
    if not r["evaluable"]:
        return "no_local_evaluation", "no_local_evaluation"
    point = "validated" if r["lift"] >= LIFT_LEVEL else "not_validated"
    lo = r["lift_ci_lo"]
    margin = ("validated" if (point == "validated" and lo is not None and lo >= LIFT_LEVEL)
              else "not_validated")
    return point, margin


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------
def outcome_block(sub: pd.DataFrame) -> dict:
    b = lh.share_block(sub.holds_model.astype(bool))
    b["median_later_lift"] = (round(float(sub.lift_eval_model.median()), 3)
                              if len(sub) else None)
    return b


def spearman_block(x: pd.Series, y: pd.Series, seed: int) -> dict:
    m = x.notna() & y.notna()
    x, y = x[m].to_numpy(float), y[m].to_numpy(float)
    n = int(len(x))
    if n < 3:
        return {"n": n, "rho": None, "p": None, "ci_95": [None, None]}
    rho, p = spearmanr(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(N_BOOT_RHO):
        i = rng.integers(0, n, n)
        if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
            boots.append(spearmanr(x[i], y[i])[0])
    return {"n": n, "rho": round(float(rho), 3), "p": float(f"{float(p):.3g}"),
            "ci_95": ([round(float(np.quantile(boots, 0.025)), 3),
                       round(float(np.quantile(boots, 0.975)), 3)] if boots else [None, None])}


def paired_spearman(a: pd.Series, b: pd.Series, y: pd.Series, names: tuple[str, str],
                    seed: int) -> dict:
    """Spearman of a and of b with y on the pairs where both exist, and the
    bootstrap interval of the difference rho(a) - rho(b) (pairs resampled)."""
    m = a.notna() & b.notna() & y.notna()
    A, B, Y = a[m].to_numpy(float), b[m].to_numpy(float), y[m].to_numpy(float)
    n = int(len(Y))
    if n < 3:
        return {"n": n}
    ra, rb = spearmanr(A, Y)[0], spearmanr(B, Y)[0]
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(N_BOOT_RHO):
        i = rng.integers(0, n, n)
        if np.ptp(A[i]) > 0 and np.ptp(B[i]) > 0 and np.ptp(Y[i]) > 0:
            d.append(spearmanr(A[i], Y[i])[0] - spearmanr(B[i], Y[i])[0])
    return {"n": n, f"rho_{names[0]}": round(float(ra), 3), f"rho_{names[1]}": round(float(rb), 3),
            "difference": round(float(ra - rb), 3),
            "difference_ci_95": ([round(float(np.quantile(d, 0.025)), 3),
                                  round(float(np.quantile(d, 0.975)), 3)] if d else [None, None])}


def flag_block(ev: pd.DataFrame, flag: pd.Series) -> dict:
    h = ev.holds_model.astype(bool)
    k = int((flag & h).sum())
    return {"n_flagged": int(flag.sum()), "n_flagged_hold": k,
            "share_hold": round(k / int(flag.sum()), 4) if flag.sum() else None,
            "wilson_95": lh.wilson(k, int(flag.sum())),
            "share_of_holding_pairs_flagged": round(k / int(h.sum()), 4),
            "median_later_lift": (round(float(ev.lift_eval_model[flag].median()), 3)
                                  if flag.any() else None)}


def newcombe_difference(k1: int, n1: int, k2: int, n2: int) -> dict:
    """Difference of two independent shares with Newcombe's hybrid score interval
    (built from the two Wilson intervals)."""
    if n1 == 0 or n2 == 0:
        return {"difference": None, "ci_95": [None, None]}
    p1, p2 = k1 / n1, k2 / n2
    (l1, u1), (l2, u2) = lh.wilson(k1, n1), lh.wilson(k2, n2)
    d = p1 - p2
    return {"difference": round(d, 4),
            "ci_95": [round(d - np.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), 4),
                      round(d + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2), 4)]}


def window_sweep(ev: pd.DataFrame, single: dict) -> tuple[dict, list[dict]]:
    """Direct local validation of the 91 pairs with the outcome window restricted
    to the last W quarters before 2017, and with the full pre-2017 record."""
    hold = ev.holds_model.astype(bool)
    full_cov = int(ev.all_evaluable.astype(bool).sum())
    block, csv_rows = {}, []
    for W in W_LIST + ("all",):
        bounds = None if W == "all" else last_quarters(W)
        res = []
        for r in ev.itertuples():
            d = single[r.campus]
            x = local_lift(window_rows(d[d.system == r.system], bounds))
            pv, mv = verdicts(x)
            res.append({**x, "point": pv, "margin": mv})
        R = pd.DataFrame(res, index=ev.index)
        name = "all" if W == "all" else f"last{W}"
        if name in WINDOWS:            # the windows of the pair table are reproduced
            same = (R.lift.astype(float).fillna(-1).to_numpy()
                    == ev[f"{name}_lift"].astype(float).fillna(-1).to_numpy())
            assert same.all() and (R.point == ev[f"_{name}_point"]).all() \
                and (R.margin == ev[f"_{name}_margin"]).all(), f"window {name} not reproduced"
        loc = R.evaluable.astype(bool)
        val, valm = R.point == "validated", R.margin == "validated"
        width = (R.lift_ci_hi.astype(float) - R.lift_ci_lo.astype(float))[loc].dropna()
        lift_loc = R.lift.astype(float)
        rho_loc = spearman_block(lift_loc, ev.lift_eval_model, C.RANDOM_SEED)
        paired = paired_spearman(lift_loc, ev.S_peer.astype(float), ev.lift_eval_model,
                                 ("direct_local_lift", "peer_evidence_score"), C.RANDOM_SEED)
        ci = paired.get("difference_ci_95", [None, None])
        dist = (None if ci[0] is None else bool(ci[0] > 0 or ci[1] < 0))
        comp = ~loc
        sup, nsup = comp & ev.supported_peer, comp & ~ev.supported_peer
        k_s, n_s = int((sup & hold).sum()), int(sup.sum())
        k_n, n_n = int((nsup & hold).sum()), int(nsup.sum())
        k_a, n_a = int((comp & hold).sum()), int(comp.sum())
        camp = sorted(set(ev.campus[loc]), key=int)
        b = {
            "window": "all pre-2017 quarters" if W == "all" else f"last {W} quarters before 2017",
            "anchor_quarters": None if W == "all" else
                [f"{bounds[0] // 4}Q{bounds[0] % 4 + 1}", f"{bounds[1] // 4}Q{bounds[1] % 4 + 1}"],
            "n_local_evaluation": int(loc.sum()),
            "share_of_full_record_coverage": round(int(loc.sum()) / full_cov, 4),
            "campuses_with_a_local_evaluation": camp,
            "n_campuses_with_a_local_evaluation": len(camp),
            "validated": outcome_block(ev[val]),
            "not_validated": outcome_block(ev[R.point == "not_validated"]),
            "validated_with_margin": outcome_block(ev[valm]),
            "not_validated_with_margin": outcome_block(ev[loc & ~valm]),
            "median_interval_width": round(float(width.median()), 3) if len(width) else None,
            "n_interval_defined": int(len(width)),
            "rank_correlation_direct_local_lift": rho_loc,
            "same_pairs_direct_local_vs_peer": {**paired, "distinguishable": dist},
            "without_local_evaluation": {
                "accept_all": outcome_block(ev[comp]),
                "peer_supported": outcome_block(ev[sup]),
                "peer_not_supported": outcome_block(ev[nsup]),
                "peer_no_verdict": int((comp & ~ev.verdict_peer).sum()),
                "supported_minus_not_supported": newcombe_difference(k_s, n_s, k_n, n_n),
                "supported_minus_accept_all": (round(k_s / n_s - k_a / n_a, 4)
                                               if n_s and n_a else None),
                "holding_pairs_left_out_by_acting_on_supported_only": k_n,
                "rank_correlation_peer_evidence_score": spearman_block(
                    ev.S_peer.astype(float).where(comp), ev.lift_eval_model, C.RANDOM_SEED),
            },
            "pairs": [{"campus": ev.campus[i], "system": ev.system[i], "lift": R.lift[i],
                       "lift_ci_lo": R.lift_ci_lo[i], "lift_ci_hi": R.lift_ci_hi[i],
                       "verdict": LABEL[R.point[i]], "verdict_margin": LABEL[R.margin[i]]}
                      for i in R.index[loc]],
        }
        block[name] = b
        wb = b["without_local_evaluation"]
        csv_rows.append({
            "window": name, "W": None if W == "all" else W,
            "n_local_evaluation": b["n_local_evaluation"],
            "share_of_full_record_coverage": b["share_of_full_record_coverage"],
            "n_campuses_with_local_evaluation": b["n_campuses_with_a_local_evaluation"],
            **{f"{g}_{x}": (b[g][x] if x != "wilson_lo" and x != "wilson_hi" else
                            b[g]["wilson_95"][0 if x == "wilson_lo" else 1])
               for g in ("validated", "not_validated", "validated_with_margin",
                         "not_validated_with_margin")
               for x in ("n", "n_hold", "share_hold", "wilson_lo", "wilson_hi")},
            "median_interval_width": b["median_interval_width"],
            "rho_local": rho_loc["rho"], "rho_local_n": rho_loc["n"],
            "rho_local_ci_lo": rho_loc["ci_95"][0], "rho_local_ci_hi": rho_loc["ci_95"][1],
            "common_n": paired["n"],
            "common_rho_local": paired.get("rho_direct_local_lift"),
            "common_rho_peer": paired.get("rho_peer_evidence_score"),
            "common_difference": paired.get("difference"),
            "common_difference_ci_lo": ci[0], "common_difference_ci_hi": ci[1],
            "common_distinguishable": dist,
            **{f"nolocal_{g}_{x}": (wb[g][x] if x not in ("wilson_lo", "wilson_hi") else
                                    wb[g]["wilson_95"][0 if x == "wilson_lo" else 1])
               for g in ("accept_all", "peer_supported", "peer_not_supported")
               for x in ("n", "n_hold", "share_hold", "wilson_lo", "wilson_hi")},
            "nolocal_supported_minus_not_supported": wb["supported_minus_not_supported"]["difference"],
            "nolocal_supported_minus_not_supported_ci_lo": wb["supported_minus_not_supported"]["ci_95"][0],
            "nolocal_supported_minus_not_supported_ci_hi": wb["supported_minus_not_supported"]["ci_95"][1],
            "nolocal_rho_peer": wb["rank_correlation_peer_evidence_score"]["rho"],
        })

    # summary statements
    cov = {W: block[f"last{W}"]["n_local_evaluation"] for W in W_LIST}
    shortest = lambda share: next((W for W in W_LIST if cov[W] >= share * full_cov), None)
    dist = {W: block[f"last{W}"]["same_pairs_direct_local_vs_peer"]["distinguishable"]
            for W in W_LIST}
    dist_full = block["all"]["same_pairs_direct_local_vs_peer"]["distinguishable"]
    from_w = None
    if dist_full:
        for W in sorted(W_LIST, reverse=True):
            if dist[W]:
                from_w = W
            else:
                break
    block["statements"] = {
        "full_record_coverage": full_cov,
        "shortest_W_covering_half_of_full_record_coverage": shortest(0.5),
        "shortest_W_covering_90pct_of_full_record_coverage": shortest(0.9),
        "difference_distinguishable_by_W": {str(W): dist[W] for W in W_LIST},
        "difference_distinguishable_full_record": dist_full,
        "shortest_W_from_which_difference_is_distinguishable_at_every_longer_window": from_w,
        "note": ("distinguishable means the 95% bootstrap interval of rho(direct local lift) - "
                 "rho(peer evidence score), computed on the pairs where both exist at that "
                 "window, excludes zero; None for the shortest W means that the difference is "
                 "distinguishable with the full pre-2017 record only"),
    }
    return block, csv_rows


def main() -> None:
    t_start = time.time()
    panel = _panel()
    campuses = sorted(panel["university"].unique())
    assert len(campuses) == 9

    check_reference(panel)
    print("[68] held-out map verdicts reproduced (cert_corrected.json, fold_gate_verdicts.csv)",
          flush=True)

    lhj = json.load(open(C.METRICS / "local_history_baseline.json"))
    train_sets = check_training_sets(lhj["fits"]["single"], campuses)
    single = {}
    for c in campuses:
        d = pd.read_parquet(lh.SINGLE_DIR / f"{c}.parquet")
        assert (d.held_out == c).all() and (d.n_held_out == 1).all() and (d.university == c).all()
        single[c] = d
    thresholds = check_labels(panel, single, campuses)
    print("[68] f_h^(-c) excludes c and every row after 2016; stored labels equal the peer "
          "pre-2017 p75 threshold", flush=True)

    stored = pd.read_csv(C.TABLES / "local_history_baseline.csv", dtype={"campus": str})
    reproduce_forward_verdicts(panel, single, campuses, stored)
    print("[68] peer and local-history evidence scores of the forward test reproduced", flush=True)

    # ---- the outcome: evaluable set and 2017-2019 lift, asserted equal ----
    st = stored.set_index(["campus", "system"])
    seen = set()
    for c in campuses:
        s = single[c]
        yr = s.period_q // 4
        for k in sorted(s.system.unique()):
            e = s[(s.system == k) & yr.isin(EVAL_YEARS)]
            assert e.empty or int(e.period_q.min()) > LAST8_LAST_Q
            ev = bool(e.y.sum() >= MIN_POS and len(e) >= MIN_ROWS)
            r = st.loc[(c, k)]
            assert ev == bool(r.evaluable), (c, k)
            if ev:
                lm = round(float(MET.lift_at_topk(e.y.to_numpy(), e.score.to_numpy(),
                                                  C.TOPK_BUDGET)), 3)
                assert abs(lm - float(r.lift_eval_model)) < 1e-9, (c, k, lm, r.lift_eval_model)
            seen.add((c, k))
    assert seen == set(st.index), "campus-system pairs differ from the stored table"
    assert int(stored.evaluable.sum()) == 91
    print("[68] the 91 evaluable pairs and their 2017-2019 lifts equal the stored table",
          flush=True)

    # ---- direct local lift per pair and window ----
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    heldout_keys = set(zip(fv.campus, fv.system))
    rows = []
    for r in stored.itertuples():
        d = single[r.campus]
        d = d[d.system == r.system]
        row = {"campus": r.campus, "system": r.system, "system_desc": r.system_desc,
               "in_heldout_maps": (r.campus, r.system) in heldout_keys,
               "evaluable_2017_2019": bool(r.evaluable),
               "lift_2017_2019": r.lift_eval_model, "holds_2017_2019": r.holds_model,
               "S_peer": r.S_peer, "supported_peer": bool(r.cert_peer),
               "verdict_peer": bool(r.verdict_peer),
               "S_local_history": r.S_local, "supported_local_history": bool(r.cert_local),
               "verdict_local_history": bool(r.verdict_local)}
        for w in WINDOWS:
            res = local_lift(window_rows(d, w))
            pv, mv = verdicts(res)
            row.update({f"{w}_{x}": v for x, v in res.items()})
            row[f"{w}_verdict"] = LABEL[pv]
            row[f"{w}_verdict_margin"] = LABEL[mv]
            row[f"_{w}_point"], row[f"_{w}_margin"] = pv, mv
        rows.append(row)
    tab = pd.DataFrame(rows)
    print(f"[68] direct local lifts computed for {len(tab)} pairs x {len(WINDOWS)} windows",
          flush=True)

    # ---- summaries over the 91 pairs evaluable in 2017 to 2019 ----
    ev = tab[tab.evaluable_2017_2019].copy()
    ev["holds_model"] = ev.holds_2017_2019.astype(bool)
    ev["lift_eval_model"] = ev.lift_2017_2019.astype(float)
    ev["lh_status"] = np.where(ev.supported_local_history, "supported",
                               np.where(ev.verdict_local_history, "not_supported_with_verdict",
                                        "no_verdict"))
    # the stored local-history figures this comparison is set against
    assert int(ev.supported_local_history.sum()) == 22 and \
        int(ev[ev.supported_local_history].holds_model.sum()) == 22
    assert int((~ev.verdict_local_history).sum()) == 34

    by_window = {}
    for w in WINDOWS:
        by_def = {}
        for dname in ("point", "margin"):
            stt = ev[f"_{w}_{dname}"]
            blk = {"counts": {s: int((stt == s).sum()) for s in STATUSES},
                   "by_campus": {c: {s: int(((ev.campus == c) & (stt == s)).sum())
                                     for s in STATUSES} for c in campuses},
                   "later_outcome": {s: outcome_block(ev[stt == s]) for s in STATUSES},
                   "cross_peer": {}, "cross_local_history": {}}
            for pk, pm in (("peer_supported", ev.supported_peer),
                           ("peer_not_supported", ~ev.supported_peer),
                           ("peer_not_supported_with_verdict", ~ev.supported_peer & ev.verdict_peer),
                           ("peer_no_verdict", ~ev.verdict_peer)):
                blk["cross_peer"][pk] = {s: outcome_block(ev[pm & (stt == s)]) for s in STATUSES}
            for lk in LH_STATUSES:
                blk["cross_local_history"][lk] = {
                    s: outcome_block(ev[(ev.lh_status == lk) & (stt == s)]) for s in STATUSES}
            # what the peer verdict adds where the campus has no local evaluation
            none = stt == "no_local_evaluation"
            blk["peer_added_coverage"] = {
                "peer_supported_without_local_evaluation": outcome_block(ev[none & ev.supported_peer]),
                "peer_supported_total": int(ev.supported_peer.sum()),
                "share_of_peer_supported_without_local_evaluation":
                    round(float((none & ev.supported_peer).sum() / ev.supported_peer.sum()), 4),
                "peer_supported_and_locally_validated": int((ev.supported_peer & (stt == "validated")).sum()),
                "share_of_peer_supported_locally_validated":
                    round(float((ev.supported_peer & (stt == "validated")).sum()
                                / ev.supported_peer.sum()), 4)}
            by_def[dname] = blk
        lo = ev[f"{w}_lift"].astype(float)
        by_def["rank_correlation_with_2017_2019_lift"] = {
            "direct_local_lift": spearman_block(lo, ev.lift_eval_model, C.RANDOM_SEED),
            "peer_evidence_score": spearman_block(ev.S_peer.astype(float), ev.lift_eval_model,
                                                  C.RANDOM_SEED),
            "local_history_evidence_score": spearman_block(ev.S_local_history.astype(float),
                                                           ev.lift_eval_model, C.RANDOM_SEED),
            "peer_evidence_score_where_no_local_evaluation": spearman_block(
                ev.S_peer.astype(float).where(lo.isna()), ev.lift_eval_model, C.RANDOM_SEED),
            "same_pairs_direct_local_vs_peer": paired_spearman(
                lo, ev.S_peer.astype(float), ev.lift_eval_model,
                ("direct_local_lift", "peer_evidence_score"), C.RANDOM_SEED),
        }
        by_window[w] = by_def

    flags = {"peer_supported": ev.supported_peer,
             "local_history_supported": ev.supported_local_history}
    for w in WINDOWS:
        for dname in ("point", "margin"):
            flags[f"direct_{w}_{dname}_validated"] = ev[f"_{w}_{dname}"] == "validated"
            flags[f"peer_supported_or_direct_{w}_{dname}_validated"] = \
                ev.supported_peer | (ev[f"_{w}_{dname}"] == "validated")
            flags[f"peer_supported_where_no_direct_{w}_evaluation"] = \
                ev.supported_peer & (ev[f"_{w}_point"] == "no_local_evaluation")
    flag_summary = {k: flag_block(ev, v) for k, v in flags.items()}

    sweep, sweep_rows = window_sweep(ev, single)
    print(f"[68] outcome-window sweep done for W in {W_LIST} and the full record", flush=True)

    # ---- coverage of a direct local evaluation ----
    coverage = {}
    for w in WINDOWS:
        e_ = tab[f"{w}_evaluable"].astype(bool)
        camp = sorted({c for c, x in zip(tab.campus, e_) if x}, key=int)
        coverage[w] = {
            "campuses_with_a_local_evaluation": camp,
            "n_campuses_with_a_local_evaluation": len(camp),
            "n_campuses": len(campuses),
            "pairs_all_159": {"n": int(len(tab)), "n_local_evaluation": int(e_.sum())},
            "pairs_heldout_maps_141": {
                "n": len(heldout_keys),
                "n_in_forward_table": int(tab.in_heldout_maps.sum()),
                "n_local_evaluation": int((e_ & tab.in_heldout_maps).sum())},
            "pairs_evaluable_2017_2019_91": {"n": int(len(ev)),
                                             "n_local_evaluation": int(ev[f"{w}_evaluable"].sum())},
            "by_campus": {c: {"quarters_in_window": int(window_rows(single[c], w).period_q.nunique()),
                              "pairs_local_evaluation": int(e_[tab.campus == c].sum()),
                              "pairs_evaluable_2017_2019": int((ev.campus == c).sum()),
                              "evaluable_2017_2019_with_local_evaluation":
                                  int(ev[(ev.campus == c)][f"{w}_evaluable"].sum())}
                          for c in campuses}}

    # ---- outputs ----
    out_tab = tab.drop(columns=[c for c in tab.columns if c.startswith("_")])
    out_tab.to_csv(C.TABLES / "direct_local_validation.csv", index=False)
    pd.DataFrame(sweep_rows).to_csv(C.TABLES / "direct_local_validation_by_window.csv", index=False)

    out = {
        "design": ("Top-decile lift of the peer-trained ranking f_h^(-c) of the forward test on "
                   "campus c's own outcomes anchored in 2016 or earlier, per building system, "
                   "over all pre-2017 quarters and over the eight quarters 2015 Q1 to 2016 Q4, "
                   "with a building bootstrap interval; the resulting direct local verdicts are "
                   "compared with the peer and local-history verdicts of the forward test "
                   "against the lift realized in 2017 to 2019."),
        "definitions": {
            "scores": ("f_h^(-c): trained on the other campuses' rows anchored in 2016 or "
                       "earlier (deployed configuration severity_labour|p75|M1|gbdt), scoring "
                       "campus c; stored predictions of the forward test, not refitted"),
            "labels": ("high-burden unit: reactive labour in quarter t+1 above the "
                       "system-specific 75th percentile of positive next-quarter reactive "
                       "labour, fitted on the training rows of f_h^(-c) (the other campuses' "
                       "rows anchored in 2016 or earlier); no outcome of campus c enters it"),
            "windows": {"all": "every anchor quarter of campus c in 2016 or earlier",
                        "last8": "the eight anchor quarters 2015 Q1 to 2016 Q4"},
            "label_dating": ("a row anchored in 2016 Q4 carries the label of 2017 Q1; the "
                             "2017 to 2019 outcome starts with the label of 2017 Q2"),
            "evaluable": f">= {MIN_POS} high-burden units and >= {MIN_ROWS} units in the window",
            "lift": "top-decile lift within (campus, building system, window)",
            "interval": (f"95% percentile interval, buildings resampled with replacement, "
                         f"{N_BOOT} resamples, seed {C.RANDOM_SEED}; resamples without a "
                         f"high-burden unit are dropped (n_boot records the rest)"),
            "validated_point": "evaluable and lift >= 2",
            "validated_margin": "evaluable, lift >= 2 and lower end of the interval >= 2",
            "not_validated": "evaluable and the validation condition not met",
            "no_local_evaluation": "not evaluable in the window",
            "holds": "top-decile lift in 2017 to 2019 of at least 2 (pairs evaluable then)",
            "peer_supported": "evidence score from peers' pre-2017 records S_peer >= 0.67",
            "local_history_supported": ("evidence score from local history S_local >= 0.67; "
                                        "no verdict when the stratum has < 20 high-burden "
                                        "units or < 200 units before 2017"),
            "rank_correlation": (f"Spearman, over the pairs where the predictor exists; "
                                 f"95% interval from {N_BOOT_RHO} resamples of pairs, seed "
                                 f"{C.RANDOM_SEED}"),
            "wilson_95": "Wilson score interval for the share that holds",
        },
        "checks": {
            "heldout_map_verdicts_reproduced": True,
            "forward_peer_and_local_history_scores_reproduced": True,
            "evaluable_91_and_lifts_equal_stored_table": True,
            "training_sets": train_sets,
            "labels_equal_peer_pre2017_threshold": True,
            "max_anchor_quarter_in_local_windows": "2016 Q4",
        },
        "thresholds_p75_by_campus": thresholds,
        "n_pairs": int(len(tab)),
        "n_evaluable_2017_2019": int(len(ev)),
        "n_hold_2017_2019": int(ev.holds_model.sum()),
        "all_evaluable_hold": outcome_block(ev),
        "coverage": coverage,
        "by_window": by_window,
        "flags": flag_summary,
        "outcome_window_sweep": sweep,
        "runtime_seconds": None,
    }
    out["runtime_seconds"] = round(time.time() - t_start, 1)
    json.dump(out, open(C.METRICS / "direct_local_validation.json", "w"), indent=2, default=str)

    # ---- console ----
    print(f"\nevaluable 2017-2019: {len(ev)}, holding {out['all_evaluable_hold']}")
    for w in WINDOWS:
        cv = coverage[w]
        print(f"\n[{w}] local evaluation: campuses {cv['n_campuses_with_a_local_evaluation']}/9, "
              f"pairs {cv['pairs_all_159']['n_local_evaluation']}/159, "
              f"{cv['pairs_heldout_maps_141']['n_local_evaluation']}/141, "
              f"{cv['pairs_evaluable_2017_2019_91']['n_local_evaluation']}/91")
        for dname in ("point", "margin"):
            b = by_window[w][dname]
            print(f"  {dname}: " + ", ".join(
                f"{s} {b['later_outcome'][s]['n']} (hold {b['later_outcome'][s]['n_hold']}, "
                f"{b['later_outcome'][s]['share_hold']})" for s in STATUSES))
            for pk in ("peer_supported", "peer_not_supported"):
                print(f"    {pk}: " + ", ".join(
                    f"{s} {b['cross_peer'][pk][s]['n']}/{b['cross_peer'][pk][s]['n_hold']}"
                    for s in STATUSES))
        rc = by_window[w]["rank_correlation_with_2017_2019_lift"]
        print(f"  rho local {rc['direct_local_lift']}, peer {rc['peer_evidence_score']}, "
              f"same pairs {rc['same_pairs_direct_local_vs_peer']}")
    print(f"[68] done in {out['runtime_seconds']} s")


if __name__ == "__main__":
    main()
