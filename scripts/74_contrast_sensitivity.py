#!/usr/bin/env python
"""
74_contrast_sensitivity.py -- how far the main held-out, forward and cold-start
contrasts rest on single campuses or building systems; a clustered interval for
the difference between two rank correlations; the inputs that differ between
the constructions of the evidence gating; and how often the prior-record-trace
gate decides a verdict.

No model is fitted.  The analysis reads the stored tables, the stored
predictions and the panel.

(1) LEAVE ONE OUT.  For each contrast, the value with all campus-system pairs
    of one campus removed, and with all pairs of one building system removed,
    in turn:
      held-out contrast   share of supported minus share of not supported pairs
                          whose top-decile lift on the held-out campus reaches
                          2 (74 of 77 against 14 of 18; fold_gate_verdicts.csv)
      held-out Spearman   Spearman correlation of S with the held-out lift
                          (0.26, same 95 pairs)
      forward contrast    the same difference in the forward test, lift in
                          2017 to 2019, verdict from peer records through 2016
                          (54 of 61 against 24 of 27; local_history_baseline.csv);
                          the three evaluable pairs without a peer verdict are
                          reported separately and, as a variant, counted as not
                          supported
      cold-start contrast the forward contrast on the pairs without a direct
                          local evaluation (15 of 22 against 7 of 9;
                          direct_local_validation.csv, all pre-2017 quarters);
                          the not supported group includes the pairs without a
                          peer verdict, as stored; a variant leaves them out
    A removal "changes the sign" when the value has the opposite sign of the
    full value, and "removes the difference" when the value is at most zero
    for a positive full value (at least zero for a negative one) or is
    undefined because a group becomes empty.

(2) DIFFERENCE OF TWO RANK CORRELATIONS.  On the 59 pairs with both a direct
    local lift (all pre-2017 quarters) and a peer evidence score, the Spearman
    correlation of each with the 2017 to 2019 lift and their difference
    (0.77 against 0.52), with 95% intervals from the two-way cluster bootstrap
    over campuses and building systems of 70_clustered_uncertainty.py, the two
    one-way cluster bootstraps and the pair-level bootstrap (5,000 replicates
    each), beside the stored pair-level interval of script 68.

(3) INPUTS OF THE CONSTRUCTIONS.  The stored high-burden labels of every
    prediction file are recomputed from the panel with the 75th-percentile
    thresholds fitted on the stated campuses and asserted equal: f^(-c) on
    C\\{c}, f^(-{a,b}) on C\\{a,b}, and the forward-test fits on the same sets
    restricted to rows anchored in 2016 or earlier; the prior-year count rule
    of 48_learner_invariance.py carries the labels of pred_single and
    pred_pair unchanged.  For every target campus and building system, the
    peers' high-burden quarters, units and evaluable campuses are counted
    under single nesting, double nesting, the nearest-size matched control of
    script 57 and the count rule, and every difference in score eligibility
    and data sufficiency is listed with the per-peer counts behind it.

(4) PRIOR-RECORD TRACE.  From the stored gate scores of the 141 pairs and the
    16 systems of the pooled map: S and the verdict with the trace gate
    omitted (mean of the other four scores, prerequisite rule unchanged), with
    its score set to pass, to caution and to fail, and with a variant scored on
    the ratio of the no-antecedent share to its frequency-matched reference
    (script 61; pass below 0.5, caution below 1, fail at or above 1, caution
    below 30 extreme events), computed from the peers' records of each map.
    The route counts of every variant are recomputed with route() of script
    45 on the stored predictions.

Output -> results/metrics/contrast_sensitivity.json
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FMSCREEN_DEVICE", "cpu")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "FMSCREEN_THREADS"):
    os.environ.setdefault(_v, "1")
_THREADS = os.environ["OMP_NUM_THREADS"]

import sys, json, time, itertools, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import features as FE


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


cc = _load("cert_corrected", "45_certification_corrected.py")
sr = _load("fold_verdicts", "56_fold_verdicts_and_simple_rule.py")
cu = _load("clustered_uncertainty", "70_clustered_uncertainty.py")
li = _load("learner_invariance", "48_learner_invariance.py")
tf = _load("trace_frequency_null", "61_trace_frequency_null.py")
# scripts 48 and 61 set their own thread caps on import; restore those of this run
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
           "FMSCREEN_THREADS"):
    os.environ[_v] = _THREADS

GATES, HARD = cc.GATES, cc.HARD
FAIL, CAUTION, PASS = cc.FAIL, cc.CAUTION, cc.PASS
S_CUT = cc.S_CUT
RCOL = "R_quarter"
SEED = C.RANDOM_SEED
N_BOOT = cu.N_BOOT
SCHEMES = cu.SCHEMES
TWO_FOLD = 2.0
PCTL = 75
RATIO_PASS, RATIO_FAIL = 0.5, 1.0
LEVEL = {FAIL: "fail", CAUTION: "caution", PASS: "pass"}
ELIG_POS, ELIG_ROWS = 20, 200
TWO_SYSTEMS = ("C20", "D10")          # stairs and conveying
OUT_JSON = C.METRICS / "contrast_sensitivity.json"


def rnd(x, d=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), d)


# --------------------------------------------------------------------------
# (1) leave one out
# --------------------------------------------------------------------------
def share_difference(d: pd.DataFrame) -> dict:
    s, n = d[d.group], d[~d.group]
    if len(s) == 0 or len(n) == 0:
        return {"value": None, "n_supported": int(len(s)), "k_supported": int(s.hold.sum()),
                "n_not_supported": int(len(n)), "k_not_supported": int(n.hold.sum())}
    return {"value": float(s.hold.mean() - n.hold.mean()),
            "n_supported": int(len(s)), "k_supported": int(s.hold.sum()),
            "n_not_supported": int(len(n)), "k_not_supported": int(n.hold.sum())}


def spearman_value(d: pd.DataFrame) -> dict:
    if len(d) < 3 or d.score.nunique() < 2 or d.lift.nunique() < 2:
        return {"value": None, "n": int(len(d))}
    r, p = spearmanr(d.score, d.lift)
    return {"value": float(r), "p_naive": float(f"{p:.4g}"), "n": int(len(d))}


def leave_one_out(d: pd.DataFrame, stat, label: str) -> dict:
    full = stat(d)
    f = full["value"]
    out = {"full": {**full, "value": rnd(f)}}
    for col, name in (("campus", "campus_removed"), ("system", "system_removed")):
        rows = {}
        for u in sorted(d[col].unique(), key=lambda x: (len(x), x)):
            r = stat(d[d[col] != u])
            v = r["value"]
            sign = None if v is None else bool(np.sign(v) != np.sign(f) and v != 0 and f != 0)
            removes = (v is None) or (v <= 0 if f > 0 else v >= 0)
            rows[u] = {**r, "value": rnd(v), "n_pairs_removed": int((d[col] == u).sum()),
                       "changes_sign": sign, "removes_difference": bool(removes),
                       "below_half_of_full": (None if v is None else bool(abs(v) < 0.5 * abs(f)))}
        vals = [r["value"] for r in rows.values() if r["value"] is not None]
        out[name] = {"by_removal": rows,
                     "range": [rnd(min(vals)), rnd(max(vals))] if vals else None,
                     "removals_changing_sign": [u for u, r in rows.items() if r["changes_sign"]],
                     "removals_removing_difference": [u for u, r in rows.items() if r["removes_difference"]],
                     "removals_below_half_of_full": [u for u, r in rows.items() if r["below_half_of_full"]]}
    # the two systems that hold the not supported pairs short of two-fold on the held-out campus
    r = stat(d[~d.system.isin(TWO_SYSTEMS)])
    v = r["value"]
    out["stairs_and_conveying_removed"] = {
        **r, "value": rnd(v), "n_pairs_removed": int(d.system.isin(TWO_SYSTEMS).sum()),
        "changes_sign": None if v is None else bool(np.sign(v) != np.sign(f) and v != 0 and f != 0),
        "removes_difference": bool((v is None) or (v <= 0 if f > 0 else v >= 0))}
    out["label"] = label
    return out


def contrast_frames() -> dict:
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    e = fv[fv.heldout_evaluable].copy()
    held = pd.DataFrame({"campus": e.campus, "system": e.system, "group": e.certified.astype(bool),
                         "hold": e.heldout_lift >= TWO_FOLD, "lift": e.heldout_lift, "score": e.S})
    assert (len(held), int(held.group.sum()), int(held[held.group].hold.sum()),
            int(held[~held.group].hold.sum())) == (95, 77, 74, 14)
    assert round(spearmanr(held.score, held.lift)[0], 4) == 0.2628

    lh = pd.read_csv(C.TABLES / "local_history_baseline.csv", dtype={"campus": str})
    ev = lh[lh.evaluable].copy()
    assert ((ev.lift_eval_model >= TWO_FOLD) == ev.holds_model).all()
    fw = ev[ev.verdict_peer]
    fwd = pd.DataFrame({"campus": fw.campus, "system": fw.system, "group": fw.cert_peer.astype(bool),
                        "hold": fw.holds_model.astype(bool), "lift": fw.lift_eval_model, "score": fw.S_peer})
    assert (len(fwd), int(fwd.group.sum()), int(fwd[fwd.group].hold.sum()),
            int(fwd[~fwd.group].hold.sum())) == (88, 61, 54, 24)
    nov = ev[~ev.verdict_peer]
    fwd_all = pd.DataFrame({"campus": ev.campus, "system": ev.system,
                            "group": (ev.cert_peer & ev.verdict_peer).astype(bool),
                            "hold": ev.holds_model.astype(bool), "lift": ev.lift_eval_model,
                            "score": ev.S_peer})

    dl = pd.read_csv(C.TABLES / "direct_local_validation.csv", dtype={"campus": str})
    cs = dl[dl.evaluable_2017_2019 & ~dl.all_evaluable.astype(bool)]
    cold = pd.DataFrame({"campus": cs.campus, "system": cs.system, "group": cs.supported_peer.astype(bool),
                         "hold": cs.holds_2017_2019.astype(bool), "lift": cs.lift_2017_2019,
                         "score": cs.S_peer, "verdict": cs.verdict_peer.astype(bool)})
    assert (len(cold), int(cold.group.sum()), int(cold[cold.group].hold.sum()),
            int(cold[~cold.group].hold.sum())) == (31, 22, 15, 7)
    return {"held": held, "fwd": fwd, "fwd_all": fwd_all, "cold": cold,
            "no_verdict": [{"campus": r.campus, "system": r.system, "holds": bool(r.holds_model),
                            "lift": float(r.lift_eval_model)} for r in nov.itertuples()]}


# --------------------------------------------------------------------------
# (2) difference of two rank correlations under clustering
# --------------------------------------------------------------------------
def rho_difference(seed_key: int = 74) -> dict:
    dl = pd.read_csv(C.TABLES / "direct_local_validation.csv", dtype={"campus": str})
    d = dl[dl.evaluable_2017_2019 & dl.all_lift.notna() & dl.S_peer.notna()].reset_index(drop=True)
    stored = json.load(open(C.METRICS / "direct_local_validation.json"))["by_window"]["all"][
        "rank_correlation_with_2017_2019_lift"]["same_pairs_direct_local_vs_peer"]
    a, b, y = d.all_lift.to_numpy(float), d.S_peer.to_numpy(float), d.lift_2017_2019.to_numpy(float)
    n = len(d)
    ones = np.ones(n, dtype=int)
    ra, rb = cu.wspearman(ones, a, y), cu.wspearman(ones, b, y)
    assert n == stored["n"] == 59
    assert round(ra, 3) == stored["rho_direct_local_lift"] and round(rb, 3) == stored["rho_peer_evidence_score"]
    camps, ci = np.unique(d.campus.to_numpy(), return_inverse=True)
    syss, si = np.unique(d.system.to_numpy(), return_inverse=True)
    out = {"n_pairs": n, "n_campuses": int(len(camps)), "n_building_systems": int(len(syss)),
           "campuses": [str(c) for c in camps], "building_systems": [str(s) for s in syss],
           "rho_direct_local_lift": rnd(ra), "rho_peer_evidence_score": rnd(rb),
           "difference": rnd(ra - rb),
           "stored_pair_level_interval_script_68": {"ci95": stored["difference_ci_95"],
                                                    "n_resamples": 2000}}
    for b_i, scheme in enumerate(SCHEMES):
        rng = np.random.default_rng([SEED, seed_key, b_i])
        W = cu.weights(scheme, ci, si, len(camps), len(syss), rng, N_BOOT)
        va = np.array([cu.wspearman(W[r], a, y) for r in range(N_BOOT)])
        vb = np.array([cu.wspearman(W[r], b, y) for r in range(N_BOOT)])
        out[f"{scheme}_bootstrap"] = {"difference": cu.interval(va - vb),
                                      "rho_direct_local_lift": cu.interval(va),
                                      "rho_peer_evidence_score": cu.interval(vb),
                                      "mean_pairs_per_replicate": rnd(W.sum(axis=1).mean(), 1)}
    # leave one campus or one building system out
    loo = {}
    for col in ("campus", "system"):
        rows = {}
        for u in sorted(d[col].unique(), key=lambda x: (len(x), x)):
            m = (d[col] != u).to_numpy()
            if m.sum() < 3:
                continue
            w = m.astype(int)
            x1, x2 = cu.wspearman(w, a, y), cu.wspearman(w, b, y)
            rows[u] = {"n_pairs": int(m.sum()), "rho_direct_local_lift": rnd(x1),
                       "rho_peer_evidence_score": rnd(x2), "difference": rnd(x1 - x2)}
        vals = [r["difference"] for r in rows.values() if r["difference"] is not None]
        loo[f"{col}_removed"] = {"by_removal": rows, "range": [min(vals), max(vals)],
                                 "removals_at_or_below_zero": [u for u, r in rows.items()
                                                               if r["difference"] is not None and r["difference"] <= 0]}
    out["leave_one_out"] = loo
    return out


# --------------------------------------------------------------------------
# (3) labels, thresholds and eligibility in each construction
# --------------------------------------------------------------------------
def check_labels(panel: pd.DataFrame, campuses: list[str], single: pd.DataFrame,
                 pairs: dict) -> dict:
    """Recompute the stored labels of every prediction file from the panel with
    the 75th-percentile thresholds fitted on the stated campuses, and check that
    the count-rule arm carries them unchanged."""
    univ = panel[C.COL_UNIV].astype(str).to_numpy()
    year = (panel["period_q"] // 4).to_numpy()
    bld = np.char.add(np.char.add(univ.astype(str), "/"), panel[C.COL_BUILDING].astype(str).to_numpy().astype(str))
    system = panel[C.COL_SYSTEM].astype(str).to_numpy()
    q = panel["period_q"].to_numpy()

    def compare(stored: pd.DataFrame, test_mask: np.ndarray, train_mask: np.ndarray) -> int:
        """The prediction files keep the panel order of the scored rows (asserted
        when they were written), so the rows are compared position by position."""
        thr, gthr = FE.severity_thresholds(panel[train_mask], target="labour", pctl=PCTL)
        y = FE.apply_severity(panel[test_mask], thr, gthr, "labour")
        assert len(stored) == int(test_mask.sum())
        assert (stored.university.astype(str).to_numpy() == univ[test_mask]).all()
        assert (stored.building.astype(str).to_numpy() == bld[test_mask]).all()
        assert (stored.system.astype(str).to_numpy() == system[test_mask]).all()
        assert (stored.period_q.to_numpy() == q[test_mask]).all()
        assert (stored.y.to_numpy() == y).all()
        return int(len(stored))

    n = {"pred_single": 0, "pred_pair": 0, "pred_single_pre2017": 0, "pred_pair_pre2017": 0}
    for c in campuses:
        st = pd.read_parquet(C.DATA_PROCESSED / "pred_single" / f"{c}.parquet")
        n["pred_single"] += compare(st, univ == c, univ != c)
        st = pd.read_parquet(C.DATA_PROCESSED / "pred_single_pre2017" / f"{c}.parquet")
        n["pred_single_pre2017"] += compare(st, univ == c, (univ != c) & (year <= 2016))
    n_pair_pre = 0
    for a, b in itertools.combinations(campuses, 2):
        st = pd.read_parquet(C.DATA_PROCESSED / "pred_pair" / f"{a}__{b}.parquet")
        hold = np.isin(univ, [a, b])
        n["pred_pair"] += compare(st, hold, ~hold)
        f = C.DATA_PROCESSED / "pred_pair_pre2017" / f"{a}__{b}.parquet"
        if f.exists():
            st = pd.read_parquet(f)
            n["pred_pair_pre2017"] += compare(st, hold & (year <= 2016), ~hold & (year <= 2016))
            n_pair_pre += 1
    rs, rp, _ = li.rule_arm(single, pairs)
    n_rule = 0
    for f, g in [(rs, single)] + [(rp[k], pairs[k]) for k in pairs]:
        li.assert_same_units(f, g, "count-rule arm")
        n_rule += int(len(f))
    return {"rows_checked": n, "n_pair_pre2017_files": n_pair_pre, "count_rule_rows_checked": n_rule,
            "statement": ("every stored label equals the label recomputed with the 75th-percentile thresholds "
                          "fitted on: C\\{c} for pred_single/c; C\\{a,b} for pred_pair/a__b; the same sets "
                          "restricted to rows anchored in 2016 or earlier for the forward-test fits; the "
                          "prior-year count rule of script 48 carries the labels of pred_single and pred_pair "
                          "on the same rows")}


def eligibility(frame: pd.DataFrame) -> dict:
    out = {}
    for s, g in frame.groupby("system", observed=True):
        per = {}
        for d, gg in g.groupby("university", observed=True):
            per[str(d)] = (int(gg.y.sum()), int(len(gg)))
        npos, n = int(g.y.sum()), int(len(g))
        nc = sum(1 for p_, r_ in per.values() if p_ >= sr.CTX_MIN_POS and r_ >= sr.CTX_MIN_ROWS)
        g1 = PASS if (nc >= 6 and npos >= 500) else (FAIL if (nc < 3 or npos < 100) else CAUTION)
        out[str(s)] = {"n_pos": npos, "n_units": n, "n_evaluable": nc,
                       "eligible": bool(npos >= ELIG_POS and n >= ELIG_ROWS),
                       "data_sufficiency": g1, "per_peer": per}
    return out


def construction_inputs(single, pairs, campuses) -> dict:
    nest = json.load(open(C.METRICS / "nesting_size_control.json"))
    partner = {tuple(k.split("->")): v for k, v in nest["deterministic_matched"]["partner"].items()}
    rs, rp, _ = li.rule_arm(single, pairs)
    arms = {}
    for c in campuses:
        arms.setdefault("single_nesting", {})[c] = eligibility(cc.dev_corpus(c, campuses, single, pairs, "single"))
        arms.setdefault("double_nesting", {})[c] = eligibility(cc.dev_corpus(c, campuses, single, pairs, "double"))
        fr = []
        for d in campuses:
            if d == c:
                continue
            pf = pairs[frozenset((d, partner[(c, d)]))]
            fr.append(pf[pf.university == d])
        arms.setdefault("matched_nearest_size", {})[c] = eligibility(pd.concat(fr, ignore_index=True))
        arms.setdefault("count_rule", {})[c] = eligibility(cc.dev_corpus(c, campuses, rs, rp, "double"))
    # every arm must reproduce its stored number of scored pairs
    n_scored = {a: sum(v["eligible"] for c in campuses for v in arms[a][c].values()) for a in arms}
    assert n_scored["double_nesting"] == 141 and n_scored["single_nesting"] == 142
    assert n_scored["matched_nearest_size"] == nest["constructions"]["matched_deterministic"]["n_stratum_folds"]
    assert n_scored["count_rule"] == json.load(open(C.METRICS / "learner_agreement.json"))["n_stratum_folds"]["rule"]
    # the double-nested data sufficiency equals the stored gate score
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    for r in fv.itertuples():
        e = arms["double_nesting"][r.campus][r.system]
        assert e["eligible"] and e["data_sufficiency"] == r.data_sufficiency and \
            e["n_evaluable"] == r.n_campuses and e["n_pos"] == r.n_pos, (r.campus, r.system)

    def label_set(arm: str, c: str, d: str) -> str:
        if arm == "single_nesting":
            return f"all campuses except {d}"
        if arm == "matched_nearest_size":
            return f"all campuses except {d} and {partner[(c, d)]}"
        return f"all campuses except {c} and {d}"      # double nesting and the count rule

    def summ(e):
        if not e:
            return None
        return {"n_pos": e["n_pos"], "n_units": e["n_units"], "n_evaluable": e["n_evaluable"],
                "eligible": e["eligible"], "data_sufficiency": LEVEL[e["data_sufficiency"]]}

    diffs = {}
    for arm in ("single_nesting", "matched_nearest_size", "count_rule"):
        rows = []
        for c in campuses:
            A, B = arms[arm][c], arms["double_nesting"][c]
            for s in sorted(set(A) | set(B)):
                a, b = A.get(s), B.get(s)
                ea, eb = bool(a and a["eligible"]), bool(b and b["eligible"])
                ds_diff = ea and eb and a["data_sufficiency"] != b["data_sufficiency"]
                if ea != eb or ds_diff:
                    peers = sorted(set((a or {}).get("per_peer", {})) | set((b or {}).get("per_peer", {})),
                                   key=lambda x: (len(x), x))
                    rows.append({
                        "target_campus": c, "system": s,
                        "kind": ("scored only here" if ea and not eb else
                                 "scored only under double nesting" if eb and not ea else
                                 "data sufficiency differs"),
                        "here": summ(a), "double_nesting": summ(b),
                        "per_peer_high_burden_quarters": {
                            d: {"here": (a or {}).get("per_peer", {}).get(d, (0, 0))[0],
                                "double_nesting": (b or {}).get("per_peer", {}).get(d, (0, 0))[0],
                                "units": (b or a)["per_peer"].get(d, (0, 0))[1],
                                "labels_here_fitted_on": label_set(arm, c, d),
                                "labels_double_nesting_fitted_on": label_set("double_nesting", c, d)}
                            for d in peers}})
        diffs[arm] = rows
    # the count rule reads the units and labels of double nesting
    assert not diffs["count_rule"], "the count-rule arm differs from double nesting in its labels"
    return {"n_scored_pairs": n_scored, "differences_from_double_nesting": diffs}


def thresholds_for(panel: pd.DataFrame, system: str, sets: dict) -> dict:
    """The 75th-percentile threshold of one building system fitted on each named
    set of campuses."""
    univ = panel[C.COL_UNIV].astype(str).to_numpy()
    out = {}
    for name, camps in sets.items():
        thr, gthr = FE.severity_thresholds(panel[np.isin(univ, camps)], target="labour", pctl=PCTL)
        out[name] = rnd(float(thr.get(system, gthr)), 3)
    return out


# --------------------------------------------------------------------------
# (4) prior-record trace
# --------------------------------------------------------------------------
def s_without(sc: dict, drop: str) -> float:
    """S with one gate left out: mean of the remaining scores, prerequisite rule unchanged."""
    keep = [g for g in GATES if g != drop]
    S = float(np.mean([sc[g] for g in keep]))
    nhf = sum(1 for g in HARD if sc[g] == FAIL)
    if nhf >= 2:
        return 0.0
    if nhf >= 1:
        return min(S, 0.50)
    return S


def ratio_gate(ratio: float | None, n_events: int) -> float:
    if n_events < 30 or ratio is None or not np.isfinite(ratio):
        return CAUTION
    return PASS if ratio < RATIO_PASS else (CAUTION if ratio < RATIO_FAIL else FAIL)


def trace_ratios(panel: pd.DataFrame, exclude: str | None, systems: list[str], camp_start: dict) -> dict:
    """No-antecedent share and its frequency-matched reference per building
    system, from the records of every campus except `exclude` (stratum_block of
    script 61, point estimates)."""
    p = panel if exclude is None else panel[panel[C.COL_UNIV].astype(str) != exclude]
    f = tf.trace_frame(p)
    tr = cc.trace_for(exclude)
    out = {}
    for s in systems:
        g = f[f.system == s]
        b = tf.stratum_block(g, np.random.default_rng(SEED), camp_start)
        t = tr[tr.system == s]
        assert b["n_events"] == int(t.n_events.sum()) and b["n_zero"] == int(t.n_zero.sum()), (exclude, s)
        out[s] = {"n_events": b["n_events"], "share": b["share"], "reference": b["reference_active"],
                  "ratio": b["ratio_active"]}
    return out


def trace_variants(panel, single, campuses) -> dict:
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    ge = pd.read_csv(C.TABLES / "gate_evidence_corrected.csv")
    ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]
    t61 = json.load(open(C.METRICS / "trace_frequency_null.json"))["strata"]
    camp_start = panel.groupby(panel[C.COL_UNIV].astype(str))["period_q"].min().to_dict()
    tf.N_BOOT = 2          # point estimates only; the bootstrap of script 61 is not needed here

    pooled_sys = sorted(ge.system)
    ratios = {"pooled": trace_ratios(panel, None, pooled_sys, camp_start)}
    for s, v in ratios["pooled"].items():
        if s in t61:
            assert v["share"] == t61[s]["share"] and v["ratio"] == t61[s]["ratio_active"], s
    for c in campuses:
        ratios[c] = trace_ratios(panel, c, sorted(fv[fv.campus == c].system), camp_start)

    variants = ("stored", "omitted", "trace_pass", "trace_caution", "trace_fail", "frequency_matched_ratio")

    def scores_for(sc: dict, rt: dict) -> dict:
        out = {"stored": cc.sufficiency(sc), "omitted": s_without(sc, "prior_record_trace")}
        for lab, val in (("trace_pass", PASS), ("trace_caution", CAUTION), ("trace_fail", FAIL)):
            out[lab] = cc.sufficiency({**sc, "prior_record_trace": val})
        out["frequency_matched_ratio"] = cc.sufficiency(
            {**sc, "prior_record_trace": ratio_gate(rt["ratio"], rt["n_events"])})
        return out

    res = {}
    held = {}
    for r in fv.itertuples():
        sc = {g: float(getattr(r, g)) for g in GATES}
        S = scores_for(sc, ratios[r.campus][r.system])
        assert abs(round(S["stored"], 2) - r.S) < 1e-9
        held[(r.campus, r.system)] = {"S": S, "trace": LEVEL[sc["prior_record_trace"]],
                                      "ratio": ratios[r.campus][r.system]["ratio"],
                                      "ratio_gate": LEVEL[ratio_gate(ratios[r.campus][r.system]["ratio"],
                                                                     ratios[r.campus][r.system]["n_events"])],
                                      "zero_prior_share": None if pd.isna(r.zero_prior_share) else float(r.zero_prior_share)}
    pool = {}
    for r in ge.itertuples():
        sc = {g: float(getattr(r, g)) for g in GATES}
        S = scores_for(sc, ratios["pooled"][r.system])
        assert abs(S["stored"] - ref["pooled"][r.system]["S"]) < 1e-9
        pool[r.system] = {"S": S, "trace": LEVEL[sc["prior_record_trace"]],
                          "ratio": ratios["pooled"][r.system]["ratio"],
                          "ratio_gate": LEVEL[ratio_gate(ratios["pooled"][r.system]["ratio"],
                                                         ratios["pooled"][r.system]["n_events"])]}

    def block(items: dict, names) -> dict:
        out = {}
        for v in variants:
            sup = {k: x["S"][v] >= S_CUT for k, x in items.items()}
            st = {k: x["S"]["stored"] >= S_CUT for k, x in items.items()}
            ch = [k for k in items if sup[k] != st[k]]
            out[v] = {"n_supported": int(sum(sup.values())),
                      "n_verdicts_changed": len(ch),
                      "gained_support": [names(k) for k in ch if sup[k]],
                      "lost_support": [names(k) for k in ch if not sup[k]]}
        dep = [k for k, x in items.items() if (x["S"]["trace_pass"] >= S_CUT) != (x["S"]["trace_fail"] >= S_CUT)]
        out["verdict_depends_on_trace_score"] = {
            "definition": "verdict with the trace gate at pass differs from the verdict with it at fail",
            "n": len(dep), "pairs": [names(k) for k in dep],
            "of_which_stored_trace": {lv: int(sum(items[k]["trace"] == lv for k in dep))
                                      for lv in ("pass", "caution", "fail")}}
        out["trace_gate_score_counts"] = {lv: int(sum(x["trace"] == lv for x in items.values()))
                                          for lv in ("pass", "caution", "fail")}
        out["ratio_gate_score_counts"] = {lv: int(sum(x["ratio_gate"] == lv for x in items.values()))
                                          for lv in ("pass", "caution", "fail")}
        return out

    res["held_out_maps"] = block(held, lambda k: f"{k[0]}/{k[1]}")
    res["pooled_map"] = block(pool, lambda k: k)
    res["held_out_maps"]["per_pair"] = {f"{c}/{s}": {"trace": x["trace"], "ratio": x["ratio"],
                                                     "ratio_gate": x["ratio_gate"],
                                                     "zero_prior_share": x["zero_prior_share"],
                                                     **{f"S_{v}": rnd(x["S"][v], 3) for v in variants}}
                                        for (c, s), x in held.items()
                                        if any((x["S"][v] >= S_CUT) != (x["S"]["stored"] >= S_CUT)
                                               for v in variants) or
                                        (x["S"]["trace_pass"] >= S_CUT) != (x["S"]["trace_fail"] >= S_CUT)}
    res["pooled_map"]["per_system"] = {s: {"trace": x["trace"], "ratio": x["ratio"], "ratio_gate": x["ratio_gate"],
                                           **{f"S_{v}": rnd(x["S"][v], 3) for v in variants}}
                                       for s, x in pool.items()}

    # route counts of every variant on the held-out maps
    routes = {}
    for v in variants:
        outer = {c: {s: {"S": held[(c, s)]["S"][v]} for (cc_, s) in held if cc_ == c} for c in campuses}
        r = cc.route(single, outer, campuses, RCOL)
        routes[v] = {"decision_tasks": r["route_counts"]["pma"], "verification_tasks": r["route_counts"]["verify"],
                     "standard_maintenance": r["route_counts"]["continue"], "watch_list": r["route_counts"]["watch"],
                     "share_high_risk_with_decision_task": r["share_high_risk_cleared"]}
    assert routes["stored"]["decision_tasks"] == ref["routing"]["route_counts"]["pma"]
    assert routes["stored"]["verification_tasks"] == ref["routing"]["route_counts"]["verify"]
    res["route_counts"] = routes
    res["ratios_pooled_map"] = ratios["pooled"]
    return res


# --------------------------------------------------------------------------
def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT_JSON), help="output file (default results/metrics/)")
    out_json = Path(ap.parse_args().out)
    t0 = time.time()
    assert not out_json.exists(), f"{out_json.name} exists; this script does not overwrite stored results"

    # ---------------- (1) ----------------
    fr = contrast_frames()
    loo = {
        "held_out_contrast": leave_one_out(fr["held"], share_difference,
                                           "share of supported minus share of not supported pairs at two-fold, held-out check"),
        "held_out_spearman": leave_one_out(fr["held"], spearman_value,
                                           "Spearman correlation of S with the held-out lift"),
        "forward_contrast": leave_one_out(fr["fwd"], share_difference,
                                          "forward test, pairs with a peer verdict"),
        "forward_contrast_no_verdict_as_not_supported": leave_one_out(
            fr["fwd_all"], share_difference, "forward test, pairs without a peer verdict counted as not supported"),
        "cold_start_contrast": leave_one_out(fr["cold"], share_difference,
                                             "pairs without a direct local evaluation; not supported includes no peer verdict"),
        "cold_start_contrast_with_peer_verdict_only": leave_one_out(
            fr["cold"][fr["cold"].verdict], share_difference,
            "pairs without a direct local evaluation, pairs without a peer verdict left out"),
    }
    loo["forward_pairs_without_peer_verdict"] = fr["no_verdict"]
    print(f"[74] leave-one-out done ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- (2) ----------------
    rho = rho_difference()
    print(f"[74] rank-correlation difference done ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- (3) ----------------
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    labels = check_labels(panel, campuses, single, pairs)
    print(f"[74] stored labels recomputed: {labels['rows_checked']}", flush=True)
    inputs = construction_inputs(single, pairs, campuses)
    others = [d for d in campuses if d != "11"]
    inputs["thresholds_G30_target_11"] = {
        "system": "G30",
        "p75_fitted_on": {"all nine campuses": thresholds_for(panel, "G30", {"x": campuses})["x"],
                          "all except 11 (peer campuses)": thresholds_for(panel, "G30", {"x": others})["x"]},
        "per_peer": {d: thresholds_for(panel, "G30", {
            "single_nesting_all_except_peer": [e for e in campuses if e != d],
            "double_nesting_all_except_11_and_peer": [e for e in campuses if e not in ("11", d)]})
            for d in others}}
    print(f"[74] construction inputs compared ({time.time() - t0:.0f} s)", flush=True)

    # ---------------- (4) ----------------
    trace = trace_variants(panel, single, campuses)
    print(f"[74] prior-record-trace variants done ({time.time() - t0:.0f} s)", flush=True)

    out = {
        "design": ("Leave-one-campus-out and leave-one-building-system-out values of the held-out, forward "
                   "and cold-start contrasts and of the Spearman correlation of S with held-out lift; "
                   "clustered intervals for the difference between the rank correlations of the direct "
                   "local lift and of the peer evidence score with the 2017 to 2019 lift; the labels, "
                   "thresholds and eligibility counts of each construction of the gating; the verdicts and "
                   "route counts with the prior-record-trace gate omitted, fixed, or scored on its "
                   "frequency-matched reference. No model is fitted."),
        "definitions": {
            "removal": "all campus-system pairs of the removed campus (as target) or of the removed building system leave the set; verdicts are not recomputed",
            "stairs_and_conveying_removed": "the pairs of stairs (C20) and conveying (D10) removed together",
            "changes_sign": "the value has the opposite sign of the full value",
            "removes_difference": "the value is at most zero for a positive full value (at least zero for a negative one), or undefined because a group is empty",
            "below_half_of_full": "the absolute value is less than half the absolute full value",
            "two_way_cluster": "campuses and building systems among the analysed pairs resampled independently; pair (c, k) enters a_c x b_k times (script 70)",
            "bootstrap": f"{N_BOOT} replicates per scheme, 95% percentile intervals, seeds [42, 74, scheme]",
            "omitted": "S as the mean of the other four gate scores, prerequisite rule and support level 2/3 unchanged",
            "trace_pass|caution|fail": "the trace gate score replaced by the stated level",
            "frequency_matched_ratio": (f"trace gate scored on the ratio of the no-antecedent share to its "
                                        f"frequency-matched reference (script 61), from the peers' records of the "
                                        f"map: pass below {RATIO_PASS}, caution below {RATIO_FAIL}, fail at or "
                                        f"above {RATIO_FAIL}, caution below 30 extreme events; levels set for "
                                        f"this analysis"),
            "verdict_depends_on_trace_score": "the verdict differs between the trace gate at pass and at fail; only these verdicts can change under any rescoring of that gate",
        },
        "leave_one_out": loo,
        "rank_correlation_difference": rho,
        "construction_inputs": {"label_check": labels, **inputs},
        "prior_record_trace": trace,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    json.dump(sr.jsonable(out), open(out_json, "w"), indent=2)

    for k, v in loo.items():
        if not isinstance(v, dict):
            continue
        print(f"\n=== {k}: full {v['full']['value']}")
        for col in ("campus_removed", "system_removed"):
            b = v[col]
            print(f"  {col}: range {b['range']}, sign changes {b['removals_changing_sign']}, "
                  f"removes {b['removals_removing_difference']}, below half {b['removals_below_half_of_full']}")
        print(f"  stairs and conveying removed: {v['stairs_and_conveying_removed']}")
    print(f"\n=== rho difference: {rho['difference']} " + " ".join(
        f"{s}: {rho[f'{s}_bootstrap']['difference']['ci95']}" for s in SCHEMES))
    print(f"\n=== scored pairs: {inputs['n_scored_pairs']}")
    for arm, rows in inputs["differences_from_double_nesting"].items():
        for r in rows:
            print(f"  {arm}: {r['target_campus']}/{r['system']} {r['kind']}: here {r['here']} double {r['double_nesting']}")
    print(f"  G30 thresholds: {inputs['thresholds_G30_target_11']}")
    for m in ("held_out_maps", "pooled_map"):
        print(f"\n=== trace {m}: " + "; ".join(f"{v}: {trace[m][v]['n_supported']} sup, {trace[m][v]['n_verdicts_changed']} changed"
                                            for v in ("stored", "omitted", "trace_pass", "trace_fail", "frequency_matched_ratio")))
        print(f"  depends: {trace[m]['verdict_depends_on_trace_score']}")
    print(f"  routes: {trace['route_counts']}")
    print(f"\n[74] done in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
