#!/usr/bin/env python
"""
60_local_history_baseline.py -- certification from a campus's own history
against certification from its peers.

A campus could validate the screen on its own past quarters instead of on the
other campuses.  This script measures what the peer pool adds over that local
alternative, with everything dated 2016 or earlier and the outcome taken in
2017 to 2019, the calendar of scripts/31_temporal_louo.py
(year = period_q // 4; history window year <= 2016; evaluation window
2017 <= year <= 2019).

FITS (CPU, the deployed configuration severity_labour|p75|M1|gbdt, through
fmscreen.engine.evaluate_split as in script 44; thresholds are fitted on the
training rows of each split):
  single  f_h^(-c)      trained on the other campuses' history-window rows,
                        scoring every row of c          (9 fits)
  pair    f_h^(-{c,d})  trained on the remaining campuses' history-window
                        rows, scoring the history-window rows of c and d
                                                        (36 fits)
Every training set is asserted to hold no row after 2016 and no row of a
held-out campus.  A split with nothing to score, or with no positives to learn
or to score, is skipped and recorded.

CERTIFICATES per (campus c, stratum k), all from history-window data:
  peer        script 45's gates_for on the peers' history rows, each peer d
              read through f_h^(-{c,d}), with the prior-record-trace table built
              from the peers' history rows (p90 threshold fitted on them).
  local       the same gates on c's own history rows scored by f_h^(-c), with
              calendar years in place of contexts: the frame is passed to
              gates_for with the year as its context column, so a year is
              evaluable with >= 10 positives and >= 100 rows, data sufficiency
              counts evaluable years (pass >= 6 years and >= 500 positives,
              fail < 3 years or < 100), and transfer stability is the share of
              evaluable years clearing two-fold, capped at caution below six.
              The risk gradient reads the within-campus-quarter percentile; the
              trace table holds c's own history events with the p90 threshold
              fitted on c's history rows (caution below 30 events).
  rule-peer, rule-local
              the same two constructions with the unfitted prior-year reactive
              work-order count (upm_cnt_w4) as the score, on the same rows and
              with the labels of the corresponding model arm.
A stratum with fewer than 20 positives or 200 rows receives no verdict, as in
script 45.  Certified means S >= 0.67.

OUTCOME: top-decile lift of f_h^(-c) within (c, k) over the evaluation window,
evaluable with >= 10 positives and >= 100 rows; the stratum holds when the lift
is at least 2.  The rule's own lift on the same rows is reported beside it.

Outputs -> data/processed/pred_single_pre2017/{c}.parquet
           data/processed/pred_pair_pre2017/{a}__{b}.parquet
           results/metrics/local_history_baseline.json
           results/tables/local_history_baseline.csv
"""
from __future__ import annotations
import os

# CPU only, with the thread caps set before numpy or fmscreen is imported
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["FMSCREEN_DEVICE"] = "cpu"
os.environ["FMSCREEN_THREADS"] = "4"
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "4"
N_JOBS = 2

import sys, json, time, itertools, importlib.util
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import engine as E
from fmscreen import metrics as MET

_spec = importlib.util.spec_from_file_location(
    "cert_corrected", ROOT / "scripts" / "45_certification_corrected.py")
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)

LAYER, MODEL, TARGET, PCTL = "M1", "gbdt", "severity_labour", 75
HIST_END = 2016
EVAL_YEARS = (2017, 2018, 2019)
MIN_POS, MIN_ROWS = 10, 100          # evaluability of a (campus, stratum) outcome
R_CUT, S_CUT = cc.R_CUT, cc.S_CUT
SINGLE_DIR = C.DATA_PROCESSED / "pred_single_pre2017"
PAIR_DIR = C.DATA_PROCESSED / "pred_pair_pre2017"
KEYS = ["university", "building", "system", "period_q"]
PRED_COLS = ["held_out", "n_held_out", "university", "building", "system",
             "period_q", "y", "score"]
ARMS = ("peer", "local", "rule_peer", "rule_local")


def _panel() -> pd.DataFrame:
    p = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    assert (p["period_q"] // 4 == p["year"]).all(), "calendar year does not match period_q"
    return p


# --------------------------------------------------------------------------
# fits
# --------------------------------------------------------------------------
def _fit(held_out: tuple[str, ...], scope: str) -> dict:
    """Train on the history-window rows of every campus outside `held_out`.
    scope='all' scores every row of the held-out campus; scope='history' scores
    the held-out campuses' history-window rows only."""
    t0 = time.time()
    panel = _panel()
    ucol = panel[C.COL_UNIV].astype(str).to_numpy()
    year = (panel["period_q"] // 4).to_numpy()
    hold = np.isin(ucol, list(held_out))
    train = panel[~hold & (year <= HIST_END)]
    test = panel[hold] if scope == "all" else panel[hold & (year <= HIST_END)]

    # the two guarantees the design rests on
    assert int((train["period_q"] // 4).max()) <= HIST_END, f"training row after {HIST_END}"
    assert not (set(train[C.COL_UNIV].astype(str)) & set(held_out)), "held-out campus in training"

    rec = {"held_out": "+".join(held_out), "scope": scope,
           "n_train": int(len(train)), "train_max_year": int((train["period_q"] // 4).max()),
           "train_campuses": sorted(set(train[C.COL_UNIV].astype(str))),
           "n_test": int(len(test)),
           "n_test_by_campus": {c: int((test[C.COL_UNIV].astype(str) == c).sum()) for c in held_out}}
    if len(test) == 0:
        rec["skipped"] = "no rows to score"
        return rec
    r = E.evaluate_split(train, test, LAYER, MODEL, TARGET, pctl=PCTL,
                         compute_ci=False, return_scores=True)
    if r is None:
        rec["skipped"] = "degenerate split (no positives to learn or to score)"
        return rec
    # assemble_Xy must preserve test row order before period_q is attached
    assert np.array_equal(np.asarray(r["_test_univ"]).astype(str),
                          test[C.COL_UNIV].astype(str).to_numpy()), \
        f"test row order not preserved for held_out={held_out}"
    df = pd.DataFrame({
        "held_out": "+".join(held_out),
        "n_held_out": len(held_out),
        "university": np.asarray(r["_test_univ"]).astype(str),
        "building": np.asarray(r["_test_building"]).astype(str),
        "system": np.asarray(r["_test_system"]).astype(str),
        "period_q": test["period_q"].to_numpy(),
        "y": np.asarray(r["_y"]).astype(np.int8),
        "score": np.asarray(r["_scores"], dtype=np.float32),
    })
    name = f"{held_out[0]}.parquet" if len(held_out) == 1 else f"{held_out[0]}__{held_out[1]}.parquet"
    df.to_parquet((SINGLE_DIR if scope == "all" else PAIR_DIR) / name, index=False)
    rec.update({"n_train_pos": int(r["n_train"] * r["train_base_rate"] + 0.5),
                "n_test_pos": int(df["y"].sum()), "file": name,
                "secs": round(time.time() - t0, 1)})
    return rec


# --------------------------------------------------------------------------
# prior-record-trace table on an arbitrary panel subset (the construction of
# script 45's prior_trace_table, asserted equal to trace_for on the full panel)
# --------------------------------------------------------------------------
def trace_table(p: pd.DataFrame) -> pd.DataFrame:
    p = p.copy()
    thr90 = p[p["upm_labour"] > 0].groupby(C.COL_SYSTEM, observed=True)["upm_labour"].quantile(0.90)
    p["p90"] = (p["upm_labour"] > p[C.COL_SYSTEM].map(thr90).astype(float).fillna(np.inf)).astype(int)
    p = p.sort_values(cc.CELL + ["period_q"]).reset_index(drop=True)
    g = p.groupby(cc.CELL, observed=True)["upm_count"]
    p["prior_total"] = g.shift(1).rolling(2, min_periods=2).sum() + \
                       g.shift(3).rolling(2, min_periods=2).sum()
    p["has_history"] = (p["period_q"] - 4 >= p["cell_first_q"]).astype(int)
    ev = p[(p["p90"] == 1) & (p["has_history"] == 1)].copy()
    ev["zero_prior"] = (ev["prior_total"] == 0).astype(int)
    t = (ev.groupby([C.COL_UNIV, C.COL_SYSTEM], observed=True)
           .agg(n_events=("zero_prior", "size"), n_zero=("zero_prior", "sum")).reset_index())
    t.columns = ["campus", "system", "n_events", "n_zero"]
    t["campus"] = t["campus"].astype(str)
    return t


def _same_trace(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    k = ["campus", "system"]
    a = a.assign(system=a.system.astype(str)).sort_values(k).reset_index(drop=True)
    b = b.assign(system=b.system.astype(str)).sort_values(k).reset_index(drop=True)
    return a[k + ["n_events", "n_zero"]].equals(b[k + ["n_events", "n_zero"]])


def check_reference(panel: pd.DataFrame) -> None:
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
    assert res["routing"]["route_counts"] == ref["routing"]["route_counts"]
    # the subset trace construction must equal script 45's on the full panel
    u = panel[C.COL_UNIV].astype(str)
    assert _same_trace(trace_table(panel), cc.trace_for(None)), "trace_table != trace_for(None)"
    for j in campuses:
        assert _same_trace(trace_table(panel[u != j]), cc.trace_for(j)), f"trace_table != trace_for({j})"


# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963984540054) -> list:
    if n == 0:
        return [None, None]
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def share_block(holds: pd.Series) -> dict:
    n, k = int(len(holds)), int(holds.sum())
    return {"n": n, "n_hold": k, "share_hold": round(k / n, 4) if n else None,
            "wilson_95": wilson(k, n)}


def gate_cols(prefix: str, g: dict | None) -> dict:
    out = {f"{prefix}_{x}": (None if g is None else g["scores"][x]) for x in cc.GATES}
    out[f"{prefix}_n_contexts"] = None if g is None else g["n_campuses"]
    out[f"{prefix}_n_pos"] = None if g is None else g["n_pos"]
    return out


def main() -> None:
    t_start = time.time()
    panel = _panel()
    panel["university"] = panel[C.COL_UNIV].astype(str)
    panel["building"] = panel["university"] + "/" + panel[C.COL_BUILDING].astype(str)
    panel["system"] = panel[C.COL_SYSTEM].astype(str)
    year = (panel["period_q"] // 4).to_numpy()
    univ = panel["university"].to_numpy()
    campuses = sorted(panel["university"].unique())
    check_reference(panel)
    print("[60] reference pipeline reproduced (141 fold certificates, 12/16 pooled, routes); "
          "trace construction equals trace_for on the full panel", flush=True)

    # ---- history available to each campus ----
    hist = panel[year <= HIST_END]
    history = {}
    for c in campuses:
        h = hist[hist.university == c]
        history[c] = {"quarters_before_2017": int(h.period_q.nunique()),
                      "rows_before_2017": int(len(h)),
                      "first_quarter": (None if h.empty else
                                        f"{int(h.period_q.min()) // 4}Q{int(h.period_q.min()) % 4 + 1}"),
                      "years_with_rows": sorted({int(x) for x in (h.period_q // 4).unique()})}

    # ---- 45 fits ----
    for d in (SINGLE_DIR, PAIR_DIR):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.parquet"):        # this script owns these directories
            f.unlink()
    pairs = list(itertools.combinations(campuses, 2))
    print(f"[60] fitting {len(campuses)} single + {len(pairs)} pair models on CPU "
          f"(n_jobs={N_JOBS}, threads={os.environ['FMSCREEN_THREADS']})", flush=True)
    t0 = time.time()
    fit_single = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_fit)((c,), "all") for c in campuses)
    fit_pair = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_fit)((a, b), "history") for a, b in pairs)
    fit_secs = round(time.time() - t0, 1)
    skipped = [r for r in fit_single + fit_pair if "skipped" in r]
    print(f"[60] fits done in {fit_secs} s; skipped: "
          f"{[(r['held_out'], r['skipped']) for r in skipped]}", flush=True)
    assert all(r["train_max_year"] <= HIST_END for r in fit_single + fit_pair)

    # ---- load predictions, attach the rule score ----
    rule = panel[KEYS + ["upm_cnt_w4"]]

    def attach(df):
        n = len(df)
        df = df.merge(rule, on=KEYS, how="left", validate="one_to_one")
        assert len(df) == n and df.upm_cnt_w4.notna().all()
        return df

    single = {}
    for c in campuses:
        f = SINGLE_DIR / f"{c}.parquet"
        assert f.exists(), f"single fit for campus {c} missing"
        single[c] = attach(pd.read_parquet(f))
    pair_pred = {}
    for a, b in pairs:
        f = PAIR_DIR / f"{a}__{b}.parquet"
        pair_pred[frozenset((a, b))] = (attach(pd.read_parquet(f)) if f.exists() else
                                        pd.DataFrame({k: pd.Series(dtype=t) for k, t in
                                                      [("held_out", str), ("n_held_out", int),
                                                       ("university", str), ("building", str),
                                                       ("system", str), ("period_q", int),
                                                       ("y", np.int8), ("score", np.float32),
                                                       ("upm_cnt_w4", float)]}))

    # ---- reproduction: the evaluation-window labels are those of the temporal transfer ----
    tl = {r["held_out_university"]: r for r in
          json.load(open(C.METRICS / "temporal_louo.json"))["per_campus"]}
    repro = []
    for c in campuses:
        e = single[c][(single[c].period_q // 4).isin(EVAL_YEARS)]
        lift = float(MET.lift_at_topk(e.y.to_numpy(), e.score.to_numpy(), C.TOPK_BUDGET))
        assert len(e) == tl[c]["n_test"] and int(e.y.sum()) == tl[c]["n_pos"], \
            (c, len(e), int(e.y.sum()), tl[c]["n_test"], tl[c]["n_pos"])
        repro.append({"campus": c, "n_test": int(len(e)), "n_pos": int(e.y.sum()),
                      "lift_top10_cpu": round(lift, 4),
                      "lift_top10_temporal_louo": round(tl[c]["lift_top10"], 4),
                      "delta": round(lift - tl[c]["lift_top10"], 4)})
    print("[60] evaluation-window rows and positives equal temporal_louo.json for every campus",
          flush=True)

    # ---- certificates ----
    def rule_frame(d):
        return cc._add_ranks(d.assign(score=d["upm_cnt_w4"].astype(np.float32)))

    def as_years(d):
        d = d.copy()
        d["university"] = (d["period_q"] // 4).astype(str)   # years take the place of contexts
        return d

    cert = {a: {} for a in ARMS}
    for c in campuses:
        tr_peer = trace_table(panel[(univ != c) & (year <= HIST_END)])
        tr_local = trace_table(panel[(univ == c) & (year <= HIST_END)])
        dev = cc.dev_corpus(c, campuses, None, pair_pred, "double")
        assert (dev.university != c).all() and int((dev.period_q // 4).max()) <= HIST_END
        cert["peer"][c] = cc.gates_for(dev, tr_peer, "R_quarter", True)
        cert["rule_peer"][c] = cc.gates_for(rule_frame(dev), tr_peer, "R_quarter", True)
        own = single[c][(single[c].period_q // 4) <= HIST_END]
        cert["local"][c] = cc.gates_for(as_years(cc._add_ranks(own)), tr_local,
                                        "R_quarter", True)
        cert["rule_local"][c] = cc.gates_for(as_years(rule_frame(own)), tr_local,
                                             "R_quarter", True)

    # ---- one row per campus-stratum pair ----
    names = (panel[["system", C.COL_SYSTEM_DESC]].astype(str).drop_duplicates("system")
             .set_index("system")[C.COL_SYSTEM_DESC].str.strip().to_dict())
    rows = []
    for c in campuses:
        s = single[c]
        yr = s.period_q // 4
        for k in sorted(s.system.unique()):
            e = s[(s.system == k) & yr.isin(EVAL_YEARS)]
            h = s[(s.system == k) & (yr <= HIST_END)]
            evaluable = bool(e.y.sum() >= MIN_POS and len(e) >= MIN_ROWS)
            lm = (float(MET.lift_at_topk(e.y.to_numpy(), e.score.to_numpy(), C.TOPK_BUDGET))
                  if evaluable else None)
            lr = (float(MET.lift_at_topk(e.y.to_numpy(), e.upm_cnt_w4.to_numpy(float), C.TOPK_BUDGET))
                  if evaluable else None)
            row = {"campus": c, "system": k, "system_desc": names.get(k, k),
                   "n_history_rows": int(len(h)), "n_history_pos": int(h.y.sum()),
                   "n_eval_rows": int(len(e)), "n_eval_pos": int(e.y.sum()),
                   "evaluable": evaluable,
                   "lift_eval_model": None if lm is None else round(lm, 3),
                   "holds_model": None if lm is None else bool(lm >= 2.0),
                   "lift_eval_rule": None if lr is None else round(lr, 3),
                   "holds_rule": None if lr is None else bool(lr >= 2.0)}
            for a in ARMS:
                g = cert[a][c].get(k)
                row[f"S_{a}"] = None if g is None else g["S"]
                row[f"verdict_{a}"] = g is not None
                row[f"cert_{a}"] = bool(g is not None and g["S"] >= S_CUT)
            row.update(gate_cols("peer", cert["peer"][c].get(k)))
            row.update(gate_cols("local", cert["local"][c].get(k)))
            rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(C.TABLES / "local_history_baseline.csv", index=False)

    # ---- summaries over evaluable pairs ----
    ev = tab[tab.evaluable].copy()

    def arm_summary(a: str, outcome: str) -> dict:
        h = ev[outcome].astype(bool)
        certd = ev[f"cert_{a}"]
        verd = ev[f"verdict_{a}"]
        return {"certified": share_block(h[certd]),
                "withheld_with_verdict": share_block(h[verd & ~certd]),
                "no_verdict": share_block(h[~verd])}

    def cross(a: str, b: str, outcome: str) -> dict:
        h = ev[outcome].astype(bool)
        out = {}
        for ca in (True, False):
            for cb in (True, False):
                m = (ev[f"cert_{a}"] == ca) & (ev[f"cert_{b}"] == cb)
                out[f"{a}_{'cert' if ca else 'not'}__{b}_{'cert' if cb else 'not'}"] = share_block(h[m])
        status = lambda x: np.where(ev[f"cert_{x}"], "certified",
                                    np.where(ev[f"verdict_{x}"], "withheld", "no_verdict"))
        sa, sb = status(a), status(b)
        three = {}
        for u in ("certified", "withheld", "no_verdict"):
            for v in ("certified", "withheld", "no_verdict"):
                m = (sa == u) & (sb == v)
                if m.any():
                    three[f"{a}_{u}__{b}_{v}"] = share_block(h[m])
        return {"two_by_two": out, "three_by_three": three}

    def _nn(x):
        return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)

    def only(a: str, b: str, outcome: str) -> dict:
        m = ev[f"cert_{a}"] & ~ev[f"cert_{b}"]
        sub = ev[m]
        return {**share_block(sub[outcome].astype(bool)),
                "pairs": [{"campus": r.campus, "system": r.system,
                           f"S_{a}": _nn(getattr(r, f"S_{a}")), f"S_{b}": _nn(getattr(r, f"S_{b}")),
                           "lift_eval_model": _nn(r.lift_eval_model),
                           "lift_eval_rule": _nn(r.lift_eval_rule)}
                          for r in sub.itertuples()]}

    summary = {
        "n_campus_stratum_pairs": int(len(tab)),
        "n_evaluable": int(len(ev)),
        "all_evaluable_hold": share_block(ev.holds_model.astype(bool)),
        "by_arm_outcome_model": {a: arm_summary(a, "holds_model") for a in ARMS},
        "rule_arms_outcome_rule": {a: arm_summary(a, "holds_rule") for a in ("rule_peer", "rule_local")},
        "all_evaluable_hold_rule": share_block(ev.holds_rule.astype(bool)),
        "cross_peer_local": cross("peer", "local", "holds_model"),
        "cross_rule_peer_rule_local": cross("rule_peer", "rule_local", "holds_model"),
        "peer_not_local": only("peer", "local", "holds_model"),
        "local_not_peer": only("local", "peer", "holds_model"),
        "rule_peer_not_rule_local": only("rule_peer", "rule_local", "holds_model"),
        "rule_local_not_rule_peer": only("rule_local", "rule_peer", "holds_model"),
    }

    per_campus = {}
    for c in campuses:
        t = tab[tab.campus == c]
        per_campus[c] = {**history[c],
                         **{f"n_strata_{a}_verdict": int(t[f"verdict_{a}"].sum()) for a in ARMS},
                         **{f"n_strata_{a}_certified": int(t[f"cert_{a}"].sum()) for a in ARMS},
                         "n_evaluable_pairs": int(t.evaluable.sum()),
                         "n_evaluable_hold": int(t.holds_model.fillna(False).astype(bool).sum())}

    # local data-sufficiency verdicts, which carry the history-length effect
    lv = tab[tab.verdict_local]
    ds_local = {str(k): int(v) for k, v in lv.local_data_sufficiency.value_counts().sort_index().items()}

    out = {
        "design": ("Certificates for each campus and stratum computed from data dated 2016 or "
                   "earlier, either from the peers' history (double nesting, script 45's gates) or "
                   "from the campus's own history with calendar years as contexts, with the screen "
                   "and with the prior-year count rule as the score; each is checked against the "
                   "top-decile lift the screen realizes on the campus in 2017 to 2019."),
        "definitions": {
            "calendar": "year = period_q // 4; history year <= 2016; evaluation 2017 <= year <= 2019",
            "history_labels": ("a history row at quarter t carries the label of quarter t+1, so the "
                               "last history labels describe 2017 Q1; evaluation labels start at "
                               "2017 Q2, so the two windows share no labelled quarter"),
            "local_contexts": "calendar years, passed to gates_for as the context column",
            "rule_score": "upm_cnt_w4 on the same rows; labels of the corresponding model arm",
            "realized": ("top-decile lift of f_h^(-c) within (c, k), 2017-2019, evaluable with "
                         ">= 10 positives and >= 100 rows; holds when lift >= 2"),
            "certified": f"S >= {S_CUT}; no verdict when the stratum has < 20 positives or < 200 rows",
            "wilson_95": "Wilson score interval for the share that holds",
        },
        "fits": {"config": f"{TARGET}|p{PCTL}|{LAYER}|{MODEL}", "device": "cpu",
                 "n_jobs": N_JOBS, "threads_per_job": int(os.environ["FMSCREEN_THREADS"]),
                 "wall_seconds": fit_secs,
                 "n_single_written": sum("file" in r for r in fit_single),
                 "n_pair_written": sum("file" in r for r in fit_pair),
                 "skipped": [{"held_out": r["held_out"], "scope": r["scope"],
                              "reason": r["skipped"], "n_test": r["n_test"]} for r in skipped],
                 "single": fit_single, "pair": fit_pair},
        "reproduction_vs_temporal_louo": {
            "note": ("rows and positives of each campus's evaluation window equal "
                     "temporal_louo.json exactly (asserted); lifts differ only by the training "
                     "device (CPU here)"),
            "max_abs_lift_delta": round(max(abs(r["delta"]) for r in repro), 4),
            "per_campus": repro},
        "per_campus": per_campus,
        "local_data_sufficiency_scores": ds_local,
        "summary": summary,
        "runtime_seconds": None,
    }
    out["runtime_seconds"] = round(time.time() - t_start, 1)
    json.dump(out, open(C.METRICS / "local_history_baseline.json", "w"), indent=2, default=str)

    # ---- console ----
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print("\nper campus:")
    print(pd.DataFrame(per_campus).T[["quarters_before_2017", "n_strata_peer_verdict",
                                      "n_strata_local_verdict", "n_strata_peer_certified",
                                      "n_strata_local_certified", "n_evaluable_pairs"]].to_string())
    print(f"\nevaluable pairs: {summary['n_evaluable']}, holding {summary['all_evaluable_hold']}")
    for a in ARMS:
        b = summary["by_arm_outcome_model"][a]
        print(f"  {a:10s} certified {b['certified']['n']:3d} hold {b['certified']['share_hold']} "
              f"{b['certified']['wilson_95']} | withheld {b['withheld_with_verdict']['n']:3d} hold "
              f"{b['withheld_with_verdict']['share_hold']} | no verdict {b['no_verdict']['n']:3d} "
              f"hold {b['no_verdict']['share_hold']}")
    print("\npeer x local (certified):", {k: (v["n"], v["share_hold"]) for k, v in
                                          summary["cross_peer_local"]["two_by_two"].items()})
    print(f"peer, not local: n={summary['peer_not_local']['n']} hold={summary['peer_not_local']['share_hold']} "
          f"{summary['peer_not_local']['wilson_95']}")
    print(f"[60] done in {out['runtime_seconds']} s")


if __name__ == "__main__":
    main()
