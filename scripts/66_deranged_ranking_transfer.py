#!/usr/bin/env python
"""
66_deranged_ranking_transfer.py -- what the gates and the count-based rule do,
on the real records, when the ranking of one building system carries no
information on a share of the peer campuses.

On the recorded data the ranking transfers wherever the data suffice, so the
count-based rule of 56_fold_verdicts_and_simple_rule.py (at least three
evaluable peer campuses, at least 100 high-burden quarters, no-antecedent
share below 0.25) reproduces most verdicts.  This script removes the transfer
on purpose and measures which verdicts change.

DERANGEMENT.  For a building system k that is supported on a map and evaluable
on at least six of its campuses, and for each k_d = 0, 1, ..., n (n the number
of evaluable campuses of k on that map), 50 draws are made.  Each draw picks
k_d of the n evaluable campuses at random and, on each picked campus, permutes
the risk scores of the units of system k within every quarter.  The scores
keep their distribution on that campus and in that quarter but no longer say
which unit has the high-burden quarter.  The within-campus-quarter percentile
R travels with its score, which is exact: the permutation leaves the set of
scores of every campus-quarter unchanged, so every score keeps its percentile
(checked once per map against a full recomputation of the ranks).  Units of
campuses that are not evaluable for k, and every other system, are left as
they are.  The high-burden labels, the counts and the prior-record trace are
not touched.

MAPS.  The pooled map reads all nine campuses through f^(-d), as in script 45.
The nine held-out maps read the eight peers of target campus c through
f^(-{c,d}) with the prior-record-trace table that excludes c.

RECOMPUTED.  The five gate scores, the evidence score S and the verdict, with
gates_for of 45_certification_corrected.py at the main setting (decision-time
percentile, transfer gate capped below six evaluable campuses), and the
count-based rule with simple_rule of script 56.  The count-based rule reads the
evaluable-campus count, the positive count and the no-antecedent share only,
so it cannot change under a permutation of scores; the runner asserts that it
does not.

INVARIANTS (the runner aborts on a mismatch).
  - The full pipeline reproduces the stored pooled map (cert_corrected.json,
    gate_evidence_corrected.csv) and the 141 verdicts of the held-out maps
    (fold_gate_verdicts.csv): gate scores, S, verdict and the gate quantities.
  - The gates computed on the rows of one system equal those of the full map.
  - k_d = 0 reproduces the stored gate scores, S and verdict of every selected
    system on every map.
  - The count-based rule gives the same verdict in every draw as without
    derangement.

A verdict is withheld when S < 2/3.  The gate that withholds it is the first
prerequisite gate scoring fail, in the order data sufficiency, transfer
stability, risk-gradient reliability; when no prerequisite fails, the verdict
is withheld because the mean of the five scores is below the support level.
Because the order puts transfer stability first, the share of draws in which
each gate scores fail is reported beside it.

Beside the two rules, the share of draws whose pooled top-decile lift stays at
or above 2 shows what a rule reading the pooled lift alone would accept.

Seeds.  Draw j of system k on map m with k_d deranged campuses uses
numpy.random.default_rng([RANDOM_SEED, m, k, k_d, j]); m is 0 for the pooled
map and 1 to 9 for the held-out maps of the campuses in sorted order, and k is
the position of the system among all systems in sorted order, so each draw is
fixed independently of the order in which draws are computed.

Usage:  python scripts/66_deranged_ranking_transfer.py [--draws 50] [--workers 3]

Outputs -> results/metrics/deranged_ranking_transfer.json
           results/tables/deranged_ranking_transfer.csv     (one row per draw)
           results/figures/fig_deranged_transfer.{pdf,png}
"""
from __future__ import annotations
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("FMSCREEN_DEVICE", "cpu")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "3")

import sys, json, time, argparse, importlib.util
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fmscreen import config as C
from fmscreen import metrics as MET
from fmscreen import figstyle
from fmscreen.figstyle import NAVY_D, CORAL_D, SLATE, INK, GRIDC


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cc = _load("cert_corrected", "45_certification_corrected.py")
sr = _load("fold_verdicts", "56_fold_verdicts_and_simple_rule.py")

GATES, HARD = cc.GATES, cc.HARD
FAIL, CAUTION, PASS = cc.FAIL, cc.CAUTION, cc.PASS
S_CUT = cc.S_CUT
RCOL = "R_quarter"
MIN_EVALUABLE = 6                      # systems evaluable on at least six campuses
CTX_MIN_POS, CTX_MIN_ROWS = 10, 100    # campus evaluability, as in gates_for
LEVEL = {FAIL: "fail", CAUTION: "caution", PASS: "pass"}
BINS = [(0.0, 0.25, "(0, 1/4]"), (0.25, 0.5, "(1/4, 1/2]"),
        (0.5, 0.75, "(1/2, 3/4]"), (0.75, 1.0, "(3/4, 1]")]
REASONS = ["data_sufficiency", "transfer_stability", "risk_gradient", "mean_below_support_level"]
GRID = np.linspace(0.0, 1.0, 201)       # common share axis for the figure

OUT_JSON = C.METRICS / "deranged_ranking_transfer.json"
OUT_CSV = C.TABLES / "deranged_ranking_transfer.csv"
OUT_FIG = C.FIGURES / "fig_deranged_transfer"


# --------------------------------------------------------------------------
# one building system on one map
# --------------------------------------------------------------------------
@dataclass
class Cell:
    map_kind: str            # "pooled" or "held_out"
    target: str              # "all" for the pooled map, else the target campus
    map_i: int
    system: str
    sys_i: int
    frame: pd.DataFrame      # rows of this system on this map, in map order
    score: np.ndarray
    R: np.ndarray
    y: np.ndarray
    univ: np.ndarray
    evaluable: list          # evaluable campuses, sorted
    rows: dict               # campus -> (row positions sorted by quarter, their quarters)
    trace: pd.DataFrame
    zp: float | None         # unrounded no-antecedent share
    count_rule0: bool
    base: dict               # unperturbed gate result


CELLS: dict[tuple, Cell] = {}           # shared with the forked workers


def build_cell(map_kind, target, map_i, system, sys_i, mapframe, trace, base) -> Cell:
    f = mapframe[mapframe.system == system][["university", "system", "period_q", "y",
                                             "score", RCOL]].copy()
    # the gates on the rows of one system equal the gates of the full map
    sub = cc.gates_for(f, trace, RCOL, True)
    assert set(sub) == {system} and sub[system] == base, \
        f"{map_kind}/{target}/{system}: gates on the system's rows differ from the full map"
    univ = f["university"].astype(str).to_numpy()
    y = f["y"].to_numpy().astype(int)
    q = f["period_q"].to_numpy()
    ev, rows = [], {}
    for d in sorted(np.unique(univ)):
        m = univ == d
        if y[m].sum() >= CTX_MIN_POS and m.sum() >= CTX_MIN_ROWS:
            ev.append(d)
            pos = np.flatnonzero(m)
            o = np.argsort(q[pos], kind="stable")
            rows[d] = (pos[o], q[pos][o])
    assert len(ev) == base["n_campuses"], f"{map_kind}/{target}/{system}: evaluable count"
    zp = sr.exact_no_antecedent(trace, system)
    rule0 = sr.simple_rule(base["n_campuses"], base["n_pos"], zp, **sr.PRESPECIFIED)
    return Cell(map_kind, target, map_i, system, sys_i, f, f["score"].to_numpy().copy(),
                f[RCOL].to_numpy().copy(), y, univ, ev, rows, trace, zp, rule0, base)


def deranged_scores(cell: Cell, picked: list, rng: np.random.Generator):
    """Scores and percentiles with those of the picked campuses permuted
    within each quarter."""
    score, R = cell.score.copy(), cell.R.copy()
    for d in picked:
        pos, qs = cell.rows[d]
        src = pos[np.lexsort((rng.random(len(pos)), qs))]   # same quarters, random order within each
        score[pos] = cell.score[src]
        R[pos] = cell.R[src]
    return score, R


def pick(cell: Cell, k: int, rng: np.random.Generator) -> list:
    if k == 0:
        return []
    idx = rng.choice(len(cell.evaluable), size=k, replace=False)
    return sorted(cell.evaluable[i] for i in idx)


def withheld_by(sc: dict, S: float) -> str | None:
    if S >= S_CUT:
        return None
    failed = [g for g in HARD if sc[g] == FAIL]
    return failed[0] if failed else "mean_below_support_level"


def draw(key: tuple, k: int, j: int) -> dict:
    cell = CELLS[key]
    rng = np.random.default_rng([C.RANDOM_SEED, cell.map_i, cell.sys_i, k, j])
    picked = pick(cell, k, rng)
    score, R = deranged_scores(cell, picked, rng)
    frame = cell.frame.assign(score=score, **{RCOL: R})
    v = cc.gates_for(frame, cell.trace, RCOL, True)[cell.system]
    sc, S = v["scores"], v["S"]
    rule = sr.simple_rule(v["n_campuses"], v["n_pos"], cell.zp, **sr.PRESPECIFIED)
    # the count-based rule reads counts and the no-antecedent share only
    assert rule == cell.count_rule0 and v["n_campuses"] == cell.base["n_campuses"] \
        and v["n_pos"] == cell.base["n_pos"], f"count-based rule changed: {key} k={k} draw={j}"
    own = {d: MET.lift_at_topk(cell.y[cell.univ == d], score[cell.univ == d], C.TOPK_BUDGET)
           for d in cell.evaluable}
    pk = set(picked)
    lifts_d = [own[d] for d in cell.evaluable if d in pk]
    lifts_i = [own[d] for d in cell.evaluable if d not in pk]
    pos_d = int(sum(cell.y[cell.univ == d].sum() for d in picked))
    n = len(cell.evaluable)
    return {
        "map": cell.map_kind, "target_campus": cell.target, "system": cell.system,
        "n_evaluable": n, "k_deranged": k, "share_deranged": round(k / n, 4), "draw": j,
        "deranged_campuses": ";".join(picked),
        "share_pos_on_deranged": round(pos_d / v["n_pos"], 4),
        "median_own_lift_deranged": round(float(np.median(lifts_d)), 3) if lifts_d else None,
        "median_own_lift_intact": round(float(np.median(lifts_i)), 3) if lifts_i else None,
        "pooled_lift": v["pooled_lift"], "frac_2x": v["frac_2x"],
        "top_band": v.get("top_band"), "rho": v.get("rho"),
        **{g: sc[g] for g in GATES},
        "S": round(float(S), 4), "supported": bool(S >= S_CUT),
        "withheld_by": withheld_by(sc, S) or "none",
        "count_rule_supported": bool(rule),
    }


def _limit_threads() -> None:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)


def _task(args):
    return draw(*args)


# --------------------------------------------------------------------------
# reference
# --------------------------------------------------------------------------
def _same(a, b) -> bool:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None or (isinstance(b, float) and np.isnan(b))
    if b is None or (isinstance(b, float) and np.isnan(b)):
        return False
    return abs(float(a) - float(b)) < 1e-9


def check_reference(outer: dict, pooled: dict, campuses: list) -> dict:
    ref = json.load(open(C.METRICS / "cert_corrected.json"))["corrected"]
    assert set(pooled) == set(ref["pooled"]), "pooled map strata differ"
    for s, v in pooled.items():
        r = ref["pooled"][s]
        assert v["scores"] == r["scores"] and abs(v["S"] - r["S"]) < 1e-9, f"pooled {s}"
        for q in ("n_campuses", "n_pos", "pooled_lift", "frac_2x", "zero_prior_share"):
            assert _same(v[q], r[q]), f"pooled {s} {q}: {v[q]} != {r[q]}"
    ge = pd.read_csv(C.TABLES / "gate_evidence_corrected.csv")
    for row in ge.itertuples():
        v = pooled[row.system]
        assert _same(v.get("top_band"), row.top_band_enrichment) and _same(v.get("rho"), row.band_spearman), \
            f"pooled {row.system} risk-gradient quantities"
    fv = pd.read_csv(C.TABLES / "fold_gate_verdicts.csv", dtype={"campus": str})
    assert len(fv) == sum(len(outer[c]) for c in campuses) == 141, "held-out verdict count"
    for row in fv.itertuples():
        v = outer[row.campus][row.system]
        for g in GATES:
            assert v["scores"][g] == getattr(row, g), f"held-out {row.campus}/{row.system} {g}"
        assert abs(round(v["S"], 2) - row.S) < 1e-9 and bool(v["S"] >= S_CUT) == bool(row.certified), \
            f"held-out {row.campus}/{row.system} S"
        for q in ("n_campuses", "n_pos", "pooled_lift", "frac_2x", "top_band", "rho", "zero_prior_share"):
            assert _same(v.get(q), getattr(row, q)), f"held-out {row.campus}/{row.system} {q}"
    n_sup = sum(v["S"] >= S_CUT for v in pooled.values())
    n_fold = int(fv.certified.sum())
    return {"pooled_map_supported": f"{n_sup}/{len(pooled)}",
            "held_out_verdicts": int(len(fv)), "held_out_supported": n_fold,
            "matches_cert_corrected_json": True, "matches_gate_evidence_corrected_csv": True,
            "matches_fold_gate_verdicts_csv": True}


def check_rank_transport(cell: Cell, mapframe: pd.DataFrame, k: int) -> None:
    """Permuting the percentile with its score equals recomputing the ranks of
    the whole map after the permutation."""
    rng = np.random.default_rng([C.RANDOM_SEED, cell.map_i, cell.sys_i, k, 0])
    picked = pick(cell, k, rng)
    score, R = deranged_scores(cell, picked, rng)
    full = mapframe.copy()
    m = (full.system == cell.system).to_numpy()
    sc = full["score"].to_numpy().copy(); sc[m] = score
    full["score"] = sc
    full = cc._add_ranks(full)
    assert np.array_equal(full.loc[m, RCOL].to_numpy(), R), "percentile does not travel with its score"
    a = cc.gates_for(full, cell.trace, RCOL, True)[cell.system]
    b = cc.gates_for(cell.frame.assign(score=score, **{RCOL: R}), cell.trace, RCOL, True)[cell.system]
    assert a == b, "gates differ between the full map and the system's rows after derangement"


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------
def _bin(share: float) -> str | None:
    for lo, hi, lab in BINS:
        if lo < share <= hi + 1e-12:
            return lab
    return None


def cell_summary(d: pd.DataFrame) -> dict:
    """Summary of a set of draws."""
    n = len(d)
    wh = ~d.supported
    out = {
        "n_draws": int(n),
        "share_withheld": round(float(wh.mean()), 4),
        "share_withheld_count_rule": round(float((~d.count_rule_supported).mean()), 4),
        "withheld_by": {r: round(float((d.withheld_by == r).mean()), 4) for r in REASONS},
        "share_fail": {g: round(float((d[g] == FAIL).mean()), 4) for g in
                       ("risk_concentration", "transfer_stability", "risk_gradient")},
        "share_transfer_stability_not_pass": round(float((d.transfer_stability < PASS).mean()), 4),
        "median_pooled_lift": round(float(d.pooled_lift.median()), 3),
        "share_pooled_lift_ge_2": round(float((d.pooled_lift >= 2.0).mean()), 4),
        "share_withheld_with_pooled_lift_ge_2": round(float((wh & (d.pooled_lift >= 2.0)).mean()), 4),
        "share_of_withheld_with_pooled_lift_ge_2": (round(float((d.pooled_lift[wh] >= 2.0).mean()), 4)
                                                    if wh.any() else None),
        "median_frac_2x": round(float(d.frac_2x.median()), 3),
        "median_top_band": round(float(d.top_band.median()), 3),
        "median_rho": round(float(d.rho.median()), 3),
        "mean_S": round(float(d.S.mean()), 4),
        "median_own_lift_deranged": (round(float(d.median_own_lift_deranged.median()), 3)
                                     if d.median_own_lift_deranged.notna().any() else None),
    }
    return out


def threshold_90(by_k: pd.DataFrame) -> dict:
    """Smallest share deranged at which support is withheld in at least 90 % of
    draws, and whether every larger share also reaches 90 %."""
    t = by_k.sort_values("share_deranged")
    hit = t[t.share_withheld >= 0.9]
    if hit.empty:
        return {"share": None, "k": None, "holds_for_every_larger_share": None}
    first = hit.iloc[0]
    later = t[t.share_deranged >= first.share_deranged]
    return {"share": round(float(first.share_deranged), 4), "k": int(first.k_deranged),
            "holds_for_every_larger_share": bool((later.share_withheld >= 0.9).all())}


def summarize(dd: pd.DataFrame, case_cols: list) -> dict:
    """dd: draws of one map kind.  A case is one building system on one map:
    a building system on the pooled map, a campus-system pair on the held-out
    maps.  case_cols: the columns that name a case."""
    cases = {}
    curves = []
    for key, g in dd.groupby(case_cols, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        name = "/".join(str(x) for x in key)
        rows = []
        per_k = {}
        for k, gk in g.groupby("k_deranged"):
            s = cell_summary(gk)
            per_k[str(int(k))] = {"share_deranged": float(gk.share_deranged.iloc[0]), **s}
            rows.append({"k_deranged": int(k), "share_deranged": float(gk.share_deranged.iloc[0]),
                         "share_withheld": s["share_withheld"],
                         "share_withheld_count_rule": s["share_withheld_count_rule"],
                         "share_pooled_lift_below_2": round(1.0 - s["share_pooled_lift_ge_2"], 4)})
        bk = pd.DataFrame(rows)
        per_bin = {}
        g2 = g[g.k_deranged > 0].assign(bin=lambda x: x.share_deranged.map(_bin))
        for lo, hi, lab in BINS:
            gb = g2[g2.bin == lab]
            if len(gb):
                per_bin[lab] = cell_summary(gb)
        cases[name] = {"n_evaluable": int(g.n_evaluable.iloc[0]), "by_k": per_k, "by_bin": per_bin,
                       "threshold_90": threshold_90(bk)}
        curves.append((name, bk))
    # pooled over cases, by bin (every draw counts once)
    allb = dd[dd.k_deranged > 0].assign(bin=lambda x: x.share_deranged.map(_bin))
    by_bin = {lab: cell_summary(allb[allb.bin == lab]) for _, _, lab in BINS
              if (allb.bin == lab).any()}
    full = cell_summary(dd[dd.k_deranged == dd.n_evaluable])
    # equal weight per case on a common share axis, linear between attainable shares
    G = np.vstack([np.interp(GRID, c.share_deranged, c.share_withheld) for _, c in curves])
    Gr = np.vstack([np.interp(GRID, c.share_deranged, c.share_withheld_count_rule) for _, c in curves])
    Gl = np.vstack([np.interp(GRID, c.share_deranged, c.share_pooled_lift_below_2) for _, c in curves])
    mean_curve = G.mean(axis=0)
    hit = np.flatnonzero(mean_curve >= 0.9)
    th = [u["threshold_90"]["share"] for u in cases.values()]
    th_ok = [x for x in th if x is not None]
    overall = {
        "n_cases": len(cases),
        "n_cases_reaching_90": len(th_ok),
        "threshold_90_min": min(th_ok) if th_ok else None,
        "threshold_90_median": round(float(np.median(th_ok)), 4) if th_ok else None,
        "threshold_90_max": max(th_ok) if th_ok else None,
        "threshold_90_mean_curve": round(float(GRID[hit[0]]), 4) if len(hit) else None,
        "all_cases_withheld_at_full_derangement": bool(all(
            c.loc[c.share_deranged == 1.0, "share_withheld"].iloc[0] >= 0.9 for _, c in curves)),
    }
    return {"cases": cases, "pooled_over_cases_by_bin": by_bin,
            "pooled_over_cases_full_derangement": full, "overall": overall,
            "_curves": {"mean": mean_curve, "min": G.min(axis=0), "max": G.max(axis=0),
                        "count_rule_mean": Gr.mean(axis=0), "count_rule_max": Gr.max(axis=0),
                        "lift_below_2_mean": Gl.mean(axis=0)}}


def by_system(dd: pd.DataFrame) -> dict:
    """Held-out maps grouped by system, all targets together."""
    out = {}
    for s, g in dd.groupby("system", sort=True):
        g2 = g[g.k_deranged > 0].assign(bin=lambda x: x.share_deranged.map(_bin))
        out[s] = {"n_targets": int(g.target_campus.nunique()),
                  "by_bin": {lab: cell_summary(g2[g2.bin == lab]) for _, _, lab in BINS
                             if (g2.bin == lab).any()}}
    return out


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
# figure
# --------------------------------------------------------------------------
def figure(panels: list) -> None:
    figstyle.apply()
    fig, axes = plt.subplots(1, len(panels), figsize=(3.4 * len(panels), 2.6), squeeze=False,
                             gridspec_kw={"wspace": 0.34})
    for ax, (title, cv) in zip(axes[0], panels):
        band = ax.fill_between(GRID, cv["min"], cv["max"], color="#d3e3f2", lw=0, zorder=1)
        mean, = ax.plot(GRID, cv["mean"], color=NAVY_D, lw=1.6, zorder=3)
        lift, = ax.plot(GRID, cv["lift_below_2_mean"], color=SLATE, lw=1.3, ls=(0, (1, 1.6)), zorder=3)
        rule, = ax.plot(GRID, cv["count_rule_mean"], color=CORAL_D, lw=1.4, ls=(0, (4, 2.5)), zorder=4)
        ax.set_xlim(0, 1); ax.set_ylim(-0.03, 1.03)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_xticklabels(["0", "1/4", "1/2", "3/4", "1"])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xlabel("Share of evaluable campuses with permuted scores", fontsize=9)
        ax.set_ylabel("Share of draws not supported", fontsize=9)
        if len(panels) > 1:
            ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold")
        ax.grid(False); ax.grid(axis="y", color=GRIDC, lw=0.5); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8.5)
    h = [mean, band, lift, rule]
    lab = ["Evidence gating, mean", "Evidence gating, range", "Pooled-lift rule (lift ≥ 2)",
           "Count-based rule"]
    leg = fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=len(h), fontsize=8,
                     frameon=True, edgecolor=INK, fancybox=False, framealpha=1.0,
                     columnspacing=1.4, handlelength=2.4)
    leg.get_frame().set_linewidth(0.5)
    fig.savefig(OUT_FIG.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT_FIG.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=50)
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    assert 1 <= args.workers <= 3
    for p in (OUT_JSON, OUT_CSV, OUT_FIG.with_suffix(".pdf"), OUT_FIG.with_suffix(".png")):
        assert not p.exists(), f"{p.name} exists; this script writes new files only"
    t0 = time.time()

    single, pairs = cc.load_single(), cc.load_pairs()
    campuses = sorted(single.university.unique())
    systems = sorted(single.system.unique())
    outer = {c: cc.gates_for(cc.dev_corpus(c, campuses, single, pairs, "double"),
                             cc.trace_for(c), RCOL, True) for c in campuses}
    pooled = cc.gates_for(single, cc.trace_for(None), RCOL, True)
    ref = check_reference(outer, pooled, campuses)
    print(f"[66] stored verdicts reproduced: {ref}", flush=True)

    # ---------------- the system-on-map cases ----------------
    maps = [("pooled", "all", single, cc.trace_for(None), pooled)]
    for c in campuses:
        maps.append(("held_out", c, cc.dev_corpus(c, campuses, single, pairs, "double"),
                     cc.trace_for(c), outer[c]))
    selection = {}
    for mi, (kind, tgt, frame, trace, res) in enumerate(maps):
        chosen = [s for s in sorted(res) if res[s]["S"] >= S_CUT and res[s]["n_campuses"] >= MIN_EVALUABLE]
        selection[tgt] = chosen
        for s in chosen:
            CELLS[(kind, tgt, s)] = build_cell(kind, tgt, mi, s, systems.index(s), frame, trace, res[s])
        # once per map: the percentile travels with its score
        if chosen:
            cell = CELLS[(kind, tgt, chosen[0])]
            check_rank_transport(cell, frame, max(1, len(cell.evaluable) // 2))
    n_cases = {"pooled": len(selection["all"]),
               "held_out": sum(len(selection[c]) for c in campuses)}
    print(f"[66] cases: {n_cases}; rank transport checked on all 10 maps ({time.time() - t0:.0f} s)",
          flush=True)

    # ---------------- draws ----------------
    tasks = []
    for key, cell in CELLS.items():
        tasks.append((key, 0, 0))                       # no derangement: one deterministic draw
        for k in range(1, len(cell.evaluable) + 1):
            tasks += [(key, k, j) for j in range(args.draws)]
    tb = time.time()
    recs = []
    ctx = mp.get_context("fork")
    with ctx.Pool(args.workers, initializer=_limit_threads) as pool:
        for i, r in enumerate(pool.imap_unordered(_task, tasks, chunksize=16), 1):
            recs.append(r)
            if i % max(1, len(tasks) // 10) == 0:
                print(f"[66] {i}/{len(tasks)} draws ({time.time() - tb:.0f} s)", flush=True)
    draw_secs = time.time() - tb
    dd = (pd.DataFrame(recs)
          .sort_values(["map", "target_campus", "system", "k_deranged", "draw"])
          .reset_index(drop=True))

    # ---------------- invariant: no derangement reproduces the stored verdicts ----------------
    z = dd[dd.k_deranged == 0]
    assert len(z) == len(CELLS)
    for r in z.itertuples():
        b = CELLS[(r.map, r.target_campus, r.system)].base
        for g in GATES:
            assert getattr(r, g) == b["scores"][g], f"k=0 {r.map}/{r.target_campus}/{r.system} {g}"
        assert abs(r.S - round(b["S"], 4)) < 1e-9 and r.supported, f"k=0 {r.map}/{r.target_campus}/{r.system} S"
        for q in ("pooled_lift", "frac_2x", "top_band", "rho"):
            assert _same(getattr(r, q), b.get(q)), f"k=0 {r.map}/{r.target_campus}/{r.system} {q}"
    assert (dd.groupby(["map", "target_campus", "system"]).count_rule_supported.nunique() == 1).all(), \
        "the count-based rule changed within a case"
    print(f"[66] k = 0 reproduces the stored gate scores and verdicts of all {len(z)} cases; "
          f"the count-based rule never changes ({draw_secs:.0f} s for {len(tasks)} draws)", flush=True)

    # ---------------- summaries ----------------
    pm = summarize(dd[dd["map"] == "pooled"], ["system"])
    ho = summarize(dd[dd["map"] == "held_out"], ["target_campus", "system"])
    ho_sys = by_system(dd[dd["map"] == "held_out"])
    curves = {"pooled": pm.pop("_curves"), "held_out": ho.pop("_curves")}

    burden = pd.read_csv(C.TABLES / "burden_by_system.csv")
    name = {k: str(v).strip() for k, v in burden[["SystemCode", "SystemDescription"]].values}
    unperturbed = {s: {"system_desc": name.get(s, s),
                       "n_evaluable": pooled[s]["n_campuses"], "n_pos": pooled[s]["n_pos"],
                       "pooled_lift": pooled[s]["pooled_lift"], "frac_2x": pooled[s]["frac_2x"],
                       "top_band": pooled[s].get("top_band"), "rho": pooled[s].get("rho"),
                       "scores": {g: LEVEL[pooled[s]["scores"][g]] for g in GATES},
                       "S": pooled[s]["S"], "count_rule_supported": CELLS[("pooled", "all", s)].count_rule0}
                   for s in selection["all"]}

    total = time.time() - t0
    out = {
        "design": ("For each building system supported on a map and evaluable on at least six of its "
                   "campuses, k of its n evaluable campuses (k = 0 to n, 50 draws per k >= 1, one draw "
                   "at k = 0) have the risk scores of the system's units permuted within every quarter; "
                   "the five gates, S and the verdict are recomputed with gates_for of "
                   "45_certification_corrected.py at the main setting, and the count-based rule with "
                   "simple_rule of 56_fold_verdicts_and_simple_rule.py.  Pooled map: nine campuses read "
                   "through f^(-d).  Held-out maps: the eight peers of each target campus read through "
                   "f^(-{c,d})."),
        "definitions": {
            "share_deranged": "k / n, the share of the system's evaluable campuses whose scores are permuted",
            "withheld": "S < 2/3, the verdict is not supported",
            "withheld_by": ("the first prerequisite gate scoring fail, in the order data sufficiency, "
                            "transfer stability, risk-gradient reliability; mean_below_support_level when "
                            "no prerequisite fails and the mean of the five scores is below 2/3"),
            "share_withheld_with_pooled_lift_ge_2": ("share of all draws in the cell that are not supported "
                                                     "while the pooled top-decile lift is at least 2"),
            "share_of_withheld_with_pooled_lift_ge_2": ("the same draws as a share of the draws that are not "
                                                        "supported"),
            "share_pos_on_deranged": "share of the system's high-burden quarters that lie on the permuted campuses",
            "median_own_lift_deranged": "median top-decile lift of the permuted campuses on their own units",
            "threshold_90": ("smallest share deranged at which support is withheld in at least 90 % of draws, "
                             "per system on a map; holds_for_every_larger_share tells whether all larger "
                             "shares also reach 90 %"),
            "threshold_90_mean_curve": ("the same level read from the mean over cases of the withheld share, "
                                        "each case linearly interpolated between its attainable shares k / n"),
            "case": ("one building system on one map: a building system on the pooled map, a campus-system "
                     "pair (target campus, building system) on the held-out maps"),
            "bins": "share deranged in (0, 1/4], (1/4, 1/2], (1/2, 3/4], (3/4, 1]; k = 0 is not in a bin",
            "pooled_over_cases_full_derangement": "the draws with every evaluable campus permuted (k = n)",
            "share_fail": "share of draws in which the gate scores fail, whatever the other gates score",
            "share_pooled_lift_ge_2": ("share of draws whose pooled top-decile lift is at least 2, the draws a "
                                       "rule reading the pooled lift alone would accept"),
            "figure": ("withheld share against share deranged; the line is the mean over cases, each case "
                       "interpolated linearly between its attainable shares k / n, the band the range over "
                       "cases; the dotted line is the share of draws with pooled lift below 2, the dashed "
                       "line the share the count-based rule does not support"),
        },
        "reference_check": ref,
        "checks": [
            "the gates on the rows of one system equal those of the full map, for every case",
            "permuting the percentile with its score equals recomputing the ranks of the whole map "
            "(checked once on each of the ten maps)",
            "k = 0 reproduces the stored gate scores, S, verdict, pooled lift, share at two-fold, "
            "top-band ratio and band correlation of every case",
            "the count-based rule gives its unperturbed verdict in every draw",
        ],
        "draws_per_k": args.draws,
        "seeds": "numpy.random.default_rng([RANDOM_SEED, map index, system index, k, draw]); RANDOM_SEED = 42",
        "cases_selected": {"pooled_map": selection["all"],
                           "held_out_maps": {c: selection[c] for c in campuses},
                           "n_cases": n_cases},
        "pooled_map": {"unperturbed": unperturbed, **pm},
        "held_out_maps": {**ho, "by_system": ho_sys},
        "runtime": {"total_seconds": round(total, 1), "draw_seconds": round(draw_secs, 1),
                    "n_draws": len(tasks), "workers": args.workers, "threads_per_worker": 1},
    }
    json.dump(jsonable(out), open(OUT_JSON, "w"), indent=2)
    dd.to_csv(OUT_CSV, index=False)
    figure([("(a) Pooled map", curves["pooled"]), ("(b) Held-out maps", curves["held_out"])])

    # ---------------- console ----------------
    for lab, blk in (("pooled map", pm), ("held-out maps", ho)):
        print(f"\n=== {lab}: pooled over cases, by share deranged ===")
        for b, s in blk["pooled_over_cases_by_bin"].items():
            print(f"  {b:11s} withheld {s['share_withheld']:.3f} (count rule {s['share_withheld_count_rule']:.3f}) "
                  f"by {s['withheld_by']} | median pooled lift {s['median_pooled_lift']} | "
                  f"withheld with lift>=2 {s['share_withheld_with_pooled_lift_ge_2']}")
        f = blk["pooled_over_cases_full_derangement"]
        print(f"  full      withheld {f['share_withheld']:.3f} | fails {f['share_fail']} | median pooled lift "
              f"{f['median_pooled_lift']} | pooled lift>=2 {f['share_pooled_lift_ge_2']}")
        print(f"  overall {blk['overall']}")
    print("\n=== pooled map: threshold for 90 % withheld, per system ===")
    for s, u in pm["cases"].items():
        print(f"  {s} n={u['n_evaluable']} {u['threshold_90']}")
    print(f"\n[66] done in {total:.0f} s")


if __name__ == "__main__":
    main()
