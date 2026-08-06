#!/usr/bin/env python
"""
WP1 (plan sec 20.1 items 6-9; sec 20.8 order/gap permutations) -- does chronology pay?

The feasibility gate. Everything is held identical to the incumbent aggregate
screen except the representation:

  A0  quarterly aggregate features, gradient-boosted        (the incumbent)
  A1  event sequence: type, subsystem, component, labour mark, elapsed time
  A2  A1 with event order shuffled inside each history      (order removed)
  A3  A1 with elapsed-time tokens removed                   (time removed)
  A4  bag of events, mean-pooled                            (order and time removed)

Same universities, same decision unit, same anchor rows, same target, same
leave-one-university-out folds, same top-decile lift metric, same leakage rules.
A1 vs A2 isolates order; A1 vs A3 isolates elapsed time; A1 vs A4 isolates
chronology as a whole; A4 vs A0 checks the encoder is not simply weaker than
the tree.

Gate rule, fixed before the run (see notes/WP1_feasibility_gate_decisions.md):
chronology counts as adding value only if A1 beats A4 by a paired per-campus
median of at least +0.25 top-decile lift with a bootstrap interval excluding
zero.

Outputs -> results/wp1/chronology_value.json
           results/wp1/chronology_folds.csv
"""
from __future__ import annotations
import os
THREADS = int(os.environ.get("WP1_THREADS", "8"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))

import sys, json, time, argparse
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C
from fmscreen import features as FE
from fmscreen import metrics as MET
from fmscreen import models as MD

torch.set_num_threads(THREADS)

OUT = C.ROOT / "results" / "wp1"
CELL = [C.COL_UNIV, C.COL_BUILDING, C.COL_SYSTEM]
MAXLEN = 64
SEED = C.RANDOM_SEED
TOPK = 0.10
PCTL = 75

GAP_EDGES = np.array([0.0, 1.0, 3.0, 7.0, 14.0, 30.0, 90.0, 180.0, 365.0])
LAB_EDGES = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 24.0, 80.0])


def _atomic(obj, path: Path, csv=False):
    tmp = path.with_suffix(path.suffix + ".tmp")
    if csv:
        obj.to_csv(tmp, index=False)
    else:
        json.dump(obj, open(tmp, "w"), indent=2, default=str)
    os.replace(tmp, path)


# --------------------------------------------------------------- event tensors
def build_event_index(panel: pd.DataFrame):
    """Map every panel anchor to the window of events available to it.

    Returns int32 gather indices (n_anchor, MAXLEN) into the event feature
    arrays, a validity mask, and the tokenised event features.
    """
    ev = pd.read_parquet(OUT / "event_sequences.parquet")
    ev[C.COL_UNIV] = ev[C.COL_UNIV].astype(str)
    ev = ev.sort_values(CELL + [C.COL_START, "event_ix"]).reset_index(drop=True)

    # ---- tokenise events ----------------------------------------------------
    f_type = np.where(ev["is_upm"].to_numpy(), 2, 1).astype(np.int16)
    sub = ev[C.COL_SUBSYS].astype("string").fillna("?")
    comp = ev[C.COL_COMPONENT].astype("string").fillna("?")
    f_sub = (pd.Categorical(sub).codes + 1).astype(np.int16)
    f_comp = (pd.Categorical(comp).codes + 1).astype(np.int16)
    lab = ev[C.COL_LABORHOURS].to_numpy(dtype="float64")
    f_lab = np.where(np.isnan(lab), 1, np.digitize(np.nan_to_num(lab), LAB_EDGES) + 2
                     ).astype(np.int16)
    gap = ev["gap_days"].to_numpy(dtype="float64")
    f_gap = np.where(np.isnan(gap), 1, np.digitize(np.nan_to_num(gap), GAP_EDGES) + 2
                     ).astype(np.int16)
    feats = dict(etype=f_type, sub=f_sub, comp=f_comp, lab=f_lab, gap=f_gap)
    sizes = dict(etype=3, sub=int(f_sub.max()) + 1, comp=int(f_comp.max()) + 1,
                 lab=int(f_lab.max()) + 1, gap=int(f_gap.max()) + 1)

    # ---- cell offsets in the globally sorted event array --------------------
    cellkey = pd.MultiIndex.from_frame(ev[CELL])
    first = ~cellkey.duplicated()
    cell_ids = pd.DataFrame({"k": cellkey[first]})
    cell_offset = pd.Series(np.flatnonzero(first), index=cellkey[first])

    # events per (cell, quarter) -> cumulative count within the cell
    cq = (ev.groupby(CELL + ["period_q"], observed=True).size()
            .rename("n").reset_index().sort_values(CELL + ["period_q"]))
    cq["cum"] = cq.groupby(CELL, observed=True)["n"].cumsum()

    anc = panel[CELL + ["period_q"]].copy()
    anc[C.COL_UNIV] = anc[C.COL_UNIV].astype(str)
    anc["_row"] = np.arange(len(anc))
    # forward-fill the cumulative count to quarters with no events
    merged = (pd.concat([cq[CELL + ["period_q", "cum"]],
                         anc[CELL + ["period_q"]].assign(cum=np.nan)])
                .sort_values(CELL + ["period_q"], kind="stable"))
    merged["cum"] = merged.groupby(CELL, observed=True)["cum"].ffill()
    merged = merged.drop_duplicates(subset=CELL + ["period_q"], keep="last")
    anc = anc.merge(merged, on=CELL + ["period_q"], how="left")
    anc = anc.sort_values("_row")
    assert len(anc) == len(panel), "anchor join changed the row count"

    cum = anc["cum"].fillna(0).to_numpy(dtype=np.int64)
    off = pd.MultiIndex.from_frame(anc[CELL]).map(cell_offset).to_numpy(dtype="float64")
    off = np.nan_to_num(off, nan=0.0).astype(np.int64)

    end = off + cum - 1                       # last usable event, global index
    steps = np.arange(MAXLEN - 1, -1, -1, dtype=np.int64)
    idx = end[:, None] - steps[None, :]       # (n, MAXLEN)
    valid = (idx >= off[:, None]) & (cum[:, None] > 0)
    idx = np.clip(idx, 0, len(ev) - 1).astype(np.int32)

    print(f"[wp1] event windows: {valid.sum():,} usable event slots, "
          f"{(valid.sum(1) == 0).mean():.1%} of anchors have no prior event")
    return idx, valid, feats, sizes


# --------------------------------------------------------------------- model
class EventGRU(nn.Module):
    def __init__(self, sizes, hidden=64, bag=False, use_gap=True):
        super().__init__()
        d = dict(etype=8, sub=16, comp=24, lab=8, gap=8)
        self.emb = nn.ModuleDict({k: nn.Embedding(sizes[k], d[k], padding_idx=0)
                                  for k in d})
        self.use_gap, self.bag = use_gap, bag
        din = sum(d.values())
        self.gru = None if bag else nn.GRU(din, hidden, batch_first=True)
        # the set encoder is given the event count explicitly: mean-pooling alone
        # would hide how much history there is, and history volume is precisely
        # what the aggregate panel carries. Without it A4 would lose to A1 for a
        # reason that has nothing to do with chronology.
        self.head = nn.Sequential(nn.Linear((din + 1) if bag else hidden, 32),
                                  nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, mask):
        # x: (B, L, 5) long in the order type, sub, comp, lab, gap
        parts = []
        for j, k in enumerate(["etype", "sub", "comp", "lab", "gap"]):
            e = self.emb[k](x[:, :, j])
            if k == "gap" and not self.use_gap:
                e = e * 0.0
            parts.append(e)
        h = torch.cat(parts, dim=-1) * mask.unsqueeze(-1)
        n = mask.sum(1, keepdim=True)
        if self.bag:
            pooled = torch.cat([h.sum(1) / n.clamp(min=1.0), torch.log1p(n)], -1)
            return self.head(pooled).squeeze(-1)
        out, _ = self.gru(h)
        last = mask.sum(1).long().clamp(min=1) - 1
        pooled = out[torch.arange(len(out)), last]
        return self.head(pooled).squeeze(-1)


def make_batch(rows, idx, valid, feats, shuffle_order, rng):
    ii = idx[rows]
    mm = valid[rows]
    if shuffle_order:
        # permute the valid slots only, leaving padding left-aligned: give each
        # valid slot a random key and the padding a key of -1, then sort.
        keys = rng.random(ii.shape)
        keys[~mm] = -1.0
        order = np.argsort(keys, axis=1, kind="stable")
        ii = np.take_along_axis(ii, order, axis=1)
    x = np.stack([feats[k][ii] for k in ["etype", "sub", "comp", "lab", "gap"]], -1)
    x = np.where(mm[:, :, None], x, 0)
    return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(mm.astype(np.float32))


def _score(net, rows, idx, valid, feats, shuffle, rng):
    net.eval(); out = np.zeros(len(rows), dtype=np.float64)
    with torch.no_grad():
        for b in range(0, len(rows), 2048):
            sel = slice(b, b + 2048)
            xb, mb = make_batch(rows[sel], idx, valid, feats, shuffle, rng)
            out[sel] = net(xb, mb).numpy()
    return out


def train_arm(arm, tr_rows, te_rows, y, idx, valid, feats, sizes, epochs, seed,
              val_rows=None, verbose=False):
    """Train one arm. If val_rows is given, report lift on it after every epoch
    so the epoch count can be chosen from source campuses only (plan sec 20.7);
    the held-out campus is never consulted for model selection."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = EventGRU(sizes, bag=(arm == "A4"), use_gap=(arm not in ("A3", "A4")))
    nparam = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    ytr = torch.from_numpy(y[tr_rows].astype(np.float32))
    shuffle = (arm in ("A2", "A4"))
    bs, curve = 512, []
    for ep in range(epochs):
        net.train()
        order = rng.permutation(len(tr_rows))
        for b in range(0, len(order), bs):
            sel = order[b:b + bs]
            xb, mb = make_batch(tr_rows[sel], idx, valid, feats, shuffle, rng)
            opt.zero_grad()
            loss = lossf(net(xb, mb), ytr[sel])
            loss.backward(); opt.step()
        if val_rows is not None:
            vl = MET.lift_at_topk(y[val_rows],
                                  _score(net, val_rows, idx, valid, feats, shuffle, rng),
                                  TOPK)
            curve.append(float(vl))
            if verbose:
                print(f"      {arm} epoch {ep+1}: source-validation lift {vl:.3f}",
                      flush=True)
    scores = _score(net, te_rows, idx, valid, feats, shuffle, rng)
    return scores, nparam, curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--folds", type=int, default=9)
    ap.add_argument("--arms", default="A0,A1,A2,A3,A4")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--select-epochs", type=int, default=0)
    a = ap.parse_args()
    arms = a.arms.split(",")

    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    univ = panel[C.COL_UNIV].astype(str).to_numpy()
    campuses = sorted(pd.unique(univ))
    assert len(campuses) == 9, f"expected the 9 panel campuses, got {len(campuses)}"
    print(f"[wp1] {len(panel):,} anchors, {len(campuses)} campuses, arms={arms}")

    idx, valid, feats, sizes = build_event_index(panel)

    # ---- epoch selection on source campuses only (plan sec 20.7) ------------
    if a.select_epochs:
        u = campuses[0]
        te = np.flatnonzero(univ == u); tr = np.flatnonzero(univ != u)
        ytr, yte, _ = FE.make_labels(panel.iloc[tr], panel.iloc[te],
                                     "severity_labour", PCTL)
        y = np.zeros(len(panel), dtype=int); y[tr] = ytr; y[te] = yte
        # hold out one source campus as validation; the test campus is untouched
        vcamp = campuses[1]
        va = np.flatnonzero(univ == vcamp)
        fit = np.setdiff1d(tr, va)
        print(f"[wp1] epoch selection: fit on 7 campuses, validate on campus "
              f"{vcamp}, test campus {u} never consulted")
        sel_curves = {}
        for arm in ["A1", "A4"]:
            _, _, curve = train_arm(arm, fit, te, y, idx, valid, feats, sizes,
                                    a.select_epochs, SEED, val_rows=va, verbose=True)
            sel_curves[arm] = curve
        best = int(np.argmax(sel_curves["A1"])) + 1
        _atomic({"validation_campus": vcamp, "curves": sel_curves,
                 "chosen_epochs": best,
                 "note": "chosen by source-campus validation lift; the held-out "
                         "campus was not used for model selection"},
                OUT / "epoch_selection.json")
        print(f"[wp1] chosen epochs = {best} (A1 source-validation argmax)")
        return

    rows, t_start = [], time.time()
    for fi, u in enumerate(campuses[:a.folds]):
        te = np.flatnonzero(univ == u); tr = np.flatnonzero(univ != u)
        # label: threshold fit on the training campuses only (incumbent rule)
        ytr, yte, _ = FE.make_labels(panel.iloc[tr], panel.iloc[te],
                                     "severity_labour", PCTL)
        y = np.zeros(len(panel), dtype=int); y[tr] = ytr; y[te] = yte
        base = yte.mean()

        for arm in arms:
            t0 = time.time()
            if arm == "A0":
                sc = MD.fit_predict_gbdt(panel.iloc[tr], panel.iloc[te], ytr,
                                         layer="M1") if hasattr(MD, "fit_predict_gbdt") else None
                if sc is None:
                    from fmscreen import runner as R
                    cfg = R.standard_configs(targets=[("severity_labour", PCTL)],
                                             layers=["M1"], models=["gbdt"])
                    r, pr = R.run_split(panel.iloc[tr], panel.iloc[te], cfg,
                                        regime="wp1", extra={"held_out_university": u},
                                        score_cfg_ids={f"severity_labour|p{PCTL}|M1|gbdt"},
                                        n_boot=0)
                    sc = pr["score"].to_numpy()
                npar = -1
                lifts = [MET.lift_at_topk(yte, sc, TOPK)]
            else:
                # plan sec 23.4 / sec 24: repeat the neural arms across seeds so
                # seed variation can be separated from campus variation
                lifts, npar = [], -1
                for s in range(a.seeds):
                    sc, npar, _ = train_arm(arm, tr, te, y, idx, valid, feats,
                                            sizes, a.epochs, SEED + s)
                    lifts.append(MET.lift_at_topk(yte, sc, TOPK))
            lift = float(np.median(lifts))
            rows.append({"campus": u, "arm": arm, "lift_top10": lift,
                         "lift_seeds": ";".join(f"{v:.4f}" for v in lifts),
                         "seed_sd": float(np.std(lifts)) if len(lifts) > 1 else 0.0,
                         "base_rate": float(base), "n_test": int(len(te)),
                         "n_param": int(npar), "seconds": round(time.time() - t0, 1)})
            print(f"  [{fi+1}/{a.folds}] campus {u:>3s}  {arm}  "
                  f"lift={lift:5.2f}  (n={a.seeds} seeds, {time.time()-t0:.0f}s)",
                  flush=True)

    df = pd.DataFrame(rows)
    _atomic(df, OUT / "chronology_folds.csv", csv=True)

    # ---- assert the sequence arms are the same size, so the comparison is fair
    npars = {r["arm"]: r["n_param"] for r in rows if r["n_param"] > 0}
    if all(k in npars for k in ("A1", "A2", "A3")):
        assert npars["A1"] == npars["A2"] == npars["A3"], \
            f"sequence arms differ in size: {npars}"

    piv = df.pivot_table(index="campus", columns="arm", values="lift_top10")
    out = {"per_campus": piv.round(3).to_dict(),
           "median": piv.median().round(3).to_dict(),
           "n_param": npars, "epochs": a.epochs, "maxlen": MAXLEN, "seed": SEED}

    def paired(x, b):
        if x not in piv or b not in piv:
            return None
        d = (piv[x] - piv[b]).to_numpy()
        rng = np.random.default_rng(SEED)
        bs = [np.median(rng.choice(d, len(d), replace=True)) for _ in range(4000)]
        return {"median_delta": float(np.median(d)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n_campuses_positive": int((d > 0).sum()), "n": int(len(d))}

    out["contrasts"] = {"A1_minus_A4_chronology": paired("A1", "A4"),
                        "A1_minus_A2_order": paired("A1", "A2"),
                        "A1_minus_A3_elapsed_time": paired("A1", "A3"),
                        "A4_minus_A0_encoder_check": paired("A4", "A0"),
                        "A1_minus_A0_vs_incumbent": paired("A1", "A0")}
    g = out["contrasts"]["A1_minus_A4_chronology"]
    out["gate"] = {
        "rule": "A1 - A4 paired median >= +0.25 lift and CI excludes zero",
        "passes": bool(g and g["median_delta"] >= 0.25 and g["ci95"][0] > 0) if g else None,
    }
    _atomic(out, OUT / "chronology_value.json")

    print("\n=== median top-decile lift across nine campuses ===")
    for k, v in out["median"].items():
        print(f"  {k}: {v:.2f}")
    print("\n=== paired contrasts (per-campus deltas) ===")
    for k, v in out["contrasts"].items():
        if v:
            print(f"  {k:32s} {v['median_delta']:+.3f}  "
                  f"CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]  "
                  f"{v['n_campuses_positive']}/{v['n']} campuses positive")
    print(f"\n=== GATE: {'PASS' if out['gate']['passes'] else 'FAIL'} "
          f"({out['gate']['rule']}) ===")
    print(f"[wp1] total {time.time()-t_start:.0f}s -> {OUT/'chronology_value.json'}")


if __name__ == "__main__":
    main()
