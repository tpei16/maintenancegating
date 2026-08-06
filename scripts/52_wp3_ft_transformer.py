#!/usr/bin/env python
"""
WP3/WP5 -- transformer arms on the aggregate representation, and the
context-adaptive transport head (plan C1) with its negative control (plan S20.8).

See notes/WP3_transformer_decisions.md for the pre-launch reasoning. The short
version: WP1 measured that event ORDER carries no value on this corpus, so the
transformer is placed over the FEATURE set (an FT-Transformer: tokenize each M1
feature, self-attention across the feature tokens, read a [CLS] token), not over
a fabricated event order.

Arms
  T0   gradient-boosted trees (incumbent)
  T1   FT-Transformer over the M1 feature set, global
  T2   T1 + context-adaptive FiLM transport on the feature tokens (plan C1)
  T3   incumbent base score + transformer transport head, context supplied
  T3n  identical head, context replaced by the pooled global descriptor
       (same architecture, same parameter count; only the information varies)
  T3p  identical head, campus->descriptor map deranged (negative control)

T3 - T3n isolates whether context adds anything at all; T3 - T3p is the control
plan S20.8 requires and this pipeline has never had. All arms are scored on the
same held-out campus with the same top-decile lift from fmscreen.metrics, and
the base scores the transport head consumes are always out-of-fold: for the
training campuses they come from an inner leave-one-campus-out loop, so the head
never sees a base score the base model fitted on.

Outputs -> results/wp3/base_scores_<u>.parquet   (cached, reused across arms)
           results/wp3/transformer_folds.csv
           results/wp3/transformer.json
           results/wp3/epoch_selection.json      (--select-epochs)
"""
from __future__ import annotations
import os
THREADS = int(os.environ.get("WP3_THREADS", "4"))
os.environ.setdefault("OMP_NUM_THREADS", str(THREADS))
os.environ.setdefault("MKL_NUM_THREADS", str(THREADS))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREADS))
os.environ.setdefault("FMSCREEN_THREADS", str(THREADS))

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
from fmscreen import models as M

torch.set_num_threads(THREADS)

OUT = C.ROOT / "results" / "wp3"; OUT.mkdir(parents=True, exist_ok=True)
SEED = C.RANDOM_SEED
TOPK = C.TOPK_BUDGET
PCTL = 75

# Context descriptor: campus operational scale, every entry past-observable and
# never an outcome, so the held-out campus's descriptor is computed from its own
# past inputs while the transport map is fitted on the training campuses only.
CTX_COLS = ["ppm_upm_ratio", "upm_cnt_cum", "ppm_cnt_cum", "upm_labour_cum",
            "ppm_labour_cum", "active_share_hist", "time_since_upm",
            "upm_labour_y1", "react_burden_trend"]


def _atomic(obj, path, csv=False):
    tmp = path.with_suffix(path.suffix + ".tmp")
    obj.to_csv(tmp, index=False) if csv else json.dump(obj, open(tmp, "w"), indent=2, default=str)
    os.replace(tmp, path)


# ------------------------------------------------------------- preprocessing
def prep_numeric(Xtr_num, Xte_num):
    """Impute (train median) + missingness indicators + robust clip + standardize.

    The one representational difference from the GBDT, which handles NaN
    natively. No information is dropped: missingness becomes its own feature."""
    tr = Xtr_num.replace([np.inf, -np.inf], np.nan).astype("float64")
    te = Xte_num.replace([np.inf, -np.inf], np.nan).astype("float64")
    miss_cols = [c for c in tr.columns if tr[c].isna().any() or te[c].isna().any()]
    med = tr.median()
    lo = tr.quantile(0.005); hi = tr.quantile(0.995)

    def apply(df):
        m = df[miss_cols].isna().astype("float64").add_suffix("__miss") if miss_cols else None
        d = df.fillna(med).clip(lower=lo, upper=hi, axis=1)
        return d, m

    trd, trm = apply(tr); ted, tem = apply(te)
    mu = trd.mean(); sd = trd.std().replace(0, 1.0)
    trd = (trd - mu) / sd; ted = (ted - mu) / sd
    if miss_cols:
        trd = pd.concat([trd, trm], axis=1); ted = pd.concat([ted, tem], axis=1)
    return trd.to_numpy("float32"), ted.to_numpy("float32"), list(trd.columns)


def campus_profiles(num_arr, univ, cols):
    """Per-campus mean over CTX_COLS on the standardized scale -> {campus: vec}."""
    idx = [cols.index(c) for c in CTX_COLS if c in cols]
    df = pd.DataFrame(num_arr[:, idx], columns=[cols[i] for i in idx])
    df["_u"] = univ
    prof = df.groupby("_u").mean()
    return {u: prof.loc[u].to_numpy("float32") for u in prof.index}, len(idx)


def subsample(y, frac, seed):
    """Keep every positive and a fraction of the negatives. Ranking is unaffected
    by the negative base rate; the loss weight is recomputed on what is kept."""
    if frac >= 1.0:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    keep = rng.choice(neg, size=int(len(neg) * frac), replace=False)
    sel = np.concatenate([pos, keep]); sel.sort()
    return sel


# --------------------------------------------------------------------- model
class FTTransformer(nn.Module):
    """Feature tokenizer + transformer encoder + [CLS] readout.

    Each scalar feature j becomes a token x_j * W_j + b_j, so self-attention
    runs across features rather than across time. context_adaptive adds the
    transport layer: the context descriptor produces a scale and a shift applied
    to every token before the encoder, which is how one shared rule is
    translated into a campus whose operating scale differs."""

    def __init__(self, n_num, n_sys, ctx_dim, d=32, heads=4, layers=2,
                 context_adaptive=False, dropout=0.1):
        super().__init__()
        self.n_num = n_num
        self.W = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.sys_emb = nn.Embedding(n_sys, d)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        enc = nn.TransformerEncoderLayer(d, heads, dim_feedforward=2 * d,
                                         dropout=dropout, batch_first=True,
                                         activation="gelu", norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.context_adaptive = context_adaptive
        # The transport map exists in every arm of the T3 family so the parameter
        # count is identical whether context is supplied, withheld or deranged;
        # only the tensor fed to it changes.
        self.film = nn.Sequential(nn.Linear(ctx_dim, d), nn.GELU(),
                                  nn.Linear(d, 2 * d))
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)

    def forward(self, xnum, xsys, xctx):
        B = xnum.shape[0]
        num_tok = xnum.unsqueeze(-1) * self.W + self.b        # (B, n_num, d)
        sys_tok = self.sys_emb(xsys).unsqueeze(1)             # (B, 1, d)
        cls = self.cls.expand(B, -1, -1)                      # (B, 1, d)
        tok = torch.cat([cls, num_tok, sys_tok], dim=1)       # (B, n_num+2, d)
        if self.context_adaptive:
            g, sh = self.film(xctx).chunk(2, dim=-1)
            tok = tok * (1.0 + g).unsqueeze(1) + sh.unsqueeze(1)
        h = self.enc(tok)
        return self.head(self.norm(h[:, 0])).squeeze(-1)


def train_net(net, num_tr, sys_tr, ctx_tr, ytr, epochs, seed, lr=1e-3,
              bs=2048, val=None, predict_sets=()):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1.0)))
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5)
    Xn = torch.from_numpy(num_tr); Xs = torch.from_numpy(sys_tr)
    Xc = torch.from_numpy(ctx_tr); Y = torch.from_numpy(ytr.astype("float32"))

    def predict(num, sys, ctx):
        net.eval(); out = np.zeros(len(num), "float64")
        with torch.no_grad():
            for b in range(0, len(num), 8192):
                s = slice(b, b + 8192)
                out[s] = net(torch.from_numpy(num[s]), torch.from_numpy(sys[s]),
                             torch.from_numpy(ctx[s])).numpy()
        return out

    curve = []
    for ep in range(epochs):
        net.train(); order = rng.permutation(len(Y))
        for b in range(0, len(order), bs):
            sel = order[b:b + bs]
            opt.zero_grad()
            loss = lossf(net(Xn[sel], Xs[sel], Xc[sel]), Y[sel])
            loss.backward(); opt.step()
        if val is not None:
            vn, vs, vc, vy = val
            curve.append(float(MET.lift_at_topk(vy, predict(vn, vs, vc), TOPK)))
            print(f"      epoch {ep+1}: val lift {curve[-1]:.2f}", flush=True)
    return [predict(*p) for p in predict_sets], curve


# ------------------------------------------------------------- fold assembly
def build_fold(panel, univ, u):
    """Feature matrices, system codes and campus descriptors for outer fold u."""
    te_m = univ == u; tr_m = ~te_m
    art = FE.assemble_Xy(panel[tr_m], panel[te_m], "M1", "severity_labour", PCTL)
    num_cols = art["num_cols"]
    ntr, nte, cols = prep_numeric(art["X_train"][num_cols], art["X_test"][num_cols])
    sys_tr = art["X_train"][C.COL_SYSTEM].cat.codes.to_numpy()
    sys_te = art["X_test"][C.COL_SYSTEM].cat.codes.to_numpy()
    n_sys = int(max(sys_tr.max(), sys_te.max())) + 1
    sys_tr = np.clip(sys_tr, 0, n_sys - 1).astype("int64")
    sys_te = np.clip(sys_te, 0, n_sys - 1).astype("int64")
    utr = panel.loc[tr_m, C.COL_UNIV].to_numpy()
    ute = panel.loc[te_m, C.COL_UNIV].to_numpy()
    prof_tr, cdim = campus_profiles(ntr, utr, cols)
    prof_te, _ = campus_profiles(nte, ute, cols)
    prof = dict(prof_tr); prof.update(prof_te)
    return dict(ntr=ntr, nte=nte, sys_tr=sys_tr, sys_te=sys_te, n_sys=n_sys,
                cols=cols, cdim=cdim, prof=prof, utr=utr, ute=ute,
                ytr=np.asarray(art["y_train"]), yte=np.asarray(art["y_test"]),
                base=float(np.mean(art["y_test"])))


def ctx_matrix(prof, univ_arr, mode="real", derange=None):
    """Row-wise context descriptor. 'real' uses each row's own campus; 'pooled'
    gives every row the mean descriptor (context withheld, architecture
    unchanged); 'derange' applies a fixed campus->campus permutation."""
    if mode == "pooled":
        mean = np.mean(np.stack(list(prof.values())), axis=0).astype("float32")
        return np.repeat(mean[None, :], len(univ_arr), axis=0)
    src = prof if mode == "real" else {k: prof[derange[k]] for k in prof}
    return np.stack([src[x] for x in univ_arr]).astype("float32")


def base_scores(panel, univ, u, cache=True):
    """Out-of-fold incumbent scores: an outer fit for the held-out campus, inner
    leave-one-campus-out fits for the training campuses. Rows are returned in
    train-then-test order, matching build_fold."""
    path = OUT / f"base_scores_{u}.parquet"
    if cache and path.exists():
        d = pd.read_parquet(path)
        return d["score"].to_numpy(), d["is_test"].to_numpy().astype(bool)
    t0 = time.time()
    te_m = univ == u
    order = np.concatenate([np.flatnonzero(~te_m), np.flatnonzero(te_m)])
    sc = np.zeros(len(panel), "float64")
    others = sorted(pd.unique(univ[~te_m]))
    for v in others:  # inner LOUO: fit without {u, v}, score v
        fit_m = (~te_m) & (univ != v)
        art = FE.assemble_Xy(panel[fit_m], panel[univ == v], "M1", "severity_labour", PCTL)
        mdl = M.MODEL_FACTORY["gbdt"](art["num_cols"], art["cat_cols"])
        sc[univ == v] = M.fit_predict(mdl, art["X_train"], art["y_train"], art["X_test"])
    art = FE.assemble_Xy(panel[~te_m], panel[te_m], "M1", "severity_labour", PCTL)
    mdl = M.MODEL_FACTORY["gbdt"](art["num_cols"], art["cat_cols"])
    sc[te_m] = M.fit_predict(mdl, art["X_train"], art["y_train"], art["X_test"])
    sc = sc[order]; ist = te_m[order]
    if cache:
        pd.DataFrame({"score": sc, "is_test": ist}).to_parquet(path, index=False)
    print(f"    base scores for {u}: {len(others)+1} fits, {time.time()-t0:.0f}s", flush=True)
    return sc, ist


def transport_inputs(sc, univ_arr):
    """Tokens the transport head reads: the base score as a logit, and its
    percentile inside its own campus, which is the within-context quantity the
    routing rule spends its budget on. No labels enter either."""
    p = np.clip(sc, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).astype("float64")
    pct = np.zeros(len(sc), "float64")
    for uu in np.unique(univ_arr):
        m = univ_arr == uu
        pct[m] = pd.Series(sc[m]).rank(pct=True).to_numpy()
    z = (logit - logit.mean()) / (logit.std() + 1e-8)
    return np.stack([z, pct - 0.5], axis=1).astype("float32")


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--head-epochs", type=int, default=15)
    ap.add_argument("--folds", type=int, default=9)
    ap.add_argument("--arms", default="T0,T1,T2,T3,T3n,T3p")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--negfrac", type=float, default=0.25)
    ap.add_argument("--select-epochs", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    arms = a.arms.split(",")
    sfx = f"_{a.tag}" if a.tag else ""

    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    panel[C.COL_UNIV] = panel[C.COL_UNIV].astype(str)
    univ = panel[C.COL_UNIV].to_numpy()
    campuses = sorted(pd.unique(univ))
    assert len(campuses) == 9, f"expected 9 campuses, got {len(campuses)}"
    # fixed derangement of the campus->descriptor map for the negative control
    rot = {campuses[i]: campuses[(i + 4) % 9] for i in range(9)}
    print(f"[wp3] {len(panel):,} anchors, {len(campuses)} campuses, arms={arms}, "
          f"threads={THREADS}, negfrac={a.negfrac}", flush=True)

    # ---- epoch selection on source campuses only (plan S20.7) -------------
    if a.select_epochs:
        u, vc = campuses[0], campuses[1]
        f = build_fold(panel, univ, u)
        vm = f["utr"] == vc; fit = ~vm
        sel = subsample(f["ytr"][fit], a.negfrac, SEED)
        ctr = ctx_matrix(f["prof"], f["utr"], "real")
        print(f"[wp3] epoch selection: fit on 7 campuses, validate on {vc}; "
              f"test campus {u} untouched", flush=True)
        curves = {}
        for arm in [x for x in ("T1", "T2") if x in arms]:
            net = FTTransformer(f["ntr"].shape[1], f["n_sys"], f["cdim"],
                                context_adaptive=(arm == "T2"))
            print(f"  {arm} ({sum(p.numel() for p in net.parameters()):,} params)", flush=True)
            _, cv = train_net(net, f["ntr"][fit][sel], f["sys_tr"][fit][sel],
                              ctr[fit][sel], f["ytr"][fit][sel], a.select_epochs, SEED,
                              val=(f["ntr"][vm], f["sys_tr"][vm], ctr[vm], f["ytr"][vm]))
            curves[arm] = cv
        best = int(np.argmax(curves[list(curves)[0]])) + 1
        _atomic({"validation_campus": vc, "test_campus_untouched": u,
                 "curves": curves, "chosen_epochs": best}, OUT / f"epoch_selection{sfx}.json")
        print(f"[wp3] chosen epochs = {best}", flush=True)
        return

    # ---- full leave-one-campus-out --------------------------------------
    rows, t0 = [], time.time()
    for fi, u in enumerate(campuses[:a.folds]):
        f = build_fold(panel, univ, u)
        sc_tr = sc_te = None
        if any(x.startswith("T3") for x in arms) or "T0" in arms:
            sc, ist = base_scores(panel, univ, u)
            sc_tr, sc_te = sc[~ist], sc[ist]
            ntr_h = transport_inputs(sc_tr, f["utr"])
            nte_h = transport_inputs(sc_te, f["ute"])
        for arm in arms:
            ta = time.time()
            if arm == "T0":
                lifts = [MET.lift_at_topk(f["yte"], sc_te, TOPK)]; npar = -1
            elif arm in ("T1", "T2"):
                lifts, npar = [], -1
                ctr = ctx_matrix(f["prof"], f["utr"], "real")
                cte = ctx_matrix(f["prof"], f["ute"], "real")
                for s in range(a.seeds):
                    sel = subsample(f["ytr"], a.negfrac, SEED + s)
                    net = FTTransformer(f["ntr"].shape[1], f["n_sys"], f["cdim"],
                                        context_adaptive=(arm == "T2"))
                    npar = sum(p.numel() for p in net.parameters())
                    (pred,), _ = train_net(net, f["ntr"][sel], f["sys_tr"][sel],
                                           ctr[sel], f["ytr"][sel], a.epochs, SEED + s,
                                           predict_sets=[(f["nte"], f["sys_te"], cte)])
                    lifts.append(MET.lift_at_topk(f["yte"], pred, TOPK))
            else:
                mode = {"T3": "real", "T3n": "pooled", "T3p": "derange"}[arm]
                ctr = ctx_matrix(f["prof"], f["utr"], mode, rot)
                cte = ctx_matrix(f["prof"], f["ute"], mode, rot)
                lifts, npar = [], -1
                for s in range(a.seeds):
                    net = FTTransformer(ntr_h.shape[1], f["n_sys"], f["cdim"],
                                        d=32, heads=4, layers=2, context_adaptive=True)
                    npar = sum(p.numel() for p in net.parameters())
                    (pred,), _ = train_net(net, ntr_h, f["sys_tr"], ctr, f["ytr"],
                                           a.head_epochs, SEED + s, bs=4096,
                                           predict_sets=[(nte_h, f["sys_te"], cte)])
                    lifts.append(MET.lift_at_topk(f["yte"], pred, TOPK))
            rows.append({"campus": u, "arm": arm, "lift_top10": float(np.median(lifts)),
                         "seed_sd": float(np.std(lifts)) if len(lifts) > 1 else 0.0,
                         "n_param": int(npar), "base_rate": f["base"],
                         "seconds": round(time.time() - ta, 1)})
            _atomic(pd.DataFrame(rows), OUT / f"transformer_folds{sfx}.csv", csv=True)
            print(f"  [{fi+1}/{a.folds}] {u:>3s} {arm:4s} lift={np.median(lifts):5.2f} "
                  f"({time.time()-ta:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    piv = df.pivot_table(index="campus", columns="arm", values="lift_top10")
    out = {"median": piv.median().round(3).to_dict(),
           "per_campus": piv.round(3).to_dict(),
           "epochs": a.epochs, "head_epochs": a.head_epochs, "seeds": a.seeds,
           "negfrac": a.negfrac,
           "n_param": df.groupby("arm")["n_param"].max().to_dict()}

    def paired(x, b):
        if x not in piv or b not in piv:
            return None
        d = (piv[x] - piv[b]).to_numpy(); rng = np.random.default_rng(SEED)
        bs = [np.median(rng.choice(d, len(d), replace=True)) for _ in range(4000)]
        return {"median_delta": float(np.median(d)),
                "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                "n_pos": int((d > 0).sum()), "n": int(len(d))}

    out["contrasts"] = {
        "T1_vs_incumbent": paired("T1", "T0"),
        "T2_vs_T1_film_transport": paired("T2", "T1"),
        "T3_vs_incumbent": paired("T3", "T0"),
        "T3_vs_T3n_context_value": paired("T3", "T3n"),
        "T3_vs_T3p_permutation_control": paired("T3", "T3p"),
        "T3p_vs_T3n_control_null": paired("T3p", "T3n"),
    }
    _atomic(out, OUT / f"transformer{sfx}.json")

    print("\n=== median top-decile lift ===")
    for k, v in out["median"].items():
        print(f"  {k:4s}: {v:.2f}")
    print("\n=== contrasts (paired per campus, bootstrap 95%) ===")
    for k, v in out["contrasts"].items():
        if v:
            print(f"  {k:32s} {v['median_delta']:+.3f}  "
                  f"CI [{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}]  {v['n_pos']}/{v['n']}")
    print(f"[wp3] {time.time()-t0:.0f}s -> {OUT/f'transformer{sfx}.json'}")


if __name__ == "__main__":
    main()
