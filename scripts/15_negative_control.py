#!/usr/bin/env python
"""
Negative control — label-permutation test (leakage sanity check).

If the pipeline is leakage-free, permuting the TRAIN labels (so features carry no
information about the shuffled target) must collapse held-out top-10% lift to ~1.
A lift that stays high under shuffling would betray a leakage path. We run the
primary config (severity p75, GBDT M1) on the temporal split with real vs
permuted training labels, repeated with several seeds.

Outputs -> results/metrics/negative_control.json
"""
from __future__ import annotations
import os
os.environ.setdefault("FMSCREEN_THREADS", "24")
import sys, json
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, validation as V, features as FE, models as M, metrics as MET


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    train, test = V.temporal_split(panel)
    art = FE.assemble_Xy(train, test, "M1", "severity_labour", 75)
    ytr, yte = art["y_train"], art["y_test"]

    # real
    model = M.make_gbdt(art["num_cols"], art["cat_cols"])
    s = M.fit_predict(model, art["X_train"], ytr, art["X_test"])
    real_lift = MET.lift_at_topk(yte, s, C.TOPK_BUDGET)

    # permuted train labels
    perm_lifts = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        yperm = rng.permutation(ytr)
        if yperm.sum() == 0:
            continue
        mp = M.make_gbdt(art["num_cols"], art["cat_cols"])
        sp = M.fit_predict(mp, art["X_train"], yperm, art["X_test"])
        perm_lifts.append(MET.lift_at_topk(yte, sp, C.TOPK_BUDGET))

    out = {"real_lift_top10": float(real_lift),
           "permuted_lift_top10_mean": float(np.mean(perm_lifts)),
           "permuted_lift_top10_max": float(np.max(perm_lifts)),
           "permuted_lifts": [float(x) for x in perm_lifts],
           "PASS_no_leakage": bool(np.mean(perm_lifts) < 1.3 and real_lift > 3)}
    json.dump(out, open(C.METRICS / "negative_control.json", "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"\n[neg-control] real lift {real_lift:.2f} vs permuted ~{np.mean(perm_lifts):.2f} "
          f"-> {'PASS (no leakage)' if out['PASS_no_leakage'] else 'INVESTIGATE'}")


if __name__ == "__main__":
    main()
