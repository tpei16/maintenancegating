#!/usr/bin/env python
"""
Phase 0b pilot — confirm a CRISP local-history
calibration curve emerges before the full multi-campus run. A mushy calibration
result is the main thing that would weaken the paper, so we de-risk it here.

Runs the PRIMARY analysis config (severity_labour, p75, known-system, M1) on two
well-covered held-out campuses, for both GPU-XGBoost and logistic regression.

Outputs -> results/pilot/*.csv, json
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C, validation as V, engine as E

OUT = C.RESULTS / "pilot"; OUT.mkdir(parents=True, exist_ok=True)


def main():
    panel = pd.read_parquet(C.DATA_PROCESSED / "panel_quarter.parquet")
    print(f"[pilot] panel {panel.shape}", flush=True)

    # (1) within-sample temporal sanity
    tr, te = V.temporal_split(panel)
    rows = []
    for model in ("gbdt", "logreg"):
        t0 = time.time()
        r = E.evaluate_split(tr, te, "M1", model, "severity_labour", pctl=75, n_boot=200)
        dt = time.time() - t0
        print(f"[temporal] {model:6s} M1 sev-p75: base={r['base_rate']:.3f} "
              f"lift@10={r['lift_top10']:.2f} [{r['lift_top10_ci_lo']:.2f},{r['lift_top10_ci_hi']:.2f}] "
              f"PR-AUC={r['pr_auc']:.3f} cap@10={r['capture_top10']:.2f} ({dt:.1f}s)", flush=True)
        rows.append({"regime": "temporal", "model": model, **{k: v for k, v in r.items() if not k.startswith("_")}})
    pd.DataFrame(rows).to_csv(OUT / "pilot_temporal.csv", index=False)

    # (2) local-history calibration curve on two campuses
    calib_rows = []
    for u in ("11", "5"):
        for model in ("gbdt", "logreg"):
            print(f"--- calibration campus={u} model={model} (severity p75, M1) ---", flush=True)
            for frac, trn, tst, naug in V.local_calibration(panel, u):
                t0 = time.time()
                r = E.evaluate_split(trn, tst, "M1", model, "severity_labour", pctl=75, compute_ci=False)
                if r is None:
                    print(f"  frac={frac:.2f} DEGENERATE", flush=True); continue
                dt = time.time() - t0
                print(f"  frac={frac:.2f} aug={naug:6d} test_n={r['n']:6d} base={r['base_rate']:.3f} "
                      f"lift@10={r['lift_top10']:.2f} cap@10={r['capture_top10']:.2f} PR-AUC={r['pr_auc']:.3f} ({dt:.1f}s)", flush=True)
                calib_rows.append({"campus": u, "model": model, "frac": frac, "aug_rows": naug,
                                   **{k: v for k, v in r.items() if not k.startswith("_")}})
    cc = pd.DataFrame(calib_rows)
    cc.to_csv(OUT / "pilot_calibration.csv", index=False)

    # crispness check: does lift@10 rise from frac=0 to frac=0.20 for GBDT?
    crisp = {}
    for u in ("11", "5"):
        sub = cc[(cc.campus == u) & (cc.model == "gbdt")].sort_values("frac")
        if len(sub) >= 2:
            crisp[u] = {"lift_at_0": float(sub.iloc[0]["lift_top10"]),
                        "lift_at_20": float(sub.iloc[-1]["lift_top10"]),
                        "delta": float(sub.iloc[-1]["lift_top10"] - sub.iloc[0]["lift_top10"]),
                        "monotone_nondec": bool(np.all(np.diff(sub["lift_top10"].to_numpy()) >= -0.05))}
    json.dump(crisp, open(OUT / "pilot_crispness.json", "w"), indent=2)
    print("[pilot] crispness:", json.dumps(crisp, indent=2), flush=True)
    print("[pilot] done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
