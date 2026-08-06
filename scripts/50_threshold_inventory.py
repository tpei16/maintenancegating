#!/usr/bin/env python
"""
50_threshold_inventory.py -- every numerical constant in the pipeline, with its
provenance.

The point of the artifact is not the list but the classification.  Each constant
is tagged with where it came from, so a reader can see which numbers could have
been tuned to produce the reported result and which could not:

  capacity    fixed by operational or engineering requirement, not by the data
  literature  taken from prior work or standard practice
  pragmatic   chosen by the authors without optimisation against the outcome
  fitted      estimated from data, inside the training fold only
  recalibrate must be reset for a new portfolio

Outputs -> results/tables/threshold_inventory.csv
           manuscript-ready LaTeX rows on stdout
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fmscreen import config as C

# name, value, where it is used, provenance class, note
ROWS = [
    # ---- decision rule ----
    ("Risk cut $\\rho$", "0.90", "routing", "capacity",
     "the top decile is the inspection capacity a campus can absorb; set from workload, not from accuracy"),
    ("Certification cut $\\sigma$", "2/3", "routing", "pragmatic",
     "lies in an empty region of the pooled score distribution; consequential under the nested map, swept in Robustness"),
    ("Risk-concentration pass", "lift $\\ge 2.0$", "gate 2", "capacity",
     "the study's pre-specified screening criterion, recorded before any held-out result was read"),
    ("Risk-concentration caution", "lift $\\ge 1.5$", "gate 2", "pragmatic", "midpoint below the pass bar"),
    ("Transfer pass / caution", "0.80 / 0.60", "gate 3", "pragmatic",
     "share of contexts clearing the two-fold bar; swept one gate at a time"),
    ("Data-sufficiency pass", "$\\ge 6$ contexts, $\\ge 500$ events", "gate 1", "pragmatic",
     "also the floor below which the transfer share is capped at caution"),
    ("Data-sufficiency fail", "$<3$ contexts or $<100$ events", "gate 1", "pragmatic", "below this nothing else is estimable"),
    ("Risk-gradient pass", "top band $\\ge 2\\times$, $\\rho_s \\ge 0.9$", "gate 4", "pragmatic", "monotone bands over quintiles of $R$"),
    ("Prior-record pass / caution", "0.10 / 0.25", "gate 5", "pragmatic",
     "no-antecedent share; the only gate whose scale is set by the record rather than the model"),
    ("Hard-fail counts", "$h\\ge2 \\to 0$; $h=1 \\to \\min(\\tfrac12,\\bar g)$", "Eq. (5)", "pragmatic",
     "non-compensatory rule; the hard set is the minimal one catching all three uninterpretable failures"),
    # ---- target and panel ----
    ("High-burden percentile", "75th", "label", "pragmatic", "50th and 90th reported as sensitivity"),
    ("Extreme percentile", "90th", "trajectory analysis", "literature", "conventional tail definition"),
    ("Quarterly aggregation", "1 quarter", "panel", "capacity", "matches the planning period a portfolio actually uses"),
    ("Inclusion rule", "known-system", "panel", "pragmatic", "recent-activity rule reported as sensitivity"),
    # ---- features ----
    ("Trailing windows", "4 and 8 quarters", "features", "pragmatic",
     "right-aligned and ending at the anchor; no window was selected against held-out performance"),
    ("Recency cap", "20 quarters", "features", "pragmatic", "censoring point for time-since-last-event"),
    ("Event-sequence length", "64 events", "capacity escalation", "pragmatic", "covers the 95th percentile of history length"),
    # ---- evaluability filters ----
    ("Stratum eligibility", "$n\\ge200$, $n_{+}\\ge20$", "gating", "pragmatic", "below this a gate quantity is not computed"),
    ("Per-context evaluability", "$n\\ge100$, $n_{+}\\ge10$", "gate 3", "pragmatic", "a context below this does not contribute a lift"),
    ("Trajectory minimum", "$\\ge30$ p90 events", "gate 5", "pragmatic", "below this the gate scores caution"),
    # ---- fitted, never chosen ----
    ("System high-burden thresholds", "per system", "label", "fitted", "fitted inside the training fold of each split"),
    ("Premium features", "per system", "features", "fitted", "fitted inside the training fold of each split"),
    ("Boosting rounds / depth", "500 / 6", "screen", "literature", "standard conservative defaults, fixed a priori, not tuned per fold"),
    ("Bootstrap resamples", "400", "confidence intervals", "literature", "clustered by building within a held-out campus"),
    # ---- synthetic generator ----
    ("Contexts per stratum", "8 (2 when scarce)", "benchmark", "pragmatic", "chosen to mirror the nine-campus panel"),
    ("Replicates", "50", "benchmark", "pragmatic", "gives 600 certification decisions"),
    ("Base seed", "fixed", "benchmark", "pragmatic", "reported so the benchmark is reproducible"),
]

CLASS_NOTE = {
    "capacity": "set by an operational or engineering requirement",
    "literature": "taken from prior work or standard practice",
    "pragmatic": "chosen by the authors without optimisation against the reported outcome",
    "fitted": "estimated from data, inside the training fold only",
}


def main() -> None:
    df = pd.DataFrame(ROWS, columns=["constant", "value", "used_in", "provenance", "note"])
    df.to_csv(C.TABLES / "threshold_inventory.csv", index=False)

    counts = df.provenance.value_counts().to_dict()
    print("=== THRESHOLD INVENTORY ===")
    print(f"  {len(df)} constants")
    for k, v in counts.items():
        print(f"    {k:11s} {v:2d}   ({CLASS_NOTE.get(k, '')})")
    print(f"\n  none is fitted against a held-out result; the {counts.get('fitted', 0)} "
          f"fitted quantities are fitted inside the training fold of each split.")

    print("\n=== condensed LaTeX rows (decision rule only) ===")
    for r in df[df.used_in.str.contains("gate|routing|Eq")].itertuples():
        print(f"{r.constant} & {r.value} & {r.provenance} \\\\")


if __name__ == "__main__":
    main()
