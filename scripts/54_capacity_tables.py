#!/usr/bin/env python
"""
Emit the LaTeX rows for the model-capacity results, read straight from the run
records so no number is transcribed by hand.

  results/wp1/chronology_value.json   representation escalation
  results/wp3/transformer.json        architecture and adaptation escalation

Prints (a) the per-campus supplement table and (b) the contrast lines used in
the main text, with the campuses-positive counts.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ARM_LABEL = {
    "T0": "Incumbent, boosted trees",
    "T1": "Feature-tokenizer transformer",
    "T3": "Context-adaptive transport head",
    "T3n": "\\quad context withheld (control)",
    "T3p": "\\quad context deranged (control)",
}
CONTRAST_LABEL = {
    "T1_vs_incumbent": ("Transformer $-$ incumbent", "base learner"),
    "T3_vs_incumbent": ("Transport head $-$ incumbent", "adaptation on top"),
    "T3_vs_T3n_context_value": ("Transport $-$ context withheld", "context information"),
    "T3_vs_T3p_permutation_control": ("Transport $-$ context deranged", "context identity"),
    "T3p_vs_T3n_control_null": ("Deranged $-$ withheld", "control null"),
}


def main():
    p = ROOT / "results" / "wp3" / "transformer.json"
    if not p.exists():
        print("results/wp3/transformer.json not present yet")
        return
    d = json.load(open(p))
    pc, med = d["per_campus"], d["median"]
    campuses = sorted(pc[next(iter(pc))].keys(), key=lambda x: int(x))

    print("% ---- per-campus table rows (supplement) ----")
    print("Arm & " + " & ".join(campuses) + " & Median \\\\")
    for arm in ["T0", "T1", "T3", "T3n", "T3p"]:
        if arm not in pc:
            continue
        row = " & ".join(f"\\num{{{pc[arm][c]:.2f}}}" for c in campuses)
        print(f"{ARM_LABEL[arm]} & {row} & \\textbf{{\\num{{{med[arm]:.2f}}}}} \\\\")

    print("\n% ---- contrasts ----")
    for k, v in d["contrasts"].items():
        if not v or k not in CONTRAST_LABEL:
            continue
        lab, iso = CONTRAST_LABEL[k]
        lo, hi = v["ci95"]
        print(f"{lab} & {iso} & \\num{{{v['median_delta']:+.2f}}} & "
              f"\\numrange{{{lo:+.2f}}}{{{hi:+.2f}}} & {v['n_pos']}/{v['n']} \\\\")

    print("\n% ---- prose numbers ----")
    for arm in ["T0", "T1", "T3", "T3n", "T3p"]:
        if arm in med:
            print(f"  median {arm:4s} = {med[arm]:.2f}")
    for k, v in d["contrasts"].items():
        if v:
            print(f"  {k:32s} {v['median_delta']:+.3f} "
                  f"[{v['ci95'][0]:+.3f},{v['ci95'][1]:+.3f}] {v['n_pos']}/{v['n']}")
    print(f"  parameters: {d.get('n_param')}")
    print(f"  epochs={d.get('epochs')} head_epochs={d.get('head_epochs')} "
          f"seeds={d.get('seeds')} negfrac={d.get('negfrac')}")


if __name__ == "__main__":
    main()
