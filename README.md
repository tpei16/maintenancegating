# Peer-campus work-order evidence for building-system-level screening of reactive maintenance workload

Reproduction code and result files for the paper *Assessing Peer-Campus Work-Order Evidence for
Building-System-Level Screening of Reactive Maintenance Workload*.

A risk screening model trained on the work orders of a computerized maintenance management
system (CMMS) is usually validated by one accuracy figure pooled over a portfolio, whereas a
maintenance planner acts on its ranking one building system at a time. The code in this
repository computes two quantities for a target campus. The first is a risk percentile *R* for
each unit (one building system of one building in one quarter), which ranks the units of the
campus within a quarter by the score of a model trained on the other campuses. The second is an
evidence score *S* for each building system, computed from the records of the other
campuses by five gates: data sufficiency, risk concentration, transfer stability, risk-gradient
reliability and prior-record trace. Three of the gates are prerequisites, and the scores are
combined by a non-compensatory rule. Every model and every threshold behind the verdict
for a campus excludes that campus, so no outcome of a portfolio informs the rule applied to it.
A building system is referred to as supported when *S* is at least 2/3. A high-risk unit (*R* of
at least 0.90) of a supported building system opens a maintenance decision task. A high-risk
unit of a building system that is not supported enters condition verification first, in a queue ordered by *R*.

The verdict states whether the work-order records of peer campuses meet stated criteria for
acting on the ranking for a building system. It does not assess the physical condition of any asset.

## Data

The analysis uses the openly available Facility Management Unified Classification Database
(FMUCD): <https://doi.org/10.17632/cb8d2nsjss.1> (Mendeley Data `cb8d2nsjss`, 3.73 M work
orders, 12 North-American universities, 2002–2021). The dataset is **not** redistributed here.
Download it to `data/raw/FMUCD.csv` before running the pipeline; `scripts/00_phase0_inspect.py`
reports its row count, date range and field coverage.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.13, pinned versions
```

`RANDOM_SEED = 42` throughout (`src/fmscreen/config.py`). The models are fitted on a GPU by
default; set `FMSCREEN_DEVICE=cpu` to fit them on the CPU.

## Reproducing the results

Scripts are numbered in execution order and write to `results/`. The pathway from the raw file
to the evidence gating and the routing is:

```bash
python scripts/00_phase0_inspect.py              # verify the raw file, coverage, base rates
python scripts/01_build_panel.py                 # panel of units, features from past records only
python scripts/05_louo_transfer.py               # screening performance with one campus left out
python scripts/44_double_holdout_fits.py         # models for the evidence gating: one campus left out (9 fits) and two left out (36 fits)
python scripts/45_certification_corrected.py     # five gates and S for the pooled map and the nine held-out maps
python scripts/46_manuscript_numbers.py          # gating and routing quantities quoted in the paper
python scripts/55_route_yield_corrected.py       # next-quarter outcome rate on each route
python scripts/56_fold_verdicts_and_simple_rule.py   # gate verdicts per held-out map, count-based rule, held-out check
```

The analyses that examine the evidence gating are:

| Script | Analysis |
|---|---|
| `43`, `63` | synthetic benchmark with known ground truth, with the shared-attribute construction and the two comparison rules |
| `66` | permuted-score test: gates recomputed on the real records after the scores of a building system are permuted on a share of the peer campuses |
| `67` | partially pooled rule: threshold on the predictive probability of two-fold lift from a random-effects model across peer campuses |
| `73` | graded loss of transfer: partial permutation and reversal of the scores, with the transfer-only, pooled-lift, partially pooled and count-based rules scored on the same draws |
| `57` | single nesting, size-matched control and double nesting |
| `58` | bootstrap probability of a positive verdict |
| `48`, `59` | evidence gating under two other risk scores and its agreement with the boosted model |
| `60`, `64` | evidence gating from a campus's own history against gating from its peers, and the figure |
| `68` | direct validation of the ranking on a campus's own earlier outcomes, by length of the outcome record |
| `61` | no-antecedent share against service frequency |
| `62` | order of the verification queue |
| `69` | flags of the routing rule by building system, pooling by percentile, and resolution of ties |
| `70` | intervals under clustering by campus and building system |
| `74` | sensitivity of the main contrasts to the removal of one campus or building system, where each threshold is fitted, and variants of the prior-record-trace gate |
| `71` | outcome-free comparability of a campus with its peers, labor-recording audit, and record flow |
| `72` | verdicts under other outcome definitions (work-order count, campus-specific labor threshold, occurrence) |
| `65` | risk-gradient gate with campus-specific and with pooled base rates |
| `49` | sensitivity to the gate levels, the support level and the set of prerequisite gates |
| `51` | score distribution of missed high-burden units and one campus-quarter worked through |
| `47`, `38` | gate-score figure |

For a complete reproduction of every number, table and figure in the paper and its Supplemental
Materials, run all numbered scripts in order. The remaining scripts cover the descriptive and
sensitivity analyses of the Supplemental Materials (burden concentration and persistence `03`,
`17`, `18`; temporal split `04`, `31`; label-permutation control `15`; cross-system association
`16`, `22`, `36`; excluded-campus comparison `24`; building-feature ablation `27`; thresholds
and base rates `28`; missed high-burden units `29`; seasonality `41`; inventory of numerical
constants `50`; model-capacity comparisons `51` to `54`).

On eight CPU cores the
nine leave-one-campus-out fits and the 36 pair fits take about 90 seconds; the bootstrap, the
nesting control and the synthetic benchmark take between 15 and 45 minutes each.

### Names in the code and in the paper

The result files keep the identifiers under which the analysis was written.

| Identifier in code and result files | Term in the paper |
|---|---|
| `stratum`, `system` | building system |
| `context`, `university`, `campus` | campus |
| cell-quarter, unit-quarter, row | unit |
| `S`, sufficiency score, certification score | evidence score |
| `certified`, `cert_*` | supported (evidence score of at least 2/3) |
| hard gates | prerequisite gates |
| `calibration_reliability`, `risk_gradient` | risk-gradient reliability gate |
| `antecedent_signal`, `zero_prior`, no-antecedent share | prior-record trace gate and its quantity |
| `pma`, `verify`, `continue`, `watch` | maintenance decision task, condition verification, standard maintenance, watch list |
| `W`, `B` (bootstrap) | buildings resampled within campuses; campuses resampled, then buildings |
| `deranged` | scores permuted within campus and quarter (permuted-score test) |
| `count_rule_*` in `route_aligned_evaluation.csv` | the prior-year count of reactive work orders used as a risk score |

## Leakage discipline

All predictive results are produced under four rules, enforced without exception:

1. every feature uses only information available through the anchor quarter *t*;
2. the label uses quarter *t*+1 only;
3. system-specific high-burden thresholds and reactive-premium features are fit on the training
   fold within each validation split, never on the full data;
4. university identity is never a model feature.

As an empirical check, a label-permutation negative control collapses top-10 % lift to 0.83,
confirming that the features carry no spurious information about the shuffled target.

## Layout

```
scripts/          numbered pipeline, one step per file
src/fmscreen/     panel construction, features, validation splits, models, metrics, figure style
results/          metrics (JSON/CSV), tables (CSV), figures (PDF/PNG)
```

Analysis settings, including the random seed and the model hyperparameters, live in
`src/fmscreen/config.py`.

## Citation

Ku Chia, T. P., Fu, Y., He, X., and Zhang, Z. *Assessing peer-campus work-order evidence for
building-system-level screening of reactive maintenance workload.* 2026. Submitted to
the *Journal of Computing in Civil Engineering*.
