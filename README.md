# Cross-context evidence gating

Reproduction code for the paper *Cross-Context Evidence Gating: Certifying Where
Work-Order Records Support Maintenance Decisions in Building Portfolios*.

Condition evidence is uneven across building portfolios. Some buildings have no asset-level
monitoring, while others monitor only selected assets. Many portfolios nevertheless maintain a
computerized maintenance management system (CMMS) that logs every work order, and such records can
support forward-looking risk screening. A screening model, however, is reported as one
portfolio-wide accuracy figure and deployed one asset class at a time, and the two need not agree.

The method in this repository therefore reads that record layer twice. It scores each
building-system-quarter for the risk of a heavy unplanned repair workload next quarter and
expresses the score as a within-campus percentile *R*. It then asks a separate question about the
*deployment stratum*, here the building system: has record-based prioritization for this kind of
system demonstrated enough support, cross-context transfer, score reliability, and prior-warning
trace to be acted on directly? Five auditable gates answer that and combine, under a
non-compensatory rule, into a certificate *S*. Critically, the certificate governing a campus is
computed from the other campuses only, so no outcome from a portfolio informs the rule applied to
it. High-risk units in certified strata open a **maintenance decision task**; high-risk units in
uncertified strata enter a **condition-verification queue** ranked by the evidence gap,
CEPI = *R*(1 − *S*).

## Data

The analysis uses the openly available Facility Management Unified Classification Database
(FMUCD): <https://doi.org/10.17632/cb8d2nsjss.1> (Mendeley Data `cb8d2nsjss`, 3.73 M work
orders, 12 North-American universities, 2002–2021). The dataset is **not** redistributed here.
Download it to `data/raw/FMUCD.csv` before running the pipeline; `scripts/00_phase0_inspect.py`
verifies the file hash.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.13, pinned versions
```

XGBoost uses the GPU when one is available and falls back to CPU otherwise. `RANDOM_SEED = 42`
throughout (`src/fmscreen/config.py`): the same data and code give the same results.

## Reproducing the results

Scripts are numbered in execution order and write to `results/`. The core
pathway from raw file to routed portfolio is:

```bash
python scripts/00_phase0_inspect.py          # verify the raw file, coverage, base rates
python scripts/01_build_panel.py             # building x system x quarter panel, past-only features
python scripts/03_premiums.py                # reactive-burden premiums, burden concentration
python scripts/04_temporal_ablation.py       # temporal split, model-layer ablation
python scripts/05_louo_transfer.py           # leave-one-university-out screening: the risk term R
python scripts/07_decomposition_heterogeneity.py
python scripts/16_coescalation.py            # cross-system association
python scripts/36_within_building_placebo.py # within-building control and permutation placebo
python scripts/37_cepi.py                    # five evidence gates, pooled certificate S
python scripts/39_queue.py                   # gate-then-rank routing, pooled deployment map
python scripts/40_outer_s.py                 # nested outer-fold S and the nested route replay
python scripts/42_route_yield.py             # retrospective outcome rate on each route
```

For a complete reproduction of every number, table, and figure in the paper
(including the supplementary material), run all numbered scripts in order, `00` through
`43`; each writes its outputs to `results/metrics/`, `results/tables/`, and
`results/figures/`. The remaining scripts cover the sensitivity and appendix
analyses the paper quotes (recent-activity rule `12`, work-order text `14`,
label-permutation control `15`, robustness strata `22`, excluded-campus
comparison `24`, building-feature ablation `27`, absolute thresholds and base
rates `28`, extreme-tail capture `29`, temporal-LOUO `31`, building-rule yield
`33`, variance decomposition `35`, and the synthetic certification benchmark
`43`, whose four generative regimes fix evidential sufficiency by construction
so that recovery can be scored against a known truth). The manuscript figures
are drawn by `09`, `21`, `26`, `32`, `38`, and `41`.

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

Ku Chia, T. P., Fu, Y., and He, X. *Cross-context evidence gating: certifying where work-order
records support maintenance decisions in building portfolios.* 2026. Submitted to the
*Journal of Computing in Civil Engineering*.
