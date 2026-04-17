# Mixed Effect Operator Calibration Results

These CSV files were generated from `experimental/mixed_effect_operator_calibration.R` on 2026-04-16 against the current development tree.

## Run set

- `interaction_null.csv`: `Rscript experimental/mixed_effect_operator_calibration.R null experimental/results/interaction_null.csv nsim=25 nperm=99 p=100`
- `interaction_missing_null.csv`: `Rscript experimental/mixed_effect_operator_calibration.R missing experimental/results/interaction_missing_null.csv nsim=25 nperm=99 p=100 missing_mid_prob=0.25`
- `within_null.csv`: `Rscript experimental/mixed_effect_operator_calibration.R within_null experimental/results/within_null.csv nsim=25 nperm=99 p=100`
- `between_null.csv`: `Rscript experimental/mixed_effect_operator_calibration.R between_null experimental/results/between_null.csv nsim=25 nperm=99 p=100`
- `interaction_rank1.csv`: `Rscript experimental/mixed_effect_operator_calibration.R rank1 experimental/results/interaction_rank1.csv nsim=25 nperm=99 p=100 signal_strength=1.0`
- `interaction_rank2.csv`: `Rscript experimental/mixed_effect_operator_calibration.R rank2 experimental/results/interaction_rank2.csv nsim=20 nperm=99 p=100 signal_strength=1.1`
- `within_rank1.csv`: `Rscript experimental/mixed_effect_operator_calibration.R within_rank1 experimental/results/within_rank1.csv nsim=25 nperm=99 p=100 signal_strength=1.0`
- `within_rank2.csv`: `Rscript experimental/mixed_effect_operator_calibration.R within_rank2 experimental/results/within_rank2.csv nsim=20 nperm=99 p=100 signal_strength=1.1`
- `between_rank1.csv`: `Rscript experimental/mixed_effect_operator_calibration.R between_rank1 experimental/results/between_rank1.csv nsim=25 nperm=99 p=100 signal_strength=1.5`
- `benchmark_grid.csv`: `Rscript experimental/mixed_effect_operator_calibration.R benchmark experimental/results/benchmark_grid.csv n_subject=20,50 p=100,1000 k=5,10 nperm=49 signal_term=interaction term=group:level`

Each run also writes a `_summary.csv` sidecar file.

## High-level findings

- Interaction omnibus calibration is close to nominal in this run set: `0.04`.
- Missing-visit interaction omnibus null is conservative: `0.00`, but the sequential rank selector still produces nonzero selections in some null runs.
- Within-subject omnibus null is still hot in this run set: `0.12`.
- Between-subject omnibus null improves to `0.00` under the subject-mean permutation path.
- Rank-1 recovery is now strong across the updated low-rank paths:
  - interaction: mean selected rank `1.2`, exact rank match `0.8`
  - within-subject: mean selected rank `1.24`, exact rank match `0.76`
  - between-subject: mean selected rank `1.0`, exact rank match `1.0`
- Rank-2 recovery is now exact in the current interaction and within-subject runs: mean selected rank `2.0`, exact rank match `1.0`.

## Immediate interpretation

- The omnibus trace test looks usable for the interaction path under the current one-grouping-variable Gaussian engine.
- The structural sequential-rank failure for one-df terms and last-axis tests is fixed: low-rank components are no longer blocked by a degenerate relative-eigenvalue statistic.
- The main remaining weakness is now clearly on the omnibus side for within-subject null calibration, not on rank-2 recoverability.
- The between-subject path is materially better after switching from whole-trajectory permutation to subject-mean permutation, but it should still be treated as a calibrated-on-this-design result rather than a universal guarantee.
- The benchmark grid suggests the high-dimensional path scales reasonably with basis rank, but `p = 1000` and `n_subject = 50` already push elapsed time into the multi-second range even with `nperm = 49`.
