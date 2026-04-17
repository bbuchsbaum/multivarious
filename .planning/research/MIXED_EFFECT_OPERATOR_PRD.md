# PRD: Effect-Operator Mixed Regression For `multivarious`

**Project:** multivarious  
**Date:** 2026-04-14  
**Status:** product requirements draft  
**Related design note:** `.planning/research/MIXED_EFFECT_OPERATOR_DESIGN.md`

## Product Vision

One fit. One family of named effects. One resampling engine.

Build `mixed_regress()` so that every fixed-effect term in a repeated-measures or mixed design becomes a first-class multivariate object, not just a p-value. That object should be interpretable, low-rank, reconstructable, and compatible with the existing `multivarious` verbs.

Core product idea:

- `mixed_regress()` fits the row-side geometry of the design.
- `effect(fit, term)` returns an `effect_operator`.
- `effect_operator` behaves like a `bi_projector`.
- `perm_test()` and `bootstrap()` work on that effect object.

This means the package should not sprout separate front ends called `rm_asca`, `lmm_pca`, and `hd_test`. Those are presets or viewpoints on one underlying object.

## 1. Problem Statement

Users have multivariate responses with:

- repeated measures within subject, such as low/mid/high,
- between-subject factors,
- random effects,
- potentially missing or unbalanced visits,
- often `p >> n`.

Classical MANOVA is not the right package-level abstraction because it ties users to low-dimensional covariance inversion and omnibus testing. The desired abstraction is a named effect in feature space, with a clean decomposition, reconstruction, and resampling-based inference.

## 2. Product Goals

The feature must satisfy five goals.

1. Unification  
   A single model object handles fixed effects, repeated measures, random effects, and high-dimensional regularization.

2. Interpretability  
   Every term of the design yields a low-rank effect decomposition with loadings, scores, singular values, and reconstruction back into original variables.

3. High-dimensional viability  
   No dense `p x p` covariance inverse is required. The implementation must work in dual space or score space when `p >> n`.

4. Native integration with `multivarious`  
   New objects reuse existing verbs: `project`, `reconstruct`, `truncate`, `components`, `ncomp`, `perm_test`, `bootstrap`, `residuals`.

5. Resampling-first inference  
   Omnibus significance, rank selection, and stability come from exchangeability-respecting permutation and cluster bootstrap.

## 3. Non-Goals For v1

To keep the first version coherent:

- no non-Gaussian outcomes,
- no full Bayesian model,
- no fully arbitrary crossed random-effects structures,
- no feature-wise missingness patterns where different variables are missing in different rows,
- no separate method families with separate APIs.

v1 should focus on continuous Gaussian matrix responses with one main clustering variable, repeated rows per subject, and optional random slopes for within-subject factors.

## 4. Statistical Abstraction

Let:

- `Y` be the stacked observation-by-feature response matrix,
- `Omega` be the observation-space covariance induced by random effects and residual correlation,
- `P_H` be the whitened incremental projector for fixed-effect term `H`,
- `B` be a feature basis, with `B = I` in the full-rank case and `k << p` in the regularized case.

Define the term-specific effect matrix in basis space:

\[
M_H = P_H \Omega^{-1/2} Y B
\]

and the corresponding positive effect operator:

\[
T_H = M_H^\top M_H
\]

Take the SVD:

\[
M_H = U_H D_H V_H^\top
\]

This decomposition is the product.

Product requirements implied by this definition:

- `trace(T_H)` is omnibus effect energy,
- `ncomp(effect)` is bounded by hypothesis degrees of freedom and basis rank,
- `components(effect)` returns feature directions `V_H`,
- `scores(effect)` returns observation-side effect scores,
- `reconstruct(effect)` returns the contribution of term `H` in original variable space,
- `truncate(effect, m)` keeps the first `m` interpretable effect axes.

This is the contract the implementation must preserve.

## 5. User-Facing API

Primary API:

```r
fit <- mixed_regress(
  Y,
  design = design,
  fixed  = ~ group * level,
  random = ~ 1 + level | subject,
  basis  = shared_pca(ncomp = 20),
  preproc = center()
)

E <- effect(fit, "group:level")

pt <- perm_test(E, nperm = 999, scheme = "reduced_model")
k  <- ncomp(pt)

E_sig <- truncate(E, k)

components(E_sig)
scores(E_sig)
reconstruct(E_sig)
bootstrap(E_sig, nboot = 500, resample = "subject")
```

Convenience array path:

```r
fit <- mixed_regress(
  Y_array,
  fixed  = ~ group * level,
  random = ~ 1 + level | subject,
  basis  = shared_pca(ncomp = 20)
)
```

Internally, all input forms normalize to the same long observation-by-feature representation.

## 6. Primary Classes

### `mixed_fit`

Top-level fitted model.

Fields:

- `design`
- `fixed_terms`
- `random_spec`
- `row_engine`
- `basis`
- `preproc`
- `effects_meta`
- cached row projectors and reduced-model residual objects
- original dimension metadata

Methods:

- `print()`
- `summary()`
- `effect()`
- `effects()` later if needed
- `residuals()`

### `effect_operator`

This should inherit from `bi_projector`.

Core fields:

- `v` feature loadings,
- `s` effect scores,
- `sdev` singular values,
- `term`,
- `df_term`,
- `effect_matrix`,
- `row_projector`,
- `row_metric`,
- `basis`,
- `fitted_contribution`,
- permutation and bootstrap metadata.

Because it is a `bi_projector`, the current package grammar stays intact.

### `perm_test_effect_operator`

Subclass of the existing permutation result class.

It should store:

- omnibus statistic,
- component-wise observed statistics,
- permutation distributions,
- selected rank,
- p-values,
- exchangeability scheme used,
- seed and reproducibility metadata.

`ncomp()` on this object should return the selected number of significant effect axes.

## 7. Input Support

The function should accept three equivalent representations:

1. primary: `Y` as `n_obs x p` matrix plus design data frame,
2. convenience: `Y` as `n_subject x n_within x p` array,
3. convenience: named list or block-wise repeated-condition structure, converted internally to stacked observations.

The package already has multiblock infrastructure, so block semantics are familiar. The new feature should use that idea only for input normalization, not as the core statistical abstraction.

## 8. Basis Strategy

This is where low-dimensional and high-dimensional use cases become one method.

Allowed basis options:

- `identity_basis()` for low or moderate `p`,
- `shared_pca(ncomp = k)` for `p >> n`,
- user-supplied `projector` or `bi_projector`,
- later: sparse, structured, or graph-smoothed basis.

Rules:

- in low dimension, `basis = identity_basis()` recovers the exact effect decomposition,
- in high dimension, the implementation operates in score space or dual space and reconstructs loadings to original variables only at the end,
- the basis is shared across terms by default for comparability,
- effect-specific basis fitting can come later.

Hard requirement:

- no dense `p x p` inversion in the main path.

## 9. Row Engine

The row engine should be a thin adapter layer so the algebra of effect operators stays independent of the fitting backend.

Required v1 capabilities:

- fixed effects from an R formula,
- one grouping variable for repeated observations,
- random intercept,
- optional random slopes for within-subject terms,
- support for unbalanced numbers of observations per subject,
- row-wise missing visits allowed.

Internal responsibilities:

- estimate row metric `Omega`,
- build reduced and full model hypothesis projectors,
- supply reduced-model residuals for permutation,
- expose whiten and unwhiten operations on observation space.

Once the row engine emits `P_H` and `Omega^{-1/2}`, the rest of the framework is pure operator algebra.

## 10. Inference Design

`multivarious` already has a generic permutation-testing workflow based on shuffling, refitting or reprojecting, and measuring a statistic, and the current PCA method already uses Vitale-style sequential logic. The new feature should extend `perm_test()` rather than invent a second inference API.

For `effect_operator`, use two statistics:

\[
\tau_H = tr(T_H)
\]

for the omnibus effect, and

\[
F_{H,a} = \frac{d_a^2}{\sum_{j \ge a} d_j^2}
\]

for sequential rank selection.

Rank-testing logic should adapt the practical lessons from Vitale et al.:

- resampling breaks relevant covariance while preserving marginal scale,
- testing proceeds by sequential deflation,
- permutation-induced rank mismatch is corrected by projection,
- the projection subspace is re-estimated from each permuted residual rather than reused from the observed residual space.

Default exchangeability rules:

- between-subject term: permute subjects,
- within-subject term: resample in subject blocks in contrast space,
- interaction term: reduced-model residual permutation with subject blocks,
- bootstrap: cluster bootstrap at subject level only.

For bootstrap, loadings and scores must be aligned to the reference solution by sign correction or Procrustes alignment before computing stability summaries.

## 11. Core Product Behavior

The following must work in v1.

Named effect extraction:

- `effect(fit, "level")`
- `effect(fit, "group")`
- `effect(fit, "group:level")`

Reconstruction:

- `reconstruct(effect(fit, "group:level"))`
- `truncate(effect(fit, "group:level"), 2)`

Ordered-factor interpretability:

- for ordered low/mid/high factors, the framework respects the contrast system used in the design,
- with orthogonal polynomial contrasts, the within-subject effect becomes immediately interpretable as linear trend and curvature.

Existing generic compatibility:

- `components`
- `scores`
- `project`
- `reconstruct`
- `truncate`
- `ncomp`
- `perm_test`
- `bootstrap`

That compatibility matters more than exporting many new helpers.

## 12. File-Level Implementation Plan

Follow the current package pattern of focused source files.

Add:

- `R/mixed_regress.R`
- `R/effect_operator.R`
- `R/mixed_row_engine.R`
- `R/mixed_permutation.R`
- `R/mixed_bootstrap.R`
- `R/mixed_utils.R`

Add tests:

- `tests/testthat/test_effect_operator.R`
- `tests/testthat/test_mixed_regress.R`
- `tests/testthat/test_mixed_permutation.R`
- `tests/testthat/test_mixed_bootstrap.R`

Add docs:

- `vignettes/Mixed_Regress.Rmd`
- update `vignettes/PermutationTesting.Rmd`
- update `vignettes/Regress.Rmd`
- optionally add a short effect-operators article

For heavy Monte Carlo calibration, keep full verification scripts outside the normal unit-test path under `experimental/` or similar.

## 13. Testing Plan

This feature needs stronger algebraic and statistical verification than the current smoke-test style in `test_regress.R`.

### Unit Tests

`test_effect_operator.R`

- constructor returns class `c("effect_operator", "bi_projector", ...)`,
- `ncomp(effect) <= min(df_term, basis_rank)`,
- `truncate()` reduces dimensions correctly,
- `components()`, `scores()`, `reconstruct()` return correct shapes,
- dual and primal implementations yield identical nonzero singular values on small problems,
- reconstruction is invariant to harmless sign flips.

`test_mixed_regress.R`

- fixed-only design works with `random = NULL`,
- random-intercept model fits and extracts effects,
- random slope on ordered within-subject factor fits,
- missing visits do not crash extraction,
- array input and long-matrix input produce identical results,
- `basis = identity_basis()` and `basis = shared_pca(k = p)` match on small full-rank examples.

### Integration Tests

- `perm_test(effect)` returns a valid permutation result object,
- `ncomp(perm_test_result)` returns selected rank,
- `bootstrap(effect)` returns aligned stability summaries,
- composed behavior with existing generics does not break,
- vignette code runs on small data.

### Regression Tests

- seeded examples reproduce omnibus statistics and selected rank,
- object printing and summary output remain snapshot-stable enough for tests,
- docs examples stay synchronized with the API.

## 14. Statistical Verification Plan

Use simulation families with known truth.

Family 1: pure null

- subject random intercept,
- repeated low/mid/high rows,
- no fixed effect,
- moderate and high-dimensional settings.

Checks:

- omnibus p-values approximately uniform,
- first-axis false positive rate near nominal,
- selected rank mostly zero under null.

Acceptance target:

- empirical Type I error in the range `0.03` to `0.07` at nominal `0.05` for omnibus and first-axis tests.

Family 2: main-effect rank recovery

- level main effect of true rank 1 or 2,
- known feature loadings,
- random intercept and optional slope,
- moderate and high noise.

Checks:

- selected rank tracks truth,
- estimated feature subspace matches truth via principal angles,
- power rises monotonically with signal strength.

Acceptance target:

- rank-1 recovery at moderate SNR at least `0.8`,
- rank-2 recovery at moderate SNR at least `0.7`.

Family 3: interaction recovery

- `group x level` effect of known rank,
- between-subject groups,
- repeated measures within subject,
- unbalanced sample sizes.

Checks:

- interaction p-value calibrated under null and powered under signal,
- reconstructed interaction pattern matches generating loadings,
- basis path and identity path agree when both are feasible.

Family 4: `p >> n`

- small `n_obs`, very large `p`,
- true effect contained in a low-rank feature subspace.

Checks:

- no memory explosion,
- dual versus primal equality on toy problems,
- recovered singular spectrum stable,
- runtime scales with basis rank rather than `p^3`.

Family 5: missing and unbalanced visits

- some subjects missing one of low/mid/high,
- heterogeneous numbers of observations per subject.

Checks:

- fit succeeds,
- effect extraction succeeds,
- calibration remains acceptable,
- bootstrap does not silently resample invalid rows.

Family 6: ordered-factor contrasts

- low/mid/high generated from linear and quadratic trajectories.

Checks:

- linear and curvature directions are separated when polynomial contrasts are used,
- reconstructed effects follow the generating trend structure.

## 15. Mathematical Verification Requirements

These should be exact or near-exact identities on toy problems.

1. Rank bound  
   Extracted effect rank never exceeds term degrees of freedom or basis rank.

2. Decomposition identity  
   The fitted contribution of term `H` equals reconstruction from its singular system.

3. Reduction to univariate LMM  
   When `p = 1`, `mixed_regress()` reduces to a univariate repeated-measures mixed model on that single response.

4. Reduction to fixed-effect operator ANOVA  
   When `random = NULL`, the effect operator matches direct fixed-design algebra.

5. Dual and primal equivalence  
   Nonzero singular values agree between direct and dual implementations.

6. Permutation invariance under relabeling  
   Reordering subjects or features changes only the same ordering in the output.

## 16. Performance Requirements

These are hard product requirements.

- no dense `p x p` inverse in the main path,
- high-dimensional mode works from dual matrices or basis scores,
- permutation testing supports parallel execution through the package's existing strategy,
- full Monte Carlo verification stays outside CRAN tests,
- smoke-size permutation tests still run in CI.

Developer benchmark grid:

- `n_subj in {20, 50, 100}`
- `q in {3, 5}`
- `p in {100, 1000, 10000}`
- `basis rank k in {5, 10, 20}`
- `permutations in {99, 199}`

Track:

- wall time,
- peak memory,
- selected rank,
- omnibus p-value stability across seeds.

## 17. Milestones

Milestone 1: algebraic core

- implement `mixed_fit`,
- implement `effect_operator`,
- support `random = NULL`,
- support identity and shared-PCA basis,
- add reconstruction and truncation methods.

Milestone 2: repeated-measures and random-effects row engine

- random intercept,
- optional random slope,
- subject-block metadata,
- missing-visit support.

Milestone 3: inference

- omnibus permutation test,
- sequential P3-style rank test,
- subject bootstrap with alignment,
- `ncomp()` on permutation result.

Milestone 4: docs and verification

- mixed-model vignette,
- simulation verification scripts,
- benchmarks,
- pkgdown integration,
- CRAN-facing smoke tests.

## 18. Definition Of Done

This feature is done when all of the following are true.

- a user can fit a low/mid/high repeated-measures design with a subject random effect and `p >> n` using one function,
- they can extract a named interaction effect as a `bi_projector`-compatible object,
- they can reconstruct that effect in original variable space,
- they can run `perm_test()` to get an omnibus p-value and selected number of significant axes,
- they can run `bootstrap()` to assess stability,
- calibration and rank-recovery simulations meet the thresholds above,
- a vignette demonstrates the complete workflow on a realistic repeated-measures dataset.

One housekeeping task belongs in the same PR or milestone set: version and documentation surfaces are out of sync. The mixed-model work should include a version and docs synchronization pass so the new feature lands on a consistent public surface.

## 19. Natural Follow-On Work

After the design note and PRD, the next clean project-management move is to split implementation into four linked issues:

- algebra core,
- row engine,
- resampling,
- verification and docs.

