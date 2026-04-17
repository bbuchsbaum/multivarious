# Design Note: Operator-Valued Mixed ANOVA In `multivarious`

**Project:** multivarious  
**Date:** 2026-04-14  
**Status:** design proposal and implementation research  
**Confidence:** high for object model and package fit; medium for edge-case inference details

## Executive Summary

The cleanest unification is to stop treating repeated-measures ASCA, mixed-model PCA, and high-dimensional omnibus testing as separate methods. In `multivarious`, they should be exposed as different queries on one fitted object.

The conceptual spine is:

- ANOVA decomposes observation space into hypothesis subspaces.
- Mixed models do not change that idea; they change the metric on observation space.
- High-dimensional multivariate analysis does not change it either; it changes how we read those subspaces in feature space.

So the right object is not "MANOVA with extras". It is operator-valued ANOVA.

At the implementation level, the central effect matrix is:

\[
M_H = P_H^{(\Omega)} Y Q
\]

where:

- `Y` is the long-format response matrix (`N x p`, one row per observation, one column per feature),
- `P_H^{(\Omega)}` is the covariance-adjusted hypothesis projector for effect `H`,
- `Q` is an optional feature-side basis or regularizer.

The derived feature-side operator is:

\[
S_H = M_H^\top M_H
\]

This gives one fit, one effect object, and one inference engine. The three main user questions become:

1. Is there an effect?
   Test `||M_H||_F^2 = tr(S_H)`.
2. How many effect dimensions are real?
   Test singular values of `M_H` sequentially.
3. What does the effect look like?
   Inspect the SVD `M_H = U_H D_H V_H^\top` and reconstruct on the original variable scale.

This formulation fits `multivarious` unusually well, because the package already organizes multivariate analysis around:

- projector objects,
- bi-projector objects,
- generic verbs like `project()`, `components()`, `reconstruct()`, and `residuals()`,
- permutation testing via `perm_test()`,
- bootstrap resampling via `bootstrap()`.

The implementation path should therefore be: introduce a mixed-model fit object plus an effect-operator object, then extend the existing projector-style verbs and inference generics.

In one phrase:

- the design supplies projectors,
- the data push them into feature space,
- mixed models bend the metric,
- the spectrum is the analysis.

## Why This Fits `multivarious`

The current architecture already contains the right abstractions:

- `R/projector.R` defines the base projection object and preprocessing pipeline.
- `R/bi_projector.R` defines a two-sided decomposition with row scores, column loadings, and singular-value-like summaries.
- `R/all_generic.R` already exports `components()`, `reconstruct()`, `residuals()`, `perm_test()`, and `bootstrap()`.
- `R/pca.R` already implements sequential permutation logic influenced by Vitale-style rank testing.
- `R/plsc_inference.R` already implements a deflation-based permutation ladder plus bootstrap stability summaries.

That means the package does not need a new philosophical layer. It needs one new model family that uses the existing geometry more consistently.

The key package-level insight is:

- the feature-side regularizer `Q` can be represented as an existing `projector` or `bi_projector`,
- the effect decomposition itself can be represented as a specialized `bi_projector`,
- permutation and bootstrap should be method extensions on the new effect class, not a separate inference subsystem.

## Core Mathematical Formulation

Let:

- `Y` be viewed as a linear operator from feature space to observation space,
- `Y` be the `N x p` observation-by-feature response matrix in long format,
- `Omega` be the row covariance implied by the mixed model,
- `W` be a whitening operator such that `W^\top W = Omega^{-1}`,
- `Y_w = W Y`,
- `X = [X_N, X_H]` be nuisance plus hypothesis columns in the fixed-effect design,
- `X_w = W X`.

Define the incremental hypothesis projector in whitened observation space:

\[
P_H = P_{[X_{N,w}, X_{H,w}]} - P_{X_{N,w}}
\]

Then the whitened effect matrix is:

\[
M_H = P_H Y_w B
\]

where `B` is a feature basis. If `B = I`, then `Q = I`. If `B = V_k`, then `Q = V_k V_k^\top`.

Equivalent positive operator on feature space:

\[
T_H = M_H^\top M_H
\]

Equivalent dual row-side operator:

\[
K_H = M_H M_H^\top
\]

Written directly in the original metric, the pure operator is:

\[
T_H =
Q^{1/2} Y^\top \Omega^{-1/2} P_H \Omega^{-1/2} Y Q^{1/2}
\]

The nonzero eigenvalues of `T_H` and `K_H` are identical, so high-dimensional computation can remain in observation space.

Important structural fact:

\[
rank(T_H) = rank(M_H) \le rank(P_H)
\]

So the intrinsic effect dimension is constrained by the design, not by `p`. This is the core reason the framework behaves well when `p >> N`.

For any feature direction `v`,

\[
\langle v, T_H v \rangle = ||P_H \Omega^{-1/2} Y Q^{1/2} v||^2 \ge 0
\]

So `T_H` is positive, its spectrum is the effect, and its trace is the multivariate sum of squares for hypothesis `H`.

## Tensor View And Observation Geometry

The most conceptually useful formulation is to treat the data as living in:

\[
\mathcal Y \in \mathcal S \otimes \mathcal W \otimes \mathcal F
\]

where:

- `S` is subject space,
- `W` is within-subject condition space,
- `F` is feature space.

Collapse subject and within-subject modes into one observation space:

\[
\mathcal O = \mathcal S \otimes \mathcal W
\]

Now view the data as an operator:

\[
Y: \mathcal F \to \mathcal O
\]

That is the clean conceptual move. The data are no longer "a big matrix". They are a map from feature directions to subject-by-condition patterns.

Once the mixed model defines `Omega` on observation space and we whiten to `Y_w = Omega^{-1/2} Y`, random effects have become geometry.

## Operator-Valued ANOVA

In a balanced orthogonal design, ANOVA decomposes the identity on observation space:

\[
I_{\mathcal O} = \sum_H P_H
\]

Push that decomposition through the data operator and you obtain:

\[
\sum_H T_H = Q^{1/2} Y^\top \Omega^{-1} Y Q^{1/2}
\]

So the ANOVA table is not fundamentally a table of scalars. It is a table of positive operators.

Classical summaries are recovered as views of those operators:

- scalar ANOVA table: traces,
- MANOVA summaries: spectral functionals,
- ASCA: effect-matrix decompositions,
- high-dimensional omnibus tests: trace statistics and resampling,
- LiMM-PCA-like variants: structured choices of `Q` or dual truncation.

This is why the package should expose one object with multiple summaries, not multiple method families.

## The Right Package Objects

### 1. `mixed_regress_fit`

This is the top-level fitted model returned by `mixed_regress()`.

It should store:

- original call,
- response matrix `Y`,
- data or row index mapping,
- fixed-effect design matrix `X`,
- random-effect design information `Z`,
- grouping structure,
- covariance representation for `Omega`,
- whitening operator or precision-factor representation,
- hypothesis registry,
- basis specification and fitted basis object,
- resampling specification,
- optional cached effect objects.

This object is not itself the inferential target. It is the geometry factory.

### 2. `effect_operator`

This should be the main effect object returned by `effect(fit, H)`.

Recommendation: make it inherit from `bi_projector`.

Why:

- `v` can store feature loadings `V_H`,
- `s` can store row patterns `U_H D_H`,
- `sdev` can store singular values `d_H`,
- `components()`, `scores()`, and `truncate()` semantics already line up,
- it keeps the new class aligned with the rest of the package.

Extra fields needed beyond a plain `bi_projector`:

- `hypothesis`,
- `term_label`,
- `row_basis`,
- `basis`,
- `metric`,
- `M_whitened`,
- `effect_energy`,
- `rank_limit`,
- `fit_ref`,
- `row_index`,
- `col_index`,
- `reconstruct_map`.

### 3. `basis_spec`

The feature-side `Q` should not be a loose matrix argument. It should be an explicit basis specification with at least these presets:

- `basis_identity()`
- `shared_basis(k, fit_on = c("full", "nuisance_residual", "whitened_residual"))`
- `supplied_basis(projector)`

Internally, this should resolve to a `projector`-compatible object with loadings `V`.

### 4. `exchangeability_scheme`

Permutation validity depends on the hypothesis, not just the fit. This needs an explicit object or structured list describing:

- block structure,
- whole-subject permutation permissions,
- within-subject sign-flip permissions,
- restricted within-subject shuffles,
- bootstrap resampling units.

This object can be stored on the fit and specialized per effect.

## Public API Recommendation

The mathematically clean interface is:

```r
fit <- mixed_regress(
  Y,
  fixed  = ~ group * level,
  random = ~ 1 + level | subject,
  data   = dat,
  basis  = shared_basis(k = 20, fit_on = "nuisance_residual"),
  infer  = permute(nperm = 1000, scheme = "reduced_model")
)

E <- effect(fit, "group:level")

perm_test(E, type = "global")
perm_test(E, type = "rank")
components(E)
reconstruct(E, comp = 1:2, scale = "original")
bootstrap(E, nboot = 500, resample = "subject")
```

This is better than introducing many new top-level verbs because it reuses the package's existing projector conventions.

### Avoid these API traps

- Do not add `rank()` as a package generic. Base R already has `rank()`, and masking it will be ugly.
- Do not make `test()` the primary entry point. It is too vague and does not align with current `multivarious` naming.
- Do not expose three method families (`rm_asca`, `lmm_pca`, `hd_test`) as parallel user concepts.

If convenience wrappers are wanted later, use:

- `effect_test()`
- `effect_rank()`
- `stability_summary()`

But the canonical path should remain `effect()` plus generic verbs.

## Mapping To Existing `multivarious` Abstractions

| Existing abstraction | Current role | New role in mixed-effect framework |
|---|---|---|
| `projector` | generic feature projection map | shared basis or regularizer for `Q` |
| `bi_projector` | row/column decomposition | representation for each `effect_operator` |
| `cross_projector` | paired decompositions across two domains | not primary, but useful precedent for dual-side metadata |
| `regress` | multi-output coefficient model | useful design precedent, but not the right effect object |
| `perm_test()` | generic permutation interface | should gain an `effect_operator` method |
| `bootstrap()` | generic bootstrap interface | should gain an `effect_operator` method |
| `reconstruct()` | low-rank reconstruction | reconstruct effect matrices on whitened or original scale |

The biggest design choice is this one:

- `mixed_regress_fit` is a model object.
- `effect_operator` is the analysis object.

This keeps fitting and effect extraction separate and matches the package's current style.

## Implementation Strategy

### Phase 1: Fixed-Effect And GLS Prototype

Goal: build the operator machinery before full mixed-model support.

Deliver:

- `mixed_regress()` with `random = NULL`,
- hypothesis parsing from fixed-effect formulas,
- GLS or OLS row projector construction,
- `effect()` returning `effect_operator`,
- `perm_test.effect_operator()` for global and sequential tests.

Reason:

- This validates the operator abstraction and package API first.
- It lets the current projector machinery carry more of the weight early.
- It reduces mixed-model debugging during the first object-model pass.

### Phase 2: Mixed-Model Row Geometry

Goal: replace OLS row geometry with covariance-adjusted mixed-model geometry.

Recommended implementation target: `lme4`, initially in `Suggests`.

Use `lme4` for:

- formula parsing,
- sparse fixed/random design construction,
- fitted random-effect covariance parameterization,
- extraction of row-side model geometry.

Implementation options:

1. Formula-first wrapper:
   `mixed_regress(Y, fixed, random, data, ...)`

2. Matrix-first core:
   internal engine working from `X`, `Z`, grouping, and `Omega` representation

Recommendation:

- implement the matrix-first engine internally,
- expose a formula wrapper externally.

This keeps the computational core package-native and easier to test.

### Phase 3: Feature Basis And High-Dimensional Path

Goal: support `Q != I` without changing the object model.

Recommended defaults:

- `basis_identity()` for full-rank effects,
- `shared_basis()` estimated from nuisance-adjusted or reduced-model residuals.

Implementation path:

- leverage current `pca()` or `svd_wrapper()` infrastructure for basis estimation,
- store the fitted basis as a `projector` or `bi_projector`,
- compute `M_H` in basis coordinates to avoid `p x p` operations.

When `p` is very large:

- compute via the dual operator `K_H`,
- recover feature loadings by backprojection through the shared basis or `Y_w^\top U_H D_H^{-1}`.

### Phase 4: Inference Engine

Goal: make permutation and bootstrap native queries on `effect_operator`.

Permutation should answer:

- is there an effect?
- how many dimensions are real?

Bootstrap should answer:

- how stable are the effect subspaces and loadings?

This is already consistent with the package's split between `perm_test()` and `bootstrap()`.

### Phase 5: Documentation And Presets

Goal: expose named presets without fragmenting the conceptual model.

Examples:

- repeated-measures ASCA preset = `basis_identity()` plus contrast-aware effect extraction,
- LiMM-PCA-like preset = shared low-rank basis,
- high-dimensional omnibus preset = global trace test only.

These should be wrappers or argument presets around `mixed_regress()`, not separate model classes.

## Concrete Algorithms

### 1. Hypothesis Registry

Each fixed-effect term should map to:

- a term label,
- full and reduced design matrices,
- a whitened incremental row subspace basis,
- rank limit implied by the design.

Implementation sketch:

1. build the full fixed-effect model matrix,
2. determine term-to-column mapping from `terms()` and `assign`,
3. for each term `H`, define nuisance columns `X_N` and effect columns `X_H`,
4. whiten both,
5. compute orthonormal bases with QR or sparse QR,
6. define the incremental projector by difference of orthogonal projectors.

### 2. Whitening And Row Geometry

For the first mixed-model implementation, avoid explicit dense `Omega`.

Preferred internal representation:

- sparse precision factor,
- sparse Cholesky factor,
- or a linear operator interface with methods `whiten_rows()` and `unwhiten_rows()`.

The package only needs a stable row-whitening API, not a literal dense covariance matrix.

### 3. Effect Extraction

Given whitened `Y` and effect projector `P_H`:

1. compute `A_H = P_H Y_w`,
2. apply basis `B` if present,
3. factorize `A_H` by SVD or eigendecomposition,
4. package result as `effect_operator`.

Suggested internal fields:

- `u`: optional left singular vectors,
- `s`: `U_H D_H`,
- `v`: `V_H`,
- `sdev`: `d_H`,
- `energy`: `d_H^2`,
- `rank_limit`.

Interpretation follows directly from the SVD:

\[
M_H = U_H D_H V_H^\top
\]

where:

- `V_H` are feature directions for hypothesis `H`,
- `U_H` are design-side patterns inside the hypothesis subspace,
- `D_H^2` are effect energies.

Back on the original scale, the reconstructed effect is:

\[
\hat E_H = \Omega^{1/2} U_H D_H V_H^\top
\]

This is cleaner than global PCA followed by post hoc interpretation because each axis belongs to a named effect.

### 4. Reconstruction

The `effect_operator` method for `reconstruct()` should support:

- `scale = "whitened"`: return `U_H D_H V_H^\top`,
- `scale = "original"`: return `W^{-1} U_H D_H V_H^\top`,
- optional component subset `comp`,
- optional row and column subsets.

This differs from ordinary `bi_projector` reconstruction enough that a specialized method is warranted.

### 5. Global Test

Observed statistic:

\[
T_H = ||M_H||_F^2 = \sum_a d_a^2
\]

This is the right default omnibus statistic for all settings, including high-dimensional ones.

### 6. Sequential Rank Test

Observed statistic at step `a`:

\[
F_{H,a} = \frac{d_a^2}{\sum_{q \ge a} d_q^2}
\]

This mirrors the relative-eigenvalue logic already used in the package's PCA permutation testing and should become the default rank-selection statistic for effect operators.

### 7. Permutation Scheme

Reduced-model residual resampling should be the default.

Pipeline:

1. fit reduced model under `H_0`,
2. form whitened reduced-model residual matrix,
3. resample only within valid exchangeability blocks,
4. rebuild pseudo-data under the null,
5. recompute `M_H`,
6. compute global and stepwise rank statistics from the same pipeline.

The P3-style lesson from Vitale et al. should be lifted directly:

- sequentially deflate,
- correct rank mismatch under permutation,
- re-estimate the projected residual subspace per permutation rather than locking it to the observed one.

This is not a cosmetic detail. It should be the default calibration strategy for rank testing.

Permutation answers existence.

- Is there an effect?
- How many effect dimensions are real?

### 8. Bootstrap Scheme

Subject-level bootstrap should be the default stability procedure.

Pipeline:

1. resample subjects,
2. refit row covariance geometry,
3. recompute each target effect,
4. Procrustes-align singular vectors or loading subspaces,
5. summarize stability of singular values, feature loadings, and reconstructed effects.

Bootstrap answers persistence.

- How stable is the geometry under subject resampling?
- Which axes and loadings recur reliably?

## Dependency Strategy

### Core

Keep the core operator algebra inside `multivarious`.

Existing useful core dependencies already present:

- `Matrix`
- `chk`
- existing SVD/PCA infrastructure

### Recommended new `Suggests`

- `lme4` for formula parsing and sparse mixed-model geometry
- `pbkrtest` for validation and small-sample benchmarking against known mixed-model tests
- `permuco` for checking repeated-measures permutation schemes and exchangeability conventions
- `lmeresampler` for bootstrap workflow comparison

Recommendation:

- do not make `pbkrtest`, `permuco`, or `lmeresampler` hard dependencies,
- probably keep `lme4` in `Suggests` for the first implementation unless `mixed_regress()` becomes central enough to justify `Imports`.

## Immediate Code Layout Proposal

Suggested new files:

- `R/mixed_regress.R`
- `R/effect_operator.R`
- `R/hypothesis_projector.R`
- `R/mixed_basis.R`
- `R/mixed_inference.R`
- `tests/testthat/test_mixed_regress.R`
- `tests/testthat/test_effect_operator.R`
- `tests/testthat/test_mixed_inference.R`

Suggested generic additions in `R/all_generic.R`:

- `effect()`
- possibly `row_basis()`
- possibly `effect_energy()`

Do not add:

- `rank()`
- `test()`

## Recommended MVP

The first real implementation slice should be deliberately narrow:

1. Gaussian responses only.
2. Long-format rows only.
3. One grouping factor.
4. Random intercepts first.
5. `basis_identity()` and `shared_basis(k)` only.
6. `perm_test()` support for:
   - global trace test,
   - sequential rank test.
7. `bootstrap()` support for subject-level stability.

This is enough to validate the framework without overcommitting to every covariance structure immediately.

## Main Risks And Open Questions

### 1. Exchangeability Is Effect-Specific

Whole-subject permutation is fine for between-subject effects, but repeated-measures terms need restricted schemes. This should be encoded explicitly, not inferred ad hoc inside `perm_test()`.

### 2. Shared Basis Estimation Can Leak Information

If the shared basis is estimated once from observed data, sequential permutation tests can become anti-conservative. The conservative default is:

- estimate basis from reduced-model or nuisance-adjusted data,
- optionally re-estimate within each permutation when calibration matters more than speed.

This tradeoff should be user-visible.

### 3. Mixed-Model Row Covariance Is Shared Across Features

The framework assumes one row-side covariance geometry across all features. That is reasonable and useful, but it should be stated explicitly in documentation because some users may expect feature-specific residual structures.

### 4. Missingness Needs A Policy

The formulation handles imbalance and missing rows naturally on the row side, but the first implementation still needs a clear policy for:

- missing features within a row,
- dropped rows due to formula evaluation,
- permutation eligibility under partially observed repeated-measures trajectories.

### 5. Naming Matters

The conceptual pitch is strongest when the package does not fragment into legacy method names again. The note should therefore be implemented with:

- one top-level fit function,
- one effect object,
- one inference engine.

## Recommended Decision

Build this in `multivarious` as:

- `mixed_regress()` returning `mixed_regress_fit`,
- `effect()` returning `effect_operator`,
- `perm_test()` and `bootstrap()` as the main inferential verbs,
- `components()` and `reconstruct()` as the interpretive verbs.

That preserves the package's current architecture and gives the unification you want:

- not three methods,
- one geometry with three queries.

## Sources

These are the main implementation references for the first pass:

- `multivarious` source files:
  - `R/projector.R`
  - `R/bi_projector.R`
  - `R/all_generic.R`
  - `R/pca.R`
  - `R/plsc_inference.R`
- `lme4` package site and vignettes:
  - https://lme4.github.io/lme4/
  - https://lme4.github.io/lme4/articles/lmer.html
- `pbkrtest` CRAN page:
  - https://cran.r-project.org/web/packages/pbkrtest/index.html
- `permuco` package documentation:
  - https://jaromilfrossard.github.io/permuco/
  - https://search.r-project.org/CRAN/refmans/permuco/html/00Index.html
- `lmeresampler` CRAN page:
  - https://cran.r-project.org/web/packages/lmeresampler/index.html
