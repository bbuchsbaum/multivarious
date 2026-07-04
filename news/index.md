# Changelog

## multivarious 0.4.0

### Sparse preprocessing

- Added sparse-aware preprocessing for `Matrix` inputs.
  [`pass()`](https://bbuchsbaum.github.io/multivarious/reference/pass.md)
  and sparse-safe
  [`colscale()`](https://bbuchsbaum.github.io/multivarious/reference/colscale.md)
  transformations preserve sparse storage, while
  [`center()`](https://bbuchsbaum.github.io/multivarious/reference/center.md)
  and
  [`standardize()`](https://bbuchsbaum.github.io/multivarious/reference/standardize.md)
  now use a lazy affine representation instead of materializing dense
  centered matrices by default.
- Added explicit sparse preprocessing policies: `sparse = "operator"`
  uses the lazy representation, `sparse = "preserve"` errors before a
  densifying transform, and `sparse = "allow_dense"` opts into dense
  materialization.
- Added matrix-like S4 classes for lazy sparse affine data and its
  transpose, with methods for `%*%`,
  [`crossprod()`](https://rdrr.io/r/base/crossprod.html),
  [`tcrossprod()`](https://rdrr.io/r/base/crossprod.html),
  [`t()`](https://rdrr.io/r/base/t.html), subsetting, dimensions, and
  [`as.matrix()`](https://rdrr.io/r/base/matrix.html). Downstream code
  can now multiply or form cross-products from transformed sparse data
  without first densifying it.
- Updated
  [`svd_wrapper()`](https://bbuchsbaum.github.io/multivarious/reference/svd_wrapper.md)
  and
  [`pca()`](https://bbuchsbaum.github.io/multivarious/reference/pca.md)
  sparse/lazy paths to use implicit sparse SVD backends where possible,
  avoiding dense centered intermediates for common sparse PCA workflows.
- Updated projector and partial-projection paths to multiply through
  lazy sparse preprocessing views directly.

### Behavior changes

- [`transform()`](https://bbuchsbaum.github.io/multivarious/reference/transform.md)
  on a sparse input with centering or standardization may now return a
  `sparse_affine_matrix` object. Use regular matrix algebra methods on
  that object to stay lazy, or call
  [`as.matrix()`](https://rdrr.io/r/base/matrix.html) when explicit
  dense materialization is desired.

### Tests

- Added regression coverage for sparse preprocessing policies,
  sparse-safe scaling, lazy centered and standardized sparse transforms,
  implicit sparse PCA, projector multiplication, and public algebra
  methods on the lazy sparse affine classes.

## multivarious 0.3.2

CRAN release: 2026-06-05

### Dependencies

- Removed the optional `PRIMME` backend from
  [`geneig()`](https://bbuchsbaum.github.io/multivarious/reference/geneig.md)
  and
  [`cPCAplus()`](https://bbuchsbaum.github.io/multivarious/reference/cPCAplus.md)
  (and dropped `PRIMME` from `Suggests`), as the `PRIMME` package is
  scheduled for archival on CRAN. The iterative `"rspectra"` and
  `"subspace"` backends and the dense `"geigen"`/`"robust"`/`"sdiag"`
  backends cover the same generalized eigenproblems.

### Mixed-effect inference

- Added explicit `term_scopes` and `exchangeability` overrides to
  [`mixed_regress()`](https://bbuchsbaum.github.io/multivarious/reference/mixed_regress.md),
  with exchangeability metadata now carried through
  [`summary()`](https://rdrr.io/r/base/summary.html),
  [`effect()`](https://bbuchsbaum.github.io/multivarious/reference/effect.md),
  and effect-operator printing.
- Fixed grouped row-metric whitening/unwhitening for random-effect
  designs by using the Cholesky orientation implied by the row metric.
- Hardened
  [`perm_test.effect_operator()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.effect_operator.md)
  by making the supported one-sided alternative explicit, preserving
  seed metadata, honoring explicit exchangeability schemes, and using a
  fixed statistic family for each sequential permutation step.
- Improved
  [`bootstrap.effect_operator()`](https://bbuchsbaum.github.io/multivarious/reference/bootstrap.effect_operator.md)
  for grouped designs by relabeling duplicated bootstrap clusters as
  distinct resampled groups, aligning multi-component loadings with an
  orthogonal Procrustes rotation, and preserving seed metadata.

### Tests

- Added regression tests for grouped whitening, explicit term-scope and
  exchangeability overrides, effect-operator permutation statistic
  selection, subject bootstrap relabeling, seed metadata, and PRIMME
  removal.

## multivarious 0.3.1

CRAN release: 2026-01-21

### Behavior Changes

- Changed
  [`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)
  default `least_squares` from `TRUE` to `FALSE`.
- Changed
  [`project_block.multiblock_projector()`](https://bbuchsbaum.github.io/multivarious/reference/project_block.multiblock_projector.md)
  default `least_squares` from `TRUE` to `FALSE` (now matches
  [`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)).

### Bug Fixes

- Fixed `reconstruct_new.bi_projector()` double-preprocessing bug that
  caused incorrect reconstruction when applied to held-out data.
- Fixed blockwise preprocessing paths that could allocate very large
  temporary matrices with multiblock data, and hardened
  [`standardize()`](https://bbuchsbaum.github.io/multivarious/reference/standardize.md)
  for missing and zero-variance columns.

### Vignette Improvements

- Rewrote CrossValidation vignette with working examples (fixed broken
  [`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
  usage and results extraction).
- Cleaned up PermutationTesting vignette: improved structure, replaced
  dense tables with readable prose.
- Cleaned up Regress vignette: broke up long code block into focused
  subsections.
- Cleaned up Extending vignette: removed commented-out code walls,
  simplified examples.

### Tests

- Added regression tests for
  [`reconstruct_new()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct_new.md)
  on held-out data.

## multivarious 0.3.0

CRAN release: 2026-01-21

### Bug Fixes

- Fixed T/F shorthand to TRUE/FALSE in
  [`pca()`](https://bbuchsbaum.github.io/multivarious/reference/pca.md)
  for CRAN compliance.
- Converted `\dontrun{}` to `\donttest{}` for executable but slow
  examples.
- Fixed
  [`bootstrap.plsc()`](https://bbuchsbaum.github.io/multivarious/reference/bootstrap.md)
  duplicate argument handling when called with named X/Y arguments.
- Fixed
  [`regress()`](https://bbuchsbaum.github.io/multivarious/reference/regress.md)
  PLS method dimension mismatch.
- Fixed iris data frame to matrix conversion in examples.

### Internal Changes

- Registered S3 methods: `classifier.projector`,
  `inverse_projection.projector`, `perm_ci.pca`.
- Added missing `importFrom` directives for `coefficients` and `combn`.
- Replaced non-ASCII characters with ASCII equivalents in documentation.

### Deprecated

- [`prep()`](https://bbuchsbaum.github.io/multivarious/reference/prep.md)
  is deprecated in favor of
  [`fit()`](https://bbuchsbaum.github.io/multivarious/reference/fit.md)
  for preprocessing pipelines.
- [`perm_ci.pca()`](https://bbuchsbaum.github.io/multivarious/reference/perm_ci.md)
  is deprecated.
- [`perm_test.plsc()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.plsc.md)
  is deprecated.

## multivarious 0.2.0

CRAN release: 2024-03-28

- Initial CRAN release.
