# Changelog

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
