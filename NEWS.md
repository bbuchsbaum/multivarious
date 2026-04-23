# multivarious 0.3.1

## Behavior Changes

* Changed `partial_project()` default `least_squares` from `TRUE` to `FALSE`.
* Changed `project_block.multiblock_projector()` default `least_squares` from `TRUE` to `FALSE` (now matches `partial_project()`).

## Bug Fixes

* Fixed `reconstruct_new.bi_projector()` double-preprocessing bug that caused incorrect reconstruction when applied to held-out data.
* Fixed blockwise preprocessing paths that could allocate very large temporary matrices with multiblock data, and hardened `standardize()` for missing and zero-variance columns.

## Vignette Improvements

* Rewrote CrossValidation vignette with working examples (fixed broken `reconstruct()` usage and results extraction).
* Cleaned up PermutationTesting vignette: improved structure, replaced dense tables with readable prose.
* Cleaned up Regress vignette: broke up long code block into focused subsections.
* Cleaned up Extending vignette: removed commented-out code walls, simplified examples.

## Tests

* Added regression tests for `reconstruct_new()` on held-out data.

# multivarious 0.3.0

## Bug Fixes

* Fixed T/F shorthand to TRUE/FALSE in `pca()` for CRAN compliance.
* Converted `\dontrun{}` to `\donttest{}` for executable but slow examples.
* Fixed `bootstrap.plsc()` duplicate argument handling when called with named X/Y arguments.
* Fixed `regress()` PLS method dimension mismatch.
* Fixed iris data frame to matrix conversion in examples.

## Internal Changes

* Registered S3 methods: `classifier.projector`, `inverse_projection.projector`, `perm_ci.pca`.
* Added missing `importFrom` directives for `coefficients` and `combn`.
* Replaced non-ASCII characters with ASCII equivalents in documentation.

## Deprecated

* `prep()` is deprecated in favor of `fit()` for preprocessing pipelines.
* `perm_ci.pca()` is deprecated.
* `perm_test.plsc()` is deprecated.

# multivarious 0.2.0

* Initial CRAN release.
