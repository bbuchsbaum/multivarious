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
