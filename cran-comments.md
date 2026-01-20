## Resubmission

This is a resubmission. In this version I have:

* Fixed T/F shorthand usage to TRUE/FALSE for CRAN compliance
* Converted `\dontrun{}` to `\donttest{}` for executable examples
* Fixed `bootstrap.plsc()` duplicate argument handling
* Fixed S3 method registration (classifier.projector, inverse_projection.projector, perm_ci.pca)
* Added missing `importFrom` directives for `coefficients` and `combn`
* Fixed vignette YAML headers to use standard multi-line format
* Created NEWS.md documenting all changes

## R CMD check results

0 errors | 0 warnings | 2 notes

* NOTE: Escaped LaTeX specials in plsc.Rd
  - Escaped underscores are intentional for correct rendering of variable names with subscripts (e.g., X_scores, Y_loadings).

* NOTE: Unstated dependencies in vignettes - 'albersdown'
  - albersdown is used conditionally for vignette styling. It is listed in Config/Needs/website, not as a runtime dependency. The vignettes check for its availability with `requireNamespace()` before use.

## Package dependencies

This package imports from 27 non-default packages. This package provides a comprehensive framework for multivariate analysis methods, and each import is actively used:

- **Matrix**: Sparse matrix operations for efficient computation
- **RSpectra/irlba/PRIMME/rsvd**: Efficient eigendecomposition for large matrices
- **pls**: Partial least squares algorithms
- **glmnet**: Regularized regression methods
- **corpcor**: Correlation and covariance estimation
- **future.apply/future**: Parallel processing support
- **geigen/GPArotation**: Generalized eigenvalue problems and factor rotation
- **ggplot2/ggrepel**: Visualization methods
- **matrixStats**: Efficient row/column statistics
- **proxy**: Distance and similarity computations
- **dplyr/tibble**: Data manipulation utilities
- **rlang/chk/assertthat/cli/crayon**: Input validation and messaging
- **withr**: Safe temporary state management
- **lifecycle**: Function deprecation support
- **methods/MASS/svd**: Core statistical methods

Reducing imports would require removing core functionality that users depend on.

## Reverse dependencies

There are currently no reverse dependencies for this package.

## Test environments

* Local: macOS Sonoma 14.3, R 4.5.1 (aarch64-apple-darwin20)
