## Resubmission (0.3.1)

This is a patch release to fix issues discovered after submitting version 0.3.0:

* **Bug fix**: `reconstruct_new.bi_projector()` had a double-preprocessing bug that caused incorrect results when reconstructing held-out data. This affected cross-validation workflows.

* **Vignette corrections**: Several vignettes contained broken or misleading examples that we discovered after submission. The CrossValidation vignette had non-working code that produced warnings/errors. Other vignettes had excessive commented-out code and unclear explanations. All vignettes have been reviewed and corrected.

* **Added regression tests** to prevent the `reconstruct_new()` bug from recurring.

We apologize for the oversight in 0.3.0. The vignettes were not sufficiently tested before submission.

## Previous submission (0.3.0)

* Fixed T/F shorthand usage to TRUE/FALSE for CRAN compliance
* Converted `\dontrun{}` to `\donttest{}` for executable examples
* Fixed `bootstrap.plsc()` duplicate argument handling
* Fixed S3 method registration (classifier.projector, inverse_projection.projector, perm_ci.pca)
* Added missing `importFrom` directives for `coefficients` and `combn`
* Fixed vignette YAML headers to use standard multi-line format
* Fixed `requireNamespace()` parameter from `quiet` to `quietly` (R-devel strict checking)
* Removed escaped underscores from plsc.R documentation
* Removed albersdown theme references from vignettes
* Created NEWS.md documenting all changes

## R CMD check results

0 errors | 0 warnings | 0 notes

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
* win-builder: Windows Server 2022, R-release
* win-builder: Windows Server 2022, R-devel
* mac-builder: macOS, R-release (Apple Silicon)
