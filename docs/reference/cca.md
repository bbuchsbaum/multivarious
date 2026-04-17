# Canonical Correlation Analysis (CCA)

Reference implementation of two-block canonical correlation analysis
that returns a `cross_projector`. The within-block covariance matrices
are ridge-regularized before whitening, which keeps the fit well-defined
even when \\p \> n\\ or either block is rank-deficient.

## Usage

``` r
cca(
  X,
  Y,
  ncomp = NULL,
  preproc_x = center(),
  preproc_y = center(),
  lambda = 1e-04,
  lambda_x = lambda,
  lambda_y = lambda,
  tol = sqrt(.Machine$double.eps),
  ...
)
```

## Arguments

- X:

  Numeric matrix of predictors (n x p_x).

- Y:

  Numeric matrix of outcomes (n x p_y). Must have the same number of
  rows as `X`.

- ncomp:

  Number of canonical dimensions to return. Defaults to
  `min(ncol(X), ncol(Y), nrow(X) - 1)` after preprocessing.

- preproc_x:

  Preprocessor for the X block (default:
  [`center()`](https://bbuchsbaum.github.io/multivarious/reference/center.md)).

- preproc_y:

  Preprocessor for the Y block (default:
  [`center()`](https://bbuchsbaum.github.io/multivarious/reference/center.md)).

- lambda:

  Shared ridge shrinkage level used when `lambda_x` and `lambda_y` are
  not supplied. The effective ridge added to each block is
  `lambda * mean(diag(S))`, where `S` is the block covariance.

- lambda_x:

  Ridge shrinkage level for the X block covariance.

- lambda_y:

  Ridge shrinkage level for the Y block covariance.

- tol:

  Eigenvalue floor used when whitening regularized covariance matrices.
  Defaults to `sqrt(.Machine$double.eps)`.

- ...:

  Extra arguments stored on the returned object.

## Value

A `cross_projector` with class `"cca"` containing canonical coefficients
(`vx`, `vy`), block scores (`sx`, `sy`), canonical correlations (`cor`),
and the regularized block covariance matrices used to fit the model.

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(120), 30, 4)
Y <- matrix(rnorm(90), 30, 3)
fit <- cca(X, Y, ncomp = 2)
fit$cor
#> [1] 0.5485591 0.3176881
```
