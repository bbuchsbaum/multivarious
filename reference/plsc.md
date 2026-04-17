# Partial Least Squares Correlation (PLSC)

Reference implementation of symmetric brain-behavior PLS (a.k.a.
Behavior PLSC). It finds paired weight vectors for X and Y that maximize
their cross-block covariance, obtained from the SVD of the
cross-covariance (or correlation) matrix \\C\_{XY} = X^\top Y / (n-1)\\.

## Usage

``` r
plsc(
  X,
  Y,
  ncomp = NULL,
  preproc_x = standardize(),
  preproc_y = standardize(),
  ...
)
```

## Arguments

- X:

  Numeric matrix of predictors (n x p_x).

- Y:

  Numeric matrix of outcomes/behaviors (n x p_y). Must have the same
  number of rows as `X`.

- ncomp:

  Number of latent variables to return. Defaults to
  `min(nrow(X), ncol(X), ncol(Y))`.

- preproc_x:

  Preprocessor for the X block (default:
  [`standardize()`](https://bbuchsbaum.github.io/multivarious/reference/standardize.md)).
  Use
  [`center()`](https://bbuchsbaum.github.io/multivarious/reference/center.md)
  if you want covariance-based PLSC instead of correlation.

- preproc_y:

  Preprocessor for the Y block (default:
  [`standardize()`](https://bbuchsbaum.github.io/multivarious/reference/standardize.md)).

- ...:

  Extra arguments stored on the returned object.

## Value

A `cross_projector` with class `"plsc"` containing

- `vx`, `vy`: X and Y loading/weight matrices.

- `sx`, `sy`: subject scores for X and Y blocks.

- `singvals`: singular values of \\C\_{XY}\\ (strength of each LV).

- `explained_cov`: proportion of cross-block covariance per LV.

- `preproc_x`, `preproc_y`: fitted preprocessors for reuse.

## Examples

``` r
set.seed(1)
X <- matrix(rnorm(80), 20, 4)
Y <- matrix(rnorm(60), 20, 3)
fit <- plsc(X, Y, ncomp = 3)
fit$singvals
#> [1] 0.53051217 0.34069860 0.06144928
```
