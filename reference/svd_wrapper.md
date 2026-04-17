# Singular Value Decomposition (SVD) Wrapper

Computes the singular value decomposition of a matrix using one of the
specified methods. It is designed to be an easy-to-use wrapper for
various SVD methods available in R.

## Usage

``` r
svd_wrapper(
  X,
  ncomp = min(dim(X)),
  preproc = pass(),
  method = c("fast", "base", "irlba", "propack", "rsvd", "svds"),
  q = 2,
  p = 10,
  tol = .Machine$double.eps,
  ...
)
```

## Arguments

- X:

  the input matrix

- ncomp:

  the number of components to estimate (default: min(dim(X)))

- preproc:

  the pre-processor to apply on the input matrix (e.g.,
  [`center()`](https://bbuchsbaum.github.io/multivarious/reference/center.md),
  [`standardize()`](https://bbuchsbaum.github.io/multivarious/reference/standardize.md),
  [`pass()`](https://bbuchsbaum.github.io/multivarious/reference/pass.md))
  Can be a `prepper` object or a pre-processing function.

- method:

  the SVD method to use: 'base', 'fast', 'irlba', 'propack', 'rsvd', or
  'svds'

- q:

  parameter passed to method `rsvd` (default: 2)

- p:

  parameter passed to method `rsvd` (default: 10)

- tol:

  minimum relative tolerance for dropping singular values (compared to
  the largest). Default: `.Machine$double.eps`.

- ...:

  extra arguments passed to the selected SVD function

## Value

an SVD object that extends `bi_projector`

## Examples

``` r
# Load iris dataset and select the first four columns
data(iris)
X <- as.matrix(iris[, 1:4])

# Compute SVD using the base method and 3 components
fit <- svd_wrapper(X, ncomp = 3, preproc = center(), method = "base")
```
