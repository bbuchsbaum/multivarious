# Principal Components Analysis (PCA)

Compute the directions of maximal variance in a data matrix using the
Singular Value Decomposition (SVD).

## Usage

``` r
pca(
  X,
  ncomp = min(dim(X)),
  preproc = center(),
  method = c("fast", "base", "irlba", "propack", "rsvd", "svds"),
  ...
)
```

## Arguments

- X:

  The data matrix.

- ncomp:

  The number of requested components to estimate (default is the minimum
  dimension of the data matrix).

- preproc:

  The pre-processing function to apply to the data matrix (default is
  centering).

- method:

  The SVD method to use, passed to `svd_wrapper` (default is "fast").

- ...:

  Extra arguments to send to `svd_wrapper`.

## Value

A `bi_projector` object containing the PCA results.

## See also

[`svd_wrapper`](https://bbuchsbaum.github.io/multivarious/reference/svd_wrapper.md)
for details on SVD methods.

## Examples

``` r
data(iris)
X <- as.matrix(iris[, 1:4])
res <- pca(X, ncomp = 4)
tres <- truncate(res, 3)
```
