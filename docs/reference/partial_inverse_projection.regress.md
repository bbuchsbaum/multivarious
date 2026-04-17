# Partial Inverse Projection for a `regress` Object

This function computes a sub-block inversion of the regression
coefficients, allowing you to focus on only certain columns (e.g.
partial factors). If your coefficient matrix is not orthonormal or is
not square, we use a pseudoinverse approach (via
[`corpcor::pseudoinverse`](https://rdrr.io/pkg/corpcor/man/pseudoinverse.html))
to find a minimal-norm solution.

## Usage

``` r
# S3 method for class 'regress'
partial_inverse_projection(x, colind, ...)
```

## Arguments

- x:

  A `regress` object (created by
  [`regress`](https://bbuchsbaum.github.io/multivarious/reference/regress.md)).

- colind:

  A numeric vector specifying which columns of the *factor space* (i.e.,
  the second dimension of `x$coefficients`) you want to invert.
  Typically these refer to a subset of canonical / PCA / PLS components.

- ...:

  Further arguments passed to or used by methods (not used here).

## Value

A matrix of shape `(length(colind) x nrow(x$coefficients))`. When
multiplied by partial factor scores `(n x length(colind))`, it yields an
`(n x nrow(x$coefficients))` reconstruction in the original domain.
