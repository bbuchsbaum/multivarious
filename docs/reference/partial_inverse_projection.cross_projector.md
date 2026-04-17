# Partial Inverse Projection of a Subset of the Loading Matrix in cross_projector

This function obtains the "inverse" mapping for a columnwise subset of
the loading matrix in the specified domain. In practice, if `v_mat` is
not orthonormal or not square, we use a pseudoinverse approach (via
[`MASS::ginv`](https://rdrr.io/pkg/MASS/man/ginv.html)).

## Usage

``` r
# S3 method for class 'cross_projector'
partial_inverse_projection(x, colind, domain = c("X", "Y"), ...)
```

## Arguments

- x:

  A `cross_projector` object.

- colind:

  A numeric vector specifying the columns (indices) of the *latent
  factors* or loadings to invert. Typically these correspond to a subset
  of canonical components or principal components, etc.

- domain:

  Either `"X"` or `"Y"`, indicating which block's partial loadings we
  want to invert.

- ...:

  Additional arguments (unused by default, but may be used by
  subclasses).

## Value

A matrix of shape `(length(colind) x p_block)` that, when multiplied by
factor scores restricted to `colind` columns, yields an `(n x p_block)`
reconstruction in the original domain block.

## Details

By default, this is a minimal-norm solution for partial columns of
`v_mat`. If you need a different approach (e.g., ridge, direct solve,
etc.), you can override this method in your specific class or code.

## Examples

``` r
# Suppose 'cp' is a cross_projector, and we want only columns 1:3 of
# the Y block factors. Then:
#   inv_mat_sub <- partial_inverse_projection(cp, colind=1:3, domain="Y")
# The shape will be (3 x pY), so factor_scores_sub (n x 3) %*% inv_mat_sub => (n x pY).
```
