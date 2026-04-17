# Default inverse_projection method for cross_projector

This function obtains the matrix that maps factor scores in the latent
space back into the original domain (X or Y). By default, we assume
`v_domain` is *not* necessarily orthonormal or invertible, so we use a
pseudoinverse approach (e.g. MASS::ginv).

## Usage

``` r
# S3 method for class 'cross_projector'
inverse_projection(x, domain = c("X", "Y"), ...)
```

## Arguments

- x:

  A `cross_projector` object.

- domain:

  Either `"X"` or `"Y"`, indicating which block's inverse loading matrix
  we want (i.e., if you want to reconstruct data in the X space or Y
  space).

- ...:

  Additional arguments (currently unused, but may be used by
  subclasses).

## Value

A matrix that, when multiplied by the factor scores, yields the
reconstruction in the specified domain's original space.

## Examples

``` r
# Suppose 'cp' is a cross_projector object. If we want the
# inverse for the Y domain:
#   inv_mat <- inverse_projection(cp, domain="Y")
# Then reconstruct:  Yhat <- Fscores %*% inv_mat
```
