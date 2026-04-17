# Identify Original Variables Used by a Projector

Determines which columns from the *original* input space contribute
(have non-zero influence) to *any* of the output components of the
projector.

## Usage

``` r
variables_used(x, ...)

# S3 method for class 'composed_projector'
variables_used(x, tol = 1e-08, ...)
```

## Arguments

- x:

  A projector object (e.g., `projector`, `composed_projector`).

- ...:

  Additional arguments passed to specific methods.

- tol:

  Numeric tolerance for determining non-zero coefficients. Default is
  1e-8 for some methods. Passed via `...`.

## Value

A sorted numeric vector of unique indices corresponding to the original
input variables.
