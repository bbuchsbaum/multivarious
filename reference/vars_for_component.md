# Identify Original Variables for a Specific Component

Determines which columns from the *original* input space contribute
(have non-zero influence) to a *specific* output component of the
projector.

## Usage

``` r
vars_for_component(x, k, ...)

# S3 method for class 'composed_projector'
vars_for_component(x, k, tol = 1e-08, ...)
```

## Arguments

- x:

  A projector object (e.g., `projector`, `composed_projector`).

- k:

  The index of the output component to query.

- ...:

  Additional arguments passed to specific methods.

- tol:

  Numeric tolerance for determining non-zero coefficients. Default is
  1e-8 for some methods. Passed via `...`.

## Value

A sorted numeric vector of unique indices corresponding to the original
input variables.
