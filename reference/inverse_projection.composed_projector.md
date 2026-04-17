# Compute the Inverse Projection for a Composed Projector

Calculates the pseudo-inverse of the composed projector, mapping from
the final output space back towards the original input space. This is
computed by multiplying the pseudo-inverses of the individual projector
stages in reverse order: `V_k+ %*% ... %*% V_2+ %*% V_1+`.

## Usage

``` r
# S3 method for class 'composed_projector'
inverse_projection(x, ...)
```

## Arguments

- x:

  A `composed_projector` object.

- ...:

  Additional arguments passed to the underlying `inverse_projection`
  methods.

## Value

A matrix representing the combined pseudo-inverse.

## Details

Requires that each stage implements the `inverse_projection` method.
