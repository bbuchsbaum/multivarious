# Coefficients for a Multiblock Projector

Extracts the components (loadings) for a given block or the entire
projector.

## Usage

``` r
# S3 method for class 'multiblock_projector'
coef(object, block, ...)
```

## Arguments

- object:

  A `multiblock_projector` object.

- block:

  Optional block index. If missing, returns loadings for all variables.

- ...:

  Additional arguments.

## Value

A matrix of loadings.
