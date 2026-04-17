# Get Coefficients of a Composed Projector

Calculates the effective coefficient matrix that maps from the original
input space (of the first projector) to the final output space (of the
last projector). This is done by multiplying the coefficient matrices of
all projectors in the sequence.

## Usage

``` r
# S3 method for class 'composed_projector'
coef(object, ...)
```

## Arguments

- object:

  A `composed_projector` object.

- ...:

  Currently unused.

## Value

A matrix representing the combined coefficients.
