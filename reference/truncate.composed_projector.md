# Truncate a Composed Projector

Reduces the number of output components of the composed projector by
truncating the *last* stage in the sequence.

## Usage

``` r
# S3 method for class 'composed_projector'
truncate(x, ncomp, ...)
```

## Arguments

- x:

  A `composed_projector` object.

- ncomp:

  The desired number of final output components.

- ...:

  Currently unused.

## Value

A new `composed_projector` object with the last stage truncated.

## Details

Note: This implementation currently only supports truncating the final
stage. Truncating intermediate stages would require re-computing
subsequent stages or combined attributes and is not yet implemented.
