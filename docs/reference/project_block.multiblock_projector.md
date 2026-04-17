# Project Data onto a Specific Block

Projects the new data onto the subspace defined by a specific block of
variables.

## Usage

``` r
# S3 method for class 'multiblock_projector'
project_block(x, new_data, block, least_squares = FALSE, ...)
```

## Arguments

- x:

  A `multiblock_projector` object.

- new_data:

  The new data to be projected.

- block:

  The block index (1-based) to project onto.

- least_squares:

  Logical. If `TRUE`, use least squares projection (default FALSE).

- ...:

  Additional arguments passed to `partial_project`.

## Value

The projected scores for the specified block.
