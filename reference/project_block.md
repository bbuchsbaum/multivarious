# Project a single "block" of data onto the subspace

When observations are concatenated into "blocks", it may be useful to
project one block from the set. This function facilitates the projection
of a specific block of data onto a subspace. It is a convenience method
for multi-block fits and is equivalent to a "partial projection" where
the column indices are associated with a given block.

## Usage

``` r
project_block(x, new_data, block, least_squares, ...)
```

## Arguments

- x:

  The model fit, typically an object of a class that implements a
  `project_block` method

- new_data:

  A matrix or vector of new observation(s) with the same number of
  columns as the original data

- block:

  An integer representing the block ID to select in the block projection
  matrix. This ID corresponds to the specific block of data to be
  projected

- least_squares:

  Logical. If `TRUE` use least squares projection.

- ...:

  Additional arguments passed to the underlying `project_block` method

## Value

A matrix or vector of the projected data for the specified block

## See also

[`project`](https://bbuchsbaum.github.io/multivarious/reference/project.md)
for the generic projection function

Other project:
[`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md),
[`project.cross_projector()`](https://bbuchsbaum.github.io/multivarious/reference/project.cross_projector.md),
[`project_vars()`](https://bbuchsbaum.github.io/multivarious/reference/project_vars.md)
