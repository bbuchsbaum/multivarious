# Reconstruct the data

Reconstruct a data set from its (possibly) low-rank representation. This
can be useful when analyzing the impact of dimensionality reduction or
when visualizing approximations of the original data.

## Usage

``` r
reconstruct(x, ...)
```

## Arguments

- x:

  The model fit, typically an object of a class that implements a
  `reconstruct` method

- ...:

  Additional arguments passed to specific methods. Common parameters
  include:

  `comp`

  :   A vector of component indices to use in the reconstruction

  `rowind`

  :   The row indices to reconstruct (optional)

  `colind`

  :   The column indices to reconstruct (optional)

  `scores`

  :   (For `composed_projector` only) A numeric matrix of scores to
      reconstruct from

## Value

A reconstructed data set based on the selected components, rows, and
columns

## See also

[`bi_projector`](https://bbuchsbaum.github.io/multivarious/reference/bi_projector.md)
for an example of a two-way mapping model that can be reconstructed

Other reconstruct:
[`reconstruct_new()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct_new.md)
