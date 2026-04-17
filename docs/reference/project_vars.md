# Project one or more variables onto a subspace

This function projects one or more variables onto a subspace. It is
often called supplementary variable projection and can be computed for a
biorthogonal decomposition, such as Singular Value Decomposition (SVD).

## Usage

``` r
project_vars(x, new_data, ...)
```

## Arguments

- x:

  The model fit, typically an object of a class that implements a
  `project_vars` method

- new_data:

  A matrix or vector of new observation(s) with the same number of rows
  as the original data

- ...:

  Additional arguments passed to the underlying `project_vars` method

## Value

A matrix or vector of the projected variables in the subspace

## See also

[`project`](https://bbuchsbaum.github.io/multivarious/reference/project.md)
for the generic projection function for samples

Other project:
[`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md),
[`project.cross_projector()`](https://bbuchsbaum.github.io/multivarious/reference/project.cross_projector.md),
[`project_block()`](https://bbuchsbaum.github.io/multivarious/reference/project_block.md)
