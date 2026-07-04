# Partial Project Through a Composed Partial Projector

Applies
[`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)
through each projector in the composition. If `colind` is a single
vector, it applies to the first projector only. Subsequent projectors
apply full columns. If `colind` is a list, only the first entry is used;
non-`NULL` entries after the first currently warn and are ignored.

## Usage

``` r
# S3 method for class 'composed_partial_projector'
partial_project(x, new_data, colind = NULL, ...)
```

## Arguments

- x:

  A `composed_partial_projector` object.

- new_data:

  The input data matrix or vector.

- colind:

  A numeric vector or a list of numeric vectors/NULLs. A single vector
  applies to the first projector only. If a list is supplied, only
  `colind[[1]]` is used; non-`NULL` entries after the first warn and are
  ignored. Later stages receive the full output from the preceding
  stage.

- ...:

  Additional arguments passed to
  [`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)
  or
  [`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md)
  methods.

## Value

The partially projected data after all projectors are applied.
