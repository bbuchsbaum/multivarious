# Partial Project Through a Composed Partial Projector

Applies
[`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)
through each projector in the composition. If `colind` is a single
vector, it applies to the first projector only. Subsequent projectors
apply full columns. If `colind` is a list, each element specifies the
`colind` for the corresponding projector in the chain.

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

  A numeric vector or a list of numeric vectors/NULLs. If a single
  vector, applies to the first projector only. If a list, its length
  should ideally match the number of projectors. `colind[[i]]` specifies
  the column indices (relative to the *input* of stage `i`) to use for
  the partial projection at stage `i`. A `NULL` entry means use full
  projection for that stage. If the list is shorter than the number of
  stages, `NULL` (full projection) is assumed for remaining stages. If a
  single numeric vector is provided, it is treated as
  `list(colind, NULL, NULL, ...)` for backward compatibility (partial
  only at first stage).

- ...:

  Additional arguments passed to
  [`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)
  or
  [`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md)
  methods.

## Value

The partially projected data after all projectors are applied.
