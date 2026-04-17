# Compose Multiple Partial Projectors

Creates a `composed_partial_projector` object that applies partial
projections sequentially. If multiple projectors are composed, the
column indices (colind) used at each stage must be considered.

This infix operator provides syntactic sugar for composing projectors
sequentially. It is an alias for `compose_partial_projector`.

## Usage

``` r
compose_partial_projector(...)

lhs %>>% rhs
```

## Arguments

- ...:

  A sequence of projectors that implement
  [`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md),
  optionally named.

- lhs:

  The left-hand side projector (or a composed projector).

- rhs:

  The right-hand side projector to add to the sequence.

## Value

A `composed_partial_projector` object.

A `composed_partial_projector` object representing the combined
sequence.
