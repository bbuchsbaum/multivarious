# Compose Two Projectors

Combine two projector models into a single projector by sequentially
applying the first projector and then the second projector.

## Usage

``` r
compose_projector(x, y, ...)
```

## Arguments

- x:

  A fitted model object (e.g., `projector`) that has been fit to a
  dataset and will be applied first in the composition.

- y:

  A second fitted model object (e.g., `projector`) that has been fit to
  a dataset and will be applied after the first projector.

- ...:

  Additional arguments to be passed to the specific model implementation
  of `compose_projector`.

## Value

A new `projector` object representing the composed projector, which can
be used to project data onto the combined subspace.
