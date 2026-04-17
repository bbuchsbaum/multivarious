# Partially project a new sample onto subspace

Project a selected subset of column indices (`colind`) of `new_data`
onto the subspace defined by the model `x`. Optionally do a
ridge-regularized least-squares solve if columns are non-orthonormal.

## Usage

``` r
partial_project(
  x,
  new_data,
  colind,
  least_squares = FALSE,
  lambda = 1e-06,
  ...
)
```

## Arguments

- x:

  The fitted model, e.g. `bi_projector`, that has a partial_project
  method.

- new_data:

  A numeric matrix (n x length(colind)) or vector, representing the
  observations to be projected.

- colind:

  A numeric vector of column indices in the original data space that
  correspond to `new_data`'s columns.

- least_squares:

  Logical; if TRUE, do a ridge-regularized solve (default FALSE).

- lambda:

  Numeric; ridge penalty (default 1e-6). Ignored if
  `least_squares=FALSE`.

- ...:

  Additional arguments passed to class-specific partial_project methods.

## Value

A numeric matrix (n x d) of factor scores in the model's subspace, for
those columns only.
