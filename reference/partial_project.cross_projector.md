# Partially project data for a cross_projector

Projects new data from either the X or Y domain onto the latent
subspace, considering only a specified subset of original features
(`colind`).

## Usage

``` r
# S3 method for class 'cross_projector'
partial_project(
  x,
  new_data,
  colind,
  least_squares = FALSE,
  lambda = 1e-06,
  source = c("X", "Y"),
  ...
)
```

## Arguments

- x:

  A `cross_projector` object.

- new_data:

  A numeric matrix (n x length(colind)) or vector, representing the
  observations corresponding to the columns specified by `colind`.

- colind:

  A numeric vector of column indices in the original data space (either
  X or Y domain, specified by `source`) that correspond to `new_data`'s
  columns.

- least_squares:

  Logical; if TRUE, use ridge-regularized least squares for projection
  (default FALSE).

- lambda:

  Numeric; ridge penalty (default 1e-6). Ignored if
  `least_squares=FALSE`.

- source:

  Character, either "X" or "Y", indicating which domain `new_data` and
  `colind` belong to.

- ...:

  Additional arguments (currently ignored).

## Value

A numeric matrix (n x d) of factor scores in the latent subspace.
