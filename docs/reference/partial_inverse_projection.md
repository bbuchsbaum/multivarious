# Partial Inverse Projection of a Columnwise Subset of Component Matrix

Compute the inverse projection of a columnwise subset of the component
matrix (e.g., a sub-block). Even when the full component matrix is
orthogonal, there is no guarantee that the partial component matrix is
orthogonal.

## Usage

``` r
partial_inverse_projection(x, colind, ...)
```

## Arguments

- x:

  A fitted model object, such as a `projector`, that has been fit to a
  dataset.

- colind:

  A numeric vector specifying the column indices of the component matrix
  to consider for the partial inverse projection.

- ...:

  Additional arguments to be passed to the specific model implementation
  of `partial_inverse_projection`.

## Value

A matrix representing the partial inverse projection.
