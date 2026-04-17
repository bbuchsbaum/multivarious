# Compute reconstruction-based error metrics

Given two numeric matrices `Xtrue` and `Xrec`, compute:

- MSE (`"mse"`)

- RMSE (`"rmse"`)

- R^2 (`"r2"`)

- MAE (`"mae"`)

## Usage

``` r
measure_reconstruction_error(
  Xtrue,
  Xrec,
  metrics = c("mse", "rmse", "r2"),
  by_column = FALSE
)
```

## Arguments

- Xtrue:

  Original data matrix, shape (n x p).

- Xrec:

  Reconstructed data matrix, shape (n x p).

- metrics:

  Character vector of metric names, e.g. `c("mse","rmse","r2","mae")`.

- by_column:

  Logical, if TRUE calculate R2 metric per column and average (default:
  FALSE).

## Value

A one-row `tibble` with columns matching `metrics`.
