# Compute inter-block transfer error metrics for a cross_projector

We measure how well the model can transfer from X-\>Y or Y-\>X, e.g.
"x2y.mse".

## Usage

``` r
measure_interblock_transfer_error(Xtrue, Ytrue, model, metrics = c("x2y.mse"))
```

## Arguments

- Xtrue:

  The X block test data

- Ytrue:

  The Y block test data

- model:

  The fitted `cross_projector`

- metrics:

  A character vector like `c("x2y.mse","y2x.r2")`

## Value

A 1-row tibble with columns for each requested metric

## Details

The metric names are of the form "x2y.mse", "x2y.rmse", "y2x.r2", etc.
