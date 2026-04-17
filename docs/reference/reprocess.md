# apply pre-processing parameters to a new data matrix

Given a new dataset, process it in the same way the original data was
processed (e.g. centering, scaling, etc.)

## Usage

``` r
reprocess(x, new_data, colind, ...)
```

## Arguments

- x:

  the model fit object

- new_data:

  the new data to process

- colind:

  the column indices of the new data

- ...:

  extra args

## Value

the reprocessed data
