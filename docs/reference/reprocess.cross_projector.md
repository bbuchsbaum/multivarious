# reprocess a cross_projector instance

reprocess a cross_projector instance

## Usage

``` r
# S3 method for class 'cross_projector'
reprocess(x, new_data, colind = NULL, source = c("X", "Y"), ...)
```

## Arguments

- x:

  the model fit object

- new_data:

  the new data to process

- colind:

  the column indices of the new data

- source:

  the source of the data (X or Y block)

- ...:

  extra args

## Value

the re(pre-)processed data

## Details

When `colind` is provided, each index is validated to be within the
available coefficient rows using
[`chk::chk_subset`](https://poissonconsulting.github.io/chk/reference/chk_subset.html).
