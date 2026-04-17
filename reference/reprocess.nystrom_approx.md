# Reprocess data for Nyström approximation

Apply preprocessing to new data for projection using a Nyström
approximation. This method overrides the default `reprocess.projector`
to handle the fact that Nyström components are in kernel space (not
feature space).

## Usage

``` r
# S3 method for class 'nystrom_approx'
reprocess(x, new_data, colind = NULL, ...)
```

## Arguments

- x:

  A `nystrom_approx` object

- new_data:

  A matrix with the same number of columns as the original training data

- colind:

  Optional column indices (not typically used for Nyström)

- ...:

  Additional arguments (ignored)

## Value

Preprocessed data matrix
