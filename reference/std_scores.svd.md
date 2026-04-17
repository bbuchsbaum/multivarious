# Calculate Standardized Scores for SVD results

Computes standardized scores from an SVD result performed by
`svd_wrapper`. These scores are scaled to have approximately unit
variance, assuming the original data used for SVD was centered. They
differ from the `s` component of the `svd` object, which contains scores
scaled by singular values.

## Usage

``` r
# S3 method for class 'svd'
std_scores(x, ...)
```

## Arguments

- x:

  An object of class `svd`, typically from `svd_wrapper`.

- ...:

  Extra arguments (ignored).

## Value

A matrix of standardized scores (N x k) with columns having variance
close to 1.
