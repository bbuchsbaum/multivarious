# Permutation Confidence Intervals

Estimate confidence intervals for model parameters using permutation
testing.

## Usage

``` r
perm_ci(x, X, nperm, ...)

# S3 method for class 'pca'
perm_ci(x, X, nperm = 100, k = 4, distr = "gamma", parallel = FALSE, ...)
```

## Arguments

- x:

  A model fit object.

- X:

  The original data matrix used to fit the model.

- nperm:

  The number of permutations to perform for the confidence interval
  estimation.

- ...:

  Additional arguments to be passed to the specific model implementation
  of `perm_ci`.

- k:

  Number of components to test (default 4).

- distr:

  Distribution assumption (default "gamma"); currently ignored in
  forwarding.

- parallel:

  Logical; if TRUE, use parallel processing.

## Value

A list containing the estimated lower and upper bounds of the confidence
intervals for model parameters.
