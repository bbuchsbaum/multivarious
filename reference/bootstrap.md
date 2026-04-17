# Bootstrap Resampling for Multivariate Models

Perform bootstrap resampling on a multivariate model to estimate the
variability of components and scores.

## Usage

``` r
bootstrap(x, nboot, ...)

# S3 method for class 'plsc'
bootstrap(x, nboot = 500, ...)
```

## Arguments

- x:

  A fitted model object, such as a `projector`, that has been fit to a
  training dataset.

- nboot:

  An integer specifying the number of bootstrap resamples to perform.

- ...:

  Additional arguments to be passed to the specific model implementation
  of `bootstrap`.

## Value

A list containing the bootstrap resampled components and scores for the
model.
