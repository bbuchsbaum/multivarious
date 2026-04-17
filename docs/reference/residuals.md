# Obtain residuals of a component model fit

Calculate the residuals of a model after removing the effect of the
first `ncomp` components. This function is useful to assess the quality
of the fit or to identify patterns that are not captured by the model.

## Usage

``` r
residuals(x, ncomp, xorig, ...)
```

## Arguments

- x:

  The model fit object.

- ncomp:

  The number of components to factor out before calculating residuals.

- xorig:

  The original data matrix (X) used to fit the model.

- ...:

  Additional arguments passed to the method.

## Value

A matrix of residuals, with the same dimensions as the original data
matrix.
