# Inverse transform data using a fitted preprocessing pipeline

Reverse the preprocessing transformation, converting transformed data
back to the original scale. The preprocessing object must have been
fitted before calling this function.

## Usage

``` r
inverse_transform(object, X, ...)
```

## Arguments

- object:

  A fitted preprocessing object

- X:

  A matrix or data frame of transformed data to reverse

- ...:

  Additional arguments passed to methods

## Value

The data matrix in original scale

## See also

[`fit()`](https://bbuchsbaum.github.io/multivarious/reference/fit.md),
[`fit_transform()`](https://bbuchsbaum.github.io/multivarious/reference/fit_transform.md),
[`transform()`](https://bbuchsbaum.github.io/multivarious/reference/transform.md)

## Examples

``` r
# Inverse transform data back to original scale
X <- matrix(rnorm(100), 10, 10)
preproc <- center()
fitted_preproc <- fit(preproc, X)
X_transformed <- transform(fitted_preproc, X)
X_reconstructed <- inverse_transform(fitted_preproc, X_transformed)

# X and X_reconstructed should be approximately equal
all.equal(X, X_reconstructed)
#> [1] TRUE
```
