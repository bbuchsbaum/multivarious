# Convenience function for preprocessing workflow

This helper function provides a simple interface for the common
preprocessing workflow: fit a preprocessor to data and return both the
fitted preprocessor and the transformed data.

## Usage

``` r
preprocess(preproc, X, ...)
```

## Arguments

- preproc:

  A preprocessing object (e.g., created with
  [`center()`](https://bbuchsbaum.github.io/multivarious/reference/center.md),
  [`standardize()`](https://bbuchsbaum.github.io/multivarious/reference/standardize.md),
  etc.)

- X:

  A matrix or data frame to preprocess

- ...:

  Additional arguments passed to methods

## Value

A list with two elements:

- preproc:

  The fitted preprocessing object

- transformed:

  The transformed data matrix

## See also

[`fit()`](https://bbuchsbaum.github.io/multivarious/reference/fit.md),
[`fit_transform()`](https://bbuchsbaum.github.io/multivarious/reference/fit_transform.md),
[`transform()`](https://bbuchsbaum.github.io/multivarious/reference/transform.md),
[`inverse_transform()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_transform.md)

## Examples

``` r
# Simple preprocessing workflow
X <- matrix(rnorm(100), 10, 10)
result <- preprocess(center(), X)
fitted_preproc <- result$preproc
X_centered <- result$transformed

# Equivalent to:
# fitted_preproc <- fit(center(), X)
# X_centered <- transform(fitted_preproc, X)
```
