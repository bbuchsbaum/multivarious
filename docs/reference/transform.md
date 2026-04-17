# Transform data using a fitted preprocessing pipeline

Apply a fitted preprocessing pipeline to new data. The preprocessing
object must have been fitted using
[`fit()`](https://bbuchsbaum.github.io/multivarious/reference/fit.md) or
[`fit_transform()`](https://bbuchsbaum.github.io/multivarious/reference/fit_transform.md)
before calling this function.

## Usage

``` r
transform(object, X, ...)
```

## Arguments

- object:

  A fitted preprocessing object

- X:

  A matrix or data frame to transform

- ...:

  Additional arguments passed to methods

## Value

The transformed data matrix

## See also

[`fit()`](https://bbuchsbaum.github.io/multivarious/reference/fit.md),
[`fit_transform()`](https://bbuchsbaum.github.io/multivarious/reference/fit_transform.md),
[`inverse_transform()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_transform.md)

## Examples

``` r
# Transform new data with fitted preprocessor
X_train <- matrix(rnorm(100), 10, 10)
X_test <- matrix(rnorm(50), 5, 10)

preproc <- center()
fitted_preproc <- fit(preproc, X_train)
X_test_transformed <- transform(fitted_preproc, X_test)
```
