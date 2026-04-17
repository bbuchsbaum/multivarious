# Fit a preprocessing pipeline

Learn preprocessing parameters from training data. This function fits
the preprocessing pipeline to the provided data matrix, learning
parameters such as means, standard deviations, or other transformation
parameters.

## Usage

``` r
fit(object, X, ...)
```

## Arguments

- object:

  A preprocessing object (e.g., `prepper` or `pre_processor`)

- X:

  A matrix or data frame to fit the preprocessing pipeline to

- ...:

  Additional arguments passed to methods

## Value

A fitted preprocessing object that can be used with
[`transform()`](https://bbuchsbaum.github.io/multivarious/reference/transform.md)
and
[`inverse_transform()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_transform.md)

## See also

[`fit_transform()`](https://bbuchsbaum.github.io/multivarious/reference/fit_transform.md),
[`transform()`](https://bbuchsbaum.github.io/multivarious/reference/transform.md),
[`inverse_transform()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_transform.md)

## Examples

``` r
# Fit a centering preprocessor
X <- matrix(rnorm(100), 10, 10)
preproc <- center()
fitted_preproc <- fit(preproc, X)

# Transform new data
X_new <- matrix(rnorm(50), 5, 10)
X_transformed <- transform(fitted_preproc, X_new)
```
