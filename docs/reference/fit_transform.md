# Fit and transform data in one step

Convenience function that fits a preprocessing pipeline to data and
immediately applies the transformation. This is equivalent to calling
[`fit()`](https://bbuchsbaum.github.io/multivarious/reference/fit.md)
followed by
[`transform()`](https://bbuchsbaum.github.io/multivarious/reference/transform.md)
but is more efficient and convenient.

## Usage

``` r
fit_transform(object, X, ...)
```

## Arguments

- object:

  A preprocessing object (e.g., `prepper` or `pre_processor`)

- X:

  A matrix or data frame to fit and transform

- ...:

  Additional arguments passed to methods

## Value

A list with two elements: `preproc` (the fitted preprocessor) and
`transformed` (the transformed data)

## See also

[`fit()`](https://bbuchsbaum.github.io/multivarious/reference/fit.md),
[`transform()`](https://bbuchsbaum.github.io/multivarious/reference/transform.md),
[`inverse_transform()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_transform.md)

## Examples

``` r
# Fit and transform in one step
X <- matrix(rnorm(100), 10, 10)
preproc <- center()
result <- fit_transform(preproc, X)
fitted_preproc <- result$preproc
X_transformed <- result$transformed
```
