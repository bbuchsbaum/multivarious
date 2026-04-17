# Transpose a model

This function transposes a model by switching coefficients and scores.
It is useful when you want to reverse the roles of samples and variables
in a model, especially in the context of dimensionality reduction
methods.

## Usage

``` r
transpose(x, ...)
```

## Arguments

- x:

  The model fit, typically an object of a class that implements a
  `transpose` method

- ...:

  Additional arguments passed to the underlying `transpose` method

## Value

A transposed model with coefficients and scores switched

## See also

[`bi_projector`](https://bbuchsbaum.github.io/multivarious/reference/bi_projector.md)
for an example of a two-way mapping model that can be transposed
