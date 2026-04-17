# Retrieve the component scores

Extract the factor score matrix from a fitted model. The factor scores
represent the projections of the data onto the components, which can be
used for further analysis or visualization.

## Usage

``` r
scores(x, ...)
```

## Arguments

- x:

  The model fit object.

- ...:

  Additional arguments passed to the method.

## Value

A matrix of factor scores, with rows corresponding to samples and
columns to components.

## See also

[`project`](https://bbuchsbaum.github.io/multivarious/reference/project.md)
for projecting new data onto the components.
