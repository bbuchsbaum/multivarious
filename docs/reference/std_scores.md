# Compute standardized component scores

Calculate standardized factor scores from a fitted model. Standardized
scores are useful for comparing the contributions of different
components on the same scale, which can help in interpreting the
results.

## Usage

``` r
std_scores(x, ...)
```

## Arguments

- x:

  The model fit object.

- ...:

  Additional arguments passed to the method.

## Value

A matrix of standardized factor scores, with rows corresponding to
samples and columns to components.

## See also

[`scores`](https://bbuchsbaum.github.io/multivarious/reference/scores.md)
for retrieving the original component scores.
