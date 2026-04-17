# Reconstruct fitted or subsetted outputs for a `regress` object

For regression-based bi_projectors, reconstruction should map from the
design matrix side (scores) to the output space using the regression
coefficients, without applying any reverse preprocessing (which belongs
to the input/basis side).

## Usage

``` r
# S3 method for class 'regress'
reconstruct(
  x,
  comp = 1:ncol(x$coefficients),
  rowind = 1:nrow(scores(x)),
  colind = 1:nrow(x$coefficients),
  ...
)
```

## Arguments

- x:

  A `regress` object produced by
  [`regress()`](https://bbuchsbaum.github.io/multivarious/reference/regress.md).

- comp:

  Integer vector of component indices (columns of the design matrix /
  predictors) to use.

- rowind:

  Integer vector of row indices in the design matrix (observations) to
  reconstruct.

- colind:

  Integer vector of output indices (columns of Y) to reconstruct.

- ...:

  Ignored.
