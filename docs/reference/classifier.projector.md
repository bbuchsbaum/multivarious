# create classifier from a projector

create classifier from a projector

## Usage

``` r
# S3 method for class 'projector'
classifier(
  x,
  colind = NULL,
  labels,
  new_data = NULL,
  knn = 1,
  global_scores = TRUE,
  ...
)
```

## Arguments

- x:

  projector

- colind:

  ...

- labels:

  ...

- new_data:

  ...

- knn:

  ...

- global_scores:

  ...

- ...:

  extra args

## See also

Other classifier:
[`classifier.multiblock_biprojector()`](https://bbuchsbaum.github.io/multivarious/reference/classifier.multiblock_biprojector.md),
[`rf_classifier.projector()`](https://bbuchsbaum.github.io/multivarious/reference/rf_classifier.projector.md)

## Examples

``` r
# Assume proj is a fitted projector object
# Assume lbls are labels and dat is new data
# classifier(proj, labels = lbls, new_data = dat, knn = 3)
```
