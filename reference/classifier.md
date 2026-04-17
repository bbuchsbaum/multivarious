# Construct a Classifier

Create a classifier from a given model object (e.g., `projector`). This
classifier can generate predictions for new data points.

## Usage

``` r
classifier(x, colind, ...)

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

- ...:

  extra args

- labels:

  ...

- new_data:

  ...

- knn:

  ...

- global_scores:

  ...

## Value

A classifier function that can be used to make predictions on new data
points.

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
