# Create a random forest classifier

Uses `randomForest` to train a random forest on the provided scores and
labels.

## Usage

``` r
# S3 method for class 'projector'
rf_classifier(x, colind = NULL, labels, scores, ...)
```

## Arguments

- x:

  a projector object

- colind:

  optional col indices

- labels:

  class labels

- scores:

  reference scores

- ...:

  passed to `randomForest`

## Value

a `rf_classifier` object with rfres (rf model), labels, scores

## See also

[`randomForest`](https://rdrr.io/pkg/randomForest/man/randomForest.html)

Other classifier:
[`classifier()`](https://bbuchsbaum.github.io/multivarious/reference/classifier.md),
[`classifier.multiblock_biprojector()`](https://bbuchsbaum.github.io/multivarious/reference/classifier.multiblock_biprojector.md)

## Examples

``` r
# Assume proj is a fitted projector object
# Assume lbls are labels and sc are scores
# if (requireNamespace("randomForest", quietly = TRUE)) {
#   rf_classifier(proj, labels = lbls, scores = sc)
# }
```
