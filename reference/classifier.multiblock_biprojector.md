# Multiblock Bi-Projector Classifier

Constructs a k-Nearest Neighbors (k-NN) classifier based on a fitted
`multiblock_biprojector` model object. The classifier uses the projected
scores as the feature space for k-NN.

## Usage

``` r
# S3 method for class 'multiblock_biprojector'
classifier(
  x,
  colind = NULL,
  labels,
  new_data = NULL,
  block = NULL,
  global_scores = TRUE,
  knn = 1,
  ...
)
```

## Arguments

- x:

  A fitted `multiblock_biprojector` object.

- colind:

  An optional numeric vector specifying column indices from the original
  data space. If provided when `global_scores=FALSE`, these indices are
  used to perform a partial projection for the reference scores. If
  provided when `global_scores=TRUE`, this value is stored but does not
  affect the reference scores (which remain global); however, it may
  influence the default projection behavior during prediction unless
  overridden there. See `predict.classifier`.

- labels:

  A factor or vector of class labels for the training data.

- new_data:

  An optional data matrix used to generate reference scores when
  `global_scores=FALSE`, or when `global_scores=TRUE` but `colind` or
  `block` is also provided (overriding `global_scores`). Must be
  provided if `global_scores=FALSE`.

- block:

  An optional integer specifying a predefined block index. Used for
  partial projection if `global_scores=FALSE` or if `new_data` is also
  provided. Cannot be used simultaneously with `colind`.

- global_scores:

  Logical. **DEPRECATED** This argument is deprecated and its behavior
  has changed. Reference scores are now determined automatically:

  - If `new_data` is NULL: Uses the globally projected scores stored in
    `x` (`scores(x)`).

  - If `new_data` is provided: Always projects `new_data` to generate
    reference scores (using `partial_project`/`project_block` if
    `colind`/`block` are given, `project` otherwise).

- knn:

  The integer number of nearest neighbors (k) for the k-NN algorithm
  (default: 1).

- ...:

  Additional arguments (currently ignored).

## Value

An object of class `multiblock_classifier`, which also inherits from
`classifier`.

## Details

Users can specify whether to use the globally projected scores stored
within the model (`global_scores = TRUE`) or to generate reference
scores by projecting provided `new_data` (`global_scores = FALSE`).
Partial projections based on `colind` or `block` can be used when
`global_scores = FALSE` or when `new_data` is provided alongside
`colind`/`block`. Prediction behavior is further controlled by arguments
passed to `predict.classifier`.

## See also

Other classifier:
[`classifier()`](https://bbuchsbaum.github.io/multivarious/reference/classifier.md),
[`rf_classifier.projector()`](https://bbuchsbaum.github.io/multivarious/reference/rf_classifier.projector.md)
