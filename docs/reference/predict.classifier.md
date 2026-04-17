# Predict Class Labels using a Classifier Object

Predicts class labels and probabilities for new data using a fitted
`classifier` object. It performs k-Nearest Neighbors (k-NN)
classification in the projected component space.

## Usage

``` r
# S3 method for class 'classifier'
predict(
  object,
  new_data,
  ncomp = NULL,
  colind = NULL,
  metric = c("euclidean", "cosine", "ejaccard"),
  normalize_probs = FALSE,
  prob_type = c("knn_proportion", "avg_similarity"),
  ...
)
```

## Arguments

- object:

  A fitted object of class `classifier`.

- new_data:

  A numeric matrix or vector of new observations to classify. Rows are
  observations, columns are variables matching the original data space
  used by the projector OR matching `colind` if provided.

- ncomp:

  Optional integer; the number of components to use from the projector
  for classification (default: all components used during classifier
  creation).

- colind:

  Optional numeric vector specifying column indices from the original
  data space. If provided, `new_data` is projected using only these
  features (`partial_project`). This overrides any `colind` stored
  default in the `object`. The resulting projection is compared against
  the reference scores (`object$scores`) stored in the classifier.

- metric:

  Character string specifying the similarity or distance metric for
  k-NN. Choices: "euclidean", "cosine", "ejaccard".

- normalize_probs:

  Logical; **DEPRECATED** Normalization behavior is now implicit in
  `prob_type="avg_similarity"`.

- prob_type:

  Character string; method for calculating probabilities:

  - "knn_proportion" (default): Calculates the proportion of each class
    among the `k` nearest neighbors.

  - "avg_similarity": Calculates average similarity to all training
    points per class (uses `avg_probs` helper).

- ...:

  Extra arguments passed down to projection methods (`project`,
  `partial_project`) or potentially to distance/similarity calculations
  (e.g., for [`proxy::simil`](https://rdrr.io/pkg/proxy/man/dist.html)
  if used with `ejaccard`).

## Value

A list containing:

- class:

  A factor vector of predicted class labels for `new_data`.

- prob:

  A numeric matrix (rows corresponding to `new_data`, columns to
  classes) of estimated class probabilities.

## Details

The function first projects the `new_data` into the component space
defined by the classifier's internal projector. If `colind` is
specified, a partial projection using only those features is performed.
This projection is then compared to the reference scores stored within
the `classifier` object (`object$scores`) using the specified `metric`.
The k-NN algorithm identifies the `k` nearest reference samples (based
on similarity or distance) and predicts the class via majority vote.
Probabilities are estimated based on the average similarity/distance to
each class among the neighbors or all reference points.

## See also

[`classifier.projector`](https://bbuchsbaum.github.io/multivarious/reference/classifier.md),
[`classifier.multiblock_biprojector`](https://bbuchsbaum.github.io/multivarious/reference/classifier.multiblock_biprojector.md),
[`partial_project`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)

Other classifier predict:
[`predict.rf_classifier()`](https://bbuchsbaum.github.io/multivarious/reference/predict.rf_classifier.md)

## Examples

``` r
# Assume clf is a fitted classifier object (e.g., from classifier.projector)
# Assume new_dat is a matrix of new observations
# preds <- predict(clf, new_data = new_dat, metric = "cosine")
# print(preds$class)
# print(preds$prob)
```
