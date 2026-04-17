# Predict Class Labels using a Random Forest Classifier Object

Predicts class labels and probabilities for new data using a fitted
`rf_classifier` object. This method projects the `new_data` into the
component space and then uses the stored `randomForest` model to predict
outcomes.

## Usage

``` r
# S3 method for class 'rf_classifier'
predict(object, new_data, ncomp = NULL, colind = NULL, ...)
```

## Arguments

- object:

  A fitted object of class `rf_classifier`.

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

- ...:

  Extra arguments passed to `predict.randomForest`.

## Value

A list containing:

- class:

  Predicted class labels (typically factor) from the random forest
  model.

- prob:

  A numeric matrix of predicted class probabilities from the random
  forest model.

## See also

[`rf_classifier.projector`](https://bbuchsbaum.github.io/multivarious/reference/rf_classifier.projector.md),
[`predict.randomForest`](https://rdrr.io/pkg/randomForest/man/predict.randomForest.html)

Other classifier predict:
[`predict.classifier()`](https://bbuchsbaum.github.io/multivarious/reference/predict.classifier.md)
