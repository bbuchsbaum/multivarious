# Evaluate Feature Importance for a Classifier

Estimates the importance of features or blocks of features for the
classification performance using either a "marginal"
(leave-one-block-out) or "standalone" (use-only-one-block) approach.

## Usage

``` r
# S3 method for class 'classifier'
feature_importance(
  x,
  new_data,
  true_labels,
  ncomp = NULL,
  blocks = NULL,
  metric = c("cosine", "euclidean", "ejaccard"),
  fun = rank_score,
  fun_direction = c("lower_is_better", "higher_is_better"),
  approach = c("marginal", "standalone"),
  ...
)
```

## Arguments

- x:

  A fitted `classifier` object.

- new_data:

  The data matrix used for evaluating importance (typically validation
  or test data).

- true_labels:

  The true class labels corresponding to the rows of `new_data`.

- ncomp:

  Optional integer; the number of components to use from the projector
  for classification (default: all components used during classifier
  creation).

- blocks:

  A list where each element is a numeric vector of feature indices
  (columns in the original data space) defining a block. If `NULL`, each
  feature is treated as its own block.

- metric:

  Character string specifying the similarity or distance metric for
  k-NN. Choices: "euclidean", "cosine", "ejaccard".

- fun:

  A function to compute the performance metric (e.g., `rank_score`,
  `topk`, or a custom function). The function should take a probability
  matrix and observed labels and return a data frame where the first
  column is the metric value per observation.

- fun_direction:

  Character string, either "lower_is_better" or "higher_is_better",
  indicating whether lower or higher values of the metric calculated by
  `fun` signify better performance. This is used to interpret the
  importance score correctly.

- approach:

  Character string: "marginal" (calculates importance as change from
  baseline when block is removed) or "standalone" (calculates importance
  as performance using only the block).

- ...:

  Additional arguments passed to `predict.classifier` during internal
  predictions.

## Value

A `data.frame` with columns `block` (character representation of feature
indices in the block) and `importance` (numeric importance score).
Higher importance values generally indicate more influential blocks,
considering `fun_direction`.

## Details

Importance is measured by the change in a performance metric (`fun`)
when features are removed (marginal) or used exclusively (standalone).

## See also

[`rank_score`](https://bbuchsbaum.github.io/multivarious/reference/rank_score.md),
[`topk`](https://bbuchsbaum.github.io/multivarious/reference/topk.md)

## Examples

``` r
# Assume clf is a fitted classifier object, dat is new data, true_lbls are correct labels for dat
# Assume blocks_list defines feature groups e.g., list(1:5, 6:10)
# feature_importance(clf, new_data = dat, true_labels = true_lbls, blocks = blocks_list)
```
