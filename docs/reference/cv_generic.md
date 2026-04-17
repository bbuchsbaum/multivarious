# Generic cross-validation engine

For each fold (train/test indices):

1.  Subset `data[train, ]`

2.  Fit a model with `.fit_fun(train_data, ...)`

3.  Evaluate with `.measure_fun(model, test_data, ...)`

## Usage

``` r
cv_generic(
  data,
  folds,
  .fit_fun,
  .measure_fun,
  fit_args = list(),
  measure_args = list(),
  backend = c("serial", "future"),
  ...
)
```

## Arguments

- data:

  A matrix or data.frame of shape (n x p).

- folds:

  A list of folds, each a list with `$train` and `$test`.

- .fit_fun:

  Function: signature `function(train_data, ...){}`. Returns a fitted
  model.

- .measure_fun:

  Function: signature `function(model, test_data, ...){}`. Returns a
  tibble or named list/vector of metrics.

- fit_args:

  A list of additional named arguments passed to `.fit_fun`.

- measure_args:

  A list of additional named arguments passed to `.measure_fun`.

- backend:

  Character string: "serial" (default) or "future" for parallel
  execution using the `future` framework.

- ...:

  Currently ignored (arguments should be passed via `fit_args` or
  `measure_args`).

## Value

A tibble with columns:

- fold:

  integer fold index

- model:

  list of fitted models

- metrics:

  list of metric tibbles/lists
