# Cross-validation Framework

Generic function for performing cross-validation on various objects or
data. Specific methods should be implemented for different data types or
model types.

## Usage

``` r
cv(x, folds, ...)
```

## Arguments

- x:

  The object to perform cross-validation on (e.g., data matrix, formula,
  model object).

- folds:

  A list defining the cross-validation folds, typically containing
  `train` and `test` indices for each fold.

- ...:

  Additional arguments passed to specific methods.

## Value

The structure of the return value depends on the specific S3 method.
Typically, it will be an object containing the results of the
cross-validation, such as performance metrics per fold or aggregated
metrics.

## Details

The specific implementation details, default functions, and relevant
arguments vary by method.

**Bi-Projector Method (`cv.bi_projector`):** Relevant arguments: `x`,
`folds`, `max_comp`, `fit_fun`, `measure`, `measure_fun`,
`return_models`, `...`.

This method performs cross-validation specifically for `bi_projector`
models (or models intended to be used like them, typically from
unsupervised methods like PCA or SVD). For each fold, it fits a single
model using the training data with the maximum number of components
specified (`max_comp`). It then iterates from 1 to `max_comp`
components:

1.  It truncates the full model to `k` components using
    [`truncate()`](https://bbuchsbaum.github.io/multivarious/reference/truncate.md).
    (Requires a `truncate` method for the fitted model class).

2.  It reconstructs the held-out test data using the k-component
    truncated model via
    [`reconstruct_new()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct_new.md).

3.  It calculates reconstruction performance metrics (e.g., MSE, R2) by
    comparing the original test data to the reconstruction using the
    `measure` argument or a custom `measure_fun`.

The `fit_fun` must accept an argument `ncomp`. Additional arguments in
`...` are passed to `fit_fun` and `measure_fun`.

The return value is a `cv_fit` object (a list with class `cv_fit`),
where the `$results` element is a tibble. Each row corresponds to a
fold, containing the fold index (`fold`) and a nested tibble
(`component_metrics`). The `component_metrics` tibble has rows for each
component evaluated (1 to `max_comp`) and columns for the component
index (`comp`) plus all calculated metrics (e.g., `mse`, `r2`, `mae`) or
error messages (`comp_error`). If `return_models=TRUE`, the full model
fitted on the training data for each fold is included in a list column
`model_full`.

## See also

[`cv_generic`](https://bbuchsbaum.github.io/multivarious/reference/cv_generic.md)
