# Predict method for a discriminant_projector, supporting LDA or Euclid

This produces class predictions or posterior-like scores for new data.
We first project the data into the subspace defined by `x$v`, then
either:

1.  **LDA approach** (`method="lda"`), which uses a (simplified) linear
    discriminant formula or distance to class means in the subspace
    combined with prior probabilities.

2.  **Euclid approach** (`method="euclid"`), which uses plain Euclidean
    distance to each class mean in the subspace.

We return either a `type="class"` label or `type="prob"` posterior-like
matrix.

## Usage

``` r
# S3 method for class 'discriminant_projector'
predict(
  object,
  new_data,
  method = c("lda", "euclid"),
  type = c("class", "prob"),
  colind = NULL,
  ...
)
```

## Arguments

- object:

  A `discriminant_projector` object.

- new_data:

  A numeric matrix (or vector) with the same \# of columns as the
  original data (unless partial usage). Rows=observations,
  columns=features.

- method:

  Either `"lda"` (the default) or `"euclid"` (nearest-mean).

- type:

  `"class"` (default) for predicted class labels, or `"prob"` for
  posterior-like probabilities.

- colind:

  (optional) if partial columns are used, specify which columns map to
  the subspace. If `NULL`, assume full columns.

- ...:

  further arguments (not used or for future expansions).

## Value

If `type="class"`, a factor vector of length n (predicted classes). If
`type="prob"`, an (n x \#classes) numeric matrix of posterior-like
values, with row names matching `new_data` if available.

Predict method for a discriminant_projector

This produces class predictions or posterior-like scores for new data,
based on:

- **LDA approach** (`method="lda"`), which uses a linear discriminant
  formula with a pooled covariance matrix if `x\$Sigma` is given, or the
  identity matrix if `Sigma=NULL`. If that covariance matrix is not
  invertible, a pseudo-inverse is used and a warning is emitted.

- **Euclid approach** (`method="euclid"`), which uses plain Euclidean
  distance to each class mean in the subspace.

We return either a `type="class"` label or `type="prob"` posterior-like
matrix.

If `type="class"`, a factor vector of length n (predicted classes). If
`type="prob"`, an (n x \#classes) numeric matrix of posterior-like
values.
