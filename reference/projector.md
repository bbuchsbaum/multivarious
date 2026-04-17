# Construct a `projector` instance

A `projector` maps a matrix from an N-dimensional space to d-dimensional
space, where `d` may be less than `N`. The projection matrix, `v`, is
not necessarily orthogonal. This function constructs a `projector`
instance which can be used for various dimensionality reduction
techniques like PCA, LDA, etc.

## Usage

``` r
projector(v, preproc = prep(pass()), ..., classes = NULL)
```

## Arguments

- v:

  A matrix of coefficients with dimensions `nrow(v)` by `ncol(v)`
  (columns = components)

- preproc:

  A prepped pre-processing object (S3 class `pre_processor`). Default is
  the no-op
  [`pass()`](https://bbuchsbaum.github.io/multivarious/reference/pass.md)
  preprocessor.

- ...:

  Extra arguments to be stored in the `projector` object.

- classes:

  Additional class information used for creating subtypes of
  `projector`. Default is NULL.

## Value

An instance of type `projector`.
