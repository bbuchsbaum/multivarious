# Construct an effect operator

Construct an effect operator

## Usage

``` r
effect_operator(v, s, sdev, preproc = fit(pass(), matrix(0, 1, nrow(v))), ...)
```

## Arguments

- v:

  Feature loadings in original variable space.

- s:

  Observation-side effect scores.

- sdev:

  Singular values.

- preproc:

  Fitted response preprocessor.

- ...:

  Additional metadata stored on the object.

## Value

An `effect_operator`.
