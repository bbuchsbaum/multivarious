# Rotate a Component Solution

Perform a rotation of the component loadings to improve
interpretability.

## Usage

``` r
rotate(x, ncomp, type, ...)
```

## Arguments

- x:

  The model fit, typically a result from a dimensionality reduction
  method like PCA.

- ncomp:

  The number of components to rotate.

- type:

  The type of rotation to apply (e.g., "varimax", "quartimax",
  "promax").

- ...:

  extra args

## Value

A modified model fit with the rotated components.
