# Apply rotation

Apply a specified rotation to the fitted model

## Usage

``` r
apply_rotation(x, rotation_matrix, ...)
```

## Arguments

- x:

  A model object, possibly created using the
  [`pca()`](https://bbuchsbaum.github.io/multivarious/reference/pca.md)
  function.

- rotation_matrix:

  `matrix` reprsenting the rotation.

- ...:

  extra args

## Value

A modified object with updated components and scores after applying the
specified rotation.
