# Rotate PCA Loadings

Apply a specified rotation to the component loadings of a PCA model.
This function leverages the GPArotation package to apply orthogonal or
oblique rotations.

## Usage

``` r
# S3 method for class 'pca'
rotate(
  x,
  ncomp,
  type = c("varimax", "quartimax", "promax"),
  loadings_type = c("pattern", "structure"),
  score_method = c("auto", "recompute", "original"),
  ...
)
```

## Arguments

- x:

  A PCA model object, typically created using the
  [`pca()`](https://bbuchsbaum.github.io/multivarious/reference/pca.md)
  function.

- ncomp:

  The number of components to rotate. Must be \<= ncomp(x).

- type:

  The type of rotation to apply. Supported rotation types:

  "varimax"

  :   Orthogonal Varimax rotation

  "quartimax"

  :   Orthogonal Quartimax rotation

  "promax"

  :   Oblique Promax rotation

- loadings_type:

  For oblique rotations, which loadings to use:

  "pattern"

  :   Use pattern loadings as `v`

  "structure"

  :   Use structure loadings (`pattern_loadings %*% Phi`) as `v`

  Ignored for orthogonal rotations.

- score_method:

  How to recompute scores after rotation:

  "auto"

  :   For orthogonal rotations, use
      `scores_new = scores_original %*% t(R)`

  "recompute"

  :   Always recompute scores from `X_proc` and the pseudoinverse of
      rotated loadings.

  "original"

  :   For orth rotations, same as `auto`, but may not work for oblique
      rotations.

- ...:

  Additional arguments passed to GPArotation functions.

## Value

A modified PCA object with class `rotated_pca` and additional fields:

- v:

  Rotated loadings

- s:

  Rotated scores

- sdev:

  Updated standard deviations of rotated components

- explained_variance:

  Proportion of explained variance for each rotated component

- rotation:

  A list with rotation details: `type`, `R` (orth) or `Phi` (oblique),
  and `loadings_type`

## Examples

``` r
# Perform PCA on the iris dataset
data(iris)
X <- as.matrix(iris[,1:4])
res <- pca(X, ncomp=4)

# Apply varimax rotation to the first 3 components
rotated_res <- rotate(res, ncomp=3, type="varimax")
```
