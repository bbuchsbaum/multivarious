# Two-way (cross) projection to latent components

A projector that reduces two blocks of data, X and Y, yielding a pair of
weights for each component. This structure can be used, for example, to
store weights derived from canonical correlation analysis.

## Usage

``` r
cross_projector(
  vx,
  vy,
  preproc_x = prep(pass()),
  preproc_y = prep(pass()),
  ...,
  classes = NULL
)
```

## Arguments

- vx:

  the X coefficients. Must have the same number of columns as `vy`.

- vy:

  the Y coefficients. Must have the same number of columns as `vx`.

- preproc_x:

  the X pre-processor

- preproc_y:

  the Y pre-processor

- ...:

  extra parameters or results to store

- classes:

  additional class names

## Value

a cross_projector object

## Details

This class extends `projector` and therefore basic operations such as
`project`, `shape`, `reprocess`, and `coef` work, but by default, it is
assumed that the `X` block is primary. To access `Y` block operations,
an additional argument `source` must be supplied to the relevant
functions, e.g., `coef(fit, source = "Y")`

## Examples

``` r
# Create two scaled matrices X and Y
X <- scale(matrix(rnorm(10 * 5), 10, 5))
Y <- scale(matrix(rnorm(10 * 5), 10, 5))

# Perform canonical correlation analysis on X and Y
cres <- cancor(X, Y)
sx <- X %*% cres$xcoef
sy <- Y %*% cres$ycoef

# Create a cross_projector object using the canonical correlation analysis results
canfit <- cross_projector(cres$xcoef, cres$ycoef, cor = cres$cor,
                          sx = sx, sy = sy, classes = "cancor")
```
