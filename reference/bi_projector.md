# Construct a bi_projector instance

A bi_projector offers a two-way mapping from samples (rows) to scores
and from variables (columns) to components. Thus, one can project from
D-dimensional input space to d-dimensional subspace. And one can project
(project_vars) from n-dimensional variable space to the d-dimensional
component space. The singular value decomposition is a canonical example
of such a two-way mapping.

## Usage

``` r
bi_projector(v, s, sdev, preproc = prep(pass()), classes = NULL, ...)
```

## Arguments

- v:

  A matrix of coefficients with dimensions `nrow(v)` by `ncol(v)`
  (columns = components)

- s:

  The score matrix

- sdev:

  The standard deviations of the score matrix

- preproc:

  (optional) A pre-processing pipeline, default is prep(pass())

- classes:

  (optional) A character vector specifying the class attributes of the
  object, default is NULL

- ...:

  Extra arguments to be stored in the `projector` object.

## Value

A bi_projector object

## Examples

``` r
X <- matrix(rnorm(200), 10, 20)
svdfit <- svd(X)

p <- bi_projector(svdfit$v, s = svdfit$u %*% diag(svdfit$d), sdev=svdfit$d)
#> Warning: `prep()` was deprecated in multivarious 0.3.0.
#> ℹ Please use `fit()` instead.
#> ℹ The prep() function is deprecated. Use fit() for a more standard interface.
#> ℹ The deprecated feature was likely used in the multivarious package.
#>   Please report the issue to the authors.
```
