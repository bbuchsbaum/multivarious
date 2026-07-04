# Lazy sparse affine matrices

Objects of class `sparse_affine_matrix` represent sparse matrix
preprocessing of the form `sweep(x, 2, scale, "*") + shift` without
materializing dense centered columns. The companion
`sparse_affine_transpose` class represents its transpose.

## Usage

``` r
# S4 method for class 'sparse_affine_matrix'
dim(x)
# S4 method for class 'sparse_affine_transpose'
dim(x)

# S4 method for class 'sparse_affine_matrix'
dimnames(x)
# S4 method for class 'sparse_affine_transpose'
dimnames(x)

# S4 method for class 'sparse_affine_matrix'
as.matrix(x, ...)
# S4 method for class 'sparse_affine_transpose'
as.matrix(x, ...)

# S4 method for class 'sparse_affine_matrix,ANY,ANY,ANY'
x[i, j, ..., drop = TRUE]
# S4 method for class 'sparse_affine_matrix,ANY,missing,ANY'
x[i, j, ..., drop = TRUE]
# S4 method for class 'sparse_affine_matrix,missing,ANY,ANY'
x[i, j, ..., drop = TRUE]
# S4 method for class 'sparse_affine_matrix,missing,missing,ANY'
x[i, j, ..., drop = TRUE]

# S4 method for class 'sparse_affine_matrix'
t(x)
# S4 method for class 'sparse_affine_transpose'
t(x)

# S4 method for class 'sparse_affine_matrix,ANY'
x %*% y
# S4 method for class 'ANY,sparse_affine_matrix'
x %*% y
# S4 method for class 'sparse_affine_transpose,ANY'
x %*% y
# S4 method for class 'ANY,sparse_affine_transpose'
x %*% y

# S4 method for class 'sparse_affine_matrix,missing'
crossprod(x, y = NULL, ...)
# S4 method for class 'sparse_affine_matrix,ANY'
crossprod(x, y = NULL, ...)
# S4 method for class 'ANY,sparse_affine_matrix'
crossprod(x, y = NULL, ...)
# S4 method for class 'sparse_affine_transpose,missing'
crossprod(x, y = NULL, ...)
# S4 method for class 'sparse_affine_transpose,ANY'
crossprod(x, y = NULL, ...)

# S4 method for class 'sparse_affine_matrix,missing'
tcrossprod(x, y = NULL, ...)
# S4 method for class 'sparse_affine_matrix,ANY'
tcrossprod(x, y = NULL, ...)
# S4 method for class 'ANY,sparse_affine_matrix'
tcrossprod(x, y = NULL, ...)
# S4 method for class 'sparse_affine_transpose,missing'
tcrossprod(x, y = NULL, ...)
# S4 method for class 'sparse_affine_transpose,ANY'
tcrossprod(x, y = NULL, ...)
```

## Arguments

- x:

  A sparse affine matrix, sparse affine transpose, or compatible
  left-hand matrix argument.

- y:

  A compatible right-hand matrix argument.

- i, j:

  Row and column indices for subsetting.

- drop:

  Whether to drop dimensions when subsetting.

- ...:

  Additional arguments passed to the generic.

## Details

These classes are implementation details returned by sparse-aware
preprocessing steps such as
[`center()`](https://bbuchsbaum.github.io/multivarious/reference/center.md)
and
[`standardize()`](https://bbuchsbaum.github.io/multivarious/reference/standardize.md)
when sparse inputs would otherwise be densified by centering. Users
normally do not construct these objects directly.

The documented S4 methods provide the matrix operations used by
downstream estimators: dimensions and names, subsetting, explicit
materialization through
[`as.matrix()`](https://rdrr.io/r/base/matrix.html), transpose views,
matrix multiplication, and cross-products. The algebraic methods apply
the affine offset during the multiplication so that common projections
and decompositions can operate without first expanding the preprocessed
sparse input to a dense matrix.

## Value

Dimension and name methods return the corresponding base values.
Subsetting returns either another sparse affine view or a materialized
vector or matrix when `drop = TRUE`. Multiplication and cross-product
methods return base matrix results matching the corresponding dense
affine calculation.

## See also

[`prep()`](https://bbuchsbaum.github.io/multivarious/reference/prep.md),
[`fit_transform()`](https://bbuchsbaum.github.io/multivarious/reference/fit_transform.md),
[`transform()`](https://bbuchsbaum.github.io/multivarious/reference/transform.md)
