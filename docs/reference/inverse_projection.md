# Inverse of the Component Matrix

Return the inverse projection matrix, which can be used to map back to
data space. If the component matrix is orthogonal, then the inverse
projection is the transpose of the component matrix.

## Usage

``` r
inverse_projection(x, ...)

# S3 method for class 'projector'
inverse_projection(x, ...)
```

## Arguments

- x:

  The model fit.

- ...:

  Extra arguments.

## Value

The inverse projection matrix.

## See also

[`project`](https://bbuchsbaum.github.io/multivarious/reference/project.md)
for projecting data onto the subspace.
