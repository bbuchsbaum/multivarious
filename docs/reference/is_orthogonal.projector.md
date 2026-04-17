# Stricter check for true orthogonality

We test if v^T \* v = I (when rows \>= cols) or v \* v^T = I (when cols
\> rows).

## Usage

``` r
# S3 method for class 'projector'
is_orthogonal(x, tol = 1e-06)
```

## Arguments

- x:

  the projector object

- tol:

  tolerance for checking orthogonality
