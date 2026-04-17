# Shape of the Projector

Get the input/output shape of the projector.

## Usage

``` r
shape(x, ...)
```

## Arguments

- x:

  The model fit.

- ...:

  Extra arguments.

## Value

A vector containing the dimensions of the sample loadings matrix `v`
(number of rows and columns).

## Details

This function retrieves the dimensions of the sample loadings matrix `v`
in the form of a vector with two elements. The first element is the
number of rows in the `v` matrix, and the second element is the number
of columns.
