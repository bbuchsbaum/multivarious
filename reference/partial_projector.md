# Construct a partial projector

Create a new projector instance restricted to a subset of input columns.
This function allows for the generation of a new projection object that
focuses only on the specified columns, enabling the projection of data
using a limited set of variables.

## Usage

``` r
partial_projector(x, colind, ...)
```

## Arguments

- x:

  The original `projector` instance, typically an object of class
  `bi_projector` or any other class that implements a
  `partial_projector` method

- colind:

  A numeric vector of column indices to select in the projection matrix.
  These indices correspond to the variables used for the partial
  projector

- ...:

  Additional arguments passed to the underlying `partial_projector`
  method

## Value

A new `projector` instance, with the same class as the original object,
that is restricted to the specified subset of input columns

## See also

[`bi_projector`](https://bbuchsbaum.github.io/multivarious/reference/bi_projector.md)
for an example of a class that implements a `partial_projector` method

## Examples

``` r
# Example with the bi_projector class
X <- matrix(rnorm(10*20), 10, 20)
svdfit <- svd(X)
p <- bi_projector(svdfit$v, s = svdfit$u %*% diag(svdfit$d), sdev=svdfit$d)

# Create a partial projector using only the first 10 variables
colind <- 1:10
partial_p <- partial_projector(p, colind)
```
