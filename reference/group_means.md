# Compute column-wise mean in X for each factor level of Y

This function computes group means for each factor level of Y in the
provided data matrix X.

## Usage

``` r
group_means(Y, X)
```

## Arguments

- Y:

  a vector of labels to compute means over disjoint sets

- X:

  a data matrix from which to compute means

## Value

a matrix with row names corresponding to factor levels of Y and
column-wise means for each factor level

## Examples

``` r
# Example data
X <- matrix(rnorm(50), 10, 5)
Y <- factor(rep(1:2, each = 5))

# Compute group means
gm <- group_means(Y, X)
```
