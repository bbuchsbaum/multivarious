# Get the number of components

This function returns the total number of components in the fitted
model.

## Usage

``` r
ncomp(x)
```

## Arguments

- x:

  A fitted model object.

## Value

The number of components in the fitted model.

## Examples

``` r
# Example using the svd_wrapper function
data(iris)
X <- as.matrix(iris[, 1:4])
fit <- svd_wrapper(X, ncomp = 3, preproc = center(), method = "base")
ncomp(fit) # Should return 3
#> [1] 3
```
