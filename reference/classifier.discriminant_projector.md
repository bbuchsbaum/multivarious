# Create a k-NN classifier for a discriminant projector

Create a k-NN classifier for a discriminant projector

## Usage

``` r
# S3 method for class 'discriminant_projector'
classifier(x, colind = NULL, knn = 1, ...)
```

## Arguments

- x:

  the discriminant projector object

- colind:

  an optional vector specifying the column indices of the components

- knn:

  the number of nearest neighbors (default=1)

- ...:

  extra arguments

## Value

a classifier object

## Examples

``` r
# Assume dp is a fitted discriminant_projector object
# classifier(dp, knn = 5) # Basic example
```
