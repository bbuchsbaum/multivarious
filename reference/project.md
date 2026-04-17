# New sample projection

Project one or more samples onto a subspace. This function takes a model
fit and new observations, and projects them onto the subspace defined by
the model. This allows for the transformation of new data into the same
lower-dimensional space as the original data.

## Usage

``` r
project(x, new_data, ...)
```

## Arguments

- x:

  The model fit, typically an object of class bi_projector or any other
  class that implements a project method

- new_data:

  A matrix or vector of new observations with the same number of columns
  as the original data. Rows represent observations and columns
  represent variables

- ...:

  Extra arguments to be passed to the specific project method for the
  object's class

## Value

A matrix or vector of the projected observations, where rows represent
observations and columns represent the lower-dimensional space

## See also

[`bi_projector`](https://bbuchsbaum.github.io/multivarious/reference/bi_projector.md)
for an example of a class that implements a project method

Other project:
[`project.cross_projector()`](https://bbuchsbaum.github.io/multivarious/reference/project.cross_projector.md),
[`project_block()`](https://bbuchsbaum.github.io/multivarious/reference/project_block.md),
[`project_vars()`](https://bbuchsbaum.github.io/multivarious/reference/project_vars.md)

## Examples

``` r
# Example with the bi_projector class
X <- matrix(rnorm(10*20), 10, 20)
svdfit <- svd(X)
p <- bi_projector(svdfit$v, s = svdfit$u %*% diag(svdfit$d), sdev=svdfit$d)

# Project new_data onto the same subspace as the original data
new_data <- matrix(rnorm(5*20), 5, 20)
projected_data <- project(p, new_data)
```
