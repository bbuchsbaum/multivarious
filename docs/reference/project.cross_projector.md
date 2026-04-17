# project a cross_projector instance

project a cross_projector instance

## Usage

``` r
# S3 method for class 'cross_projector'
project(x, new_data, source = c("X", "Y"), ...)
```

## Arguments

- x:

  The model fit, typically an object of class bi_projector or any other
  class that implements a project method

- new_data:

  A matrix or vector of new observations with the same number of columns
  as the original data. Rows represent observations and columns
  represent variables

- source:

  the source of the data (X or Y block)

- ...:

  Extra arguments to be passed to the specific project method for the
  object's class

## Value

the projected data

## See also

Other project:
[`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md),
[`project_block()`](https://bbuchsbaum.github.io/multivarious/reference/project_block.md),
[`project_vars()`](https://bbuchsbaum.github.io/multivarious/reference/project_vars.md)
