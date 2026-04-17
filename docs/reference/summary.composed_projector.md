# Summarize a Composed Projector

Provides a summary of the stages within a composed projector, including
stage names, input/output dimensions, and the primary class of each
stage.

## Usage

``` r
# S3 method for class 'composed_projector'
summary(object, ...)
```

## Arguments

- object:

  A `composed_projector` object.

- ...:

  Currently unused.

## Value

A `tibble` summarizing the pipeline stages.
