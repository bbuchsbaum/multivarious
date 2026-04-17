# scale a data matrix

normalize each column by a scale factor.

## Usage

``` r
colscale(preproc = prepper(), type = c("unit", "z", "weights"), weights = NULL)
```

## Arguments

- preproc:

  the pre-processing pipeline

- type:

  the kind of scaling, `unit` norm, `z`-scoring, or precomputed
  `weights`

- weights:

  optional precomputed weights

## Value

a `prepper` list
