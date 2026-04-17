# Extract scores from a PLSC fit

Extract scores from a PLSC fit

## Usage

``` r
# S3 method for class 'plsc'
scores(x, block = c("X", "Y"), ...)
```

## Arguments

- x:

  A `plsc` object.

- block:

  Which block to return scores for: "X" (default) or "Y".

- ...:

  Ignored.

## Value

Numeric matrix of scores for the chosen block.
