# Extract scores from a CCA fit

Extract scores from a CCA fit

## Usage

``` r
# S3 method for class 'cca'
scores(x, block = c("X", "Y"), ...)
```

## Arguments

- x:

  A `cca` object.

- block:

  Which block to return scores for: "X" (default) or "Y".

- ...:

  Ignored.

## Value

Numeric matrix of canonical scores for the chosen block.
