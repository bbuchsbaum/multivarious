# Permutation test for an effect operator

Permutation test for an effect operator

## Usage

``` r
# S3 method for class 'effect_operator'
perm_test(
  x,
  nperm = 999,
  scheme = c("reduced_model"),
  parallel = FALSE,
  alpha = 0.05,
  stepwise = TRUE,
  alternative = c("greater", "less", "two.sided"),
  ...
)
```

## Arguments

- x:

  An `effect_operator`.

- nperm:

  Number of permutations.

- scheme:

  Permutation scheme. Currently only `"reduced_model"` is supported.

- parallel:

  Logical; if `TRUE`, use `future.apply`.

- alpha:

  Sequential significance threshold used to determine selected rank.

- stepwise:

  Logical; if `TRUE`, apply sequential rank testing by deflating
  previously selected effect directions before evaluating the next axis.

- alternative:

  Alternative hypothesis for empirical p-values.

- ...:

  Reserved for future extensions.

## Value

A permutation-test result object for effect operators.
