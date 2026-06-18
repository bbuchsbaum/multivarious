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
  refit_basis = FALSE,
  seed = NULL,
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

  Alternative hypothesis for empirical p-values. Only `"greater"` is
  supported for effect-operator permutation tests.

- refit_basis:

  Logical; if `TRUE`, refit the feature basis per permutation from that
  permutation's reduced-model residual (and refit the observed basis
  from the observed reduced-model residual). Uses the full whitened
  feature space. Keeps the same basis rank as the static fit.
  Experimental; intended for evaluating whether static-basis leakage
  drives miscalibration.

- seed:

  Optional integer seed for reproducibility.

- ...:

  Reserved for future extensions.

## Value

A permutation-test result object for effect operators.
