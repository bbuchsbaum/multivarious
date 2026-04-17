# Permutation test for PLSC latent variables

Uses row-wise permutation of the Y block to assess the significance of
each latent variable (LV) in a fitted `plsc` model. The test statistic
is the singular value of the cross-covariance matrix for each LV.

## Usage

``` r
# S3 method for class 'plsc'
perm_test(
  x,
  X,
  Y,
  nperm = 1000,
  comps = ncomp(x),
  stepwise = TRUE,
  shuffle_fun = NULL,
  parallel = FALSE,
  alternative = c("greater", "less", "two.sided"),
  alpha = 0.05,
  ...
)
```

## Arguments

- x:

  A fitted `plsc` model object.

- X:

  Original X block used to fit `x`.

- Y:

  Original Y block used to fit `x`.

- nperm:

  Number of permutations to perform (default 1000).

- comps:

  Number of components (LVs) to test. Defaults to `ncomp(x)`.

- stepwise:

  Logical; if TRUE (default), perform sequential testing with deflation.

- shuffle_fun:

  Optional function to permute Y; defaults to shuffling rows.

- parallel:

  Logical; if TRUE, use parallel processing via future.apply.

- alternative:

  Character string for the alternative hypothesis: "greater" (default),
  "less", or "two.sided".

- alpha:

  Significance level used to report `n_significant`; not used directly
  in p-value calculation.

- ...:

  Additional arguments (currently unused).
