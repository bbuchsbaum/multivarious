# Mixed Effect Operators with mixed_regress()

## 1. Operator-valued ANOVA

[`mixed_regress()`](https://bbuchsbaum.github.io/multivarious/reference/mixed_regress.md)
treats each named fixed-effect term as a multivariate object rather than
a scalar test. The package-level idea is:

- fit the row-side geometry once,
- extract a named effect,
- analyze that effect with the existing `multivarious` verbs.

For a term `H`, the effect matrix is

``` math
M_H = P_H^{(\Omega)} Y B
```

where:

- `Y` is the stacked observation-by-feature response matrix,
- `P_H^{(\Omega)}` is the covariance-weighted projector for the term,
- `B` is an optional shared feature basis.

The returned `effect_operator` behaves like a `bi_projector`, so you can
call
[`components()`](https://bbuchsbaum.github.io/multivarious/reference/components.md),
[`scores()`](https://bbuchsbaum.github.io/multivarious/reference/scores.md),
[`truncate()`](https://bbuchsbaum.github.io/multivarious/reference/truncate.md),
[`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md),
[`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md),
and
[`bootstrap()`](https://bbuchsbaum.github.io/multivarious/reference/bootstrap.md)
directly.

------------------------------------------------------------------------

## 2. Simulate a repeated-measures design

We generate a simple low/mid/high repeated-measures dataset with a
between-subject group factor, a random intercept, and a random slope on
the ordered within-subject effect.

``` r
set.seed(1)

n_subject <- 18
levels_within <- c("low", "mid", "high")

design <- expand.grid(
  subject = factor(seq_len(n_subject)),
  level = factor(levels_within, levels = levels_within),
  KEEP.OUT.ATTRS = FALSE
)

subject_group <- rep(c("A", "B"), length.out = n_subject)
design$group <- factor(subject_group[as.integer(design$subject)])

level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
group_num <- ifelse(design$group == "B", 1, 0)
subj_idx <- as.integer(design$subject)

b0 <- rnorm(n_subject, sd = 0.7)
b1 <- rnorm(n_subject, sd = 0.3)

n <- nrow(design)
p <- 8

Y <- cbind(
  b0[subj_idx] + 1.2 * level_num + rnorm(n, sd = 0.2),
  0.8 * group_num + rnorm(n, sd = 0.2),
  1.4 * level_num * group_num + rnorm(n, sd = 0.2),
  -0.9 * level_num + rnorm(n, sd = 0.2),
  b1[subj_idx] * level_num + rnorm(n, sd = 0.2),
  rnorm(n, sd = 0.2),
  rnorm(n, sd = 0.2),
  rnorm(n, sd = 0.2)
)

dim(Y)
#> [1] 54  8
```

------------------------------------------------------------------------

## 3. Fit the model

The current implementation supports one grouping variable and a shared
row covariance across features. You can supply either a single
random-effects bar such as `~ 1 + level | subject` or multiple bars that
collapse to the same grouping variable, such as
`~ (1 | subject) + (0 + level | subject)`.

``` r
fit <- mixed_regress(
  Y,
  design = design,
  fixed = ~ group * level,
  random = ~ 1 + level | subject,
  basis = shared_pca(ncomp = 4),
  preproc = center()
)

print(fit)
#> mixed_fit object
#> 
#> Observations: 54
#> Features: 8
#> Terms: group, level, group:level
#> Basis rank: 4
#> Row metric: grouped_lmm
#> Grouping variable: subject
summary(fit)
#>                    term df_term   scope
#> group             group       1 between
#> level             level       2  within
#> group:level group:level       2   mixed
```

The fit stores:

- the design matrix,
- the grouped row metric,
- the shared feature basis,
- metadata for named fixed-effect terms.

------------------------------------------------------------------------

## 4. Extract a named effect

Now extract the interaction effect as a first-class multivariate object.

``` r
E <- effect(fit, "group:level")

print(E)
#> effect_operator
#> 
#> Term: group:level
#> Components: 2
#> Term df: 2
#> Scope: mixed
#> Basis rank: 4
ncomp(E)
#> [1] 2
components(E)[1:8, ]
#>             [,1]        [,2]
#> [1,]  0.06948227 -0.34743768
#> [2,] -0.04199354 -0.44344259
#> [3,] -0.94008393 -0.11457637
#> [4,]  0.19045696  0.43476790
#> [5,]  0.26452948 -0.65875982
#> [6,]  0.01487796 -0.07191431
#> [7,]  0.04325303 -0.06124667
#> [8,]  0.03618121 -0.19392665
```

Because `E` is an `effect_operator` and inherits from `bi_projector`,
the familiar decomposition grammar carries over:

- `components(E)` gives feature directions,
- `scores(E)` gives observation-side effect scores,
- `truncate(E, k)` keeps only the first `k` axes.

------------------------------------------------------------------------

## 5. Reconstruct the effect

You can reconstruct the fitted contribution of the effect on different
scales.

``` r
E_proc <- reconstruct(E, scale = "processed")
E_orig <- reconstruct(E, scale = "original")

dim(E_proc)
#> [1] 54  8
dim(E_orig)
#> [1] 54  8
round(E_orig[1:6, 1:4], 3)
#>        [,1]   [,2]   [,3]   [,4]
#> [1,] -0.054  0.069  0.992 -0.224
#> [2,]  0.054 -0.069 -0.992  0.224
#> [3,] -0.054  0.069  0.992 -0.224
#> [4,]  0.054 -0.069 -0.992  0.224
#> [5,] -0.054  0.069  0.992 -0.224
#> [6,]  0.054 -0.069 -0.992  0.224
```

Typical choices:

- `scale = "whitened"` for the covariance-adjusted effect geometry,
- `scale = "processed"` for the response scale after preprocessing,
- `scale = "original"` for the final effect contribution on the original
  variables.

------------------------------------------------------------------------

## 6. Omnibus and rank inference

[`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
works directly on the extracted effect object.

``` r
set.seed(2)
pt <- perm_test(E, nperm = 99, alpha = 0.10)

print(pt)
#> 
#> Effect operator permutation test
#> 
#> Term: group:level
#> Method: Reduced-model residual permutation test for effect_operator with sequential deflation
#> Exchangeability: whole-subject trajectory permutation within equal block-size strata
#> Omnibus statistic (trace): 242.6
#> Omnibus p-value: 0.01
#> Selected rank: 1
#> 
#>   comp statistic effective_rank   lead_sv2       rel   observed pval
#> 1    1  lead_sv2              2 240.665744 0.9922062 240.665744 0.01
#> 2    2  lead_sv2              1   1.890438 1.0000000   1.890438 0.44
pt$component_results
#> # A tibble: 2 × 7
#>    comp statistic effective_rank lead_sv2   rel observed  pval
#>   <int> <chr>              <int>    <dbl> <dbl>    <dbl> <dbl>
#> 1     1 lead_sv2               2   241.   0.992   241.    0.01
#> 2     2 lead_sv2               1     1.89 1         1.89  0.44
```

The permutation result provides:

- an omnibus trace test,
- a sequential rank test based on relative singular-value statistics,
- `ncomp(pt)` as the selected number of significant effect axes.

``` r
k <- ncomp(pt)
E_sig <- truncate(E, k)

k
#> [1] 1
ncomp(E_sig)
#> [1] 1
```

------------------------------------------------------------------------

## 7. Stability by bootstrap

Permutation asks whether an effect exists. Bootstrap asks whether the
geometry is stable under subject resampling.

``` r
set.seed(3)
bres <- bootstrap(E, nboot = 49, resample = "subject")

print(bres)
#> Bootstrap stability for effect_operator
#> 
#> Term: group:level
#> Bootstrap samples: 49
#> Resampling unit: subject
#> Mean singular values: 38399626.3353, 5862255.8293
bres$singular_values_mean
#> [1] 38399626  5862256
```

The bootstrap result contains means and standard deviations for:

- singular values,
- feature loadings,
- full loading arrays across resamples.

------------------------------------------------------------------------

## 8. Array input

Repeated-measures arrays can be supplied directly. Internally they are
normalized to the same stacked representation.

``` r
Y_array <- array(NA_real_, dim = c(n_subject, length(levels_within), p))
idx <- 1
for (i in seq_len(n_subject)) {
  for (j in seq_along(levels_within)) {
    Y_array[i, j, ] <- Y[idx, ]
    idx <- idx + 1
  }
}

fit_array <- mixed_regress(
  Y_array,
  design = design,
  fixed = ~ group * level,
  random = ~ 1 | subject,
  basis = shared_pca(ncomp = 4),
  preproc = center()
)

effect(fit_array, "level")$term
#> [1] "level"
```

------------------------------------------------------------------------

## 9. Current scope

The present implementation is intentionally narrow:

- Gaussian multivariate responses,
- one grouping variable,
- random intercepts and random slopes,
- grouped permutation and subject bootstrap,
- shared feature basis or identity basis.

The calibration harness used for the empirical checks in this vignette
lives in `experimental/mixed_effect_operator_calibration.R`. Batch
outputs can be written to `experimental/results/` for larger Monte Carlo
runs outside the package test suite.

Still to come:

- richer exchangeability schemes for repeated-measures contrasts,
- broader support beyond one grouping variable,
- more extensive mixed-model examples across design families.

------------------------------------------------------------------------
