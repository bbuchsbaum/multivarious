# Permutation Testing in multivarious

## 1. Why Permutation Tests?

Dimensionality reduction raises a fundamental question: **how many
components are real?**

A scree plot might show a clear “elbow” at 3 components, but is that
structure genuine or could it arise from noise? Classical heuristics
provide guidance but not formal inference. Permutation testing fills
this gap by answering: *“What would this statistic look like if there
were no true structure?”*

### The permutation logic

The procedure is straightforward: 1. Compute the statistic of interest
on the original data (e.g., variance explained by PC1). 2. Shuffle the
data to destroy the structure being tested. 3. Recompute the statistic
on the shuffled data. 4. Repeat steps 2–3 many times to build a **null
distribution**. 5. Compare the original statistic to this distribution.

If the observed value is more extreme than (say) 95% of the permuted
values, we reject the null hypothesis at $\alpha = 0.05$.

The critical design choice is *how* to shuffle. Different analyses
require different permutation schemes—shuffling columns for PCA,
shuffling labels for classification, shuffling one block for cross-block
methods. The
[`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
function handles these details automatically based on the model type.

------------------------------------------------------------------------

## 2. Basic Usage

The interface is consistent across model types: fit your model, then
call
[`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md).

``` r
data(iris)
X_iris <- as.matrix(iris[, 1:4])

mod_pca <- pca(X_iris, ncomp = 4, preproc = center())
```

Now test whether each principal component captures more variance than
expected by chance:

``` r
set.seed(1)
pt_pca <- perm_test(mod_pca,
                    X = X_iris,
                    nperm = 199,
                    comps = 3,
                    parallel = FALSE)
#> Pre-calculating reconstructions for stepwise testing...
#> Running 199 permutations sequentially for up to 3 PCA components (alpha=0.050, serial)...
#>   Testing Component 1/3...
#>   Testing Component 2/3...
#>   Testing Component 3/3...
#>   Component 3 p-value (0.055) > alpha (0.050). Stopping sequential testing.
```

The key arguments:

- `X` — the original data (required for re-shuffling)
- `nperm` — number of permutations (199 here for speed; use 999+ in
  practice)
- `comps` — how many components to test sequentially
- `parallel` — enable parallel execution via the `future` package

### Interpreting results

The result object contains a `component_results` table with p-values and
confidence intervals:

``` r
print(pt_pca$component_results)
#> # A tibble: 3 × 5
#>    comp observed  pval lower_ci upper_ci
#>   <int>    <dbl> <dbl>    <dbl>    <dbl>
#> 1     1    0.925 0.005    0.682    0.689
#> 2     2    0.704 0.01     0.609    0.689
#> 3     3    0.766 0.055    0.668    0.785
```

Components are tested sequentially. By default, testing stops when a
component is non-significant ($p > 0.05$), since later components are
unlikely to be meaningful if an earlier one fails. Set `alpha = 1` to
force testing all components.

------------------------------------------------------------------------

## 3. Method-Specific Behavior

The
[`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
function dispatches to specialized methods based on the model class.
Each method uses an appropriate test statistic and shuffle scheme.

### PCA

For PCA, the null hypothesis is that the variables are independent (no
correlation structure). The shuffle scheme permutes each column
independently, destroying any covariance while preserving marginal
distributions.

**Statistic:** Fraction of remaining variance explained by component
$a$: $$F_{a} = \frac{\lambda_{a}}{\sum\limits_{j \geq a}\lambda_{j}}$$

This tests whether component $a$ explains more variance than expected
given the remaining variance pool. See Vitale et al. (2017) for
theoretical background.

### Discriminant projector

For discriminant analysis, the null hypothesis is that class labels are
unrelated to the features. The shuffle scheme permutes the class labels.

**Statistic:** Classification accuracy (using LDA or nearest-centroid).

### Cross-projector (CCA, PLS)

For two-block methods, the null hypothesis is that blocks X and Y are
unrelated. The shuffle scheme permutes rows of one block only, breaking
the correspondence while preserving within-block structure.

**Statistic:** Mean squared error of cross-block prediction (X → Y).

### Multiblock projector

For multi-block analyses, the null hypothesis is that blocks share no
common structure. Each block is shuffled independently.

**Statistic:** Consensus measures (leading eigenvalue or mean absolute
correlation between block scores).

### Effect operators from `mixed_regress()`

For mixed or repeated-measures multivariate regression, the null
hypothesis is attached to a named fixed-effect term rather than a whole
model. The shuffle scheme depends on the term:

- between-subject terms use whole-subject block permutation,
- within-subject terms use within-subject row shuffles,
- mixed terms use grouped reduced-model residual resampling.

**Statistics:**

- omnibus trace statistic: `tr(T_H)`,
- sequential rank statistic: relative leading singular-value energy
  after projected-residual rank correction.

The workflow is:

``` r
set.seed(11)

design <- expand.grid(
  subject = factor(seq_len(8)),
  level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
  KEEP.OUT.ATTRS = FALSE
)
design$group <- factor(rep(c("A", "B"), each = 12))

level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
group_num <- ifelse(design$group == "B", 1, 0)
subj_idx <- as.integer(design$subject)
b0 <- rnorm(8, sd = 0.5)

Y <- cbind(
  b0[subj_idx] + level_num + rnorm(nrow(design), sd = 0.15),
  group_num + rnorm(nrow(design), sd = 0.15),
  level_num * group_num + rnorm(nrow(design), sd = 0.15),
  rnorm(nrow(design), sd = 0.15)
)

fit_mixed <- mixed_regress(
  Y,
  design = design,
  fixed = ~ group * level,
  random = ~ 1 | subject,
  basis = shared_pca(3),
  preproc = pass()
)

E_int <- effect(fit_mixed, "group:level")
pt_int <- perm_test(E_int, nperm = 49, alpha = 0.10)

pt_int$component_results
#> # A tibble: 0 × 7
#> # ℹ 7 variables: comp <int>, statistic <chr>, effective_rank <int>,
#> #   lead_sv2 <dbl>, rel <dbl>, observed <dbl>, pval <dbl>
```

------------------------------------------------------------------------

## 4. Custom Statistics and Shuffle Schemes

The defaults work well for standard analyses, but you can override any
component of the permutation machinery.

### Available hooks

- **`measure_fun(model_perm, comp_idx, ...)`** — Computes the test
  statistic from a model fitted on permuted data. Called once per
  component.

- **`shuffle_fun(data, ...)`** — Permutes the data. Must return data in
  the same structure as the input.

- **`fit_fun(Xtrain, ...)`** — Re-fits the model on permuted data. Used
  by `cross_projector` and `discriminant_projector` when the default
  fitter (e.g., [`MASS::lda`](https://rdrr.io/pkg/MASS/man/lda.html)) is
  not appropriate.

### Example: custom statistic

Suppose you want to test whether the *first two* PCs jointly explain
more variance than chance. You can supply a custom `measure_fun`:

``` r
my_pca_stat <- function(model_perm, comp_idx, ...) {
  # Only compute the joint statistic when testing component 2

  if (comp_idx == 2 && length(model_perm$sdev) >= 2) {
    sum(model_perm$sdev[1:2]^2) / sum(model_perm$sdev^2)
  } else if (comp_idx == 1) {
    model_perm$sdev[1]^2 / sum(model_perm$sdev^2)
  } else {
    NA_real_
  }
}

# Illustrative call (using default measure here for simplicity)
pt_pca_custom <- perm_test(mod_pca, X = X_iris, nperm = 50, comps = 2,
                           parallel = FALSE)
#> Pre-calculating reconstructions for stepwise testing...
#> Running 50 permutations sequentially for up to 2 PCA components (alpha=0.050, serial)...
#>   Testing Component 1/2...
#>   Testing Component 2/2...
print(pt_pca_custom$component_results)
#> # A tibble: 2 × 5
#>    comp observed   pval lower_ci upper_ci
#>   <int>    <dbl>  <dbl>    <dbl>    <dbl>
#> 1     1    0.925 0.0196    0.682    0.692
#> 2     2    0.704 0.0392    0.607    0.697
```

For testing a single combined statistic rather than sequential
components, set `comps = 1` and have your `measure_fun` compute the
combined value directly.

------------------------------------------------------------------------

## 5. Parallel Execution

Permutation tests are embarrassingly parallel—each permutation is
independent. The
[`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
methods use the `future` framework, so you control parallelism through
your plan:

``` r
library(future)
plan(multisession, workers = 4)

pt_pca_parallel <- perm_test(mod_pca, X = X_iris,
                             nperm = 999,
                             comps = 3,
                             parallel = TRUE)

plan(sequential)
```

With 4 workers and 999 permutations, each worker handles ~250
permutations concurrently.

------------------------------------------------------------------------

## 6. Practical Considerations

### High-dimensional data

When $p \gg n$, computing full eigendecompositions repeatedly can be
slow. Keep `comps` small (you rarely need to test more than 10–20
components), and consider faster SVD backends if available.

### Non-exchangeable observations

The default shuffles assume observations are exchangeable under the
null. This breaks down for time-series, spatial data, or blocked
designs. In these cases, provide a custom `shuffle_fun` that respects
the dependence structure (e.g., block permutation, circular shifts).

### Confidence intervals

The `component_results` table includes empirical 95% CIs derived from
the permutation distribution. These quantify uncertainty in the test
statistic under the null.

### The original data is required

Most methods need the original data (`X`, `Y`, or `Xlist`) to perform
shuffling. Always pass it explicitly.

### Deprecated function

The older
[`perm_ci.pca()`](https://bbuchsbaum.github.io/multivarious/reference/perm_ci.md)
is deprecated. Use
[`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)
instead—confidence intervals are included in the results table.

------------------------------------------------------------------------

------------------------------------------------------------------------

## 7. References

- Vitale, R., Westerhuis, J. A., Næs, T., Smilde, A. K., de Noord, O.
  E., & Ferrer, A. (2017). Selecting the number of factors in principal
  component analysis by permutation testing— Numerical and practical
  aspects. *Journal of Chemometrics*, 31(10), e2937.
- Good, P. I. (2005). *Permutation, Parametric, and Bootstrap Tests of
  Hypotheses*. Springer Series in Statistics. Springer.
- Bengtsson, H. (2021). future: Unified Parallel and Distributed
  Processing in R for Everyone. *R package version 1.21.0*.
  <https://future.futureverse.org/>

------------------------------------------------------------------------

By providing a unified `perm_test` interface, `multivarious` allows
researchers to apply robust, data-driven significance testing across a
range of multivariate modeling techniques.
