# Generic Permutation-Based Test

This generic function implements a permutation-based test to assess the
significance of components or statistics in a fitted model. The actual
procedure depends on the method defined for the specific model class.
Typical usage:

## Arguments

- x:

  A fitted model object (e.g. `pca`, `cross_projector`,
  `discriminant_projector`, `multiblock_biprojector`).

- ...:

  Additional arguments passed down to `shuffle_fun` or `measure_fun` (if
  applicable). Note: For `multiblock` methods, `Xlist`, `comps`,
  `alpha`, and `use_rspectra` (for biprojector) are handled as direct
  named arguments, not via `...`.

- X:

  (Used by `pca`, `cross_projector`, `discriminant_projector`) The
  original primary data matrix used to fit `x`. Ignored by the
  `multiblock_biprojector` method.

- Y:

  (Used by `cross_projector`) The secondary data block (n x pY). Ignored
  by other methods.

- Xlist:

  (Used by `multiblock_biprojector` \[optional, default `NULL`\] and
  `multiblock_projector` \[required\]) List of data blocks.

- nperm:

  Integer number of permutations (Default: 1000 for PCA, 500 for
  multiblock methods, 100 otherwise).

- measure_fun:

  (Optional; Used by `pca`, `cross_projector`, `discriminant_projector`,
  `multiblock_projector`) A function for computing the statistic(s) of
  interest. Ignored by `multiblock_biprojector`. Signature/default
  varies by method (see Details).

- shuffle_fun:

  (Optional; Used by all methods) A function for permuting the data
  appropriately. Signature/default varies by method (see Details).

- fit_fun:

  (Optional; Used by `cross_projector`, `discriminant_projector`) A
  function for re-fitting a new model. Ignored by PCA and multiblock
  methods. Signature/default varies by method (see Details).

- stepwise:

  (Used by `pca`) Logical indicating if sequential testing (P3
  projection) should be performed. Default `TRUE`. (The multiblock
  methods also perform sequential testing based on `alpha` and `comps`,
  but this argument is ignored). Ignored by other methods.

- parallel:

  (Used by all methods) Logical; if `TRUE`, attempt parallel execution
  via
  [`future.apply::future_lapply`](https://future.apply.futureverse.org/reference/future_lapply.html).

- alternative:

  (Used by all methods) Character string for the alternative hypothesis:
  "greater" (default), "less", or "two.sided".

- alpha:

  (Used by `pca`, `multiblock_biprojector`, `multiblock_projector`)
  Significance level for sequential stopping rule (default 0.05). Passed
  directly as a named argument to these methods.

- comps:

  (Used by `pca`, `multiblock_biprojector`, `multiblock_projector`)
  Maximum number of components to test sequentially (default 4). Passed
  directly as a named argument to these methods.

- use_svd_solver:

  (Used by `pca`) Optional string specifying the SVD solver (default
  "fast").

- use_rspectra:

  (Used by `multiblock_biprojector`) Logical indicating whether to use
  RSpectra for eigenvalue calculation (default `TRUE`). Passed directly
  as a named argument.

- predict_method:

  (Used by `discriminant_projector`) Prediction method (`"lda"` or
  `"euclid"`) used by the default measure function (default "lda").

## Value

The structure of the return value depends on the method:

- **`cross_projector`** and **`discriminant_projector`**::

  Returns an object of class `perm_test`, a list containing:
  `statistic`, `perm_values`, `p.value`, `alternative`, `method`,
  `nperm`, `call`.

- **`pca`**, **`multiblock_biprojector`**, and
  **`multiblock_projector`**::

  Returns an object inheriting from `perm_test` (classes
  `perm_test_pca`, `perm_test_multiblock`, or `perm_test` respectively
  for multiblock_projector), a list containing: `component_results`
  (data frame with observed stat, pval, CIs per component),
  `perm_values` (matrix of permuted stats), `alpha` (if applicable),
  `alternative`, `method`, `nperm` (vector of successful permutations
  per component), `call`.

## Details

1.  Shuffle or permute the data in a way that breaks the structure of
    interest (e.g., shuffle labels for supervised methods, shuffle
    columns/rows for unsupervised).

2.  Re-fit or re-project the model on the permuted data. Depending on
    the class, this can be done via a `fit_fun` or a class-specific
    approach.

3.  Measure the statistic of interest (e.g., variance explained,
    classification accuracy, canonical correlation).

4.  Compare the distribution of permuted statistics to the observed
    statistic to compute an empirical p-value.

S3 methods define the specific defaults and required signatures for the
functions involved in shuffling, fitting, and measuring.

This function provides a framework for permutation testing in various
multivariate models. The specific implementation details, default
functions, and relevant arguments vary by method.

**PCA Method (`perm_test.pca`):** Relevant arguments: `X`, `nperm`,
`measure_fun`, `shuffle_fun`, `stepwise`, `parallel`, `alternative`,
`alpha`, `comps`, `use_svd_solver`, `...`. Assesses significance of
variance explained by each PC (Vitale et al., 2017). Default statistic:
F_a. Default shuffle: column-wise. Default uses P3 projection and
sequential stopping with `alpha`.

**Cross Projector Method (`perm_test.cross_projector`):** Relevant
arguments: `X`, `Y`, `nperm`, `measure_fun`, `shuffle_fun`, `fit_fun`,
`parallel`, `alternative`, `...`. Tests the X-Y relationship. Default
statistic: `x2y.mse`. Default shuffle: rows of Y. Default fit:
[`stats::cancor`](https://rdrr.io/r/stats/cancor.html).

**Discriminant Projector Method (`perm_test.discriminant_projector`):**
Relevant arguments: `X`, `nperm`, `measure_fun`, `shuffle_fun`,
`fit_fun`, `predict_method`, `parallel`, `alternative`, `...`. Tests
class separation. Default statistic: prediction accuracy. Default
shuffle: labels. Default fit:
[`MASS::lda`](https://rdrr.io/pkg/MASS/man/lda.html).

**Multiblock Bi-Projector Method (`perm_test.multiblock_biprojector`):**
Relevant arguments: `Xlist` (optional), `nperm`, `shuffle_fun`,
`parallel`, `alternative`, `alpha`, `comps`, `use_rspectra`, `...`.
Tests consensus using fixed internal statistic (eigenvalue) on scores
for each component. The statistic is the leading eigenvalue of the
covariance matrix of block scores for a given component (T^T, where T
columns are scores of block *b* on component *k*). By default, it
shuffles rows within each block independently (either from `Xlist` if
provided via `...`, or using the internally stored scores). It performs
sequential testing for components specified by `comps` using the
stopping rule defined by `alpha` (both passed via `...`).

**Multiblock Projector Method (`perm_test.multiblock_projector`):**
Relevant arguments: `Xlist` (required), `nperm`, `measure_fun`,
`shuffle_fun`, `parallel`, `alternative`, `alpha`, `comps`, `...`. Tests
consensus using `measure_fun` (default: mean abs corr) on scores
projected from `Xlist` using the original model `x`. Does not refit.

## References

Buja, A., & Eyuboglu, N. (1992). Remarks on parallel analysis.
*Multivariate Behavioral Research*, 27(4), 509-540. (Relevant for PCA
permutation concepts)

Vitale, R., Westerhuis, J. A., Næs, T., Smilde, A. K., de Noord, O. E.,
& Ferrer, A. (2017). Selecting the number of factors in principal
component analysis by permutation testing— Numerical and practical
aspects. *Journal of Chemometrics*, 31(10), e2937.
[doi:10.1002/cem.2937](https://doi.org/10.1002/cem.2937) (Specific to
`perm_test.pca`)

## See also

[`pca`](https://bbuchsbaum.github.io/multivarious/reference/pca.md),
[`cross_projector`](https://bbuchsbaum.github.io/multivarious/reference/cross_projector.md),
[`discriminant_projector`](https://bbuchsbaum.github.io/multivarious/reference/discriminant_projector.md),
[`multiblock_biprojector`](https://bbuchsbaum.github.io/multivarious/reference/multiblock_biprojector.md),
[`measure_interblock_transfer_error`](https://bbuchsbaum.github.io/multivarious/reference/measure_interblock_transfer_error.md)

## Examples

``` r
# PCA Example
data(iris)
X_iris <- as.matrix(iris[,1:4])
mod_pca <- pca(X_iris, ncomp=4, preproc=center()) # Ensure centering

# Test first 3 components sequentially (faster with more nperm)
# Ensure a future plan is set for parallel=TRUE, e.g., future::plan("multisession")
res_pca <- perm_test(mod_pca, X_iris, nperm=50, comps=3, parallel=FALSE)
#> Pre-calculating reconstructions for stepwise testing...
#> Running 50 permutations sequentially for up to 3 PCA components (alpha=0.050, serial)...
#>   Testing Component 1/3...
#>   Testing Component 2/3...
#>   Testing Component 3/3...
#>   Component 3 p-value (0.05882) > alpha (0.050). Stopping sequential testing.
print(res_pca)
#> 
#> PCA Permutation Test Results
#> 
#> Method:  Permutation test for PCA (Vitale et al. 2017 P3) (statistic: F_a (Fraction of Remaining Variance), stepwise: TRUE, shuffle: column-wise) 
#> Alternative:  greater 
#> 
#> Component Results:
#>   comp  observed       pval  lower_ci  upper_ci
#> 1    1 0.9246187 0.01960784 0.6817051 0.6895225
#> 2    2 0.7039743 0.01960784 0.6045818 0.6920858
#> 3    3 0.7664247 0.05882353 0.6674626 0.7726917
#> 
#> Number of successful permutations per component: 50, 50, 50 

# PCA Example with row shuffling (tests different null hypothesis)
row_shuffle <- function(dat, ...) dat[sample(nrow(dat)), ]
res_pca_row <- perm_test(mod_pca, X_iris, nperm=50, comps=3,
                         shuffle_fun=row_shuffle, parallel=FALSE)
#> Pre-calculating reconstructions for stepwise testing...
#> Running 50 permutations sequentially for up to 3 PCA components (alpha=0.050, serial)...
#>   Testing Component 1/3...
#>   Component 1 p-value (0.4314) > alpha (0.050). Stopping sequential testing.
print(res_pca_row)
#> 
#> PCA Permutation Test Results
#> 
#> Method:  Permutation test for PCA (Vitale et al. 2017 P3) (statistic: F_a (Fraction of Remaining Variance), stepwise: TRUE, shuffle: custom) 
#> Alternative:  greater 
#> 
#> Component Results:
#>   comp  observed      pval  lower_ci  upper_ci
#> 1    1 0.9246187 0.4313725 0.9246187 0.9246187
#> 
#> Number of successful permutations per component: 50 
```
