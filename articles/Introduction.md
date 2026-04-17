# Introduction to the multivarious Package

## The Goal: Unified Dimensionality Reduction

Multivariate data analysis often involves reducing dimensionality or
transforming data using techniques like Principal Component Analysis
(PCA), Partial Least Squares (PLS), Contrastive PCA (cPCA), Nyström
approximation for Kernel PCA, or representing data in a specific basis
(e.g., Fourier, splines). While each method has unique mathematical
underpinnings, they share common operational needs:

- Fitting the model to training data.
- Extracting key components (scores, loadings/coefficients).
- Projecting *new* data points into the reduced/transformed space.
- Reconstructing approximations of the original data from the reduced
  space.
- Integrating these steps with pre-processing (like centering or
  scaling).
- Comparing or tuning models using cross-validation.
- Extracting named multivariate effects from repeated-measures or mixed
  designs.

Handling these tasks consistently across different algorithms can lead
to repetitive code and complex workflows. The **`multivarious` package
aims to simplify this by providing a unified interface** centered around
the concept of a **`bi_projector`**.

## The `bi_projector`: A Two-Way Map

The `bi_projector` class is the cornerstone of `multivarious`. It
represents a linear transformation (or an approximation thereof) that
provides a **two-way mapping**:

1.  **Samples (Rows) ↔︎ Scores:** Maps data points from the original
    high-dimensional space to a lower-dimensional latent space (scores),
    and potentially back.
2.  **Variables (Columns) ↔︎ Components/Loadings:** Maps original
    variables to their representation in the latent space
    (loadings/components), and potentially back.

Think of it as encapsulating the core results of a dimensionality
reduction technique (like the U, S, V components of an SVD, or the
scores and loadings of PCA/PLS) along with any necessary pre-processing
information.

Crucially, many functions within `multivarious` (e.g.,
[`pca()`](https://bbuchsbaum.github.io/multivarious/reference/pca.md),
`pls()`,
[`cPCAplus()`](https://bbuchsbaum.github.io/multivarious/reference/cPCAplus.md),
[`nystrom_approx()`](https://bbuchsbaum.github.io/multivarious/reference/nystrom_approx.md),
[`regress()`](https://bbuchsbaum.github.io/multivarious/reference/regress.md))
return objects that inherit from `bi_projector`.

The same grammar now extends to mixed-effect operator analysis:

- [`mixed_regress()`](https://bbuchsbaum.github.io/multivarious/reference/mixed_regress.md)
  fits the row-side geometry of a repeated-measures or mixed design.
- [`effect()`](https://bbuchsbaum.github.io/multivarious/reference/effect.md)
  extracts a named fixed-effect term as an `effect_operator`.
- `effect_operator` inherits from `bi_projector`, so
  [`components()`](https://bbuchsbaum.github.io/multivarious/reference/components.md),
  [`scores()`](https://bbuchsbaum.github.io/multivarious/reference/scores.md),
  [`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md),
  [`truncate()`](https://bbuchsbaum.github.io/multivarious/reference/truncate.md),
  [`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md),
  and
  [`bootstrap()`](https://bbuchsbaum.github.io/multivarious/reference/bootstrap.md)
  all work in the same style.

## Key Actions with a `bi_projector`

Because different methods return a `bi_projector`, you can perform
common tasks using a consistent set of verbs:

- `scores(model)`: Get the scores (latent space representation) of the
  *training* data.
- `coef(model)` or `loadings(model)`: Get the loadings or coefficients
  mapping variables to components.
- `project(model, newdata)`: Project *new* samples (rows of `newdata`)
  into the latent space defined by the `model`.
- `reconstruct(model, ...)`: Reconstruct an approximation of the
  original data from the latent space (either from training scores or
  provided new scores/coefficients).
- `truncate(model, ncomp)`: Reduce the number of components kept in the
  model.
- `summary(model)`: Get a concise summary of the model dimensions.

This consistent API simplifies writing generic analysis code and makes
it easier to swap between different dimensionality reduction methods.

## Example: PCA Workflow

Let’s demonstrate a typical workflow using PCA on the classic `iris`
dataset.

``` r
# Load iris dataset and select numeric columns
data(iris)
X <- as.matrix(iris[, 1:4])

# 1. Define a pre-processor (center the data)
preproc <- center()

# 2. Fit PCA using svd_wrapper, keeping 3 components
#    The pre-processor is applied internally.
fit <- pca(X, ncomp = 3, preproc = preproc)

# The result 'fit' is a bi_projector
print(fit)
#> PCA object  -- derived from SVD
#> 
#> Data: 150 observations x 4 variables
#> Components retained: 3
#> 
#> Variance explained (per component):
#>  1 2 3  92.95  5.33  1.72%  (cumulative:  92.95 98.28   100%)

# 3. Access results
iris_scores <- scores(fit) # Scores of the centered training data (150 x 3)
iris_loadings <- loadings(fit) # Loadings (4 x 3)
cat("\nDimensions of Scores:", dim(iris_scores), "\n")
#> 
#> Dimensions of Scores: 150 3
cat("Dimensions of Loadings:", dim(iris_loadings), "\n")
#> Dimensions of Loadings:

# 4. Project new data
# Create some new iris-like samples (5 samples, 4 variables)
set.seed(123)
new_iris_data <- matrix(rnorm(5 * 4, mean = colMeans(X), sd = apply(X, 2, sd)), 
                        nrow = 5, byrow = TRUE)

# Project the new data into the PCA space defined by 'fit'
# Pre-processing (centering using training data means) is applied automatically.
projected_new_scores <- project(fit, new_iris_data)
cat("\nDimensions of Projected New Data Scores:", dim(projected_new_scores), "\n")
#> 
#> Dimensions of Projected New Data Scores: 5 3
print(head(projected_new_scores))
#>            [,1]       [,2]        [,3]
#> [1,] -2.2172144  0.8590909 -0.44924532
#> [2,] -0.3270495 -0.5478369  0.07965279
#> [3,] -1.7602954  0.9106117 -0.52932939
#> [4,]  0.2367242 -0.3204326 -0.50433574
#> [5,] -1.1529598  0.5426518  0.85478044

# 5. Reconstruct approximated original data from scores
# Reconstruct the first few original samples
reconstructed_X_approx <- reconstruct(fit, comp=1:3) # uses scores(fit) by default
cat("\nReconstructed Approximation of Original Data (first 5 rows):\n")
#> 
#> Reconstructed Approximation of Original Data (first 5 rows):
print(head(reconstructed_X_approx))
#>          [,1]     [,2]     [,3]      [,4]
#> [1,] 5.099286 3.500723 1.401086 0.1982949
#> [2,] 4.868758 3.031661 1.447517 0.1253679
#> [3,] 4.693700 3.206384 1.309582 0.1849507
#> [4,] 4.623843 3.075837 1.463736 0.2569583
#> [5,] 5.019326 3.580414 1.370606 0.2461680
#> [6,] 5.407635 3.892262 1.688387 0.4182392

print(head(X)) # Original data for comparison
#>      Sepal.Length Sepal.Width Petal.Length Petal.Width
#> [1,]          5.1         3.5          1.4         0.2
#> [2,]          4.9         3.0          1.4         0.2
#> [3,]          4.7         3.2          1.3         0.2
#> [4,]          4.6         3.1          1.5         0.2
#> [5,]          5.0         3.6          1.4         0.2
#> [6,]          5.4         3.9          1.7         0.4
```

This example shows how fitting (`pca`), accessing results (`scores`,
`loadings`), and applying the model to new data (`project`) follow a
consistent pattern, regardless of whether the underlying method was PCA,
PLS, or another technique returning a `bi_projector`.

## Example: Mixed effect operators

The same package grammar applies to repeated-measures multivariate
effects.

``` r
set.seed(99)

design_m <- expand.grid(
  subject = factor(seq_len(6)),
  level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
  KEEP.OUT.ATTRS = FALSE
)
design_m$group <- factor(rep(c("A", "B"), each = 9))

level_num <- c(low = -1, mid = 0, high = 1)[as.character(design_m$level)]
group_num <- ifelse(design_m$group == "B", 1, 0)
subj_idx <- as.integer(design_m$subject)
b0 <- rnorm(6, sd = 0.5)

Y_m <- cbind(
  b0[subj_idx] + level_num + rnorm(nrow(design_m), sd = 0.15),
  group_num + rnorm(nrow(design_m), sd = 0.15),
  level_num * group_num + rnorm(nrow(design_m), sd = 0.15),
  rnorm(nrow(design_m), sd = 0.15)
)

fit_m <- mixed_regress(
  Y_m,
  design = design_m,
  fixed = ~ group * level,
  random = ~ 1 | subject,
  basis = shared_pca(3),
  preproc = pass()
)

E_gl <- effect(fit_m, "group:level")
pt_gl <- perm_test(E_gl, nperm = 19, alpha = 0.10)

print(E_gl)
#> effect_operator
#> 
#> Term: group:level
#> Components: 0
#> Term df: 0
#> Scope: within
#> Basis rank: 3
pt_gl$component_results
#> # A tibble: 0 × 7
#> # ℹ 7 variables: comp <int>, statistic <chr>, effective_rank <int>,
#> #   lead_sv2 <dbl>, rel <dbl>, observed <dbl>, pval <dbl>
```

This is the same projector grammar in a different setting:

- fit once with
  [`mixed_regress()`](https://bbuchsbaum.github.io/multivarious/reference/mixed_regress.md),
- extract a named effect,
- decompose it,
- test it,
- reconstruct it.

## Beyond Basic Projection: The `multivarious` Ecosystem

The unified `bi_projector` interface enables several powerful features
within the package:

- **Pre-processing Pipelines:** Define reusable pre-processing steps
  (see
  [`vignette("PreProcessing")`](https://bbuchsbaum.github.io/multivarious/articles/PreProcessing.md)).
- **Model Composition:** Chain multiple `bi_projector` steps together
  (e.g., pre-processing → PCA → rotation) into a single composite
  projector (see
  [`vignette("Composing_Projectors")`](https://bbuchsbaum.github.io/multivarious/articles/Composing_Projectors.md)).
- **Cross-Validation:** Easily perform cross-validation to select
  hyperparameters (like the number of components) using helpers that
  understand the `bi_projector` structure (see
  [`vignette("CrossValidation")`](https://bbuchsbaum.github.io/multivarious/articles/CrossValidation.md)).

## Projecting Variables (`project_vars`)

While
[`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md)
operates on new samples (rows), the `bi_projector` also supports
projecting new *variables* (columns) into the component space defined by
the model’s scores (U vectors in SVD terms). This is done using
[`project_vars()`](https://bbuchsbaum.github.io/multivarious/reference/project_vars.md).

``` r
# Using the 'fit' object from the PCA example above

# Create a new variable (column) with the same number of samples as original data
set.seed(456)
new_variable <- rnorm(nrow(X))

# Project this new variable into the component space defined by the PCA scores (fit$s)
# Result shows how the new variable relates to the principal components.
projected_variable_loadings <- project_vars(fit, new_variable)
cat("\nProjection of new variable onto components:", projected_variable_loadings, "\n")
#> 
#> Projection of new variable onto components: 0.0003082567 -0.0004245081 -0.0003111904
```

## Conclusion

The `multivarious` package provides a consistent and extensible
framework for common dimensionality reduction and related linear
transformation tasks. By leveraging the `bi_projector` class, it offers
a unified API for fitting models, projecting new data, reconstruction,
and accessing key model components. This simplifies workflows, promotes
code reuse, and facilitates integration with pre-processing, model
composition, and cross-validation tools within the package ecosystem.
