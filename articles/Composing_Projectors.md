# Composing Projectors: Chaining Models

The composed partial projector (`compose_partial_projector`) lets you
snap together any number of ordinary projector objects (PCA, PLS,
cPCA++, block projectors, …) and treat the whole chain as if it were a
single map from the original input space to the final output space:

$$\left. {\mathbb{R}}^{p_{\text{orig}}}\rightarrow{\mathbb{R}}^{q_{\text{final}}} \right.$$

**Typical Motives:**

| Why compose?                                                             | What you get                                                             |
|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| Pre-whitening, centring or wavelet-decomposition before the “real” model | Keep the preparation and the model in one tidy object.                   |
| Block-wise modelling (e.g. one PCA per sensor block)                     | Treat the concatenation of block-specific results as a single projector. |
| Dimensionality milk-run – reduce \> filter \> reduce again               | A single set of scores from the final stage to feed to a classifier.     |

------------------------------------------------------------------------

## 1. Quick start – two PCAs in series

Let’s compose two PCA steps:

``` r
set.seed(1)

X  <- matrix(rnorm(30*15), 30, 15)   # raw data, 30 samples, 15 variables
p1 <- pca(X, ncomp = 8)              # first reduction: 15 -> 8 components
p2 <- pca(scores(p1), ncomp = 7)     # second reduction: 8 -> 4 components

# Compose the two projectors
pipe <- compose_partial_projector(
           first  = p1,
           second = p2)

print(pipe)
#> Composed projector object:
#>   Number of projectors:  2 
#>  Pipeline:
#>    1. first  :   15 ->    8
#>    2. second :    8 ->    7

# Project original data through the entire pipeline
S <- project(pipe, X)          # 30 × 4 scores – as if the two steps were one
dim(S)
#> [1] 30  7

# Get a summary of the pipeline stages
summary(pipe)
#> # A tibble: 2 × 5
#>   stage name   in_dim out_dim class
#>   <int> <chr>   <int>   <int> <chr>
#> 1     1 first      15       8 pca  
#> 2     2 second      8       7 pca
```

The [`summary()`](https://rdrr.io/r/base/summary.html) output provides a
clear overview of the stages, their names, input/output dimensions, and
underlying class.

------------------------------------------------------------------------

## 2. Partial projections – “zoom in” on selected variables

[`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)
works on composed projectors, allowing you to apply projections using
only a subset of variables at specific stages.

You supply the `colind` argument as either:

- A **vector**: Applies only to the *first* stage. Subsequent stages
  receive the full output from the preceding stage.
- A **list**: One entry per stage. Use `NULL` for a stage that should
  receive the full input from the previous stage.

``` r
# Example 1: Use only variables 1:5 for the *first* PCA stage.
# The second PCA stage receives the full 8 components from the (partial) first stage.
S15 <- partial_project(pipe, X[, 1:5, drop=FALSE], colind = 1:5)
cat("Dimensions after partial projection (cols 1:5 in first stage):", dim(S15), "\n")
#> Dimensions after partial projection (cols 1:5 in first stage): 30 7

# Example 2: Multi-stage pipeline (conceptual)
# Imagine a 3-stage pipeline: wavelets -> PCA (block1) -> PCA (global)
# pipe2 <- wavelet_projector(...) %>>% 
#          pca(..., ncomp = 10)   %>>% 
#          pca(..., ncomp = 3)

# To focus on coefficients 12:20 *after* the wavelet step (i.e., input to stage 2):
# S_sel <- partial_project(pipe2, X, # Assuming X is appropriate input for wavelets
#                          colind = list(NULL, 12:20, NULL))
# Note: The indices in the list always refer to the dimensions *entering* that specific stage.
```

Behind the scenes, the composed projector manages the mapping of indices
through the pipeline.

------------------------------------------------------------------------

## 3. Reconstruction & inverse projection

Since each stage typically provides a way to reverse its projection
(often via
[`inverse_projection()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_projection.md)),
the composed projector can also reconstruct the original data from the
final scores.

``` r
# Reconstruct original data from the final scores 'S'
X_hat <- reconstruct(pipe, S)
cat("Dimensions of reconstructed data:", dim(X_hat), "\n")
#> Dimensions of reconstructed data: 30 15

# Check reconstruction accuracy
# Note: Since the pipeline involves dimensionality reduction (15 -> 8 -> 4),
# reconstruction will not be exact. The error reflects the information lost.
max_reconstruction_error <- max(abs(X - X_hat))
cat("Maximum absolute reconstruction error:", format(max_reconstruction_error, digits=3), "\n")
#> Maximum absolute reconstruction error: 1.47
# stopifnot(max_reconstruction_error < 1e-5) # Removed: This check is too strict for lossy reconstruction

# Get the overall coefficient matrix (p_orig x q_final)
V <- coef(pipe)
cat("Dimensions of overall coefficient matrix:", dim(V), "\n")
#> Dimensions of overall coefficient matrix: 15 7

# Get the overall pseudo-inverse matrix (q_final x p_orig)
Vplus <- inverse_projection(pipe)
cat("Dimensions of overall inverse projection matrix:", dim(Vplus), "\n")
#> Dimensions of overall inverse projection matrix: 7 15
```

Both the forward (`coef`) and inverse (`inverse_projection`) matrices
for the *entire* pipeline are calculated and potentially cached for
efficiency.

------------------------------------------------------------------------

## 4. House-keeping helpers

Some useful helper functions:

- **`%>>%`**: A pipe operator specifically for composing projectors. It
  preserves stage names if the projectors are named.

  ``` r
  # pipe3 <- pca1 %>>% pca2 %>>% pca3
  ```

- **`truncate(pipe, ncomp = k)`**: Safely reduces the number of
  components kept from the *last* stage of the pipeline.

- **`variables_used(pipe)` / `vars_for_component(pipe, k)`**: (Potential
  future helpers) Intended to trace which original variables contribute
  to the final scores, especially useful if any stages perform variable
  selection.

------------------------------------------------------------------------

## 6. Where next?

Composed projectors open up possibilities:

- Combine pre-processing (e.g., centering, scaling), dimensionality
  reduction (PCA, PLS), and perhaps an orthogonal rotation (Varimax,
  Procrustes) into a single, deployable modeling artifact.
- Future enhancements might allow tracing the lineage of specific final
  components back to the exact original variables that contribute most
  significantly, leveraging the internal index mapping.

Happy composing!
