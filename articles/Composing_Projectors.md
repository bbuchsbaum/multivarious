# Composing Projectors: Chaining Models

[`compose_partial_projector()`](https://bbuchsbaum.github.io/multivarious/reference/compose_partial_projector.md)
chains fitted projector objects whose dimensions line up. The output
scores from one stage become the input data for the next stage, and the
composed object can then be used with the same core verbs as a single
projector:
[`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md),
[`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md),
[`coef()`](https://rdrr.io/r/stats/coef.html),
[`inverse_projection()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_projection.md),
and
[`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md).

Composition is for fitted projectors, not for standalone preprocessing
steps. Use a model’s `preproc` argument for centering, scaling, or other
preprocessing that should happen inside a stage. Use composition when
one fitted projection is itself the training data for a later fitted
projection.

## 1. Fit compatible stages

This example fits a PCA on the original variables, then fits a second
PCA on the scores from the first PCA. The first stage maps 15 variables
to 8 scores. The second stage maps those 8 scores to 7 final scores.

``` r

set.seed(1)

X <- matrix(rnorm(30 * 15), nrow = 30, ncol = 15)

p1 <- pca(X, ncomp = 8)
p2 <- pca(scores(p1), ncomp = 7)

pipe <- compose_partial_projector(
  first = p1,
  second = p2
)

summary(pipe)
#> # A tibble: 2 × 5
#>   stage name   in_dim out_dim class
#>   <int> <chr>   <int>   <int> <chr>
#> 1     1 first      15       8 pca  
#> 2     2 second      8       7 pca
```

The summary is the first check to make: adjacent stages must have
matching output and input dimensions.

## 2. Project through the whole chain

Projecting with the composed object is equivalent to applying each stage
in sequence.

``` r

scores_pipe <- project(pipe, X)
dim(scores_pipe)
#> [1] 30  7
```

The infix operator `%>>%` is a shorter spelling for the same
composition.

``` r

pipe_with_operator <- p1 %>>% p2
dim(project(pipe_with_operator, X))
#> [1] 30  7
```

## 3. Use a subset of original variables

[`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md)
on a composed projector supports partial projection at the first stage.
The `colind` vector names the original columns supplied to that first
stage; all later stages receive the full output from the preceding
stage.

``` r

cols <- 1:5
scores_subset <- partial_project(
  pipe,
  X[, cols, drop = FALSE],
  colind = cols
)

dim(scores_subset)
#> [1] 30  7
```

Later-stage column subsetting is not currently implemented for composed
projectors. Supplying a list with non-`NULL` entries after the first
stage emits a warning and those later entries are ignored.

## 4. Inspect the combined linear map

For linear projectors such as PCA,
[`coef()`](https://rdrr.io/r/stats/coef.html) returns the effective
coefficient matrix for the whole chain: original variables by final
components.

``` r

V <- coef(pipe)
dim(V)
#> [1] 15  7
```

[`variables_used()`](https://bbuchsbaum.github.io/multivarious/reference/variables_used.md)
and
[`vars_for_component()`](https://bbuchsbaum.github.io/multivarious/reference/vars_for_component.md)
inspect this combined coefficient matrix. With dense PCA stages, every
original variable usually contributes to the final scores.

``` r

c(
  variables_used = length(variables_used(pipe)),
  variables_for_first_component = length(vars_for_component(pipe, 1))
)
#>                variables_used variables_for_first_component 
#>                            15                            15
```

## 5. Reconstruct from final scores

[`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
maps final scores back toward the original variable space using the
inverse projection for each stage. Because both PCA stages reduce
dimension, the reconstruction is a lossy approximation, not an exact
inverse.

``` r

X_hat <- reconstruct(pipe, scores_pipe)

c(
  rows = nrow(X_hat),
  columns = ncol(X_hat),
  rmse = sqrt(mean((X - X_hat)^2))
)
#>       rows    columns       rmse 
#> 30.0000000 15.0000000  0.4487182
```

## 6. Truncate the final stage

[`truncate()`](https://bbuchsbaum.github.io/multivarious/reference/truncate.md)
reduces the number of components retained from the last stage of a
composed projector. It does not refit or truncate intermediate stages.

``` r

pipe4 <- truncate(pipe, ncomp = 4)

summary(pipe4)
#> # A tibble: 2 × 5
#>   stage name   in_dim out_dim class
#>   <int> <chr>   <int>   <int> <chr>
#> 1     1 first      15       8 pca  
#> 2     2 second      8       4 pca
dim(project(pipe4, X))
#> [1] 30  4
```

## 7. Practical boundaries

Composition is useful when the stages form a valid sequence of fitted
projectors. The output dimension of each stage must match the input
dimension of the next stage. For preprocessing, use each model’s
preprocessing argument. For partial projection, treat `colind` as a
first-stage feature subset; it does not target later stages.
