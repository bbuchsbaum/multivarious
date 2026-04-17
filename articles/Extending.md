# Extending multiblock: CCA and glmnet Examples

This vignette demonstrates how to wrap existing statistical models
within the `multiblock` framework, specifically using `cross_projector`
for Canonical Correlation Analysis (CCA) and `projector` for `glmnet`
(penalized regression).

## 1. Canonical Correlation Analysis with `cross_projector`

### 1.1 Data Setup

We use the classic Iris dataset and split its features into two blocks:

- **X-block**: Sepal.Length, Sepal.Width
- **Y-block**: Petal.Length, Petal.Width

``` r
data(iris)
X <- as.matrix(iris[, 1:2])
Y <- as.matrix(iris[, 3:4])

# Show first few rows of combined data
head(cbind(X, Y))
#>      Sepal.Length Sepal.Width Petal.Length Petal.Width
#> [1,]          5.1         3.5          1.4         0.2
#> [2,]          4.9         3.0          1.4         0.2
#> [3,]          4.7         3.2          1.3         0.2
#> [4,]          4.6         3.1          1.5         0.2
#> [5,]          5.0         3.6          1.4         0.2
#> [6,]          5.4         3.9          1.7         0.4
```

### 1.2 Wrap `stats::cancor()` into a `cross_projector`

We create a fitting function that runs standard CCA using
[`stats::cancor()`](https://rdrr.io/r/stats/cancor.html) and then stores
the resulting coefficients (loadings) and preprocessing steps in a
`cross_projector` object.

``` r
fit_cca <- function(Xtrain, Ytrain, ncomp = 2, ...) {
  # Step 1: Define and fit preprocessors with training data
  # Use center()+z-score for both blocks (swap to center() only if preferred)
  preproc_x <- fit(colscale(center(), type = "z"), Xtrain)
  preproc_y <- fit(colscale(center(), type = "z"), Ytrain)

  # Step 2: Transform training data with the fitted preprocessors
  Xp <- transform(preproc_x, Xtrain)
  Yp <- transform(preproc_y, Ytrain)

  # Step 3: Run CCA on the preprocessed data (no additional centering here)
  cc <- stats::cancor(Xp, Yp, xcenter = FALSE, ycenter = FALSE)

  # Step 4: Store loadings and preprocessors in a cross_projector
  cp <- cross_projector(
    vx         = cc$xcoef[, 1:ncomp, drop = FALSE],
    vy         = cc$ycoef[, 1:ncomp, drop = FALSE],
    preproc_x  = preproc_x,
    preproc_y  = preproc_y,
    classes    = "cca_cross_projector"
  )
  attr(cp, "can_cor") <- cc$cor[1:ncomp]
  cp
}

# Fit the model
cp_cca <- fit_cca(X, Y)
print(cp_cca)
#> cross projector:  cca_cross_projector cross_projector projector 
#> input dim (X):  2 
#> output dim (X):  2 
#> input dim (Y):  2 
#> output dim (Y):  2
attr(cp_cca, "can_cor") # Show canonical correlations
#> [1] 0.9409690 0.1239369
```

**What does this `cross_projector` enable?**

- **Projection:** Map new X or Y data into the shared latent space
  defined by CCA (`project(cp_cca, newX, from = "X")`).
- **Transfer/Prediction:** Predict the Y block from the X block, or
  vice-versa (`transfer(cp_cca, newX, from = "X", to = "Y")`).
- **Partial Features:** Project or transfer using only a subset of
  features from X or Y (`partial_project`).
- **Integration:** Use standardized `multiblock` tools like
  cross-validation (`cv`) and permutation testing (`perm_test`).

### 1.3 Bidirectional Prediction (Transfer)

Let’s predict the Y-block (Petal measurements) using the X-block (Sepal
measurements).

``` r
Y_hat <- transfer(cp_cca, X, from = "X", to = "Y")
head(round(Y_hat, 2))
#>      [,1] [,2]
#> [1,] 1.84 0.54
#> [2,] 1.91 0.32
#> [3,] 1.32 0.15
#> [4,] 1.21 0.05
#> [5,] 1.54 0.46
#> [6,] 2.06 0.85

measure_reconstruction_error(Y, Y_hat, metrics = c("rmse", "r2"))
#> # A tibble: 1 × 2
#>    rmse    r2
#>   <dbl> <dbl>
#> 1 0.588 0.900
```

### 1.4 Using Partial Features

What if we only have Sepal.Length (the first column of X) at prediction
time? We can still project into the latent space:

``` r
new_x_partial <- X[, 1, drop = FALSE]

scores_partial <- partial_project(cp_cca, new_x_partial,
                                  colind = 1,
                                  source = "X")
head(round(scores_partial, 3))
#>       [,1]   [,2]
#> [1,] 0.065 -0.035
#> [2,] 0.083 -0.045
#> [3,] 0.100 -0.054
#> [4,] 0.109 -0.059
#> [5,] 0.074 -0.040
#> [6,] 0.039 -0.021
```

These scores represent the best estimate of position in the latent space
given only partial input. This is useful for missing data scenarios or
when only some measurements are available.

### 1.5 Cross-validated Component Selection

Cross-validation for two-block models requires wrapper functions that
handle both X and Y blocks. The pattern is:

``` r
set.seed(1)
folds <- kfold_split(nrow(X), k = 5)

cv_fit <- function(train_data, ncomp) {
  fit_cca(train_data$X, train_data$Y, ncomp = ncomp)
}

cv_measure <- function(model, test_data) {
  measure_interblock_transfer_error(
    test_data$X, test_data$Y, model,
    metrics = c("x2y.rmse", "y2x.rmse")
  )
}

cv_res <- cv_generic(
  data = list(X = X, Y = Y),
  folds = folds,
  .fit_fun = cv_fit,
  .measure_fun = cv_measure,
  fit_args = list(ncomp = 2)
)
```

The key is packaging X and Y together so fold indices apply to both
blocks simultaneously.

### 1.6 Permutation Test

We can test the significance of the X-Y relationship using
[`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md).
The default shuffles rows of Y relative to X and measures whether the
observed transfer error is lower than expected by chance.

``` r
perm_res <- perm_test(
  cp_cca, X, Y = Y,
  nperm = 199,
  alternative = "less"
)
print(perm_res)
```

The default statistic is `x2y.mse` (mean squared error when predicting Y
from X). A significant result (low p-value) indicates the CCA
relationship is unlikely due to chance.

### 1.7 CCA Take-aways

- `cross_projector` provides a convenient wrapper for two-block methods
  like CCA.
- It enables using standard `multiblock` verbs (`project`, `transfer`,
  `partial_project`, `cv`, `perm_test`) with the wrapped model.
- Preprocessing is handled internally, reducing risk of data leakage.
- Other methods (PLS, O2PLS) can be integrated by providing a different
  `fit_fun`.

------------------------------------------------------------------------

## 2. Example: Wrapping `glmnet` as a `projector`

The `projector` object maps data from its original space to a
lower-dimensional space defined by its components (`v`). This is useful
for dimensionality reduction (like PCA) but can also represent
coefficients from supervised models like penalized regression.

Here, we fit LASSO using `glmnet` and wrap the resulting coefficients
(excluding the intercept) into a `projector`.

### 2.1 Data and Model Fitting

``` r
# Generate sample data
set.seed(123)
n_obs <- 100
n_pred <- 50
X_glm <- matrix(rnorm(n_obs * n_pred), n_obs, n_pred)
# True coefficients (sparse)
true_beta <- matrix(0, n_pred, 1)
true_beta[1:10, 1] <- runif(10, -1, 1)
# Response variable with noise
y_glm <- X_glm %*% true_beta + rnorm(n_obs, sd = 0.5)

# Fit glmnet (LASSO, alpha=1)
# Typically use cv.glmnet to find lambda, but using a fixed one for simplicity
glm_fit <- glmnet::glmnet(X_glm, y_glm, alpha = 1) # alpha=1 is LASSO

# Choose a lambda (e.g., one near the end of the path)
chosen_lambda <- glm_fit$lambda[length(glm_fit$lambda) * 0.8]

# Get coefficients for this lambda
beta_hat <- coef(glm_fit, s = chosen_lambda)
print(paste("Number of non-zero coefficients:", sum(beta_hat != 0)))
#> [1] "Number of non-zero coefficients: 44"

# Extract coefficients, excluding the intercept
v_glm <- beta_hat[-1, 1, drop = FALSE] # Drop intercept, ensure it's a matrix
dim(v_glm) # Should be n_pred x 1
#> [1] 50  1
```

### 2.2 Create the `projector`

We define a `projector` using these coefficients. The projection
`X %*% v` gives a single score per observation, representing the LASSO
linear predictor. We also include centering and scaling as
preprocessing, often recommended for `glmnet`.

``` r
# Define preprocessor
# Define and fit preprocessor with training data
preproc_glm <- fit(colscale(center(), type = "z"), X_glm)

# Create the projector
proj_glm <- projector(
  v = v_glm,
  preproc = preproc_glm,
  classes = "glmnet_projector"
)

print(proj_glm)
#> Projector object:
#>   Input dimension: 50
#>   Output dimension: 1
#>   With pre-processing:
#> A finalized pre-processing pipeline:
#>  Step 1: center
#>  Step 2: colscale
```

### 2.3 Project New Data

We can now use this `projector` to calculate the LASSO linear predictor
score for new data.

``` r
# Generate some new test data
X_glm_test <- matrix(rnorm(20 * n_pred), 20, n_pred)

# Project the test data
# project() handles applying the centering/scaling from preproc_glm
lasso_scores <- project(proj_glm, X_glm_test)

head(round(lasso_scores, 3))
#>      s=0.004767049
#> [1,]         0.660
#> [2,]        -1.142
#> [3,]         4.503
#> [4,]         0.249
#> [5,]        -2.710
#> [6,]        -0.867
dim(lasso_scores) # Should be n_test x 1 (since v_glm has 1 column)
#> [1] 20  1

# Compare with direct calculation using transform
# Apply the same preprocessing used within project()
X_glm_test_processed <- transform(preproc_glm, X_glm_test)
# Calculate scores directly using the processed data and coefficients
direct_scores <- X_glm_test_processed %*% v_glm
head(round(direct_scores, 3))
#> 6 x 1 Matrix of class "dgeMatrix"
#>      s=0.004767049
#> [1,]         0.660
#> [2,]        -1.142
#> [3,]         4.503
#> [4,]         0.249
#> [5,]        -2.710
#> [6,]        -0.867

# Check they are close
all.equal(c(lasso_scores), c(direct_scores))
#> [1] "Modes: numeric, list"               "Lengths: 20, 1"                    
#> [3] "target is numeric, current is list"
```
