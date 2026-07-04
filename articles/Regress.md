# Linear Re-representation with regress()

When you already have a known set of basis functions—such as Fourier
components, wavelets, splines, or principal components from a reference
dataset—the goal isn’t to discover a latent space, but rather to express
new data in terms of that existing basis. The
[`regress()`](https://bbuchsbaum.github.io/multivarious/reference/regress.md)
function facilitates this by wrapping several multi-output linear models
within the `bi_projector` API.

This integration means you automatically inherit standard `bi_projector`
methods for tasks like:

- [`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md):
  Map new data from the original space into the basis coefficients.
- [`inverse_projection()`](https://bbuchsbaum.github.io/multivarious/reference/inverse_projection.md):
  Map basis coefficients back to the original data space.
- [`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
  /
  [`reconstruct_new()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct_new.md):
  Reconstruct data using all or a subset of basis components.
- [`coef()`](https://rdrr.io/r/stats/coef.html): Retrieve the basis
  coefficients.
- [`truncate()`](https://bbuchsbaum.github.io/multivarious/reference/truncate.md):
  Keep only a subset of basis components.

This vignette demonstrates a typical workflow using an orthonormal
Fourier basis. The same interface also accepts non-orthogonal
dictionaries, with the chosen regression method determining how
coefficients are estimated.

------------------------------------------------------------------------

## 1. Build a design matrix of basis functions

First, let’s define our basis. We’ll use sines and cosines.

``` r

set.seed(42)
n  <- 128                       # Number of observations (e.g., signals)
p  <- 32                        # Original variables per observation (e.g., time points)
k  <- 20                        # Number of basis functions (<= p often, <= n for lm)

## Create simulated data: smooth signals plus noise
t   <- seq(0, 1, length.out = p)
Y   <- replicate(n,  3*sin(2*pi*3*t) + 2*cos(2*pi*5*t) ) + 
       matrix(rnorm(n*p, sd = 0.3), p, n) # Note: Y is p x n here

## Orthonormal Fourier basis (columns = basis functions)
# We create k/2 sine and k/2 cosine terms, plus an intercept
freqs <- 1:(k %/% 2) # Integer division for number of frequencies
B <- cbind(rep(1, p), # Intercept column
           do.call(cbind, lapply(freqs, function(f) sin(2*pi*f*t))),
           do.call(cbind, lapply(freqs, function(f) cos(2*pi*f*t))))
colnames(B) <- c("Intercept", paste0("sin", freqs), paste0("cos", freqs))

# Make columns orthonormal (length 1, orthogonal to each other)
B <- scale(B, center = FALSE, scale = sqrt(colSums(B^2))) 

cat(paste("Dimensions: Y is", nrow(Y), "x", ncol(Y), 
          ", Basis B is", nrow(B), "x", ncol(B), "\n"))
#> Dimensions: Y is 32 x 128 , Basis B is 32 x 21
# We want coefficients C (k x n) such that Y ≈ B %*% C.
```

------------------------------------------------------------------------

## 2. Fit multi-output regression

Now, we use
[`regress()`](https://bbuchsbaum.github.io/multivarious/reference/regress.md)
to find the coefficients $`C`$ that best represent each signal in $`Y`$
using the basis $`B`$.

``` r

# Fit using standard linear models (lm)
# Y is p x n (32 x 128): 32 time points x 128 signals
# B is p x k (32 x 21): 32 time points x 21 basis functions
# regress will fit 128 separate regressions, each with 32 observations and 21 predictors
fit <- regress(X = B,          # Predictors = basis functions (p x k)
               Y = Y,          # Response = signals (p x n)
               method    = "lm",
               intercept = FALSE) # Basis B already includes an intercept column

# The result is a bi_projector object
print(fit)
#> Regression bi_projector object:
#>   Method: lm
#>   Input dimension: 128
#>   Output dimension: 21
#>   Coefficients dimension:  128 x 21
```

The fitted object stores the basis matrix and the signal-by-basis
coefficient matrix. Use
[`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
for fitted values rather than relying on raw slots in downstream code.

------------------------------------------------------------------------

## 3. Go back and forth between spaces

With the fitted `bi_projector`, we can move freely between the original
data space and the basis coefficient space. This section demonstrates
the key operations.

### 3.1 Inspecting the fitted coefficients

The coefficient matrix returned by
[`coef()`](https://rdrr.io/r/stats/coef.html) has dimensions
$`n \times k`$ (signals $`\times`$ basis functions). Each row contains
the weights that express one signal as a linear combination of basis
functions.

``` r

coef_matrix_first3 <- coef(fit)[1:3, ]
cat("Coefficient matrix shape (first 3 signals):",
    nrow(coef_matrix_first3), "x", ncol(coef_matrix_first3), "\n\n")
#> Coefficient matrix shape (first 3 signals): 3 x 21

cat("Coefficients for signal 1:\n")
#> Coefficients for signal 1:
print(coef_matrix_first3[1, ])
#>    Intercept         sin1         sin2         sin3         sin4         sin5 
#>  0.135054901  0.587137215 -0.183816033 12.150464524 -0.007984793 -0.162281276 
#>         sin6         sin7         sin8         sin9        sin10         cos1 
#> -0.593723657  0.581637564 -0.370023776 -0.108499186  0.173654284  0.394261726 
#>         cos2         cos3         cos4         cos5         cos6         cos7 
#> -0.590122835  0.288835048  0.448031764  8.196393043  0.080690010 -0.612238653 
#>         cos8         cos9        cos10 
#>  0.678547189 -0.030446715  0.088104811
```

Notice that `sin3` and `cos5` have the largest coefficients—this matches
our generating function $`3\sin(2\pi \cdot 3t) + 2\cos(2\pi \cdot 5t)`$.

### 3.2 Reconstructing the fitted data

Reconstruction recovers fitted values in the original data space. The
[`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
method applies the matrix product stored in the fitted `bi_projector`
and returns a $`p \times n`$ fitted data matrix:

``` math
\hat{Y} \approx B \cdot C
```

``` r

Y_hat <- reconstruct(fit)
max_reconstruction_error <- max(abs(Y_hat - Y))

cat("Reconstruction shape:", nrow(Y_hat), "x", ncol(Y_hat), "\n")
#> Reconstruction shape: 32 x 128
cat("Maximum reconstruction error:", format(max_reconstruction_error, digits=3), "\n")
#> Maximum reconstruction error: 0.7
```

The error is non-zero because our signals contain random noise that
cannot be captured by the smooth Fourier basis. This is expected and
acceptable.

### 3.3 Projecting new data onto the basis

A key use case is expressing *new* observations in terms of the same
basis. For an orthonormal basis $`B`$, projection is straightforward:

``` math
\text{coefficients} = B^T \cdot Y_{\text{new}}
```

Let’s create a new signal with the same underlying pattern but different
noise:

``` r

Y_new_signal <- 3*sin(2*pi*3*t) + 2*cos(2*pi*5*t) + rnorm(p, sd=0.3)
Y_new_matrix <- matrix(Y_new_signal, nrow = p, ncol = 1)

coef_new <- t(fit$s) %*% Y_new_matrix

cat("Basis coefficients for new signal:\n")
#> Basis coefficients for new signal:
print(coef_new[, 1])
#>   Intercept        sin1        sin2        sin3        sin4        sin5 
#> -0.16459096  0.35921181 -0.09452109 11.90760497  0.37429968 -0.13899361 
#>        sin6        sin7        sin8        sin9       sin10        cos1 
#>  0.44535459 -0.16769033  0.02891241  0.54613517 -0.41145969  0.53832581 
#>        cos2        cos3        cos4        cos5        cos6        cos7 
#>  0.45371511  0.24009459  0.81766812  8.25333978  0.75410708 -0.42128770 
#>        cos8        cos9       cos10 
#>  0.49377610  0.45618692  0.79379189
```

Again, the `sin3` and `cos5` components dominate, as expected.

### 3.4 Reconstructing new data

Finally, we can reconstruct the new signal from its basis coefficients
to see how well the basis captures the underlying structure:

``` r

Y_new_recon <- fit$s %*% coef_new
reconstruction_error <- sqrt(mean((Y_new_matrix - Y_new_recon)^2))

cat("Reconstruction RMSE for new signal:", format(reconstruction_error, digits=3), "\n")
#> Reconstruction RMSE for new signal: 0.362
```

The reconstruction error reflects the noise that the basis cannot
represent. Because $`B`$ is orthonormal, projection and reconstruction
are exact inverses for the signal components that lie within the basis
span.

------------------------------------------------------------------------

## 4. Regularisation & PLS

If the basis is ill-conditioned or you need feature selection/shrinkage,
simply change the `method` argument.
[`regress()`](https://bbuchsbaum.github.io/multivarious/reference/regress.md)
wraps common regularized models.

``` r

# Ridge regression (requires glmnet)
fit_ridge <- regress(X = B, Y = Y, method = "mridge", lambda = 0.01, intercept = FALSE)

# Elastic Net (requires glmnet)
fit_enet  <- regress(X = B, Y = Y, method = "enet", alpha = 0.5, lambda = 0.02, intercept = FALSE)

# Partial Least Squares (requires pls package) - useful if k > p or multicollinearity
fit_pls   <- regress(X = B, Y = Y, method = "pls", ncomp = 15, intercept = FALSE)

# All these return bi_projector objects, so downstream code using 
# project(), reconstruct(), coef() etc. remains the same.
```

------------------------------------------------------------------------

## 5. Partial / custom mappings

The `bi_projector` interface allows for flexible manipulation:

``` r

# Truncate: keep only the first 5 basis functions (Intercept + 2 sine + 2 cosine)
fit5   <- truncate(fit, ncomp = 5) 
cat("Dimensions after truncating to 5 components:", 
    "Basis (s):", paste(dim(fit5$s), collapse="x"), 
    ", Coefs (v):", paste(dim(fit5$v), collapse="x"), "\n")
#> Dimensions after truncating to 5 components: Basis (s): 32x5 , Coefs (v): 128x5
# Reconstruction using only the first 5 basis functions
Y_hat5 <- reconstruct(fit5)

# Manual reconstruction example for components 2:6
coef_subset <- coef(fit)[, 2:6, drop=FALSE] # n x k_sub
basis_subset <- scores(fit)[, 2:6, drop=FALSE] # p x k_sub
Y_lowHat <- basis_subset %*% t(coef_subset) # (p x k_sub) %*% (k_sub x n) = p x n
```

------------------------------------------------------------------------

## 6. Under-the-hood: Matrix View

The core idea is to represent the $`p \times n`$ data matrix $`Y`$ as a
product of the $`p \times k`$ basis matrix and the transpose of the
$`n \times k`$ coefficient matrix:
``` math
 \underbrace{Y}_{p\times n} \approx \underbrace{s}_{p\times k} \underbrace{v^\top}_{k\times n} 
```

[`regress()`](https://bbuchsbaum.github.io/multivarious/reference/regress.md)
estimates $`v`$ (coefficients) based on the chosen method:

| `method` | Solver | Regularisation | Key Reference |
|----|----|----|----|
| “lm” | QR decomposition (`lm.fit`) | None | Classical OLS |
| “mridge” | `glmnet` (alpha=0) | Ridge ($`\lambda ||\beta||_2^2`$) | Hoerl & Kennard 1970 |
| “enet” | `glmnet` | Elastic Net ($`\alpha`$-mix) | Zou & Hastie 2005 |
| “pls” | [`pls::plsr`](https://khliland.github.io/pls/reference/mvr.html) | Latent PLS factors | Wold 1984 |

The resulting object stores: \* `v`: The estimated coefficient matrix
($`n \times k`$). \* `s`: The basis/design matrix $`B`$
($`p \times k`$), possibly centered or scaled by the underlying solver.
\* `sdev`, `center`: Potentially stores scaling/centering info related
to $`s`$.

All other `bi_projector` methods (`project`, `reconstruct`,
`inverse_projection`) are derived from these core matrices.

------------------------------------------------------------------------

## 7. Take-aways

- [`regress()`](https://bbuchsbaum.github.io/multivarious/reference/regress.md)
  provides a convenient way to fit multiple linear models simultaneously
  when expressing data $`Y`$ in a known basis $`B`$.
- It returns a `bi_projector`, giving immediate access to projection,
  reconstruction, truncation, and coefficient extraction.
- Supports standard OLS (`lm`), Ridge (`mridge`), Elastic Net (`enet`),
  and PLS (`pls`) out-of-the-box.
- Works with any basis dictionary (orthonormal or not).
- Can be integrated into larger analysis pipelines using composition
  (`%>>%`) or cross-validation helpers.
