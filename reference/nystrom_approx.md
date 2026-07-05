# Nyström approximation for kernel-based decomposition (Unified Version)

Approximate the eigen-decomposition of a large kernel matrix K using
either the standard Nyström method (Williams & Seeger, 2001) or the
Double Nyström method (Lim et al., 2015, Algorithm 3).

## Usage

``` r
nystrom_approx(
  X,
  kernel_func = NULL,
  ncomp = NULL,
  landmarks = NULL,
  nlandmarks = 10,
  preproc = pass(),
  method = c("standard", "double"),
  center = FALSE,
  l = NULL,
  use_RSpectra = TRUE,
  ...
)
```

## Arguments

- X:

  A numeric matrix or data frame of size (N x D), where N is number of
  samples.

- kernel_func:

  A kernel function with signature `kernel_func(X, Y, ...)`. If NULL,
  defaults to a linear kernel: `X %*% t(Y)`.

- ncomp:

  Number of components (eigenvectors/eigenvalues) to return. Cannot
  exceed the number of landmarks. Default capped at `length(landmarks)`.

- landmarks:

  A vector of row indices (1-based, from X) specifying the landmark
  points. If NULL, `nlandmarks` points are sampled uniformly at random.

- nlandmarks:

  The number of landmark points to sample if `landmarks` is NULL.
  Default is 10.

- preproc:

  A pre-processing pipeline object (e.g., from
  [`prep()`](https://bbuchsbaum.github.io/multivarious/reference/prep.md))
  or a pre-processing function (default
  [`pass()`](https://bbuchsbaum.github.io/multivarious/reference/pass.md))
  to apply before computing the kernel.

- method:

  Either "standard" (the classic single-stage Nyström) or "double" (the
  two-stage Double Nyström method).

- center:

  Logical. If TRUE, attempts kernel centering. Default FALSE. **Note:**
  True kernel centering (required for equivalence to Kernel PCA) is
  computationally expensive and **not fully implemented**. Setting
  `center=TRUE` currently only issues a warning. For results equivalent
  to standard PCA, use a linear kernel and center the input data `X`
  (e.g., via `preproc`). See Details.

- l:

  Intermediate rank for the double Nyström method. Ignored if
  `method="standard"`. Typically, `l < length(landmarks)` to reduce
  complexity.

- use_RSpectra:

  Logical. If TRUE, use
  [`RSpectra::svds`](https://rdrr.io/pkg/RSpectra/man/svds.html) for
  partial SVD. Recommended for large problems.

- ...:

  Additional arguments passed to `kernel_func`.

## Value

A `bi_projector` object with class "nystrom_approx" and additional
fields:

- `v`:

  The eigenvectors (N x ncomp) approximating the kernel eigenbasis.

- `s`:

  The scores (N x ncomp) = v \* diag(sdev), analogous to principal
  component scores.

- `sdev`:

  The square roots of the eigenvalues.

- `preproc`:

  The pre-processing pipeline used.

- `meta`:

  A list containing parameters and intermediate results used (method,
  landmarks, kernel_func, etc.).

## Details

The Double Nyström method introduces an intermediate step that reduces
the size of the decomposition problem, potentially improving efficiency
and scalability.

**Kernel Centering:** Standard Kernel PCA requires the kernel matrix K
to be centered in the feature space (Schölkopf et al., 1998). This
implementation currently **does not perform kernel centering** by
default (`center=FALSE`) due to computational complexity. Consequently,
with non-linear kernels, the results approximate the eigen-decomposition
of the *uncentered* kernel matrix, and are not strictly equivalent to
Kernel PCA. If using a linear kernel, centering the input data `X`
(e.g., using `preproc=prep(center())`) yields results equivalent to
standard PCA, which is often sufficient.

**Standard Nyström:** Uses the method from Williams & Seeger (2001),
including the `sqrt(m/N)` scaling for eigenvectors and `N/m` for
eigenvalues (`m` landmarks, `N` samples).

**Double Nyström:** Implements Algorithm 3 from Lim et al. (2015).

## References

Schölkopf, B., Smola, A., & Müller, K. R. (1998). Nonlinear component
analysis as a kernel eigenvalue problem. *Neural computation*, 10(5),
1299-1319.

Williams, C. K. I., & Seeger, M. (2001). Using the Nyström Method to
Speed Up Kernel Machines. In *Advances in Neural Information Processing
Systems 13* (pp. 682-688).

Lim, D., Jin, R., & Zhang, L. (2015). An Efficient and Accurate Nystrom
Scheme for Large-Scale Data Sets. *Proceedings of the Twenty-Ninth AAAI
Conference on Artificial Intelligence* (pp. 2765-2771).

## Examples

``` r
set.seed(123)
# Smaller example matrix
X <- matrix(rnorm(1000*300), 1000, 300)

# Standard Nyström
res_std <- nystrom_approx(X, ncomp=5, nlandmarks=50, method="standard")
print(res_std)
#> A bi_projector object with the following properties:
#> 
#> Dimensions of the weights (v) matrix:
#>   Rows:  1000  Columns:  5 
#> 
#> Dimensions of the scores (s) matrix:
#>   Rows:  1000  Columns:  5 
#> 
#> Length of the standard deviations (sdev) vector:
#>   Length:  5 
#> 
#> Preprocessing information:
#> A finalized pre-processing pipeline:
#>  Step 1: pass

# Double Nyström
res_db <- nystrom_approx(X, ncomp=5, nlandmarks=50, method="double", l=20)
print(res_db)
#> A bi_projector object with the following properties:
#> 
#> Dimensions of the weights (v) matrix:
#>   Rows:  1000  Columns:  5 
#> 
#> Dimensions of the scores (s) matrix:
#>   Rows:  1000  Columns:  5 
#> 
#> Length of the standard deviations (sdev) vector:
#>   Length:  5 
#> 
#> Preprocessing information:
#> A finalized pre-processing pipeline:
#>  Step 1: pass

# Projection (using standard result as example)
scores_new <- project(res_std, X[1:10,])
head(scores_new)
#>            [,1]       [,2]       [,3]        [,4]       [,5]
#> [1,] -0.3464966 -2.4798337  0.6040714 -0.62223962  0.5059013
#> [2,] -0.1623986 -0.4074079 -0.8435420 -1.38722424 -0.3488808
#> [3,]  0.6435716 -0.2531178 -0.7252837  0.35537372 -0.7208270
#> [4,]  0.1593062 -0.2865374  0.5394600 -0.02853512 -0.5659083
#> [5,] -1.5810568 -6.8534098  3.2928589 -0.59725008 -0.8724898
#> [6,] -0.6895022  1.3136927  1.0744290 -0.93430486 -1.5582248
```
