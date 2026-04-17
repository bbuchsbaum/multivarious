# Fast Matrix Approximation with the Nyström Method

## The Scalability Problem

When working with similarity or kernel matrices, the computational cost
grows rapidly:

- **Memory**: O(N²) to store an N × N matrix
- **Time**: O(N³) for eigendecomposition

For N = 10,000, that’s 100 million entries and billions of operations.
For N = 100,000, it becomes intractable.

## The Nyström Approximation

The Nyström method provides a fast approximation by using only a subset
of “landmark” points:

1.  Select m \<\< N landmark points
2.  Compute the m × m matrix between landmarks
3.  Compute the N × m matrix between all points and landmarks
4.  Use these to approximate the full eigendecomposition

This reduces complexity from O(N³) to O(Nm²) — a massive speedup when m
is small.

## Quick Start

``` r
set.seed(42)
N <- 1000
p <- 50
X <- matrix(rnorm(N * p), N, p)

# Nyström approximation with linear kernel (default)
# Uses only 100 landmarks instead of all 1000 points
ny_fit <- nystrom_approx(

  X,
  ncomp = 10,
  nlandmarks = 100,
  preproc = center()
)

print(ny_fit)
#> A bi_projector object with the following properties:
#> 
#> Dimensions of the weights (v) matrix:
#>   Rows:  1000  Columns:  10 
#> 
#> Dimensions of the scores (s) matrix:
#>   Rows:  1000  Columns:  10 
#> 
#> Length of the standard deviations (sdev) vector:
#>   Length:  10 
#> 
#> Preprocessing information:
#> A finalized pre-processing pipeline:
#>  Step 1: center

# Standard bi_projector interface
head(scores(ny_fit))
#>             [,1]        [,2]       [,3]       [,4]         [,5]        [,6]
#> [1,]  1.45270292  0.03749928 -0.2368871 -0.1187156 -0.416524468  0.58381281
#> [2,] -0.08553264  0.21098943  2.0174421 -0.2780763  0.016232223  1.46634110
#> [3,]  1.35999975  2.38423969 -0.7357389 -0.4051195  0.289000747 -0.24190588
#> [4,] -0.14581306 -0.32474655 -1.6893599 -1.0902513 -0.972903175  0.03148513
#> [5,]  1.03295739 -0.85262013 -0.8882174  0.2777049 -1.048920775  0.12657430
#> [6,] -0.98488361  0.15283689 -1.1922802  0.5915995 -0.008715646 -2.10974250
#>            [,7]        [,8]       [,9]      [,10]
#> [1,]  0.1024324 -0.06344357 -1.8224682 -0.4924368
#> [2,]  1.1993310  0.27285706  1.4943231 -0.6030422
#> [3,]  0.9969271 -0.70227209 -1.0836289  0.5889711
#> [4,]  1.1429161 -0.14968205  0.7300808  1.3913713
#> [5,] -0.6772174 -0.36304203  0.4231099  1.5737886
#> [6,]  0.0955964  1.30091786  1.7640812  0.7613047
```

## Specifying a Kernel Function

By default,
[`nystrom_approx()`](https://bbuchsbaum.github.io/multivarious/reference/nystrom_approx.md)
uses a linear kernel. You can provide any kernel function:

``` r
# RBF (Gaussian) kernel
rbf_kernel <- function(X, Y = NULL, sigma = 1) {
  if (is.null(Y)) Y <- X

  sumX2 <- rowSums(X^2)
  sumY2 <- rowSums(Y^2)
  sqdist <- outer(sumX2, sumY2, `+`) - 2 * tcrossprod(X, Y)
  sqdist[sqdist < 0] <- 0

  exp(-sqdist / (2 * sigma^2))
}

ny_rbf <- nystrom_approx(
  X,
  kernel_func = rbf_kernel,
  ncomp = 10,
  nlandmarks = 100
)
```

## Projecting New Data

The result is a `bi_projector`, so you can project new observations:

``` r
X_new <- matrix(rnorm(50 * p), 50, p)
new_scores <- project(ny_fit, X_new)
dim(new_scores)
#> [1] 50 10
```

## Double Nyström for Extra Speed

For very large datasets, the “double” method applies the approximation
twice, reducing complexity further:

``` r
# Standard method
system.time(

  ny_standard <- nystrom_approx(X, ncomp = 5, nlandmarks = 200, method = "standard")
)
#>    user  system elapsed 
#>   0.009   0.007   0.005

# Double Nyström (faster with intermediate rank l)
system.time(
  ny_double <- nystrom_approx(X, ncomp = 5, nlandmarks = 200, method = "double", l = 50)
)
#>    user  system elapsed 
#>   0.019   0.022   0.011
```

## Choosing Parameters

### Number of Landmarks

| Dataset Size | Suggested Landmarks | Notes                       |
|--------------|---------------------|-----------------------------|
| \< 1,000     | 50–200              | Larger fraction acceptable  |
| 1,000–10,000 | 200–500             | Good accuracy/speed balance |
| \> 10,000    | 500–2,000           | Consider Double Nyström     |

### Method Selection

- **Standard**: Higher accuracy, use when m \< 2,000
- **Double**: Faster for very large m, adds intermediate rank parameter
  `l`

## Technical Details

Click for mathematical details

The Nyström approximation uses the fact that a positive semi-definite
matrix K can be approximated as:

$$K \approx K_{nm}K_{mm}^{- 1}K_{mn}$$

where: - $K_{mm}$ is the m × m matrix between landmarks - $K_{nm}$ is
the N × m matrix between all points and landmarks

The eigendecomposition of this approximation can be computed
efficiently: 1. Compute eigendecomposition of
$K_{mm} = U_{m}\Lambda_{m}U_{m}^{T}$ 2. Approximate eigenvectors of K
as: $U \approx K_{nm}U_{m}\Lambda_{m}^{- 1/2}$

Double Nyström applies this approximation recursively, reducing
complexity from O(Nm² + m³) to O(Nml + l³) where l \<\< m.

## References

- Williams, C. K. I., & Seeger, M. (2001). Using the Nyström Method to
  Speed Up Kernel Machines. *NIPS*.
- Lim, D., Jin, R., & Zhang, L. (2015). An Efficient and Accurate
  Nyström Scheme for Large-Scale Data Sets. *AAAI*.

## See Also

- [`pca()`](https://bbuchsbaum.github.io/multivarious/reference/pca.md)
  for standard PCA on smaller datasets
- [`svd_wrapper()`](https://bbuchsbaum.github.io/multivarious/reference/svd_wrapper.md)
  for various SVD backends
