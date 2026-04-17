# Contrastive PCA++ (cPCA++) Performs Contrastive PCA++ (cPCA++) to find directions that capture variation enriched in a "foreground" dataset relative to a "background" dataset. This implementation follows the cPCA++ approach which directly solves the generalized eigenvalue problem Rf v = lambda Rb v, where Rf and Rb are the covariance matrices of the foreground and background data, centered using the *background mean*.

Contrastive PCA++ (cPCA++) Performs Contrastive PCA++ (cPCA++) to find
directions that capture variation enriched in a "foreground" dataset
relative to a "background" dataset. This implementation follows the
cPCA++ approach which directly solves the generalized eigenvalue problem
Rf v = lambda Rb v, where Rf and Rb are the covariance matrices of the
foreground and background data, centered using the *background mean*.

## Usage

``` r
cPCAplus(
  X_f,
  X_b,
  ncomp = NULL,
  center_background = TRUE,
  lambda = 0,
  method = c("geigen", "primme", "sdiag", "corpcor"),
  strategy = c("auto", "feature", "sample"),
  verbose = getOption("multivarious.verbose", TRUE),
  sample_rank = NULL,
  sample_oversample = 10L,
  ...
)
```

## Arguments

- X_f:

  A numeric matrix representing the foreground dataset (samples x
  features).

- X_b:

  A numeric matrix representing the background dataset (samples x
  features). `X_f` and `X_b` must have the same number of features
  (columns).

- ncomp:

  Integer. The number of contrastive components to compute. Defaults to
  `min(ncol(X_f), nrow(X_f), nrow(X_b))`, and may be further capped by
  the effective background rank (especially under the sample-space
  strategy).

- center_background:

  Logical. If TRUE (default), both `X_f` and `X_b` are centered using
  the column means of `X_b`. If FALSE, it assumes data is already
  appropriately centered.

- lambda:

  Shrinkage intensity for covariance estimation (0 \<= lambda \<= 1).
  Defaults to 0 (no shrinkage). Uses
  [`corpcor::cov.shrink`](https://rdrr.io/pkg/corpcor/man/cov.shrink.html).
  Can help stabilize results if `Rb` is ill-conditioned or singular.

- method:

  A character string specifying the primary computation method. Options
  include:

  - `"geigen"` (Default): Use `geneig` from the `geigen` package.

  - `"primme"`: Use `geneig` with the PRIMME library backend (requires
    special `geigen` build).

  - `"sdiag"`: Use `geneig` with a spectral decomposition method.

  - `"corpcor"`: Use a corpcor-based whitening approach followed by
    standard PCA.

- strategy:

  Controls the GEVD approach when `method` is not `"corpcor"`. Options
  include:

  - `"auto"` (Default): Chooses based on dimensions (feature vs. sample
    space).

  - `"feature"`: Forces direct computation via `p x p` covariance
    matrices.

  - `"sample"`: Forces sample-space computation via SVD and a smaller
    GEVD (efficient for large `p`).

- verbose:

  Logical; if TRUE (default), prints brief status messages about
  strategy selection and defaults. Set to FALSE to silence these
  messages.

- sample_rank:

  Optional integer controlling the background subspace rank used in the
  sample-space strategy. If `NULL` (default), uses the full background
  rank `min(n_b-1, p)`. If provided, the solver will target
  approximately `sample_rank + sample_oversample` and will be bounded
  above by the full background rank.

- sample_oversample:

  Integer oversampling margin (default 10) applied when `sample_rank` is
  given. Ignored when `sample_rank` is `NULL`.

- ...:

  Additional arguments passed to the underlying computation functions
  (`geigen::geneig` or
  [`irlba::irlba`](https://rdrr.io/pkg/irlba/man/irlba.html) based on
  `method` and `strategy`).

## Value

A `bi_projector`-like object with classes
`c("cPCAplus", "<method_class>", "bi_projector")` containing:

- v:

  Loadings matrix (features x ncomp). Interpretation depends on `method`
  (see Details).

- s:

  Scores matrix (samples_f x ncomp).

- sdev:

  Vector (length ncomp). Standard deviations (sqrt of generalized
  eigenvalues for `geigen` methods, PCA std devs for `corpcor`).

- values:

  Vector (length ncomp). Generalized eigenvalues (for `geigen` methods)
  or PCA eigenvalues (for `corpcor`).

- strategy:

  The strategy used ("feature" or "sample") if method was not "corpcor".

- preproc:

  The initialized `preprocessor` object used.

- method:

  The computation method used.

- ncomp:

  The number of components computed.

- nfeatures:

  The number of features.

## Details

**Preprocessing:** Following the cPCA++ paper, if
`center_background = TRUE`, both `X_f` and `X_b` are centered by
subtracting the column means calculated *only* from the background data
`X_b`. This is crucial for isolating variance specific to `X_f`.

**Core Algorithm (methods "geigen", "primme", "sdiag",
strategy="feature"):**

1.  Center `X_f` and `X_b` using the mean of `X_b`.

2.  Compute potentially shrunk \\p \times p\\ covariance matrices `Rf`
    (from centered `X_f`) and `Rb` (from centered `X_b`) using
    [`corpcor::cov.shrink`](https://rdrr.io/pkg/corpcor/man/cov.shrink.html).

3.  Solve the generalized eigenvalue problem `Rf v = lambda Rb v` for
    the top `ncomp` eigenvectors `v` using `geigen::geneig`. These
    eigenvectors are the contrastive principal components (loadings).

4.  Compute scores by projecting the centered foreground data onto the
    eigenvectors: `S = X_f_centered %*% v`.

**Core Algorithm (Large-D / Sample Space Strategy, strategy="sample"):**
When \\p \gg n\\, forming \\p \times p\\ matrices `Rf` and `Rb` is
infeasible. The "sample" strategy follows cPCA++ §3.2:

1.  Center `X_f` and `X_b` using the mean of `X_b`.

2.  Compute the SVD of centered \\X_b = Ub Sb Vb^T\\ (using `irlba` for
    efficiency).

3.  Project centered `X_f` into the background's principal subspace:
    `Zf = X_f_centered %*% Vb`.

4.  Form small \\r \times r\\ matrices: `Rf_small = cov(Zf)` and
    `Rb_small = (1/(n_b-1)) * Sb^2`.

5.  Solve the small \\r \times r\\ GEVD:
    `Rf_small w = lambda Rb_small w` using `geigen::geneig`.

6.  Lift eigenvectors back to feature space: `v = Vb %*% w`.

7.  Compute scores: `S = X_f_centered %*% v`.

**Alternative Algorithm (method "corpcor"):**

1.  Center `X_f` and `X_b` using the mean of `X_b`.

2.  Compute `Rb` and its inverse square root `Rb_inv_sqrt`.

3.  Whiten the foreground data:
    `X_f_whitened = X_f_centered %*% Rb_inv_sqrt`.

4.  Perform standard PCA
    ([`stats::prcomp`](https://rdrr.io/r/stats/prcomp.html)) on
    `X_f_whitened`.

5.  The returned `v` and `s` are the loadings and scores *in the
    whitened space*. The loadings are *not* the generalized eigenvectors
    `v`. A specific class `corpcor_pca` is added to signal this.

## References

Abid, A., Zhang, M. J., Bagaria, V. K., & Zou, J. (2018). Exploring
patterns enriched in a dataset with contrastive principal component
analysis. Nature Communications, 9(1), 2134.

Salloum, R., & Kuo, C. C. J. (2022). cPCA++: An efficient method for
contrastive feature learning. Pattern Recognition, 124, 108378.

Wu, M., Sun, Q., & Yang, Y. (2025). PCA++: How Uniformity Induces
Robustness to Background Noise in Contrastive Learning. arXiv preprint
arXiv:2511.12278.

Woller, J. P., Menrath, D., & Gharabaghi, A. (2025). Generalized
contrastive PCA is equivalent to generalized eigendecomposition. PLOS
Computational Biology, 21(10), e1013555.

## Examples

``` r
# Simulate data where foreground has extra variance in first few dimensions
set.seed(123)
n_f <- 100
n_b <- 150
n_features <- 50

# Background: standard normal noise
X_b <- matrix(rnorm(n_b * n_features), nrow=n_b, ncol=n_features)
colnames(X_b) <- paste0("Feat_", 1:n_features)

# Foreground: background noise + extra variance in first 5 features
X_f_signal <- matrix(rnorm(n_f * 5, mean=0, sd=2), nrow=n_f, ncol=5)
X_f_noise <- matrix(rnorm(n_f * (n_features-5)), nrow=n_f, ncol=n_features-5)
X_f <- cbind(X_f_signal, X_f_noise) + matrix(rnorm(n_f * n_features), nrow=n_f, ncol=n_features)
colnames(X_f) <- paste0("Feat_", 1:n_features)
rownames(X_f) <- paste0("SampleF_", 1:n_f)

# Apply cPCA++ (requires geigen and corpcor packages)
# install.packages(c("geigen", "corpcor"))
if (requireNamespace("geigen", quietly = TRUE) && requireNamespace("corpcor", quietly = TRUE)) {
  # Assuming helper constructors like bi_projector are available
  # library(multivarious) 

  res_cpca_plus <- cPCAplus(X_f, X_b, ncomp = 5, method = "geigen")

  # Scores for the foreground data (samples x components)
  print(head(res_cpca_plus$s))

  # Loadings (contrastive directions) (features x components)
  print(head(res_cpca_plus$v))
}
#> Using feature-space strategy...
#>                 cPC1       cPC2       cPC3       cPC4       cPC5
#> SampleF_1  0.6973216 -0.9298861 -2.4890872 -1.1068083  0.7499051
#> SampleF_2  5.3814450  3.7903387 -1.0576106 -1.5806685 -0.6569618
#> SampleF_3  1.0732526  0.7042730 -2.1782781  0.2180881  3.7225401
#> SampleF_4 -2.4616309 -3.5851637  3.9806244 -0.3335771 -0.1837763
#> SampleF_5  2.2755087  0.8559051 -0.1064592 -0.3198966  2.4177053
#> SampleF_6 -2.2642836  1.9850255  1.5783660 -0.9690632  1.3477944
#>               cPC1        cPC2        cPC3        cPC4       cPC5
#> Feat_1  0.12938667  0.35443654 -0.27401589  0.23486767 -0.1665203
#> Feat_2  0.28086477  0.01166486 -0.16767050  0.28665253  0.2750788
#> Feat_3  0.32662862  0.10800109  0.24508903  0.06522056 -0.1403823
#> Feat_4  0.28416270 -0.01924305 -0.01876914 -0.16614791  0.1280966
#> Feat_5 -0.05633953  0.15857353  0.42986961  0.03909660 -0.1655489
#> Feat_6  0.06099437  0.09724319  0.02361012 -0.01815817 -0.1194691

# \donttest{
# Plot example (slow graphics)
if (requireNamespace("geigen", quietly = TRUE) && requireNamespace("corpcor", quietly = TRUE)) {
  set.seed(123)
  X_b <- matrix(rnorm(150 * 50), nrow=150, ncol=50)
  X_f <- cbind(matrix(rnorm(100*5, sd=2), 100, 5), matrix(rnorm(100*45), 100, 45))
  res <- cPCAplus(X_f, X_b, ncomp = 5, method = "geigen")
  plot(res$s[, 1], res$s[, 2],
       xlab = "Contrastive Component 1", ylab = "Contrastive Component 2",
       main = "cPCA++ Scores")
}
#> Using feature-space strategy...

# }
```
