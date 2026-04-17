# Multiblock basics: one projector, many tables

## 1. Why multiblock?

Many studies collect several tables on the same samples – e.g.
transcriptomics + metabolomics, or multiple sensor blocks. Most
single-table reductions (PCA, ICA, NMF, …) ignore that structure.
`multiblock_projector` is a thin wrapper that keeps track of which
original columns belong to which block, so you can

- drop-in any existing decomposition (PCA, SVD, NMF, …)
- still know “these five loadings belong to block A, those three to
  block B”
- project or reconstruct per block effortlessly.

We demonstrate with a minimal two-block toy-set.

``` r
set.seed(1)
n  <- 100
pA <- 7; pB <- 5                    # two blocks, different widths

XA <- matrix(rnorm(n * pA), n, pA)
XB <- matrix(rnorm(n * pB), n, pB)
X  <- cbind(XA, XB)                 # global data matrix
blk_idx <- list(A = 1:pA, B = (pA + 1):(pA + pB)) # Named list is good practice
```

## 2. Wrap a single PCA as a multiblock projector

``` r
# 2-component centred PCA (using base SVD for brevity)
preproc_fitted <- fit(center(), X)
Xc        <- transform(preproc_fitted, X)          # Centered data
svd_res   <- svd(Xc, nu = 0, nv = 2)               # only V (loadings)
mb        <- multiblock_projector(
  v             = svd_res$v,                       # p × k loadings
  preproc       = preproc_fitted,                  # remembers centering
  block_indices = blk_idx
)

print(mb)
#> Projector object:
#>   Input dimension: 12
#>   Output dimension: 2
#>   With pre-processing:
#> A finalized pre-processing pipeline:
#>  Step 1: center
```

### 2.1 Project the whole data

``` r
scores_all <- project(mb, X)                       # n × 2
head(round(scores_all, 3))
#>        [,1]   [,2]
#> [1,] -0.815 -1.159
#> [2,]  1.075 -3.326
#> [3,] -0.068  1.124
#> [4,] -0.055 -0.788
#> [5,] -0.554  1.005
#> [6,] -0.942  1.565
```

### 2.2 Project one block only

``` r
# Project using only data from block A (requires original columns)
scores_A <- project_block(mb, XA, block = 1)       
# Project using only data from block B
scores_B <- project_block(mb, XB, block = 2)       

cor(scores_all[,1], scores_A[,1])                  # high (they coincide)
#> [1] 0.7971026
```

Because the global PCA treats all columns jointly, projecting only block
A gives exactly the same latent coordinates as when the whole matrix is
available – useful when a block is missing at prediction time.

### 2.3 Partial feature projection

Need to use just three variables from block B?

``` r
# Get the global indices for the first 3 columns of block B
sel_cols_global <- blk_idx[["B"]][1:3]
# Extract the corresponding data columns from the full matrix or block B
part_XB_data  <- X[, sel_cols_global, drop = FALSE] # Data must match global indices

scores_part <- partial_project(mb, part_XB_data,
                               colind = sel_cols_global)  # Use global indices
head(round(scores_part, 3))
#>        [,1]   [,2]
#> [1,] -1.045 -0.888
#> [2,] -0.142 -1.597
#> [3,]  0.469  0.647
#> [4,] -0.244 -0.410
#> [5,] -0.513  0.305
#> [6,] -0.740  0.000
```

## 3. Adding scores → multiblock_biprojector

If you also keep the sample scores (from the original fit) you get
two-way functionality: re-construct data, measure error, run permutation
tests, etc. That is one extra line when creating the object:

``` r
bi <- multiblock_biprojector(
  v             = svd_res$v,
  s             = Xc %*% svd_res$v,    # Calculate scores: Xc %*% V
  sdev          = svd_res$d[1:2] / sqrt(n-1), # SVD d are related to sdev
  preproc       = preproc_fitted,
  block_indices = blk_idx
)
print(bi)
#> Multiblock Bi-Projector object:
#>   Projection matrix dimensions:  12 x 2 
#>   Block indices:
#>     Block 1: 1,2,3,4,5,6,7
#>     Block 2: 8,9,10,11,12
```

Now you can, for instance, test whether component-wise consensus between
blocks is stronger than by chance.

``` r
# Quick permutation test (use more permutations for real analyses)
# use_rspectra=FALSE needed for this 2-block example; larger problems can use TRUE
perm_res <- perm_test(bi, Xlist = list(A = XA, B = XB), nperm = 99, use_rspectra = FALSE)
print(perm_res$component_results)
#>   comp observed pval lower_ci upper_ci
#> 1    1 84.25129  0.1 78.70594 88.96802
```

The `perm_test` method for `multiblock_biprojector` uses an eigen-based
score consensus statistic to assess whether blocks share more variance
than expected by chance.

## 4. Take-aways

- Any decomposition that delivers a loading matrix `v` (and optionally
  scores `s`) can become multiblock-aware by supplying `block_indices`.
- The wrapper introduces zero new maths – it only remembers the column
  grouping and plugs into the common verbs:

| Verb                                                                                          | What it does in multiblock context                 |
|-----------------------------------------------------------------------------------------------|----------------------------------------------------|
| [`project()`](https://bbuchsbaum.github.io/multivarious/reference/project.md)                 | whole-matrix projection (uses preprocessing)       |
| [`project_block()`](https://bbuchsbaum.github.io/multivarious/reference/project_block.md)     | scores based on one block’s data                   |
| [`partial_project()`](https://bbuchsbaum.github.io/multivarious/reference/partial_project.md) | scores from an arbitrary subset of global columns  |
| `coef(..., block=)`                                                                           | retrieve loadings for a specific block             |
| [`perm_test()`](https://bbuchsbaum.github.io/multivarious/reference/perm_test.md)             | permutation test for block consensus (biprojector) |

This light infrastructure lets you prototype block-aware analyses
quickly, while still tapping into the entire `multiblock` toolkit
(cross-validation, reconstruction metrics, composition with
`compose_projector`, etc.).

``` r
sessionInfo()
#> R version 4.5.3 (2026-03-11)
#> Platform: x86_64-pc-linux-gnu
#> Running under: Ubuntu 24.04.4 LTS
#> 
#> Matrix products: default
#> BLAS:   /usr/lib/x86_64-linux-gnu/openblas-pthread/libblas.so.3 
#> LAPACK: /usr/lib/x86_64-linux-gnu/openblas-pthread/libopenblasp-r0.3.26.so;  LAPACK version 3.12.0
#> 
#> locale:
#>  [1] LC_CTYPE=C.UTF-8       LC_NUMERIC=C           LC_TIME=C.UTF-8       
#>  [4] LC_COLLATE=C.UTF-8     LC_MONETARY=C.UTF-8    LC_MESSAGES=C.UTF-8   
#>  [7] LC_PAPER=C.UTF-8       LC_NAME=C              LC_ADDRESS=C          
#> [10] LC_TELEPHONE=C         LC_MEASUREMENT=C.UTF-8 LC_IDENTIFICATION=C   
#> 
#> time zone: UTC
#> tzcode source: system (glibc)
#> 
#> attached base packages:
#> [1] stats     graphics  grDevices utils     datasets  methods   base     
#> 
#> other attached packages:
#> [1] multivarious_0.3.1 dplyr_1.2.1       
#> 
#> loaded via a namespace (and not attached):
#>  [1] vctrs_0.7.3       cli_3.6.6         knitr_1.51        rlang_1.2.0      
#>  [5] xfun_0.57         generics_0.1.4    textshaping_1.0.5 jsonlite_2.0.0   
#>  [9] glue_1.8.0        htmltools_0.5.9   ragg_1.5.2        sass_0.4.10      
#> [13] rmarkdown_2.31    grid_4.5.3        tibble_3.3.1      evaluate_1.0.5   
#> [17] jquerylib_0.1.4   fastmap_1.2.0     yaml_2.3.12       lifecycle_1.0.5  
#> [21] chk_0.10.0        compiler_4.5.3    fs_2.0.1          pkgconfig_2.0.3  
#> [25] lattice_0.22-9    systemfonts_1.3.2 digest_0.6.39     R6_2.6.1         
#> [29] tidyselect_1.2.1  pillar_1.11.1     magrittr_2.0.5    Matrix_1.7-4     
#> [33] bslib_0.10.0      tools_4.5.3       geigen_2.3        pkgdown_2.2.0    
#> [37] cachem_1.1.0      desc_1.4.3
```
