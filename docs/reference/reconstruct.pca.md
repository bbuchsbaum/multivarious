# Reconstruct Data from PCA Results

Reconstructs the original (centered) data matrix from the PCA scores and
loadings.

## Usage

``` r
# S3 method for class 'pca'
reconstruct(x, comp = 1:ncomp(x), ...)
```

## Arguments

- x:

  A `pca` object.

- comp:

  Integer vector specifying which components to use for reconstruction
  (default: all components in `x`).

- ...:

  Extra arguments (ignored).

## Value

A matrix representing the reconstructed data in the *original* scale
(preprocessing reversed).
