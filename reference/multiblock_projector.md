# Create a Multiblock Projector

Constructs a multiblock projector using the given component matrix
(`v`), a preprocessing function, and a list of block indices. This
allows for the projection of multiblock data, where each block
represents a different set of variables or features.

## Usage

``` r
multiblock_projector(
  v,
  preproc = prep(pass()),
  ...,
  block_indices,
  classes = NULL
)
```

## Arguments

- v:

  A matrix of components with dimensions `nrow(v)` by `ncol(v)` (columns
  = number of components).

- preproc:

  A pre-processing function for the data (default: `prep(pass())`).

- ...:

  Extra arguments.

- block_indices:

  A list of numeric vectors specifying the indices of each data block.

- classes:

  (optional) A character vector specifying additional class attributes
  of the object, default is NULL.

## Value

A `multiblock_projector` object.

## See also

projector

## Examples

``` r
# Generate some example data
X1 <- matrix(rnorm(10 * 5), 10, 5)
X2 <- matrix(rnorm(10 * 5), 10, 5)
X <- cbind(X1, X2)

# Compute PCA on the combined data
pc <- pca(X, ncomp = 8)

# Create a multiblock projector using PCA components and block indices
mb_proj <- multiblock_projector(pc$v, block_indices = list(1:5, 6:10))

# Project multiblock data using the multiblock projector
mb_scores <- project(mb_proj, X)
```
