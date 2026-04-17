# A Union of Concatenated `bi_projector` Fits

This function combines a set of `bi_projector` fits into a single
`bi_projector` instance. The new instance's weights and associated
scores are obtained by concatenating the weights and scores of the input
fits.

## Usage

``` r
bi_projector_union(fits, outer_block_indices = NULL)
```

## Arguments

- fits:

  A list of `bi_projector` instances with the same row space. These
  instances will be combined to create a new `bi_projector` instance.

- outer_block_indices:

  An optional list of indices for the outer blocks. If not provided, the
  function will compute the indices based on the dimensions of the input
  fits.

## Value

A new `bi_projector` instance with concatenated weights, scores, and
other properties from the input `bi_projector` instances.

## Examples

``` r
X1 <- matrix(rnorm(5*5), 5, 5)
X2 <- matrix(rnorm(5*5), 5, 5)

bpu <- bi_projector_union(list(pca(X1), pca(X2)))
```
