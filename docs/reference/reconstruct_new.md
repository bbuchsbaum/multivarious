# Reconstruct new data in a model's subspace

This function takes a model (e.g., `projector` or `bi_projector`) and a
new dataset, and computes the rank-d approximation of the new data in
the same subspace that was defined by the model. In other words, we
**project** the new data into the fitted subspace and then **map it
back** to the original dimensionality.

## Usage

``` r
reconstruct_new(x, new_data, ...)
```

## Arguments

- x:

  The fitted model object (e.g., `bi_projector`) that defines a subspace
  or factorization.

- new_data:

  A numeric matrix (or data frame) of shape `(n x p_full)` or possibly
  fewer columns if you allow partial reconstruction.

- ...:

  Additional arguments passed to the specific `reconstruct_new` method
  for the class of `x`.

## Value

A numeric matrix (same number of rows as `new_data`, and typically the
same number of columns if you're reconstructing fully) representing the
rank-d approximation in the model's subspace.

## Details

Similar to
[`reconstruct`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
but operates on an external `new_data` rather than the original fitted
data. Often used to see how well the model's subspace explains unseen
data.

## See also

[`reconstruct`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
for reconstructing the original data in the model.

Other reconstruct:
[`reconstruct()`](https://bbuchsbaum.github.io/multivarious/reference/reconstruct.md)
