# component scales (singular values)

Return the per-component scale of a fitted projection. For the SVD-based
methods in this package (`svd_wrapper`, `pca`, `bi_projector`), this is
the vector of *singular values* of the pre-processed data matrix, i.e.
the square roots of the eigenvalues of \\X^\top X\\. It is **not** the
sample standard deviation of the score columns: those equal
`sdev(x) / sqrt(n - 1)` (the convention used by
[`stats::prcomp`](https://rdrr.io/r/stats/prcomp.html)). The two differ
only by the constant factor `sqrt(n - 1)`, so variance-explained ratios
(`sdev^2 / sum(sdev^2)`) are identical either way.

## Usage

``` r
sdev(x)
```

## Arguments

- x:

  the model fit

## Value

a numeric vector of singular values (one per component)

## Note

The name `sdev` is retained for backward compatibility and
`prcomp`-familiarity; the value stored is the raw singular value. A
future major version may switch to the `prcomp` scaling
(`d / sqrt(n - 1)`); do not rely on the absolute magnitude across
versions – use ratios, or rescale explicitly.
