# PCA Outlier Diagnostics

Calculates Hotelling T^2 (score distance) and Q-residual (orthogonal
distance) for each observation, given a chosen number of components.

## Usage

``` r
pca_outliers(x, X, ncomp, cutoff = FALSE)
```

## Arguments

- x:

  A `pca` object.

- X:

  The original data matrix used for PCA.

- ncomp:

  Number of components to consider.

- cutoff:

  Logical or numeric specifying threshold for labeling outliers. If
  `TRUE`, uses some typical statistical threshold (F-dist) for T^2, or
  sets an arbitrary Q limit. If numeric, treat it as a cutoff. Default
  is `FALSE` (no labeling).

## Value

A data frame with columns `T2` and `Q`, and optionally an outlier flag.
