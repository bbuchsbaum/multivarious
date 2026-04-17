# Compute subspace similarity

Compute subspace similarity

## Usage

``` r
subspace_similarity(
  fits,
  method = c("avg_pair", "grassmann", "worst_case"),
  ...
)
```

## Arguments

- fits:

  a list of bi_projector objects

- method:

  the method to use for computing subspace similarity

- ...:

  additional arguments to pass to the method

## Value

a numeric value representing the subspace similarity
