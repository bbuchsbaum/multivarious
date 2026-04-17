# Bootstrap stability summaries for an effect operator

Bootstrap stability summaries for an effect operator

## Usage

``` r
# S3 method for class 'effect_operator'
bootstrap(
  x,
  nboot,
  resample = c("auto", "rows", "subject"),
  parallel = FALSE,
  seed = NULL,
  ...
)
```

## Arguments

- x:

  An `effect_operator`.

- nboot:

  Number of bootstrap resamples.

- resample:

  Resampling unit. Use `"subject"` for grouped repeated-measures data or
  `"rows"` for observation-level resampling; `"auto"` selects subject
  blocks when available.

- parallel:

  Logical; if `TRUE`, use `future.apply`.

- seed:

  Optional random seed.

- ...:

  Reserved for future extensions.

## Value

A bootstrap result object with loading and singular-value summaries.
`n_failed` records the number of resamples that produced a
rank-deficient or otherwise unfittable design; failed draws are skipped.

## Note

Loadings are currently aligned to the reference solution by a per-axis
sign flip. When more than one effect axis is retained, axes can rotate
between bootstrap draws and a Procrustes-based alignment (see PRD
section 10) would give tighter stability summaries. Not implemented in
this pass.
