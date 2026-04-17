# Bootstrap inference for PLSC loadings

Provides bootstrap ratios (mean / sd) for X and Y loadings to assess
stability, mirroring common practice in Behavior PLSC.

## Usage

``` r
bootstrap_plsc(
  x,
  X,
  Y,
  nboot = 500,
  comps = ncomp(x),
  seed = NULL,
  parallel = FALSE,
  epsilon = 1e-09,
  ...
)
```

## Arguments

- x:

  A fitted `plsc` object.

- X:

  Original X block.

- Y:

  Original Y block.

- nboot:

  Number of bootstrap samples (default 500).

- comps:

  Number of components to bootstrap (default: `ncomp(x)`).

- seed:

  Optional integer seed for reproducibility.

- parallel:

  Use future.apply for parallelization (default FALSE).

- epsilon:

  Small positive constant to stabilize division for ratios.

- ...:

  Additional arguments (currently unused).
