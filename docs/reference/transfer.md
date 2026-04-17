# Transfer data from one domain/block to another via a latent space

Convert between data representations in a multiblock or
cross-decomposition model by projecting the input `new_data` from the
`from` domain/block onto a latent space and then reconstructing it in
the `to` domain/block.

## Usage

``` r
transfer(x, new_data, from, to, opts = list(), ...)
```

## Arguments

- x:

  The model fit, typically an object that implements a `transfer` method
  and ideally a `block_names` method.

- new_data:

  The data to transfer, typically matching the dimension of the `from`
  domain.

- from:

  Character string or index identifying the source domain/block. Must be
  present in `block_names(x)` if that method exists.

- to:

  Character string or index identifying the target domain/block. Must be
  present in `block_names(x)` if that method exists.

- opts:

  A list of optional arguments controlling the transfer process:

  `cols`

  :   Optional numeric vector specifying column indices of the *target*
      domain to reconstruct. If NULL (default), reconstructs all
      columns.

  `comps`

  :   Optional numeric vector specifying which latent components to use
      for the projection/reconstruction. If NULL (default), uses all
      components.

  `ls_rr`

  :   Logical; if TRUE, use a ridge-regularized LS approach for the
      initial projection from the `from` domain. Default FALSE.

  `lambda`

  :   Numeric ridge penalty (if `ls_rr=TRUE`). Default 1e-6.

- ...:

  Additional arguments passed to specific methods (discouraged, prefer
  `opts`).

## Value

A matrix or data frame representing the transferred data in the `to`
domain/block (or a subset of columns/components if specified in `opts`).
