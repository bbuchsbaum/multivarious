# Transfer from X domain to Y domain (or vice versa) in a cross_projector

Convert between data representations in a multiblock or
cross-decomposition model by projecting the input `new_data` from the
`from` domain/block onto a latent space and then reconstructing it in
the `to` domain/block.

## Usage

``` r
# S3 method for class 'cross_projector'
transfer(x, new_data, from, to, opts = list(), ...)
```

## Arguments

- x:

  A `cross_projector` object.

- new_data:

  The data to transfer.

- from:

  Source domain ("X" or "Y").

- to:

  Target domain ("X" or "Y").

- opts:

  A list of options (see `transfer` generic).

- ...:

  Ignored.

## Value

Transferred data matrix.

## Details

When `opts$ls_rr` is `TRUE`, the forward projection from the `from`
domain is computed using a ridge-regularized least squares approach. The
penalty parameter is taken from `opts$lambda`. Component subsetting via
`opts$comps` is applied after computing these ridge-based scores.
