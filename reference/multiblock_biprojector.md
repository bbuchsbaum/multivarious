# Create a Multiblock Bi-Projector

Constructs a multiblock bi-projector using the given component matrix
(`v`), score matrix (`s`), singular values (`sdev`), a preprocessing
function, and a list of block indices. This allows for two-way mapping
with multiblock data.

## Usage

``` r
multiblock_biprojector(
  v,
  s,
  sdev,
  preproc = prep(pass()),
  ...,
  block_indices,
  classes = NULL
)
```

## Arguments

- v:

  A matrix of components (nrow = number of variables, ncol = number of
  components).

- s:

  A matrix of scores (nrow = samples, ncol = components).

- sdev:

  A numeric vector of singular values or standard deviations.

- preproc:

  A pre-processing object (default: `prep(pass())`).

- ...:

  Extra arguments.

- block_indices:

  A list of numeric vectors specifying data block variable indices.

- classes:

  Additional class attributes (default NULL).

## Value

A `multiblock_biprojector` object.

## See also

bi_projector, multiblock_projector
