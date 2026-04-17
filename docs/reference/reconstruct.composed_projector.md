# Reconstruct Data from Scores using a Composed Projector

Maps scores from the final latent space back towards the original input
space using the composed projector's combined inverse projection.
Requires scores to be provided explicitly.

## Usage

``` r
# S3 method for class 'composed_projector'
reconstruct(x, scores, comp = NULL, rowind = NULL, colind = NULL, ...)
```

## Arguments

- x:

  A `composed_projector` object.

- scores:

  A numeric matrix of scores (observations x components) in the final
  latent space of the composed projector.

- comp:

  Numeric vector of component indices (columns of `scores`, rows of
  `inverse_projection`) to use for reconstruction. Defaults to all
  components.

- rowind:

  Numeric vector of row indices (observations in `scores`) to
  reconstruct. Defaults to all rows.

- colind:

  Numeric vector of original variable indices (columns of the final
  reconstructed matrix) to return. Defaults to all original variables.

- ...:

  Additional arguments (currently unused).

## Value

A matrix representing the reconstructed data, ideally in the original
data space.

## Details

Attempts to apply the `reverse_transform` of the *first* stage's
preprocessor to return data in the original units. If the first stage
preprocessor is unavailable or invalid, a warning is issued, and data is
returned in the (potentially) preprocessed space of the first stage.
