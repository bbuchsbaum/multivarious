# Construct a Discriminant Projector

A `discriminant_projector` is an instance that extends `bi_projector`
with a projection that maximizes class separation. This can be useful
for dimensionality reduction techniques that take class labels into
account, such as Linear Discriminant Analysis (LDA).

## Usage

``` r
discriminant_projector(
  v,
  s,
  sdev,
  preproc = prep(pass()),
  labels,
  classes = NULL,
  ...
)
```

## Arguments

- v:

  The projection matrix (often `X %*% v`). Rows correspond to
  observations, columns to components.

- s:

  The score matrix (often `X %*% v`). Rows correspond to observations,
  columns to components.

- sdev:

  The standard deviations associated with the scores or components
  (e.g., singular values from LDA).

- preproc:

  A `prepper` or `pre_processor` object, or a pre-processing function
  (e.g., `center`, `pass`).

- labels:

  A factor or character vector of class labels corresponding to the rows
  of `X` (and `s`).

- classes:

  Additional S3 classes to prepend.

- ...:

  Extra arguments passed to `bi_projector`.

## Value

A `discriminant_projector` object.

## See also

bi_projector

## Examples

``` r
# Simulate data and labels
set.seed(123)
X <- matrix(rnorm(100 * 10), 100, 10)
labels <- factor(rep(1:2, each = 50))

# Perform LDA and create a discriminant projector
lda_fit <- MASS::lda(X, labels)

dp <- discriminant_projector(lda_fit$scaling, X %*% lda_fit$scaling, sdev = lda_fit$svd, 
labels = labels)
```
