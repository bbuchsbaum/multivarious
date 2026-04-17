# Multi-output linear regression

Fit a multivariate regression model for a matrix of basis functions,
`X`, and a response matrix `Y`. The goal is to find a projection matrix
that can be used for mapping and reconstruction.

## Usage

``` r
regress(
  X,
  Y,
  preproc = pass(),
  method = c("lm", "enet", "mridge", "pls"),
  intercept = FALSE,
  lambda = 0.001,
  alpha = 0,
  ncomp = ceiling(ncol(X)/2),
  ...
)
```

## Arguments

- X:

  the set of independent (basis) variables

- Y:

  the response matrix

- preproc:

  A preprocessing pipeline applied to `X` before fitting the model

- method:

  the regression method: `lm`, `enet`, `mridge`, or `pls`

- intercept:

  whether to include an intercept term

- lambda:

  ridge shrinkage parameter (for methods `mridge` and `enet`)

- alpha:

  the elastic net mixing parameter if method is `enet`

- ncomp:

  number of PLS components if method is `pls`

- ...:

  extra arguments sent to the underlying fitting function

## Value

a bi-projector of type `regress`. The `sdev` component of this object
stores the standard deviations of the columns of the design matrix (`X`
potentially including an intercept) used in the fit, not the standard
deviations of latent components as might be typical in other
`bi_projector` contexts (e.g., SVD).

## Examples

``` r
# Generate synthetic data
set.seed(123) # for reproducibility
Y <- matrix(rnorm(10 * 100), 10, 100)
X <- matrix(rnorm(10 * 9), 10, 9)

# Fit regression models and reconstruct the fitted response matrix
r_lm <- regress(X, Y, intercept = FALSE, method = "lm")
recon_lm <- reconstruct(r_lm) # Reconstructs fitted Y

r_mridge <- regress(X, Y, intercept = TRUE, method = "mridge", lambda = 0.001)
recon_mridge <- reconstruct(r_mridge)

r_enet <- regress(X, Y, intercept = TRUE, method = "enet", lambda = 0.001, alpha = 0.5)
recon_enet <- reconstruct(r_enet)

r_pls <- regress(X, Y, intercept = TRUE, method = "pls", ncomp = 5)
recon_pls <- reconstruct(r_pls)
```
