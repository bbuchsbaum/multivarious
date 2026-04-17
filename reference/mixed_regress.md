# Mixed-effect multivariate regression

Fit the row-side design geometry for operator-valued ANOVA. Supports
fixed-effect designs (`random = NULL`) and grouped random-effects
designs with a single grouping variable (random intercept and optional
random slopes, shared-`Omega` across response features).

## Usage

``` r
mixed_regress(
  Y,
  design = NULL,
  fixed,
  random = NULL,
  basis = identity_basis(),
  preproc = center(),
  ...
)
```

## Arguments

- Y:

  Response matrix (`n_obs x p`) or 3D array
  (`n_subject x n_within x p`).

- design:

  Optional design data frame. Required for matrix input.

- fixed:

  Fixed-effect formula.

- random:

  Random-effect specification. Either `NULL` (fixed-effects only) or a
  one-sided formula of the form `~ ... | group` understood by
  [`reformulas::findbars()`](https://rdrr.io/pkg/reformulas/man/formfuns.html)
  / `lme4::findbars()`. All random-effects bars must share a single
  grouping variable.

- basis:

  Feature basis specification or projector-like object.

- preproc:

  Response preprocessor.

- ...:

  Reserved for future extensions.

## Value

A `mixed_fit` object.
