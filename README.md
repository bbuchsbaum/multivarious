
<!-- README.md is generated from README.Rmd. Please edit that file -->

# multivarious

<!-- badges: start -->

[![Codecov test
coverage](https://codecov.io/gh/bbuchsbaum/multivarious/branch/master/graph/badge.svg)](https://app.codecov.io/gh/bbuchsbaum/multivarious?branch=master)
<!-- badges: end -->

This package is intended to provide some basic abstractions and default
implementations of basic computational infrastructure for multivariate
component-based modeling such as principal components analysis.

The main idea is to model multivariate decompositions as involving
projections from an input data space to a lower dimensional component
space. This idea is encapsulated by the `projector` class and the
`project` function. Support for two-way mapping (row projection and
column projection) is provided by the derived class `bi-projector`.
Generic functions for common operations are included:

- `project` for mapping from input space into (usually)
  reduced-dimensional output space
- `partial_project` for mapping a subset of input space into output
  space
- `project_vars` for mapping new variables (“supplementary variables”)
  to output space
- `reconstruct` for reconstructing input data from its low-dimensional
  representation
- `residuals` for extracting residuals of a fit with `n` components.

The package now also includes a mixed-model path for operator-valued
ANOVA. With `mixed_regress()`, each named fixed-effect term in a
repeated-measures design can be extracted as an `effect_operator`, then
analyzed with the same core verbs:

- `effect` for named term extraction
- `components` and `scores` for interpretable effect axes
- `reconstruct` for effect contributions in original variable space
- `perm_test` for omnibus and rank inference
- `bootstrap` for subject-level stability

The broader calibration harness for this path lives at
`experimental/mixed_effect_operator_calibration.R`, with batch outputs
saved under `experimental/results/` when you run the simulation grid
locally.

## Installation

You can install the development version from
[GitHub](https://github.com/) with:

``` r
# install.packages("devtools")
devtools::install_github("bbuchsbaum/multivarious")
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(multivarious)
#> 
#> Attaching package: 'multivarious'
#> The following objects are masked from 'package:stats':
#> 
#>     residuals, screeplot
#> The following objects are masked from 'package:base':
#> 
#>     transform, truncate
## basic example code
```

## Mixed effect operators

``` r
set.seed(1)

design <- expand.grid(
  subject = factor(seq_len(6)),
  level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
  KEEP.OUT.ATTRS = FALSE
)
design$group <- factor(rep(c("A", "B"), each = 9))

level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
group_num <- ifelse(design$group == "B", 1, 0)
subj_idx <- as.integer(design$subject)
b0 <- rnorm(6, sd = 0.5)

Y <- cbind(
  b0[subj_idx] + level_num + rnorm(nrow(design), sd = 0.15),
  group_num + rnorm(nrow(design), sd = 0.15),
  level_num * group_num + rnorm(nrow(design), sd = 0.15),
  rnorm(nrow(design), sd = 0.15)
)

fit <- mixed_regress(
  Y,
  design = design,
  fixed = ~ group * level,
  random = ~ 1 | subject,
  basis = shared_pca(3),
  preproc = pass()
)

E <- effect(fit, "group:level")
pt <- perm_test(E, nperm = 19, alpha = 0.10)

ncomp(E)
#> [1] 0
ncomp(pt)
#> [1] 0
```

## Albers theme

This package uses the albersdown theme. Vignettes are styled with
`vignettes/albers.css` and a local `vignettes/albers.js`; the palette
family is provided via `params$family` (default ‘red’). The pkgdown site
uses `template: { package: albersdown }`.
