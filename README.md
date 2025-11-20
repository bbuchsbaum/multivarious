
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

## Albers theme

This package uses the albersdown theme. Vignettes are styled with
`vignettes/albers.css` and a local `vignettes/albers.js`; the palette
family is provided via `params$family` (default ‘red’). The pkgdown site
uses `template: { package: albersdown }`.
