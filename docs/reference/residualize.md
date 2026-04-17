# Compute a regression model for each column in a matrix and return residual matrix

Compute a regression model for each column in a matrix and return
residual matrix

## Usage

``` r
residualize(form, X, design, intercept = FALSE)
```

## Arguments

- form:

  the formula defining the model to fit for residuals

- X:

  the response matrix

- design:

  the `data.frame` containing the design variables specified in `form`
  argument.

- intercept:

  add an intercept term (default is FALSE)

## Value

a `matrix` of residuals

## Examples

``` r
X <- matrix(rnorm(20*10), 20, 10)
des <- data.frame(a=rep(letters[1:4], 5), b=factor(rep(1:5, each=4)))
xresid <- residualize(~ a+b, X, design=des)

## design is saturated, residuals should be zero
xresid2 <- residualize(~ a*b, X, design=des)
sum(xresid2) == 0
#> [1] TRUE
```
