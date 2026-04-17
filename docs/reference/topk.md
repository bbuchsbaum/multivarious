# top-k accuracy indicator

Determines if the true class label is among the top `k` predicted
probabilities for each observation.

## Usage

``` r
topk(prob, observed, k)
```

## Arguments

- prob:

  Numeric matrix of predicted probabilities (observations x classes).
  Column names must correspond to class labels.

- observed:

  Factor or vector of observed class labels. Must be present in
  `colnames(prob)`.

- k:

  Integer; the number of top probabilities to consider.

## Value

A `data.frame` with columns `topk` (logical indicator: `TRUE` if
observed class is in top-k) and `observed`.

## See also

Other classifier evaluation:
[`rank_score()`](https://bbuchsbaum.github.io/multivarious/reference/rank_score.md)

## Examples

``` r
probs <- matrix(c(0.1, 0.9, 0.8, 0.2, 0.3, 0.7), 3, 2, byrow=TRUE,
                dimnames = list(NULL, c("A", "B")))
obs <- factor(c("B", "A", "B"))
topk(probs, obs, k=1)
#>   topk observed
#> 1 TRUE        B
#> 2 TRUE        A
#> 3 TRUE        B
topk(probs, obs, k=2)
#>   topk observed
#> 1 TRUE        B
#> 2 TRUE        A
#> 3 TRUE        B
```
