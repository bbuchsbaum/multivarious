# Calculate Rank Score for Predictions

Computes the rank score (normalized rank of the true class probability)
for each observation. Lower rank scores indicate better predictions
(true class has higher probability).

## Usage

``` r
rank_score(prob, observed)
```

## Arguments

- prob:

  Numeric matrix of predicted probabilities (observations x classes).
  Column names must correspond to class labels.

- observed:

  Factor or vector of observed class labels. Must be present in
  `colnames(prob)`.

## Value

A `data.frame` with columns `prank` (the normalized rank score) and
`observed` (the input labels).

## See also

Other classifier evaluation:
[`topk()`](https://bbuchsbaum.github.io/multivarious/reference/topk.md)

## Examples

``` r
probs <- matrix(c(0.1, 0.9, 0.8, 0.2), 2, 2, byrow=TRUE,
               dimnames = list(NULL, c("A", "B")))
obs <- factor(c("B", "A"))
rank_score(probs, obs)
#>       prank observed
#> 1 0.3333333        B
#> 2 0.3333333        A
```
