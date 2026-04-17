# Summarise Benchmark Results

Summarise Benchmark Results

## Usage

``` r
summarize_geneig_benchmark(bench_result)
```

## Arguments

- bench_result:

  Output of
  [`benchmark_geneig()`](https://bbuchsbaum.github.io/multivarious/reference/benchmark_geneig.md).

## Value

Tibble with one row per method and scenario containing aggregated
statistics (mean/median runtime, standard deviation, number of runs,
residuals, availability notes).

## Examples

``` r
# \donttest{
  if (requireNamespace("RSpectra", quietly = TRUE)) {
    res <- benchmark_geneig(n = 150, reps = 1, methods = c("robust", "rspectra"))
    summarize_geneig_benchmark(res)
  }
# }
```
