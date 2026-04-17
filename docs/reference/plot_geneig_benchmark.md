# Plot Benchmark Results

Plot Benchmark Results

## Usage

``` r
plot_geneig_benchmark(
  bench_result,
  stat = c("median", "mean"),
  log_scale = TRUE
)
```

## Arguments

- bench_result:

  Output from
  [`benchmark_geneig()`](https://bbuchsbaum.github.io/multivarious/reference/benchmark_geneig.md)
  or
  [`summarize_geneig_benchmark()`](https://bbuchsbaum.github.io/multivarious/reference/summarize_geneig_benchmark.md).

- stat:

  Which summary statistic to display on the y-axis (`"median"` or
  `"mean"`).

- log_scale:

  Logical; if `TRUE` (default) the y-axis is shown on a log10 scale.

## Value

A `ggplot` object comparing runtime across methods.

## Examples

``` r
# \donttest{
  if (requireNamespace("RSpectra", quietly = TRUE)) {
    res <- benchmark_geneig(n = 150, reps = 1, methods = c("robust", "rspectra"))
    plot_geneig_benchmark(res)
  }
# }
```
