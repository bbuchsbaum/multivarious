# Benchmark `geneig()` Backends

Quickly compare the runtime and numerical accuracy of the available
[`geneig()`](https://bbuchsbaum.github.io/multivarious/reference/geneig.md)
backends on randomly generated SPD problems. The helper returns a tidy
data frame that can be summarised or visualised with
[`plot_geneig_benchmark()`](https://bbuchsbaum.github.io/multivarious/reference/plot_geneig_benchmark.md).

## Usage

``` r
benchmark_geneig(
  n = c(200, 400),
  density = NULL,
  ncomp = NULL,
  reps = 5,
  methods = c("robust", "sdiag", "geigen", "primme", "rspectra", "subspace"),
  seed = 123,
  rspectra_opts = list(),
  primme_opts = list(),
  primme_jacobi = FALSE,
  subspace_opts = list(max_iter = 300, tol = 1e-05),
  ...
)
```

## Arguments

- n:

  Integer vector with the problem sizes (matrix dimension). Each size
  defines one benchmark scenario.

- density:

  Optional numeric vector in `(0, 1]` with the proportion of non-zero
  entries to use for each scenario. Values `< 1` generate sparse
  matrices via
  [`Matrix::rsparsematrix()`](https://rdrr.io/pkg/Matrix/man/rsparsematrix.html),
  values `>= 1` (or `NULL`) generate dense matrices. A single value is
  recycled for all `n`.

- ncomp:

  Optional integer vector specifying the number of components to compute
  in each scenario. Defaults to `pmax(2, pmin(10, floor(n / 10)))`. A
  single value is recycled for all `n`.

- reps:

  Number of independent repetitions per method and scenario. The default
  (`reps = 5`) offers a reasonable compromise between speed and
  robustness of the summary statistics.

- methods:

  Character vector listing the backends to benchmark. Methods whose
  required packages are not installed are reported with
  `available = FALSE`.

- seed:

  Optional integer seed applied before each scenario for
  reproducibility. When `NULL`, the current RNG state is left unchanged.

- rspectra_opts, primme_opts:

  Named lists of additional arguments forwarded to
  [`geneig()`](https://bbuchsbaum.github.io/multivarious/reference/geneig.md)
  whenever the corresponding backend is evaluated. For example, set
  `rspectra_opts = list(opts = list(tol = 1e-10))` to tighten the
  convergence tolerance.

- primme_jacobi:

  Logical; when `TRUE` and benchmarking the PRIMME backend, add a simple
  Jacobi preconditioner based on `diag(A)` (ignored if
  `primme_opts$prec` is supplied). Defaults to `FALSE`.

- subspace_opts:

  Named list of arguments passed to
  [`geneig()`](https://bbuchsbaum.github.io/multivarious/reference/geneig.md)
  when `method = "subspace"`. Defaults to a slightly looser tolerance
  and a higher iteration cap (`list(max_iter = 300, tol = 1e-5)`) to
  balance runtime and accuracy in randomized benchmarks.

- ...:

  Additional arguments passed to every
  [`geneig()`](https://bbuchsbaum.github.io/multivarious/reference/geneig.md)
  call (e.g., `opts = list(ncv = 50)`).

## Value

A tibble with one row per repetition and method containing the raw
timings. Columns include:

- `scenario`: textual label describing the problem size and sparsity.

- `n`, `density`, `type`, `nnz`: descriptors of the generated matrices.

- `method`, `rep`, `time`: backend name, repetition index and elapsed
  time (seconds).

- `residual`: spectral residual `max(abs(A V - B V diag(lambda)))` for
  the first repetition (others contain `NA`).

- `available`: flag indicating whether the backend ran successfully.

- `note`: diagnostic note for skipped/failed backends.

## Examples

``` r
# \donttest{
  if (requireNamespace("RSpectra", quietly = TRUE)) {
    bench_res <- benchmark_geneig(
      n = c(150, 300),
      density = c(1, 0.05),
      reps = 2,
      methods = c("robust", "sdiag", "rspectra"),
      rspectra_opts = list(opts = list(tol = 1e-9))
    )
    summarize_geneig_benchmark(bench_res)
    plot_geneig_benchmark(bench_res)
  }
#> 'as(<dsCMatrix>, "dgCMatrix")' is deprecated.
#> Use 'as(., "generalMatrix")' instead.
#> See help("Deprecated") and help("Matrix-deprecated").
#> Error in grid.Call(C_stringMetric, as.graphicsAnnot(x$label)): 'S4SXP': should not happen - please report
# }
```
