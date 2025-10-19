#' Benchmark `geneig()` Backends
#'
#' Quickly compare the runtime and numerical accuracy of the available
#' `geneig()` backends on randomly generated SPD problems. The helper returns a
#' tidy data frame that can be summarised or visualised with
#' `plot_geneig_benchmark()`.
#'
#' @param n Integer vector with the problem sizes (matrix dimension). Each size
#'   defines one benchmark scenario.
#' @param density Optional numeric vector in `(0, 1]` with the proportion of
#'   non-zero entries to use for each scenario. Values `< 1` generate sparse
#'   matrices via `Matrix::rsparsematrix()`, values `>= 1` (or `NULL`) generate
#'   dense matrices. A single value is recycled for all `n`.
#' @param ncomp Optional integer vector specifying the number of components to
#'   compute in each scenario. Defaults to `pmax(2, pmin(10, floor(n / 10)))`.
#'   A single value is recycled for all `n`.
#' @param reps Number of independent repetitions per method and scenario. The
#'   default (`reps = 5`) offers a reasonable compromise between speed and
#'   robustness of the summary statistics.
#' @param methods Character vector listing the backends to benchmark. Methods
#'   whose required packages are not installed are reported with `available = FALSE`.
#' @param seed Optional integer seed applied before each scenario for
#'   reproducibility. When `NULL`, the current RNG state is left unchanged.
#' @param rspectra_opts,primme_opts Named lists of additional arguments forwarded
#'   to `geneig()` whenever the corresponding backend is evaluated. For example,
#'   set `rspectra_opts = list(opts = list(tol = 1e-10))` to tighten the
#'   convergence tolerance.
#' @param primme_jacobi Logical; when `TRUE` and benchmarking the PRIMME backend,
#'   add a simple Jacobi preconditioner based on `diag(A)` (ignored if
#'   `primme_opts$prec` is supplied). Defaults to `FALSE`.
#' @param subspace_opts Named list of arguments passed to `geneig()` when
#'   `method = "subspace"`. Defaults to a slightly looser tolerance and a higher
#'   iteration cap (`list(max_iter = 300, tol = 1e-5)`) to balance runtime and
#'   accuracy in randomized benchmarks.
#' @param ... Additional arguments passed to every `geneig()` call (e.g.,
#'   `opts = list(ncv = 50)`).
#'
#' @return A tibble with one row per repetition and method containing the raw
#'   timings. Columns include:
#'   * `scenario`: textual label describing the problem size and sparsity.
#'   * `n`, `density`, `type`, `nnz`: descriptors of the generated matrices.
#'   * `method`, `rep`, `time`: backend name, repetition index and elapsed time (seconds).
#'   * `residual`: spectral residual `max(abs(A V - B V diag(lambda)))` for the
#'     first repetition (others contain `NA`).
#'   * `available`: flag indicating whether the backend ran successfully.
#'   * `note`: diagnostic note for skipped/failed backends.
#'
#' @noRd
benchmark_geneig <- function(n = c(200, 400),
                             density = NULL,
                             ncomp = NULL,
                             reps = 5,
                             methods = c("robust", "sdiag", "geigen", "primme", "rspectra", "subspace"),
                             seed = 123,
                             rspectra_opts = list(),
                             primme_opts = list(),
                             primme_jacobi = FALSE,
                             subspace_opts = list(max_iter = 300, tol = 1e-5),
                             ...) {

  stopifnot(is.numeric(n), all(n > 0))
  n <- as.integer(n)
  ns <- length(n)

  if (is.null(density)) {
    density <- rep(1, ns)
  } else {
    stopifnot(is.numeric(density), all(density > 0))
    if (length(density) == 1) {
      density <- rep(density, ns)
    } else if (length(density) != ns) {
      stop("'density' must have length 1 or match length(n).", call. = FALSE)
    }
  }

  if (is.null(ncomp)) {
    ncomp <- pmax(2L, pmin(10L, floor(n / 10L)))
  } else {
    stopifnot(is.numeric(ncomp), all(ncomp > 0))
    if (length(ncomp) == 1) {
      ncomp <- rep(as.integer(ncomp), ns)
    } else if (length(ncomp) != ns) {
      stop("'ncomp' must have length 1 or match length(n).", call. = FALSE)
    } else {
      ncomp <- as.integer(ncomp)
    }
  }

  stopifnot(is.numeric(reps), length(reps) == 1, reps >= 1)
  reps <- as.integer(reps)

  methods <- unique(methods)
  stopifnot(all(methods %in% c("robust", "sdiag", "geigen", "primme", "rspectra", "subspace")))

  common_extra <- list(...)

  results <- vector("list", ns)

  for (i in seq_len(ns)) {
    n_i <- n[i]
    dens_i <- density[i]
    ncomp_i <- min(ncomp[i], n_i)

    if (!is.null(seed)) set.seed(seed + i - 1L)

    if (dens_i < 1) {
      A0 <- Matrix::rsparsematrix(n_i, n_i, dens_i)
      B0 <- Matrix::rsparsematrix(n_i, n_i, dens_i)
      A <- Matrix::crossprod(A0) + Matrix::Diagonal(n_i) * 1e-3
      B <- Matrix::crossprod(B0) + Matrix::Diagonal(n_i)
      type <- "sparse"
      nnz <- length(A@x)
    } else {
      A0 <- matrix(rnorm(n_i * n_i), n_i, n_i)
      B0 <- matrix(rnorm(n_i * n_i), n_i, n_i)
      A <- crossprod(A0) + diag(n_i) * 1e-3
      B <- crossprod(B0) + diag(n_i)
      type <- "dense"
      nnz <- sum(A != 0)
    }

    scenario_label <- paste0("n=", n_i, if (dens_i < 1) paste0(", density=", sprintf("%.2f", dens_i)) else " (dense)")

    method_tbls <- vector("list", length(methods))

    for (m_idx in seq_along(methods)) {
      method <- methods[m_idx]
      method_tbls[[m_idx]] <- .benchmark_geneig_method(
        method = method,
        A = A,
        B = B,
        n = n_i,
        ncomp = ncomp_i,
        reps = reps,
        scenario = scenario_label,
        density = dens_i,
        type = type,
        nnz = nnz,
        common_extra = common_extra,
        rspectra_opts = rspectra_opts,
        primme_opts = primme_opts,
        primme_jacobi = primme_jacobi,
        subspace_opts = subspace_opts
      )
    }

    results[[i]] <- dplyr::bind_rows(method_tbls)
  }

  out <- dplyr::bind_rows(results)
  out$method <- factor(out$method, levels = methods)
  out
}

# Internal helper
.benchmark_geneig_method <- function(method,
                                     A,
                                     B,
                                     n,
                                     ncomp,
                                     reps,
                                     scenario,
                                     density,
                                     type,
                                     nnz,
                                     common_extra,
                                     rspectra_opts,
                                     primme_opts,
                                     primme_jacobi,
                                     subspace_opts) {

  required_pkg <- switch(
    method,
    geigen = "geigen",
    primme = "PRIMME",
    rspectra = "RSpectra",
    NULL
  )

  if (!is.null(required_pkg) && !requireNamespace(required_pkg, quietly = TRUE)) {
    note <- paste0("Package '", required_pkg, "' not installed")
    return(tibble::tibble(
      scenario = scenario,
      n = n,
      density = density,
      type = type,
      nnz = nnz,
      method = method,
      rep = NA_integer_,
      time = NA_real_,
      residual = NA_real_,
      available = FALSE,
      note = note
    ))
  }

  call_args <- c(
    list(A = A, B = B, ncomp = ncomp, method = method, preproc = NULL),
    common_extra
  )

  if (identical(method, "primme")) {
    default_primme <- list(
      eps = 1e-6,
      maxBlockSize = max(1L, min(4L, as.integer(ncomp)))
    )
    extra <- utils::modifyList(default_primme, primme_opts)
    if (isTRUE(primme_jacobi) && is.null(extra$prec)) {
      diag_A <- if (inherits(A, "Matrix")) Matrix::diag(A) else base::diag(A)
      if (any(diag_A == 0)) {
        warning("Jacobi preconditioner skipped: diagonal entries of A include zeros.")
      } else {
        extra$prec <- function(x) x / diag_A
      }
    }
    if (!is.null(extra$method)) {
      extra$.primme_method <- extra$method
      extra$method <- NULL
    }
    call_args <- c(call_args, extra)
  } else if (identical(method, "rspectra")) {
    call_args <- c(call_args, rspectra_opts)
  } else if (identical(method, "subspace")) {
    call_args <- c(call_args, subspace_opts)
  }

  times <- rep(NA_real_, reps)
  residuals <- rep(NA_real_, reps)
  note <- NA_character_
  available <- TRUE

  for (r in seq_len(reps)) {
    gc(FALSE)
    t <- tryCatch(
      {
        timing <- system.time({
          sol <- do.call(geneig, call_args)
        })
        times[r] <- as.numeric(timing["elapsed"])
        if (r == 1) {
          Av <- A %*% sol$vectors
          Bv <- B %*% sol$vectors
          residuals[r] <- max(abs(Av - Bv %*% diag(sol$values, ncol(sol$vectors), ncol(sol$vectors))))
        }
        TRUE
      },
      error = function(e) {
        note <<- conditionMessage(e)
        available <<- FALSE
        FALSE
      }
    )

    if (!isTRUE(t)) break
  }

  tibble::tibble(
    scenario = scenario,
    n = n,
    density = density,
    type = type,
    nnz = nnz,
    method = method,
    rep = seq_len(reps),
    time = times,
    residual = residuals,
    available = available,
    note = note
  )
}

#' Summarise Benchmark Results
#'
#' @param bench_result Output of [benchmark_geneig()].
#'
#' @return Tibble with one row per method and scenario containing aggregated
#'   statistics (mean/median runtime, standard deviation, number of runs,
#'   residuals, availability notes).
#' @noRd
summarize_geneig_benchmark <- function(bench_result) {
  required_cols <- c("scenario", "method", "time", "available")
  if (!all(required_cols %in% names(bench_result))) {
    stop("Input does not look like benchmark results from benchmark_geneig().", call. = FALSE)
  }

  bench_result <- dplyr::mutate(
    bench_result,
    scenario = factor(scenario, levels = unique(scenario))
  )

  dplyr::group_by(bench_result, scenario, method) |>
    dplyr::summarise(
      n = dplyr::first(n),
      density = dplyr::first(density),
      type = dplyr::first(type),
      nnz = dplyr::first(nnz),
      available = any(available),
      n_runs = sum(!is.na(time)),
      median_time = if (all(is.na(time))) NA_real_ else stats::median(time, na.rm = TRUE),
      mean_time = if (all(is.na(time))) NA_real_ else mean(time, na.rm = TRUE),
      sd_time = if (all(is.na(time))) NA_real_ else stats::sd(time, na.rm = TRUE),
      residual = if (all(is.na(residual))) NA_real_ else stats::median(residual, na.rm = TRUE),
      note = paste(unique(stats::na.omit(note)), collapse = "; "),
      .groups = "drop"
    ) |>
    dplyr::arrange(scenario, method)
}

#' Plot Benchmark Results
#'
#' @param bench_result Output from [benchmark_geneig()] or
#'   [summarize_geneig_benchmark()].
#' @param stat Which summary statistic to display on the y-axis (`"median"` or
#'   `"mean"`).
#' @param log_scale Logical; if `TRUE` (default) the y-axis is shown on a log10 scale.
#'
#' @return A `ggplot` object comparing runtime across methods.
#' @noRd
plot_geneig_benchmark <- function(bench_result, stat = c("median", "mean"), log_scale = TRUE) {
  stat <- match.arg(stat)

  if (!all(c("median_time", "mean_time") %in% names(bench_result))) {
    bench_result <- summarize_geneig_benchmark(bench_result)
  }

  plot_data <- dplyr::filter(bench_result, available)

  value_col <- if (identical(stat, "median")) "median_time" else "mean_time"

  p <- ggplot2::ggplot(
    plot_data,
    ggplot2::aes(x = method, y = !!rlang::sym(value_col), fill = method)
  ) +
    ggplot2::geom_col(width = 0.6) +
    ggplot2::geom_errorbar(
      ggplot2::aes(
        ymin = !!rlang::sym(value_col) - sd_time,
        ymax = !!rlang::sym(value_col) + sd_time
      ),
      width = 0.2,
      colour = "#333333"
    ) +
    ggplot2::facet_wrap(~ scenario, scales = "free_y") +
    ggplot2::labs(
      x = "geneig() backend",
      y = if (identical(stat, "median")) "Median runtime (s)" else "Mean runtime (s)",
      title = "geneig() backend benchmark",
      fill = "Method"
    ) +
    ggplot2::theme_minimal(base_size = 12) +
    ggplot2::theme(legend.position = "bottom")

  if (isTRUE(log_scale)) {
    p <- p + ggplot2::scale_y_log10()
  }

  p
}
