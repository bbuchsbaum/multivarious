#' Bootstrap stability summaries for an effect operator
#'
#' @param x An `effect_operator`.
#' @param nboot Number of bootstrap resamples.
#' @param resample Resampling unit. Use `"subject"` for grouped repeated-measures data
#'   or `"rows"` for observation-level resampling; `"auto"` selects subject blocks when available.
#' @param parallel Logical; if `TRUE`, use `future.apply`.
#' @param seed Optional random seed.
#' @param ... Reserved for future extensions.
#' @return A bootstrap result object with loading and singular-value summaries.
#'   `n_failed` records the number of resamples that produced a rank-deficient
#'   or otherwise unfittable design; failed draws are skipped.
#' @note Loadings are currently aligned to the reference solution by a per-axis
#'   sign flip. When more than one effect axis is retained, axes can rotate
#'   between bootstrap draws and a Procrustes-based alignment (see PRD section
#'   10) would give tighter stability summaries. Not implemented in this pass.
#' @export
bootstrap.effect_operator <- function(x,
                                      nboot,
                                      resample = c("auto", "rows", "subject"),
                                      parallel = FALSE,
                                      seed = NULL,
                                      ...) {
  chk::chk_count(nboot)
  resample <- match.arg(resample)

  fit <- x$fit
  B <- fit$basis_matrix
  term <- x$term
  ref_v <- components(x)
  ref_d <- x$sdev
  k <- ncomp(x)
  n <- nrow(fit$Y_proc)
  p <- nrow(ref_v)

  if (!is.null(seed)) {
    set.seed(seed)
  }

  empty_draw <- list(v = matrix(0, nrow = p, ncol = k), d = numeric(k), ok = FALSE)

  worker <- function(i) {
    use_subject_blocks <- !is.null(fit$subject_blocks) && resample %in% c("auto", "subject")
    idx <- if (use_subject_blocks) {
      resample_blocks(fit$subject_blocks, replace = TRUE)
    } else {
      sample.int(n, size = n, replace = TRUE)
    }
    design_b <- fit$design[idx, , drop = FALSE]
    Y_proc_b <- fit$Y_proc[idx, , drop = FALSE]

    tryCatch({
      random_spec_b <- parse_random_spec(fit$random, design_b)
      row_engine_b <- build_row_engine(fit$fixed, design_b, random_spec_b, Y_proc_b)
      meta_b <- row_engine_b$effects_meta[[term]]
      if (is.null(meta_b)) {
        return(empty_draw)
      }
      Yw_basis_b <- row_engine_b$whiten(Y_proc_b %*% B)
      M_basis_b <- meta_b$P_term %*% Yw_basis_b
      rank_b <- min(k, qr(M_basis_b)$rank, nrow(M_basis_b), ncol(M_basis_b))

      sv <- svd_effect(M_basis_b, B, rank_b)
      vb <- matrix(0, nrow = p, ncol = k)
      db <- numeric(k)
      if (sv$rank > 0L) {
        cols <- seq_len(sv$rank)
        vb[, cols] <- sv$v
        db[cols] <- sv$sdev
        signs <- sign(colSums(vb[, cols, drop = FALSE] * ref_v[, cols, drop = FALSE]))
        signs[signs == 0] <- 1
        vb[, cols] <- sweep(vb[, cols, drop = FALSE], 2, signs, "*")
      }
      list(v = vb, d = db, ok = TRUE)
    }, error = function(e) empty_draw)
  }

  apply_fun <- if (parallel) {
    if (!requireNamespace("future.apply", quietly = TRUE)) {
      stop("future.apply is required for parallel bootstrap.")
    }
    future.apply::future_lapply
  } else {
    lapply
  }

  args <- list(X = seq_len(nboot), FUN = worker)
  if (parallel) {
    args$future.seed <- TRUE
  }
  boot_list <- do.call(apply_fun, args)

  ok_flags <- vapply(boot_list, function(b) isTRUE(b$ok), logical(1))
  n_failed <- sum(!ok_flags)
  ok_list <- boot_list[ok_flags]
  n_ok <- length(ok_list)

  v_array <- array(0, dim = c(p, k, nboot))
  d_mat <- matrix(0, nrow = nboot, ncol = k)
  for (i in seq_len(nboot)) {
    v_array[, , i] <- boot_list[[i]]$v
    d_mat[i, ] <- boot_list[[i]]$d
  }

  if (n_ok > 0L) {
    v_ok <- v_array[, , ok_flags, drop = FALSE]
    d_ok <- d_mat[ok_flags, , drop = FALSE]
    v_mean <- apply(v_ok, c(1, 2), mean)
    v_sd <- apply(v_ok, c(1, 2), stats::sd)
    d_mean <- colMeans(d_ok)
    d_sd <- apply(d_ok, 2, stats::sd)
  } else {
    v_mean <- matrix(0, nrow = p, ncol = k)
    v_sd <- matrix(NA_real_, nrow = p, ncol = k)
    d_mean <- numeric(k)
    d_sd <- rep(NA_real_, k)
  }

  out <- list(
    call = match.call(),
    term = term,
    nboot = nboot,
    n_failed = n_failed,
    resample = if (!is.null(fit$subject_blocks) && resample %in% c("auto", "subject")) "subject" else "rows",
    loadings_mean = v_mean,
    loadings_sd = v_sd,
    singular_values_mean = d_mean,
    singular_values_sd = d_sd,
    loadings_array = v_array,
    singular_values_matrix = d_mat
  )
  class(out) <- c("bootstrap_effect_operator_result", "list")
  out
}

#' @export
print.bootstrap_effect_operator_result <- function(x, ...) {
  cat("Bootstrap stability for effect_operator\n\n")
  cat("Term: ", x$term, "\n", sep = "")
  cat("Bootstrap samples: ", x$nboot, "\n", sep = "")
  if (!is.null(x$n_failed) && x$n_failed > 0L) {
    cat("Failed draws: ", x$n_failed, "\n", sep = "")
  }
  cat("Resampling unit: ", x$resample, "\n", sep = "")
  cat("Mean singular values: ", paste(round(x$singular_values_mean, 4), collapse = ", "), "\n", sep = "")
  invisible(x)
}
