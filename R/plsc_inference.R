#' Permutation test for PLSC latent variables
#'
#' Uses row-wise permutation of the Y block to assess the significance of each
#' latent variable (LV) in a fitted \code{plsc} model. The test statistic is the
#' singular value of the cross-covariance matrix for each LV.
#'
#' @inheritParams perm_test.pca
#' @param X Original X block used to fit \code{x}.
#' @param Y Original Y block used to fit \code{x}.
#' @param comps Number of components (LVs) to test. Defaults to \code{ncomp(x)}.
#' @param shuffle_fun Optional function to permute Y; defaults to shuffling rows.
#' @param alpha Significance level used to report \code{n_significant}; not used
#'   directly in p-value calculation.
#' @export
perm_test.plsc <- function(x,
                           X,
                           Y,
                           nperm = 1000,
                           comps = ncomp(x),
                           shuffle_fun = NULL,
                           parallel = FALSE,
                           alternative = c("greater", "less", "two.sided"),
                           alpha = 0.05,
                           ...) {
  alternative <- match.arg(alternative)
  chk::chk_matrix(X); chk::chk_matrix(Y)
  chk::chk_equal(nrow(X), nrow(Y))

  comps <- min(comps, ncomp(x))
  if (comps < 1) stop("Need at least one component to test.")

  # Use fitted preprocessors from the model to avoid re-learning on permutations
  Xp <- transform(x$preproc_x, X)
  Yp <- transform(x$preproc_y, Y)

  if (is.null(shuffle_fun)) {
    shuffle_fun <- function(Ymat) Ymat[sample(nrow(Ymat)), , drop = FALSE]
  }

  obs_vals <- x$singvals[seq_len(comps)]
  n <- nrow(Xp)

  one_perm <- function(i) {
    Yperm <- shuffle_fun(Yp)
    Cperm <- crossprod(Xp, Yperm) / (n - 1)
    sv <- try(svd(Cperm, nu = comps, nv = comps), silent = TRUE)
    if (inherits(sv, "try-error")) return(rep(NA_real_, comps))
    sv$d[seq_len(comps)]
  }

  apply_fun <- if (parallel) {
    if (!requireNamespace("future.apply", quietly = TRUE)) {
      stop("future.apply is required for parallel=TRUE")
    }
    future.apply::future_lapply
  } else {
    lapply
  }

  perm_list <- do.call(apply_fun, list(X = seq_len(nperm), FUN = one_perm))
  perm_mat <- do.call(rbind, perm_list)
  n_complete <- colSums(is.finite(perm_mat))

  get_p <- function(vals, obs) {
    vals <- vals[is.finite(vals)]
    if (length(vals) == 0 || is.na(obs)) return(NA_real_)
    if (alternative == "greater") {
      (sum(vals >= obs) + 1) / (length(vals) + 1)
    } else if (alternative == "less") {
      (sum(vals <= obs) + 1) / (length(vals) + 1)
    } else {
      greater <- (sum(vals >= obs) + 1) / (length(vals) + 1)
      less <- (sum(vals <= obs) + 1) / (length(vals) + 1)
      min(1, 2 * min(greater, less))
    }
  }

  pvals <- vapply(seq_len(comps), function(j) get_p(perm_mat[, j], obs_vals[j]), numeric(1))

  lower_ci <- upper_ci <- rep(NA_real_, comps)
  for (j in seq_len(comps)) {
    if (n_complete[j] > 1) {
      qs <- stats::quantile(stats::na.omit(perm_mat[, j]), probs = c(0.025, 0.975))
      lower_ci[j] <- qs[1]; upper_ci[j] <- qs[2]
    }
  }

  n_significant <- 0
  for (j in seq_len(comps)) {
    if (!is.na(pvals[j]) && pvals[j] <= alpha) {
      n_significant <- j
    } else {
      break
    }
  }

  component_results <- tibble::tibble(
    comp = seq_len(comps),
    observed = obs_vals,
    pval = pvals,
    lower_ci = lower_ci,
    upper_ci = upper_ci
  )

  out <- list(
    call = match.call(),
    component_results = component_results,
    perm_values = perm_mat,
    alpha = alpha,
    alternative = alternative,
    method = "Permutation test for PLSC (row-shuffle Y; statistic = singular value)",
    nperm = n_complete,
    n_significant = n_significant
  )
  class(out) <- c("perm_test_plsc", "perm_test")
  out
}

#' @export
print.perm_test_plsc <- function(x, ...) {
  cat("\nPLSC Permutation Test Results\n\n")
  cat("Method: ", x$method, "\n")
  cat("Alternative: ", x$alternative, "\n")
  cat("Alpha: ", x$alpha, "\n\n", sep = "")
  print(as.data.frame(x$component_results))
  cat("\nSuccessful permutations per component: ", paste(x$nperm, collapse = ", "), "\n", sep = "")
  cat("Number of significant components (sequential, alpha = ", x$alpha, "): ", x$n_significant, "\n", sep = "")
  invisible(x)
}

#' Bootstrap inference for PLSC loadings
#'
#' Provides bootstrap ratios (mean / sd) for X and Y loadings to assess stability,
#' mirroring common practice in Behavior PLSC.
#'
#' @param x A fitted \code{plsc} object.
#' @param X Original X block.
#' @param Y Original Y block.
#' @param nboot Number of bootstrap samples (default 500).
#' @param comps Number of components to bootstrap (default: \code{ncomp(x)}).
#' @param seed Optional integer seed for reproducibility.
#' @param parallel Use future.apply for parallelization (default FALSE).
#' @param epsilon Small positive constant to stabilize division for ratios.
#' @export
bootstrap_plsc <- function(x,
                           X,
                           Y,
                           nboot = 500,
                           comps = ncomp(x),
                           seed = NULL,
                           parallel = FALSE,
                           epsilon = 1e-9,
                           ...) {
  chk::chk_matrix(X); chk::chk_matrix(Y)
  chk::chk_equal(nrow(X), nrow(Y))
  comps <- min(comps, ncomp(x))
  if (comps < 1) stop("Need at least one component to bootstrap.")
  if (!is.null(seed)) set.seed(seed)

  vx_ref <- coef.cross_projector(x, source = "X")[, seq_len(comps), drop = FALSE]
  vy_ref <- coef.cross_projector(x, source = "Y")[, seq_len(comps), drop = FALSE]
  n <- nrow(X)

  align_signs <- function(vx_b, vy_b) {
    signs <- sign(colSums(vx_b * vx_ref))
    zero <- which(signs == 0)
    if (length(zero)) {
      signs[zero] <- sign(colSums(vy_b[, zero, drop = FALSE] * vy_ref[, zero, drop = FALSE]))
    }
    signs[signs == 0] <- 1
    list(
      vx = sweep(vx_b, 2, signs, "*"),
      vy = sweep(vy_b, 2, signs, "*"),
      signs = signs
    )
  }

  boot_worker <- function(i) {
    idx <- sample.int(n, n, replace = TRUE)
    # Fresh preprocessors preserve the pipeline but refit to the bootstrap sample
    px_b <- try(fresh(x$preproc_x), silent = TRUE)
    py_b <- try(fresh(x$preproc_y), silent = TRUE)
    if (inherits(px_b, "try-error") || inherits(py_b, "try-error")) {
      px_b <- x$preproc_x; py_b <- x$preproc_y
    }
    modb <- try(plsc(X[idx, , drop = FALSE],
                     Y[idx, , drop = FALSE],
                     ncomp = comps,
                     preproc_x = px_b,
                     preproc_y = py_b),
                silent = TRUE)
    if (inherits(modb, "try-error")) return(NULL)
    aligned <- align_signs(modb$vx, modb$vy)
    list(
      vx = aligned$vx,
      vy = aligned$vy,
      singvals = modb$singvals[seq_len(comps)]
    )
  }

  apply_fun <- if (parallel) {
    if (!requireNamespace("future.apply", quietly = TRUE)) {
      stop("future.apply is required for parallel=TRUE")
    }
    future.apply::future_lapply
  } else {
    lapply
  }

  res_list <- do.call(apply_fun, list(X = seq_len(nboot), FUN = boot_worker))
  res_list <- Filter(Negate(is.null), res_list)

  if (length(res_list) == 0) stop("All bootstrap replicates failed.")

  vx_arr <- simplify2array(lapply(res_list, function(z) z$vx))
  vy_arr <- simplify2array(lapply(res_list, function(z) z$vy))
  sv_arr <- do.call(rbind, lapply(res_list, function(z) z$singvals))

  E_vx <- apply(vx_arr, c(1, 2), mean)
  sd_vx <- apply(vx_arr, c(1, 2), sd)
  z_vx <- E_vx / pmax(sd_vx, epsilon)

  E_vy <- apply(vy_arr, c(1, 2), mean)
  sd_vy <- apply(vy_arr, c(1, 2), sd)
  z_vy <- E_vy / pmax(sd_vy, epsilon)

  out <- list(
    call = match.call(),
    comps = comps,
    requested_nboot = nboot,
    successful_nboot = length(res_list),
    E_vx = E_vx,
    sd_vx = sd_vx,
    z_vx = z_vx,
    E_vy = E_vy,
    sd_vy = sd_vy,
    z_vy = z_vy,
    singvals = sv_arr
  )
  class(out) <- c("bootstrap_plsc_result", "list")
  out
}

#' @export
bootstrap.plsc <- function(x, X, Y, nboot = 500, ...) {
  bootstrap_plsc(x, X = X, Y = Y, nboot = nboot, ...)
}

#' @export
print.bootstrap_plsc_result <- function(x, ...) {
  cat(crayon::bold(crayon::green("PLSC bootstrap (loadings)\n")))
  cat("Components: ", x$comps, 
      " | Successful resamples: ", x$successful_nboot, 
      "/", x$requested_nboot, "\n", sep = "")
  cat("Use $z_vx and $z_vy for bootstrap ratios (X/Y loadings).\n")
  invisible(x)
}
