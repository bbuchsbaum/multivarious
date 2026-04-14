#' Permutation test for PLSC latent variables
#'
#' Uses row-wise permutation of the Y block to assess the significance of each
#' latent variable (LV) in a fitted \code{plsc} model. The test statistic is the
#' singular value of the cross-covariance matrix for each LV.
#'
#' @param x A fitted \code{plsc} model object.
#' @param X Original X block used to fit \code{x}.
#' @param Y Original Y block used to fit \code{x}.
#' @param nperm Number of permutations to perform (default 1000).
#' @param comps Number of components (LVs) to test. Defaults to \code{ncomp(x)}.
#' @param stepwise Logical; if TRUE (default), perform sequential testing with deflation.
#' @param shuffle_fun Optional function to permute Y; defaults to shuffling rows.
#' @param parallel Logical; if TRUE, use parallel processing via future.apply.
#' @param alternative Character string for the alternative hypothesis: "greater" (default), "less", or "two.sided".
#' @param alpha Significance level used to report \code{n_significant}; not used
#'   directly in p-value calculation.
#' @param ... Additional arguments (currently unused).
#' @export
perm_test.plsc <- function(x,
                           X,
                           Y,
                           nperm = 1000,
                           comps = ncomp(x),
                           stepwise = TRUE,
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
    shuffle_fun <- function(Ymat) Ymat[sample.int(nrow(Ymat)), , drop = FALSE]
  }

  n <- nrow(Xp)

  # helper to get leading singular value of cross-covariance
  stat_lead_sv <- function(Xr, Yr) {
    Cxy <- crossprod(Xr, Yr) / (n - 1)
    sv <- try(svd(Cxy, nu = 1, nv = 1), silent = TRUE)
    if (inherits(sv, "try-error") || length(sv$d) == 0) return(NA_real_)
    sv$d[1]
  }

  # Prepare observed residuals per step (stepwise deflation like PCA P3 idea)
  X_res <- Xp
  Y_res <- Yp
  obs_vals <- numeric(comps)
  X_res_list <- vector("list", comps)
  Y_res_list <- vector("list", comps)

  for (a in seq_len(comps)) {
    obs_vals[a] <- stat_lead_sv(X_res, Y_res)
    X_res_list[[a]] <- X_res
    Y_res_list[[a]] <- Y_res
    if (stepwise && a < comps) {
      # Deflate using current LV from observed model
      vxa <- coef(x, "X")[, a, drop = FALSE]
      vya <- coef(x, "Y")[, a, drop = FALSE]
      sx_a <- X_res %*% vxa
      sy_a <- Y_res %*% vya
      X_res <- X_res - sx_a %*% t(vxa)
      Y_res <- Y_res - sy_a %*% t(vya)
    }
  }

  one_perm <- function(i, a, Xr, Yr) {
    Yperm <- shuffle_fun(Yr)
    stat_lead_sv(Xr, Yperm)
  }

  apply_fun <- if (parallel) {
    if (!requireNamespace("future.apply", quietly = TRUE)) {
      stop("future.apply is required for parallel=TRUE")
    }
    future.apply::future_lapply
  } else {
    lapply
  }

  perm_mat <- matrix(NA_real_, nrow = nperm, ncol = comps)
  n_complete <- integer(comps)
  pvals <- rep(NA_real_, comps)
  comps_tested <- 0

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

  for (a in seq_len(comps)) {
    perm_args <- list(X = seq_len(nperm), FUN = one_perm,
                      a = a,
                      Xr = if (stepwise) X_res_list[[a]] else Xp,
                      Yr = if (stepwise) Y_res_list[[a]] else Yp)
    if (parallel) perm_args$future.seed <- TRUE
    perm_vals_list <- do.call(apply_fun, perm_args)
    perm_vec <- unlist(perm_vals_list)
    perm_mat[, a] <- perm_vec
    n_complete[a] <- sum(is.finite(perm_vec))
    pvals[a] <- get_p(perm_vec, obs_vals[a])
    comps_tested <- a
    if (!is.na(pvals[a]) && pvals[a] > alpha) {
      break
    }
  }

  lower_ci <- upper_ci <- rep(NA_real_, comps_tested)
  for (j in seq_len(comps_tested)) {
    if (n_complete[j] > 1) {
      qs <- stats::quantile(stats::na.omit(perm_mat[, j]), probs = c(0.025, 0.975))
      lower_ci[j] <- qs[1]; upper_ci[j] <- qs[2]
    }
  }

  n_significant <- 0
  for (j in seq_len(comps_tested)) {
    if (!is.na(pvals[j]) && pvals[j] <= alpha) {
      n_significant <- j
    } else {
      break
    }
  }

  component_results <- tibble::tibble(
    comp = seq_len(comps_tested),
    observed = obs_vals[seq_len(comps_tested)],
    pval = pvals[seq_len(comps_tested)],
    lower_ci = lower_ci,
    upper_ci = upper_ci
  )

  out <- list(
    call = match.call(),
    component_results = component_results,
    perm_values = perm_mat[, seq_len(comps_tested), drop = FALSE],
    alpha = alpha,
    alternative = alternative,
    method = sprintf("Permutation test for PLSC (row-shuffle Y; statistic = leading singular value; stepwise=%s)", stepwise),
    nperm = n_complete[seq_len(comps_tested)],
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
#' @param ... Additional arguments (currently unused).
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

#' @rdname bootstrap
#' @export
bootstrap.plsc <- function(x, nboot = 500, ...) {
  # Extract X and Y from ... to match generic signature
  args <- list(...)
  X <- args$X
  Y <- args$Y

  if (is.null(X) || is.null(Y)) {
    stop("bootstrap.plsc requires X and Y to be passed as named arguments via ...", call. = FALSE)
  }

  # Remove X and Y from args to avoid duplicate argument error
  args$X <- NULL
  args$Y <- NULL

  do.call(bootstrap_plsc, c(list(x = x, X = X, Y = Y, nboot = nboot), args))
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
