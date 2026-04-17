#' @keywords internal
#' @noRd
orthogonal_projector <- function(X, tol = 1e-10) {
  n <- nrow(X)
  if (is.null(X) || ncol(X) == 0) {
    return(matrix(0, nrow = n, ncol = n))
  }

  qrX <- qr(X, tol = tol)
  rankX <- qrX$rank
  if (rankX == 0) {
    return(matrix(0, nrow = n, ncol = n))
  }

  Q <- qr.Q(qrX)[, seq_len(rankX), drop = FALSE]
  tcrossprod(Q)
}

#' @keywords internal
#' @noRd
make_random_effect_block <- function(random_spec, design) {
  if (is.null(random_spec$grouping_var)) {
    return(NULL)
  }

  Z_full <- stats::model.matrix(random_spec$lhs_formula, data = design)
  blocks <- unname(split(seq_len(nrow(design)), design[[random_spec$grouping_var]], drop = TRUE))
  blocks <- unname(Filter(function(idx) length(idx) > 0, blocks))
  Z_blocks <- lapply(blocks, function(idx) Z_full[idx, , drop = FALSE])

  list(
    Z = Z_full,
    Z_blocks = Z_blocks,
    q = ncol(Z_full),
    blocks = blocks
  )
}

#' @keywords internal
#' @noRd
theta_to_cov_params <- function(theta, q) {
  L <- matrix(0, nrow = q, ncol = q)
  idx <- 1L
  for (j in seq_len(q)) {
    for (i in j:q) {
      if (i == j) {
        L[i, j] <- exp(theta[idx])
      } else {
        L[i, j] <- theta[idx]
      }
      idx <- idx + 1L
    }
  }
  sigma <- exp(theta[length(theta)])
  list(G = L %*% t(L), sigma2 = sigma^2, L = L)
}

#' @keywords internal
#' @noRd
build_block_covariance <- function(Z_blocks, G, sigma2) {
  lapply(Z_blocks, function(Zi) {
    m <- nrow(Zi)
    cov_i <- Zi %*% G %*% t(Zi)
    cov_i + diag(1, m) * sigma2
  })
}

#' @keywords internal
#' @noRd
build_block_chol <- function(Omega_blocks) {
  lapply(Omega_blocks, function(Om) {
    if (!is.matrix(Om)) {
      Om <- as.matrix(Om)
    }
    if (nrow(Om) == 0L || ncol(Om) == 0L) {
      stop("Encountered an empty covariance block.")
    }

    jitter <- 0
    for (attempt in 0:6) {
      Om_try <- if (jitter > 0) Om + diag(jitter, nrow(Om)) else Om
      ch <- try(chol(Om_try), silent = TRUE)
      if (!inherits(ch, "try-error")) {
        return(ch)
      }
      base_scale <- max(diag(Om), 1e-8)
      jitter <- base_scale * 10^(-8 + attempt)
    }

    stop("Failed to compute a positive-definite Cholesky factor for a covariance block.")
  })
}

#' @keywords internal
#' @noRd
apply_block_solve <- function(block_chol, blocks, M) {
  out <- matrix(0, nrow = nrow(M), ncol = ncol(M))
  for (i in seq_along(blocks)) {
    idx <- blocks[[i]]
    U <- block_chol[[i]]
    out[idx, ] <- backsolve(U, M[idx, , drop = FALSE], transpose = FALSE)
  }
  out
}

#' @keywords internal
#' @noRd
apply_block_oinv <- function(block_chol, blocks, M) {
  out <- matrix(0, nrow = nrow(M), ncol = ncol(M))
  for (i in seq_along(blocks)) {
    idx <- blocks[[i]]
    U <- block_chol[[i]]
    tmp <- forwardsolve(t(U), M[idx, , drop = FALSE], upper.tri = FALSE, transpose = FALSE)
    out[idx, ] <- backsolve(U, tmp, transpose = FALSE)
  }
  out
}

#' @keywords internal
#' @noRd
profiled_grouped_nll <- function(theta, X, Y, Z_blocks, blocks, reml = TRUE) {
  q <- ncol(Z_blocks[[1]])
  cov_par <- theta_to_cov_params(theta, q)
  Omega_blocks <- try(build_block_covariance(Z_blocks, cov_par$G, cov_par$sigma2), silent = TRUE)
  if (inherits(Omega_blocks, "try-error")) {
    return(1e20)
  }
  block_chol <- try(build_block_chol(Omega_blocks), silent = TRUE)
  if (inherits(block_chol, "try-error")) {
    return(1e20)
  }

  OinvX <- try(apply_block_oinv(block_chol, blocks, X), silent = TRUE)
  OinvY <- try(apply_block_oinv(block_chol, blocks, Y), silent = TRUE)
  if (inherits(OinvX, "try-error") || inherits(OinvY, "try-error")) {
    return(1e20)
  }

  XtOinvX <- crossprod(X, OinvX)
  XtOinvY <- crossprod(X, OinvY)

  chol_X <- try(chol(XtOinvX), silent = TRUE)
  if (inherits(chol_X, "try-error")) {
    return(1e20)
  }

  Bhat <- try(solve(XtOinvX, XtOinvY), silent = TRUE)
  if (inherits(Bhat, "try-error")) {
    return(1e20)
  }

  E <- Y - X %*% Bhat
  OinvE <- try(apply_block_oinv(block_chol, blocks, E), silent = TRUE)
  if (inherits(OinvE, "try-error")) {
    return(1e20)
  }

  logdet_Omega <- sum(vapply(block_chol, function(U) 2 * sum(log(diag(U))), numeric(1)))
  quad <- sum(E * OinvE)

  penalty <- 0
  if (reml) {
    penalty <- ncol(Y) * (2 * sum(log(diag(chol_X))))
  }

  0.5 * (ncol(Y) * logdet_Omega + penalty + quad)
}

#' @keywords internal
#' @noRd
estimate_grouped_row_metric <- function(X, Y, random_spec, design) {
  zinfo <- make_random_effect_block(random_spec, design)
  q <- zinfo$q

  y_scale <- sqrt(max(mean(Y^2), 1e-6))
  theta0_g <- c()
  for (j in seq_len(q)) {
    for (i in j:q) {
      theta0_g <- c(theta0_g, if (i == j) log(y_scale / 2) else 0)
    }
  }
  theta0 <- c(theta0_g, log(y_scale / 2))
  obj <- function(theta) profiled_grouped_nll(theta, X, Y, zinfo$Z_blocks, zinfo$blocks, reml = TRUE)

  opt <- stats::optim(theta0, obj, method = "BFGS", control = list(maxit = 200, reltol = 1e-8))
  theta_hat <- if (is.finite(opt$value)) opt$par else theta0

  cov_par <- theta_to_cov_params(theta_hat, q)
  Omega_blocks <- build_block_covariance(zinfo$Z_blocks, cov_par$G, cov_par$sigma2)
  block_chol <- build_block_chol(Omega_blocks)

  list(
    mode = "grouped_lmm",
    grouping_var = random_spec$grouping_var,
    random_terms = random_spec$random_terms,
    Z = zinfo$Z,
    Z_blocks = zinfo$Z_blocks,
    blocks = zinfo$blocks,
    G = cov_par$G,
    sigma2 = cov_par$sigma2,
    theta = theta_hat,
    optim = opt,
    block_chol = block_chol
  )
}

#' @keywords internal
#' @noRd
identity_row_metric <- function(design, random_spec = NULL) {
  n <- nrow(design)
  list(
    mode = if (is.null(random_spec$grouping_var)) "identity" else "grouped_identity",
    grouping_var = random_spec$grouping_var,
    random_terms = random_spec$random_terms,
    blocks = random_spec$subject_blocks,
    G = NULL,
    sigma2 = 1,
    theta = NULL,
    optim = NULL,
    block_chol = NULL,
    whiten = function(M) M,
    unwhiten = function(M) M,
    solve = function(M) M,
    logdet = 0,
    n = n
  )
}

#' @keywords internal
#' @noRd
attach_metric_operators <- function(metric) {
  if (metric$mode %in% c("identity", "grouped_identity")) {
    return(identity_row_metric(
      design = data.frame(dummy = seq_len(metric$n)),
      random_spec = list(grouping_var = metric$grouping_var, random_terms = metric$random_terms, subject_blocks = metric$blocks)
    ))
  }

  metric$whiten <- function(M) apply_block_solve(metric$block_chol, metric$blocks, M)
  metric$unwhiten <- function(M) {
    out <- matrix(0, nrow = nrow(M), ncol = ncol(M))
    for (i in seq_along(metric$blocks)) {
      idx <- metric$blocks[[i]]
      out[idx, ] <- metric$block_chol[[i]] %*% M[idx, , drop = FALSE]
    }
    out
  }
  metric$solve <- function(M) apply_block_oinv(metric$block_chol, metric$blocks, M)
  metric$logdet <- sum(vapply(metric$block_chol, function(U) 2 * sum(log(diag(U))), numeric(1)))
  metric$n <- sum(vapply(metric$blocks, length, integer(1)))
  metric
}

#' @keywords internal
#' @noRd
build_row_engine <- function(fixed, design, random_spec, Y_proc) {
  Terms <- stats::terms(fixed, data = design)
  X <- stats::model.matrix(Terms, data = design)
  assign_vec <- attr(X, "assign")
  term_labels <- attr(Terms, "term.labels")
  has_intercept <- attr(Terms, "intercept") == 1L
  intercept_cols <- which(assign_vec == 0L)

  metric <- if (is.null(random_spec$grouping_var)) {
    identity_row_metric(design, random_spec = random_spec)
  } else {
    est <- estimate_grouped_row_metric(X, Y_proc, random_spec, design)
    attach_metric_operators(est)
  }

  X_w <- metric$whiten(X)

  effects_meta <- lapply(seq_along(term_labels), function(i) {
    effect_cols <- which(assign_vec == i)
    nuisance_cols <- setdiff(seq_len(ncol(X)), effect_cols)

    X_effect_w <- X_w[, effect_cols, drop = FALSE]
    X_nuis_w <- if (length(nuisance_cols)) X_w[, nuisance_cols, drop = FALSE] else matrix(0, nrow(X_w), 0)

    P_nuis <- orthogonal_projector(X_nuis_w)
    P_full <- orthogonal_projector(cbind(X_nuis_w, X_effect_w))
    P_term <- P_full - P_nuis

    df_term <- qr(cbind(X_nuis_w, X_effect_w))$rank - qr(X_nuis_w)$rank

    list(
      term = term_labels[[i]],
      term_index = i,
      effect_cols = effect_cols,
      nuisance_cols = nuisance_cols,
      intercept_cols = intercept_cols,
      has_intercept = has_intercept,
      P_term = P_term,
      P_nuis = P_nuis,
      P_full = P_full,
      df_term = df_term,
      term_scope = classify_term_scope(term_labels[[i]], design, random_spec$grouping_var)
    )
  })

  names(effects_meta) <- term_labels
  P_model <- orthogonal_projector(X_w)

  structure(
    list(
      X = X,
      X_w = X_w,
      design = design,
      terms = Terms,
      assign = assign_vec,
      term_labels = term_labels,
      has_intercept = has_intercept,
      intercept_cols = intercept_cols,
      effects_meta = effects_meta,
      P_model = P_model,
      metric = metric,
      whiten = metric$whiten,
      unwhiten = metric$unwhiten,
      solve = metric$solve
    ),
    class = "mixed_row_engine"
  )
}
