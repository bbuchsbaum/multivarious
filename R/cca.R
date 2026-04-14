#' Canonical Correlation Analysis (CCA)
#'
#' Reference implementation of two-block canonical correlation analysis that
#' returns a \code{cross_projector}. The within-block covariance matrices are
#' ridge-regularized before whitening, which keeps the fit well-defined even
#' when \eqn{p > n} or either block is rank-deficient.
#'
#' @param X Numeric matrix of predictors (n x p_x).
#' @param Y Numeric matrix of outcomes (n x p_y). Must have the same number of
#'   rows as \code{X}.
#' @param ncomp Number of canonical dimensions to return. Defaults to
#'   \code{min(ncol(X), ncol(Y), nrow(X) - 1)} after preprocessing.
#' @param preproc_x Preprocessor for the X block (default: \code{center()}).
#' @param preproc_y Preprocessor for the Y block (default: \code{center()}).
#' @param lambda Shared ridge shrinkage level used when \code{lambda_x} and
#'   \code{lambda_y} are not supplied. The effective ridge added to each block
#'   is \code{lambda * mean(diag(S))}, where \code{S} is the block covariance.
#' @param lambda_x Ridge shrinkage level for the X block covariance.
#' @param lambda_y Ridge shrinkage level for the Y block covariance.
#' @param tol Eigenvalue floor used when whitening regularized covariance
#'   matrices. Defaults to \code{sqrt(.Machine$double.eps)}.
#' @param ... Extra arguments stored on the returned object.
#'
#' @return A \code{cross_projector} with class \code{"cca"} containing
#'   canonical coefficients (\code{vx}, \code{vy}), block scores (\code{sx},
#'   \code{sy}), canonical correlations (\code{cor}), and the regularized block
#'   covariance matrices used to fit the model.
#' @export
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(120), 30, 4)
#' Y <- matrix(rnorm(90), 30, 3)
#' fit <- cca(X, Y, ncomp = 2)
#' fit$cor
cca <- function(X,
                Y,
                ncomp = NULL,
                preproc_x = center(),
                preproc_y = center(),
                lambda = 1e-4,
                lambda_x = lambda,
                lambda_y = lambda,
                tol = sqrt(.Machine$double.eps),
                ...) {
  chk::chk_matrix(X)
  chk::chk_matrix(Y)
  chk::chk_equal(nrow(X), nrow(Y))

  if (nrow(X) < 2) {
    stop("CCA requires at least two observations.", call. = FALSE)
  }
  if (!is.numeric(lambda_x) || length(lambda_x) != 1L || is.na(lambda_x) || lambda_x < 0) {
    stop("lambda_x must be a non-negative scalar.", call. = FALSE)
  }
  if (!is.numeric(lambda_y) || length(lambda_y) != 1L || is.na(lambda_y) || lambda_y < 0) {
    stop("lambda_y must be a non-negative scalar.", call. = FALSE)
  }
  if (!is.numeric(tol) || length(tol) != 1L || is.na(tol) || tol <= 0) {
    stop("tol must be a positive scalar.", call. = FALSE)
  }

  fit_block <- function(p, M) {
    if (inherits(p, "pre_processor")) {
      list(preproc = p, transformed = transform(p, M))
    } else {
      res <- fit_transform(p, M)
      list(preproc = res$preproc, transformed = res$transformed)
    }
  }

  fx_res <- fit_block(preproc_x, X)
  fy_res <- fit_block(preproc_y, Y)
  fx <- fx_res$preproc
  fy <- fy_res$preproc
  Xp <- fx_res$transformed
  Yp <- fy_res$transformed

  max_comp <- min(ncol(Xp), ncol(Yp), nrow(Xp) - 1L)
  if (max_comp < 1L) {
    stop("CCA requires at least one estimable canonical component.", call. = FALSE)
  }
  if (is.null(ncomp)) {
    ncomp <- max_comp
  } else {
    chk::chk_range(ncomp, c(1, max_comp))
  }

  n_eff <- nrow(Xp) - 1
  Sxx <- crossprod(Xp) / n_eff
  Syy <- crossprod(Yp) / n_eff
  Sxy <- crossprod(Xp, Yp) / n_eff

  penalty_x <- .cca_penalty_scale(Sxx, lambda_x)
  penalty_y <- .cca_penalty_scale(Syy, lambda_y)
  Sxx_reg <- Sxx + diag(penalty_x, nrow(Sxx))
  Syy_reg <- Syy + diag(penalty_y, nrow(Syy))

  wx <- .cca_inv_sqrt(Sxx_reg, tol = tol)
  wy <- .cca_inv_sqrt(Syy_reg, tol = tol)

  K <- wx %*% Sxy %*% wy
  sv <- svd(K, nu = ncomp, nv = ncomp)
  vx <- wx %*% sv$u[, seq_len(ncomp), drop = FALSE]
  vy <- wy %*% sv$v[, seq_len(ncomp), drop = FALSE]
  cor <- pmin(pmax(sv$d[seq_len(ncomp)], 0), 1)

  sx <- Xp %*% vx
  sy <- Yp %*% vy
  explained <- cor^2
  explained <- explained / sum(explained)

  cross_projector(
    vx,
    vy,
    preproc_x = fx,
    preproc_y = fy,
    sx = sx,
    sy = sy,
    cor = cor,
    cancor = cor,
    explained_cor = explained,
    Sxx = Sxx,
    Syy = Syy,
    Sxy = Sxy,
    Sxx_reg = Sxx_reg,
    Syy_reg = Syy_reg,
    ridge = list(
      lambda_x = lambda_x,
      lambda_y = lambda_y,
      penalty_x = penalty_x,
      penalty_y = penalty_y
    ),
    classes = "cca",
    ...
  )
}

#' @export
print.cca <- function(x, ...) {
  cat(cli::style_bold(cli::col_green("CCA object (cross-projector)\n\n")))
  cat(cli::col_cyan("Samples: "), nrow(x$sx), "\n", sep = "")
  cat(cli::col_cyan("X vars: "), nrow(x$vx), " | Y vars: ", nrow(x$vy), "\n", sep = "")
  cat(cli::col_cyan("Components: "), ncomp(x), "\n", sep = "")
  if (!is.null(x$cor)) {
    cat("Canonical correlations: ", paste(round(x$cor, 4), collapse = ", "), "\n")
  }
  invisible(x)
}

#' @export
coef.cca <- function(object, source = c("X", "Y"), ...) {
  source <- match.arg(source)
  coef.cross_projector(object, source = source, ...)
}

#' Extract scores from a CCA fit
#'
#' @param x A \code{cca} object.
#' @param block Which block to return scores for: "X" (default) or "Y".
#' @param ... Ignored.
#' @return Numeric matrix of canonical scores for the chosen block.
#' @export
scores.cca <- function(x, block = c("X", "Y"), ...) {
  block <- match.arg(block)
  if (block == "X") {
    x$sx
  } else {
    x$sy
  }
}

#' @export
reprocess.cca <- function(x, new_data, colind = NULL, source = c("X", "Y"), ...) {
  source <- match.arg(source)
  reprocess.cross_projector(x, new_data, colind = colind, source = source, ...)
}

#' @export
transfer.cca <- function(x, new_data, from, to, opts = list(), ...) {
  transfer.cross_projector(x, new_data, from = from, to = to, opts = opts, ...)
}

#' @export
truncate.cca <- function(x, ncomp) {
  old_ncomp <- ncomp(x)
  chk::chk_number(ncomp)
  if (ncomp < 1 || ncomp > old_ncomp) {
    stop("Requested ncomp must be between 1 and ", old_ncomp)
  }

  keep <- seq_len(ncomp)
  x$vx <- x$vx[, keep, drop = FALSE]
  x$vy <- x$vy[, keep, drop = FALSE]
  x$v <- x$vx

  if (!is.null(x$sx)) {
    x$sx <- x$sx[, keep, drop = FALSE]
  }
  if (!is.null(x$sy)) {
    x$sy <- x$sy[, keep, drop = FALSE]
  }
  if (!is.null(x$cor)) {
    x$cor <- x$cor[keep]
  }
  if (!is.null(x$cancor)) {
    x$cancor <- x$cancor[keep]
  }
  if (!is.null(x$explained_cor)) {
    explained <- x$explained_cor[keep]
    denom <- sum(explained)
    if (is.finite(denom) && denom > 0) {
      explained <- explained / denom
    }
    x$explained_cor <- explained
  }

  cache_env <- x$.cache
  if (is.environment(cache_env)) {
    rm(list = ls(cache_env), envir = cache_env)
  }

  x
}

.cca_penalty_scale <- function(S, lambda) {
  scale <- mean(diag(S))
  if (!is.finite(scale) || scale <= 0) {
    scale <- 1
  }
  lambda * scale
}

.cca_inv_sqrt <- function(S, tol) {
  Ssym <- (S + t(S)) / 2
  eig <- eigen(Ssym, symmetric = TRUE)
  vals <- pmax(eig$values, tol)
  eig$vectors %*% diag(1 / sqrt(vals), nrow = length(vals)) %*% t(eig$vectors)
}
