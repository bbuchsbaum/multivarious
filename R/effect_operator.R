#' Construct an effect operator
#'
#' @param v Feature loadings in original variable space.
#' @param s Observation-side effect scores.
#' @param sdev Singular values.
#' @param preproc Fitted response preprocessor.
#' @param ... Additional metadata stored on the object.
#' @return An `effect_operator`.
#' @export
effect_operator <- function(v, s, sdev, preproc = fit(pass(), matrix(0, 1, nrow(v))), ...) {
  bi_projector(
    v = v,
    s = s,
    sdev = sdev,
    preproc = preproc,
    classes = "effect_operator",
    ...
  )
}

#' @export
effect.mixed_fit <- function(x, term, ...) {
  term_label <- normalize_mixed_term(term)
  wem <- whitened_effect_matrix(x, term_label)
  meta <- wem$meta
  B <- wem$B
  M_basis <- wem$M

  rank_max <- min(meta$df_term, ncol(B), qr(M_basis)$rank)
  sv <- svd_effect(M_basis, B, rank_max)

  fitted_contribution_w <- if (sv$rank < 1L) {
    matrix(0, nrow = nrow(x$Y_proc), ncol = ncol(x$Y_proc))
  } else {
    sv$s %*% t(sv$v)
  }
  fitted_contribution <- x$row_engine$unwhiten(fitted_contribution_w)

  effect_operator(
    v = sv$v,
    s = sv$s,
    sdev = sv$sdev,
    preproc = x$preproc,
    term = term_label,
    term_scope = meta$term_scope,
    df_term = meta$df_term,
    basis = x$basis_fit,
    basis_rank = ncol(B),
    effect_matrix = M_basis,
    effect_matrix_whitened = fitted_contribution_w,
    row_projector = meta$P_term,
    row_metric = x$row_metric,
    fitted_contribution = fitted_contribution,
    fit = x,
    P_nuis = meta$P_nuis,
    P_full = meta$P_full
  )
}

#' @export
print.effect_operator <- function(x, ...) {
  cat("effect_operator\n\n")
  cat("Term: ", x$term, "\n", sep = "")
  cat("Components: ", ncomp(x), "\n", sep = "")
  cat("Term df: ", x$df_term, "\n", sep = "")
  cat("Scope: ", x$term_scope, "\n", sep = "")
  cat("Basis rank: ", x$basis_rank, "\n", sep = "")
  invisible(x)
}

#' @export
truncate.effect_operator <- function(x, ncomp) {
  old_ncomp <- ncomp(x)
  chk::chk_range(ncomp, c(0, old_ncomp))

  x$v <- x$v[, seq_len(ncomp), drop = FALSE]
  x$s <- x$s[, seq_len(ncomp), drop = FALSE]
  x$sdev <- x$sdev[seq_len(ncomp)]
  fitted_w <- if (ncomp == 0) {
    matrix(0, nrow = nrow(x$s), ncol = nrow(stats::coef(x)))
  } else {
    x$s %*% t(x$v)
  }
  x$effect_matrix_whitened <- fitted_w
  x$fitted_contribution <- x$fit$row_engine$unwhiten(fitted_w)

  cache_env <- attr(x, ".cache")
  if (!is.null(cache_env) && is.environment(cache_env)) {
    rm(list = ls(cache_env), envir = cache_env)
  }

  x
}

#' @export
reconstruct.effect_operator <- function(x,
                                        comp = 1:ncomp(x),
                                        rowind = 1:nrow(scores(x)),
                                        colind = 1:nrow(stats::coef(x)),
                                        scale = c("original", "processed", "whitened"),
                                        ...) {
  scale <- match.arg(scale)

  if (length(comp) == 0 || length(rowind) == 0 || length(colind) == 0) {
    return(matrix(0, nrow = length(rowind), ncol = length(colind)))
  }

  chk::chk_subset(comp, 1:ncomp(x))
  chk::chk_subset(rowind, 1:nrow(scores(x)))
  chk::chk_subset(colind, 1:nrow(stats::coef(x)))

  rec_proc <- scores(x)[rowind, comp, drop = FALSE] %*%
    t(stats::coef(x)[colind, comp, drop = FALSE])

  if (scale == "whitened") {
    rec_proc
  } else if (scale == "processed") {
    x$fit$row_engine$unwhiten(rec_proc)
  } else {
    inverse_transform_contribution(x$preproc, x$fit$row_engine$unwhiten(rec_proc), colind = colind)
  }
}
