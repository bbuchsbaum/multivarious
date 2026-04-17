#' @keywords internal
#' @noRd
perm_pvalue <- function(vals, obs, alternative = c("greater", "less", "two.sided")) {
  alternative <- match.arg(alternative)
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0 || !is.finite(obs)) {
    return(NA_real_)
  }

  if (alternative == "greater") {
    (sum(vals >= obs) + 1) / (length(vals) + 1)
  } else if (alternative == "less") {
    (sum(vals <= obs) + 1) / (length(vals) + 1)
  } else {
    pg <- (sum(vals >= obs) + 1) / (length(vals) + 1)
    pl <- (sum(vals <= obs) + 1) / (length(vals) + 1)
    min(1, 2 * min(pg, pl))
  }
}

#' @keywords internal
#' @noRd
centering_basis <- function(m) {
  if (m <= 1L) {
    return(matrix(0, nrow = m, ncol = 0))
  }
  H <- diag(m) - matrix(1 / m, nrow = m, ncol = m)
  eig <- eigen(H, symmetric = TRUE)
  keep <- which(eig$values > 1e-8)
  if (!length(keep)) {
    matrix(0, nrow = m, ncol = 0)
  } else {
    eig$vectors[, keep, drop = FALSE]
  }
}

#' @keywords internal
#' @noRd
permute_between_block_means <- function(M, blocks) {
  out <- matrix(0, nrow = nrow(M), ncol = ncol(M))
  sizes <- vapply(blocks, length, integer(1))
  idx_by_size <- split(seq_along(blocks), sizes)

  for (grp in idx_by_size) {
    perm_grp <- sample(grp, length(grp), replace = FALSE)
    for (j in seq_along(grp)) {
      target <- grp[[j]]
      source <- perm_grp[[j]]
      blk_target <- blocks[[target]]
      blk_source <- blocks[[source]]
      R_target <- M[blk_target, , drop = FALSE]
      R_source <- M[blk_source, , drop = FALSE]

      mean_source <- matrix(
        colMeans(R_source),
        nrow = length(blk_target),
        ncol = ncol(M),
        byrow = TRUE
      )
      centered_target <- sweep(R_target, 2, colMeans(R_target), FUN = "-")
      out[blk_target, ] <- centered_target + mean_source
    }
  }

  out
}

#' @keywords internal
#' @noRd
signflip_within_block_contrasts <- function(M, blocks) {
  out <- matrix(0, nrow = nrow(M), ncol = ncol(M))

  for (blk in blocks) {
    R_blk <- M[blk, , drop = FALSE]
    m <- nrow(R_blk)
    block_mean <- matrix(colMeans(R_blk), nrow = m, ncol = ncol(M), byrow = TRUE)
    centered <- R_blk - block_mean
    C <- centering_basis(m)

    if (ncol(C) == 0L) {
      out[blk, ] <- R_blk
      next
    }

    coeff <- crossprod(C, centered)
    signs <- sample(c(-1, 1), ncol(C), replace = TRUE)
    flipped <- C %*% (diag(signs, nrow = length(signs), ncol = length(signs)) %*% coeff)
    out[blk, ] <- block_mean + flipped
  }

  out
}

#' @keywords internal
#' @noRd
projected_effect_residual <- function(M, rank_correction = 0L) {
  rank_correction <- as.integer(rank_correction)
  max_rank <- min(nrow(M), ncol(M))
  if (max_rank < 1) {
    return(structure(
      list(
        raw = M,
        projected = M,
        subspace = matrix(0, nrow = nrow(M), ncol = 0),
        rank_correction = rank_correction
      ),
      class = "projected_effect_residual"
    ))
  }

  if (rank_correction < 1L) {
    return(structure(
      list(
        raw = M,
        projected = M,
        subspace = matrix(0, nrow = nrow(M), ncol = 0),
        rank_correction = 0L
      ),
      class = "projected_effect_residual"
    ))
  }

  rank_use <- min(rank_correction, max_rank)
  sv <- svd(M, nu = rank_use, nv = 0)
  U <- sv$u[, seq_len(rank_use), drop = FALSE]
  projected <- M - U %*% crossprod(U, M)

  structure(
    list(
      raw = M,
      projected = projected,
      subspace = U,
      rank_correction = rank_use
    ),
    class = "projected_effect_residual"
  )
}

#' @keywords internal
#' @noRd
effect_residual_stat <- function(obj) {
  M <- obj$projected
  rank_max <- min(qr(M)$rank, nrow(M), ncol(M))
  if (rank_max < 1) {
    return(list(lead_sv2 = 0, rel = 0, total = 0, effective_rank = 0L))
  }
  d <- svd(M, nu = 0, nv = 0)$d[seq_len(rank_max)]
  d2 <- d^2
  total <- sum(d2)
  rel <- if (total <= .Machine$double.eps) 0 else d2[1] / total
  list(lead_sv2 = d2[1], rel = rel, total = total, effective_rank = rank_max)
}

#' @keywords internal
#' @noRd
sequential_component_stat <- function(obj) {
  stat <- effect_residual_stat(obj)
  stat_type <- if (stat$effective_rank <= 2L) "lead_sv2" else "relative"
  observed <- if (identical(stat_type, "relative")) stat$rel else stat$lead_sv2
  list(
    observed = observed,
    statistic_type = stat_type,
    lead_sv2 = stat$lead_sv2,
    rel = stat$rel,
    total = stat$total,
    effective_rank = stat$effective_rank
  )
}

#' @keywords internal
#' @noRd
#' Pillai-style omnibus pivot: tr(H) / tr(E) where H = M'M is the hypothesis
#' operator and E is the full-model residual Gram, computed by applying
#' (I - P_full) to the whitened basis-space response Yw_basis. Applying the
#' residual projector to M directly is degenerate: since M = P_term * Yw_basis
#' and P_term = P_full - P_nuis lies inside P_full's column space,
#' (I - P_full) * M is identically zero.
omnibus_effect_stat <- function(M, P_full, Yw_basis) {
  raw <- sum(M^2)
  Rproj <- diag(nrow(P_full)) - P_full
  E_mat <- Rproj %*% Yw_basis
  residual_energy <- sum(E_mat^2)
  statistic_type <- if (residual_energy <= .Machine$double.eps) "trace" else "trace_ratio"
  observed <- if (identical(statistic_type, "trace_ratio")) raw / residual_energy else raw
  list(
    observed = observed,
    raw = raw,
    residual_energy = residual_energy,
    statistic_type = statistic_type
  )
}

#' Permutation test for an effect operator
#'
#' @param x An `effect_operator`.
#' @param nperm Number of permutations.
#' @param scheme Permutation scheme. Currently only `"reduced_model"` is supported.
#' @param parallel Logical; if `TRUE`, use `future.apply`.
#' @param alpha Sequential significance threshold used to determine selected rank.
#' @param stepwise Logical; if `TRUE`, apply sequential rank testing by deflating previously
#'   selected effect directions before evaluating the next axis.
#' @param alternative Alternative hypothesis for empirical p-values.
#' @param refit_basis Logical; if `TRUE`, refit the feature basis per permutation
#'   from that permutation's reduced-model residual (and refit the observed basis
#'   from the observed reduced-model residual). Uses the full whitened feature
#'   space. Keeps the same basis rank as the static fit. Experimental; intended
#'   for evaluating whether static-basis leakage drives miscalibration.
#' @param ... Reserved for future extensions.
#' @return A permutation-test result object for effect operators.
#' @export
perm_test.effect_operator <- function(x,
                                      nperm = 999,
                                      scheme = c("reduced_model"),
                                      parallel = FALSE,
                                      alpha = 0.05,
                                      stepwise = TRUE,
                                      alternative = c("greater", "less", "two.sided"),
                                      refit_basis = FALSE,
                                      ...) {
  scheme <- match.arg(scheme)
  alternative <- match.arg(alternative)

  fit <- x$fit
  wem <- whitened_effect_matrix(fit, x$term)
  meta <- wem$meta

  if (refit_basis) {
    k <- ncol(fit$basis_matrix)
    Yw_full <- fit$row_engine$whiten(fit$Y_proc)
    fitted0 <- meta$P_nuis %*% Yw_full
    resid0 <- Yw_full - fitted0

    fit_basis <- function(Rw) {
      p_loc <- ncol(Rw)
      k_use <- min(k, nrow(Rw), p_loc)
      if (k_use < 1L) return(matrix(0, nrow = p_loc, ncol = k))
      sv <- svd(Rw, nu = 0, nv = k_use)
      B <- sv$v
      if (ncol(B) < k) {
        B <- cbind(B, matrix(0, nrow = p_loc, ncol = k - ncol(B)))
      }
      B
    }

    B_obs <- fit_basis(resid0)
    Yw_basis_obs <- Yw_full %*% B_obs
    effect_full <- meta$P_term %*% Yw_basis_obs
  } else {
    Yw_basis_obs <- wem$Yw_basis
    effect_full <- wem$M
    fitted0 <- meta$P_nuis %*% Yw_basis_obs
    resid0 <- Yw_basis_obs - fitted0
  }

  obs_omnibus <- omnibus_effect_stat(effect_full, meta$P_full, Yw_basis_obs)
  obs_effective_rank <- min(meta$df_term, ncol(fit$basis_matrix), qr(effect_full)$rank)
  n_axes <- if (refit_basis) obs_effective_rank else ncomp(x)
  obs_steps <- lapply(seq_len(n_axes), function(a) {
    if (stepwise) projected_effect_residual(effect_full, a - 1L)
    else projected_effect_residual(effect_full, 0L)
  })
  obs_step_stats <- lapply(obs_steps, sequential_component_stat)
  obs_values <- vapply(obs_step_stats, `[[`, numeric(1), "observed")
  obs_rel <- vapply(obs_step_stats, `[[`, numeric(1), "rel")
  obs_lead_sv2 <- vapply(obs_step_stats, `[[`, numeric(1), "lead_sv2")
  obs_rank <- vapply(obs_step_stats, `[[`, integer(1), "effective_rank")
  obs_stat_type <- vapply(obs_step_stats, `[[`, character(1), "statistic_type")

  compute_stats <- function(Yb) {
    Yw_basis_b <- if (refit_basis) {
      resid_b <- Yb - meta$P_nuis %*% Yb
      B_b <- fit_basis(resid_b)
      Yb %*% B_b
    } else {
      Yb
    }
    M_full <- meta$P_term %*% Yw_basis_b
    full_stat <- omnibus_effect_stat(M_full, meta$P_full, Yw_basis_b)
    step_objs <- lapply(seq_len(n_axes), function(a) {
      if (stepwise) projected_effect_residual(M_full, a - 1L)
      else projected_effect_residual(M_full, 0L)
    })
    step_stats <- lapply(step_objs, sequential_component_stat)
    observed <- vapply(step_stats, `[[`, numeric(1), "observed")
    rel <- vapply(step_stats, `[[`, numeric(1), "rel")
    lead_sv2 <- vapply(step_stats, `[[`, numeric(1), "lead_sv2")
    rank <- vapply(step_stats, `[[`, integer(1), "effective_rank")
    statistic_type <- vapply(step_stats, `[[`, character(1), "statistic_type")
    list(
      omnibus = full_stat$observed,
      omnibus_raw = full_stat$raw,
      omnibus_residual_energy = full_stat$residual_energy,
      omnibus_type = full_stat$statistic_type,
      observed = observed,
      rel = rel,
      lead_sv2 = lead_sv2,
      effective_rank = rank,
      statistic_type = statistic_type
    )
  }

  worker <- function(i) {
    Yb <- if (!is.null(fit$subject_blocks) && identical(x$term_scope, "between")) {
      fitted0 + permute_between_block_means(resid0, fit$subject_blocks)
    } else if (!is.null(fit$subject_blocks) && identical(x$term_scope, "within")) {
      fitted0 + signflip_within_block_contrasts(resid0, fit$subject_blocks)
    } else if (!is.null(fit$subject_blocks) && identical(x$term_scope, "mixed")) {
      fitted0 + resid0[permute_blocks_same_size(fit$subject_blocks), , drop = FALSE]
    } else {
      fitted0 + resid0[sample.int(nrow(resid0)), , drop = FALSE]
    }
    compute_stats(Yb)
  }

  apply_fun <- if (parallel) {
    if (!requireNamespace("future.apply", quietly = TRUE)) {
      stop("future.apply is required for parallel permutation testing.")
    }
    future.apply::future_lapply
  } else {
    lapply
  }

  args <- list(X = seq_len(nperm), FUN = worker)
  if (parallel) {
    args$future.seed <- TRUE
  }
  perm_list <- do.call(apply_fun, args)

  omnibus_perm <- vapply(perm_list, `[[`, numeric(1), "omnibus")
  omnibus_raw_perm <- vapply(perm_list, `[[`, numeric(1), "omnibus_raw")
  omnibus_resid_perm <- vapply(perm_list, `[[`, numeric(1), "omnibus_residual_energy")
  stat_perm <- matrix(0, nrow = nperm, ncol = max(1, n_axes))
  rel_perm <- matrix(0, nrow = nperm, ncol = max(1, n_axes))
  lead_perm <- matrix(0, nrow = nperm, ncol = max(1, n_axes))
  rank_perm <- matrix(0L, nrow = nperm, ncol = max(1, n_axes))
  if (n_axes > 0) {
    for (i in seq_len(nperm)) {
      stat_i <- perm_list[[i]]$observed
      if (length(stat_i)) {
        stat_perm[i, seq_along(stat_i)] <- stat_i
      }
      rel_i <- perm_list[[i]]$rel
      if (length(rel_i)) {
        rel_perm[i, seq_along(rel_i)] <- rel_i
      }
      d2_i <- perm_list[[i]]$lead_sv2
      if (length(d2_i)) {
        lead_perm[i, seq_along(d2_i)] <- d2_i
      }
      rank_i <- perm_list[[i]]$effective_rank
      if (length(rank_i)) {
        rank_perm[i, seq_along(rank_i)] <- rank_i
      }
    }
  }

  omnibus_p <- perm_pvalue(omnibus_perm, obs_omnibus$observed, alternative = alternative)

  component_results <- if (n_axes > 0) {
    pvals <- vapply(seq_len(n_axes), function(a) {
      perm_pvalue(stat_perm[, a], obs_values[a], alternative = alternative)
    }, numeric(1))
    tibble::tibble(
      comp = seq_len(n_axes),
      statistic = obs_stat_type,
      effective_rank = obs_rank,
      lead_sv2 = obs_lead_sv2,
      rel = obs_rel,
      observed = obs_values,
      pval = pvals
    )
  } else {
    tibble::tibble(
      comp = integer(),
      statistic = character(),
      effective_rank = integer(),
      lead_sv2 = numeric(),
      rel = numeric(),
      observed = numeric(),
      pval = numeric()
    )
  }

  n_significant <- 0L
  if (nrow(component_results) > 0) {
    for (i in seq_len(nrow(component_results))) {
      if (!is.na(component_results$pval[i]) && component_results$pval[i] <= alpha) {
        n_significant <- i
      } else {
        break
      }
    }
  }

  out <- list(
    call = match.call(),
    term = x$term,
    omnibus_statistic = obs_omnibus$observed,
    omnibus_statistic_raw = obs_omnibus$raw,
    omnibus_statistic_residual_energy = obs_omnibus$residual_energy,
    omnibus_statistic_type = obs_omnibus$statistic_type,
    omnibus_p_value = omnibus_p,
    component_results = component_results,
    perm_values = list(
      omnibus = omnibus_perm,
      omnibus_raw = omnibus_raw_perm,
      omnibus_residual_energy = omnibus_resid_perm,
      rank = stat_perm,
      relative = rel_perm,
      lead_sv2 = lead_perm,
      effective_rank = rank_perm
    ),
    observed_projected_residuals = obs_steps,
    alpha = alpha,
    nperm = nperm,
    n_significant = n_significant,
    alternative = alternative,
    scheme = scheme,
    exchangeability = if (!is.null(fit$subject_blocks) && identical(x$term_scope, "between")) {
      "subject-mean permutation within equal block-size strata"
    } else if (!is.null(fit$subject_blocks) && identical(x$term_scope, "within")) {
      "within-subject contrast sign flips"
    } else if (!is.null(fit$subject_blocks) && identical(x$term_scope, "mixed")) {
      "whole-subject trajectory permutation within equal block-size strata"
    } else {
      "row-wise permutation"
    },
    method = paste0(
      if (stepwise) {
        "Reduced-model residual permutation test for effect_operator with sequential deflation"
      } else {
        "Reduced-model residual permutation test for effect_operator"
      },
      if (refit_basis) " (per-permutation basis refit)" else ""
    ),
    refit_basis = refit_basis
  )
  class(out) <- c("perm_test_effect_operator", "perm_test")
  out
}

#' @export
print.perm_test_effect_operator <- function(x, ...) {
  cat("\nEffect operator permutation test\n\n")
  cat("Term: ", x$term, "\n", sep = "")
  cat("Method: ", x$method, "\n", sep = "")
  cat("Exchangeability: ", x$exchangeability, "\n", sep = "")
  cat("Omnibus statistic (", x$omnibus_statistic_type, "): ", format(x$omnibus_statistic, digits = 4), "\n", sep = "")
  cat("Omnibus p-value: ", format(x$omnibus_p_value, digits = 4), "\n", sep = "")
  cat("Selected rank: ", x$n_significant, "\n\n", sep = "")
  if (nrow(x$component_results) > 0) {
    print(as.data.frame(x$component_results))
  }
  invisible(x)
}

#' @export
ncomp.perm_test_effect_operator <- function(x) {
  x$n_significant
}
