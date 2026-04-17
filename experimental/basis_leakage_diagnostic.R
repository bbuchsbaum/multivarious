#!/usr/bin/env Rscript
## Basis-leakage falsification test.
##
## Compares shared-PCA basis built from fit_on = "nuisance_residual" vs
## fit_on = "whitened_residual" across (null, rank-1 signal, rank-2 signal)
## arms for both "level" and "group:level" terms. Records per-replicate rows
## and collapse diagnostics for the whitened_residual path.
##
## Decision rule (from Buchsbaum, session 2026-04-17):
##   - if whitened_residual improves Type I and preserves power/subspace
##     recovery, it stays alive
##   - if it improves Type I by crushing power or rank recovery, it's dead
##   - if behavior is unstable and tied to rank/optimizer pathologies, basis
##     leakage is still unresolved, just badly probed

suppressPackageStartupMessages({
  devtools::load_all(".", quiet = TRUE)
})

## Simulator: returns Y, design, and the true generating feature subspace V
## (size p x signal_rank), used for principal-angle recovery metrics.
simulate_family <- function(
  n_subject = 24,
  p = 50,
  signal_term = c("within", "interaction"),
  signal_rank = 0,
  signal_strength = 0,
  random_slope = TRUE,
  seed = NULL
) {
  signal_term <- match.arg(signal_term)
  if (!is.null(seed)) set.seed(seed)

  levels_within <- c("low", "mid", "high")
  design <- expand.grid(
    subject = factor(seq_len(n_subject)),
    level = factor(levels_within, levels = levels_within),
    KEEP.OUT.ATTRS = FALSE
  )
  subject_group <- rep(c("A", "B"), length.out = n_subject)
  design$group <- factor(subject_group[as.integer(design$subject)])

  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  level_quad <- c(low = 1, mid = -2, high = 1)[as.character(design$level)]
  group_num <- ifelse(design$group == "B", 1, 0)
  subj_idx <- as.integer(design$subject)
  n <- nrow(design)

  b0 <- rnorm(n_subject, sd = 0.7)
  b1 <- if (random_slope) rnorm(n_subject, sd = 0.3) else rep(0, n_subject)

  V_true <- NULL
  effect_mat <- matrix(0, nrow = n, ncol = p)
  if (signal_rank >= 1) {
    base_scores <- if (signal_term == "within") {
      cbind(level_num, level_quad)
    } else {
      cbind(group_num * level_num, group_num * level_quad)
    }
    base_scores <- scale(base_scores, center = TRUE, scale = FALSE)
    scores <- signal_strength *
      base_scores[, seq_len(min(signal_rank, ncol(base_scores))), drop = FALSE]
    V_raw <- matrix(rnorm(p * signal_rank), nrow = p, ncol = signal_rank)
    V_true <- qr.Q(qr(V_raw))[, seq_len(signal_rank), drop = FALSE]
    effect_mat <- scores %*% t(V_true)
  }

  Y <- effect_mat +
    cbind(b0[subj_idx] + b1[subj_idx] * level_num,
          matrix(0, nrow = n, ncol = p - 1)) +
    matrix(rnorm(n * p, sd = 0.3), nrow = n, ncol = p)

  list(Y = Y, design = design, V_true = V_true)
}

## Principal angles between two subspaces given as matrices of any column count.
principal_angles <- function(A, B) {
  if (is.null(A) || is.null(B) || ncol(A) == 0 || ncol(B) == 0) {
    return(numeric(0))
  }
  Qa <- qr.Q(qr(A))[, seq_len(ncol(A)), drop = FALSE]
  Qb <- qr.Q(qr(B))[, seq_len(ncol(B)), drop = FALSE]
  sv <- svd(crossprod(Qa, Qb), nu = 0, nv = 0)$d
  acos(pmin(pmax(sv, -1), 1))
}

run_condition <- function(term, fit_on, signal_rank, signal_strength,
                          nsim, nperm, p, seed_offset,
                          refit_basis = FALSE) {
  out <- vector("list", nsim)
  signal_term <- if (term == "level") "within" else "interaction"
  arm_label <- if (refit_basis) paste0(fit_on, "+refit") else fit_on

  for (i in seq_len(nsim)) {
    dat <- simulate_family(
      p = p,
      signal_term = signal_term,
      signal_rank = signal_rank,
      signal_strength = signal_strength,
      random_slope = TRUE,
      seed = seed_offset + i
    )

    fit_time <- system.time({
      fit <- mixed_regress(
        dat$Y,
        design = dat$design,
        fixed = ~ group * level,
        random = ~ 1 + level | subject,
        basis = shared_pca(min(10, p), fit_on = fit_on),
        preproc = center()
      )
    })[["elapsed"]]

    eff <- effect(fit, term)

    ## Collapse diagnostics: recompute M_basis exactly as perm_test does.
    meta <- fit$effects_meta[[term]]
    B <- fit$basis_matrix
    Yw_basis <- fit$row_engine$whiten(fit$Y_proc %*% B)
    M_basis <- meta$P_term %*% Yw_basis
    M_sv <- svd(M_basis, nu = 0, nv = 0)$d
    lead_sv2_vec <- M_sv^2

    basis_rank <- qr(B)$rank
    effect_rank <- min(meta$df_term, basis_rank, qr(M_basis)$rank)

    pt_time <- system.time({
      pt <- perm_test(eff, nperm = nperm, alpha = 0.05, refit_basis = refit_basis)
    })[["elapsed"]]

    ## Principal angles (only defined with true V and non-degenerate estimate).
    angle1 <- NA_real_
    angle2 <- NA_real_
    if (!is.null(dat$V_true) && ncol(components(eff)) >= 1L) {
      k_true <- ncol(dat$V_true)
      k_use <- min(k_true, ncol(components(eff)))
      angles <- principal_angles(
        components(eff)[, seq_len(k_use), drop = FALSE],
        dat$V_true[, seq_len(k_use), drop = FALSE]
      )
      if (length(angles) >= 1) angle1 <- angles[1]
      if (length(angles) >= 2) angle2 <- angles[2]
    }

    optim_info <- fit$row_metric$optim
    opt_conv <- if (!is.null(optim_info)) optim_info$convergence else NA_integer_
    opt_iter <- if (!is.null(optim_info) && !is.null(optim_info$counts)) {
      optim_info$counts[["function"]]
    } else NA_integer_

    component_stat_type_1 <- if (nrow(pt$component_results) > 0) {
      pt$component_results$statistic[1]
    } else NA_character_

    perm_raw <- pt$perm_values$omnibus_raw
    perm_resid <- pt$perm_values$omnibus_residual_energy
    perm_pivot <- pt$perm_values$omnibus

    out[[i]] <- data.frame(
      sim = i,
      term = term,
      fit_on = fit_on,
      refit_basis = refit_basis,
      arm = arm_label,
      signal_rank = signal_rank,
      signal_strength = signal_strength,
      omnibus_p = pt$omnibus_p_value,
      selected_rank = ncomp(pt),
      omnibus_statistic = pt$omnibus_statistic,
      omnibus_statistic_type = pt$omnibus_statistic_type,
      omnibus_statistic_raw = pt$omnibus_statistic_raw,
      omnibus_statistic_residual_energy = pt$omnibus_statistic_residual_energy,
      perm_raw_mean = mean(perm_raw, na.rm = TRUE),
      perm_raw_q95 = stats::quantile(perm_raw, 0.95, na.rm = TRUE, names = FALSE),
      perm_resid_mean = mean(perm_resid, na.rm = TRUE),
      perm_resid_q95 = stats::quantile(perm_resid, 0.95, na.rm = TRUE, names = FALSE),
      perm_pivot_mean = mean(perm_pivot, na.rm = TRUE),
      perm_pivot_q95 = stats::quantile(perm_pivot, 0.95, na.rm = TRUE, names = FALSE),
      component_stat_type_1 = component_stat_type_1,
      lead_sv2_1 = if (length(lead_sv2_vec) >= 1) lead_sv2_vec[1] else 0,
      lead_sv2_2 = if (length(lead_sv2_vec) >= 2) lead_sv2_vec[2] else 0,
      m_basis_trace = sum(lead_sv2_vec),
      basis_rank = basis_rank,
      effect_rank = effect_rank,
      angle1_rad = angle1,
      angle2_rad = angle2,
      opt_convergence = opt_conv,
      opt_iter = opt_iter,
      fit_elapsed = fit_time,
      perm_elapsed = pt_time,
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, out)
}

args <- commandArgs(trailingOnly = TRUE)
nsim_null <- if (length(args) >= 1) as.integer(args[[1]]) else 80L
nsim_signal <- if (length(args) >= 2) as.integer(args[[2]]) else 60L
nperm <- if (length(args) >= 3) as.integer(args[[3]]) else 49L
p <- if (length(args) >= 4) as.integer(args[[4]]) else 50L

cat(sprintf("Contract: nsim_null=%d, nsim_signal=%d, nperm=%d, p=%d\n\n",
            nsim_null, nsim_signal, nperm, p))

## Three arms per (term, rank):
##   static   -- fit_on = "nuisance_residual", refit_basis = FALSE  (current default)
##   refit    -- fit_on = "nuisance_residual", refit_basis = TRUE   (proposed fix)
##   whitened -- fit_on = "whitened_residual", refit_basis = FALSE  (already falsified; kept as reference)
conditions <- list(
  list(term = "level",       fit_on = "nuisance_residual", refit = FALSE, rank = 0, strength = 0,   nsim = nsim_null,   seed = 10000),
  list(term = "level",       fit_on = "nuisance_residual", refit = TRUE,  rank = 0, strength = 0,   nsim = nsim_null,   seed = 10000),
  list(term = "level",       fit_on = "whitened_residual", refit = FALSE, rank = 0, strength = 0,   nsim = nsim_null,   seed = 10000),
  list(term = "group:level", fit_on = "nuisance_residual", refit = FALSE, rank = 0, strength = 0,   nsim = nsim_null,   seed = 20000),
  list(term = "group:level", fit_on = "nuisance_residual", refit = TRUE,  rank = 0, strength = 0,   nsim = nsim_null,   seed = 20000),
  list(term = "group:level", fit_on = "whitened_residual", refit = FALSE, rank = 0, strength = 0,   nsim = nsim_null,   seed = 20000),
  list(term = "level",       fit_on = "nuisance_residual", refit = FALSE, rank = 1, strength = 0.7, nsim = nsim_signal, seed = 30000),
  list(term = "level",       fit_on = "nuisance_residual", refit = TRUE,  rank = 1, strength = 0.7, nsim = nsim_signal, seed = 30000),
  list(term = "level",       fit_on = "nuisance_residual", refit = FALSE, rank = 2, strength = 0.7, nsim = nsim_signal, seed = 40000),
  list(term = "level",       fit_on = "nuisance_residual", refit = TRUE,  rank = 2, strength = 0.7, nsim = nsim_signal, seed = 40000),
  list(term = "group:level", fit_on = "nuisance_residual", refit = FALSE, rank = 1, strength = 0.7, nsim = nsim_signal, seed = 50000),
  list(term = "group:level", fit_on = "nuisance_residual", refit = TRUE,  rank = 1, strength = 0.7, nsim = nsim_signal, seed = 50000),
  list(term = "group:level", fit_on = "nuisance_residual", refit = FALSE, rank = 2, strength = 0.7, nsim = nsim_signal, seed = 60000),
  list(term = "group:level", fit_on = "nuisance_residual", refit = TRUE,  rank = 2, strength = 0.7, nsim = nsim_signal, seed = 60000)
)

## Directly measure how much the per-permutation refit basis differs from the
## observed refit basis. Projector distance: ||B_obs Bobs^T - Bp Bp^T||_F.
## Done once per refit condition on the first simulated dataset with 30 draws
## from the scope-appropriate permutation scheme.
measure_basis_stability <- function(term, fit_on, signal_rank, signal_strength,
                                    p, seed_offset, n_draws = 30L) {
  signal_term <- if (term == "level") "within" else "interaction"
  dat <- simulate_family(
    p = p, signal_term = signal_term,
    signal_rank = signal_rank, signal_strength = signal_strength,
    random_slope = TRUE, seed = seed_offset + 1L
  )
  fit <- mixed_regress(
    dat$Y, design = dat$design,
    fixed = ~ group * level, random = ~ 1 + level | subject,
    basis = shared_pca(min(10, p), fit_on = fit_on), preproc = center()
  )
  meta <- fit$effects_meta[[term]]
  Yw_full <- fit$row_engine$whiten(fit$Y_proc)
  fitted0 <- meta$P_nuis %*% Yw_full
  resid0 <- Yw_full - fitted0

  k <- ncol(fit$basis_matrix)
  fit_basis <- function(Rw) {
    k_use <- min(k, nrow(Rw), ncol(Rw))
    sv <- svd(Rw, nu = 0, nv = k_use)
    B <- sv$v
    if (ncol(B) < k) B <- cbind(B, matrix(0, nrow = ncol(Rw), ncol = k - ncol(B)))
    B
  }
  B_obs <- fit_basis(resid0)
  P_obs <- tcrossprod(B_obs)

  scope <- if (!is.null(fit$subject_blocks)) {
    scope_tag <- fit$effects_meta[[term]]$term_scope
    scope_tag
  } else {
    "ungrouped"
  }

  distances <- numeric(n_draws)
  for (j in seq_len(n_draws)) {
    Rp <- if (!is.null(fit$subject_blocks) && identical(scope, "between")) {
      multivarious:::permute_between_block_means(resid0, fit$subject_blocks)
    } else if (!is.null(fit$subject_blocks) && identical(scope, "within")) {
      multivarious:::signflip_within_block_contrasts(resid0, fit$subject_blocks)
    } else if (!is.null(fit$subject_blocks) && identical(scope, "mixed")) {
      resid0[multivarious:::permute_blocks_same_size(fit$subject_blocks), , drop = FALSE]
    } else {
      resid0[sample.int(nrow(resid0)), , drop = FALSE]
    }
    Yb_full <- fitted0 + Rp
    resid_b <- Yb_full - meta$P_nuis %*% Yb_full
    B_p <- fit_basis(resid_b)
    P_p <- tcrossprod(B_p)
    distances[j] <- sqrt(sum((P_obs - P_p)^2))
  }

  list(
    scope = scope,
    mean_proj_dist = mean(distances),
    max_proj_dist = max(distances),
    min_proj_dist = min(distances),
    sd_proj_dist = stats::sd(distances),
    n_draws = n_draws
  )
}

all_rep <- list()
summary_rows <- list()
basis_stab_rows <- list()

for (cond in conditions) {
  t0 <- proc.time()[["elapsed"]]
  df <- run_condition(cond$term, cond$fit_on, cond$rank, cond$strength,
                      cond$nsim, nperm, p, cond$seed,
                      refit_basis = isTRUE(cond$refit))
  dt <- proc.time()[["elapsed"]] - t0

  all_rep[[length(all_rep) + 1]] <- df

  type1_or_power <- mean(df$omnibus_p <= 0.05, na.rm = TRUE)
  se <- sqrt(type1_or_power * (1 - type1_or_power) / nrow(df))
  mean_selected <- mean(df$selected_rank, na.rm = TRUE)
  rank_match <- if (cond$rank > 0) {
    mean(df$selected_rank == cond$rank, na.rm = TRUE)
  } else NA_real_
  mean_angle1 <- mean(df$angle1_rad, na.rm = TRUE)
  mean_angle2 <- mean(df$angle2_rad, na.rm = TRUE)
  mean_basis_rank <- mean(df$basis_rank)
  mean_effect_rank <- mean(df$effect_rank)
  frac_collapsed <- mean(df$lead_sv2_1 < 1e-8)
  conv_fail <- sum(df$opt_convergence != 0, na.rm = TRUE)

  summary_rows[[length(summary_rows) + 1]] <- data.frame(
    term = cond$term,
    fit_on = cond$fit_on,
    refit_basis = isTRUE(cond$refit),
    arm = if (isTRUE(cond$refit)) paste0(cond$fit_on, "+refit") else cond$fit_on,
    signal_rank = cond$rank,
    n = nrow(df),
    type1_or_power = type1_or_power,
    se = se,
    mean_selected_rank = mean_selected,
    rank_match_rate = rank_match,
    mean_angle1_deg = mean_angle1 * 180 / pi,
    mean_angle2_deg = mean_angle2 * 180 / pi,
    mean_basis_rank = mean_basis_rank,
    mean_effect_rank = mean_effect_rank,
    frac_collapsed = frac_collapsed,
    conv_fail = conv_fail,
    elapsed_sec = dt,
    stringsAsFactors = FALSE
  )

  arm_print <- if (isTRUE(cond$refit)) paste0(cond$fit_on, "+refit") else cond$fit_on
  cat(sprintf("  %-12s %-26s rank=%d n=%3d  %s=%.3f (SE %.3f)  sel=%.2f  ang1=%5.1f°  collapse=%.2f  %.1fs\n",
              cond$term, arm_print, cond$rank, nrow(df),
              if (cond$rank == 0) "type1" else "power",
              type1_or_power, se, mean_selected,
              mean_angle1 * 180 / pi, frac_collapsed, dt))

  if (isTRUE(cond$refit)) {
    stab <- measure_basis_stability(cond$term, cond$fit_on, cond$rank, cond$strength,
                                    p = p, seed_offset = cond$seed)
    basis_stab_rows[[length(basis_stab_rows) + 1]] <- data.frame(
      term = cond$term,
      scope = stab$scope,
      signal_rank = cond$rank,
      n_draws = stab$n_draws,
      proj_dist_mean = stab$mean_proj_dist,
      proj_dist_max = stab$max_proj_dist,
      proj_dist_min = stab$min_proj_dist,
      proj_dist_sd = stab$sd_proj_dist,
      stringsAsFactors = FALSE
    )
    cat(sprintf("       [basis stability] scope=%-7s proj_dist: mean=%.4f  max=%.4f  min=%.4f\n",
                stab$scope, stab$mean_proj_dist, stab$max_proj_dist, stab$min_proj_dist))
  }
}

rep_out <- do.call(rbind, all_rep)
summary_out <- do.call(rbind, summary_rows)

rep_file <- "experimental/results/basis_leakage_replicates_v4.csv"
sum_file <- "experimental/results/basis_leakage_summary_v4.csv"
stab_file <- "experimental/results/basis_stability_v4.csv"
utils::write.csv(rep_out, rep_file, row.names = FALSE)
utils::write.csv(summary_out, sum_file, row.names = FALSE)
if (length(basis_stab_rows)) {
  stab_out <- do.call(rbind, basis_stab_rows)
  utils::write.csv(stab_out, stab_file, row.names = FALSE)
  cat("\nWrote: ", stab_file, "\n", sep = "")
}

cat("\nWrote: ", rep_file, "\n", sep = "")
cat("Wrote: ", sum_file, "\n", sep = "")
cat("\n=== Summary ===\n")
print(summary_out, row.names = FALSE)
if (exists("stab_out")) {
  cat("\n=== Basis stability (refit conditions) ===\n")
  print(stab_out, row.names = FALSE)
}
