#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  if (!requireNamespace("multivarious", quietly = TRUE) ||
      !"mixed_regress" %in% getNamespaceExports("multivarious")) {
    if (!requireNamespace("devtools", quietly = TRUE)) {
      stop("Either install the development version of multivarious or install devtools for load_all().")
    }
    devtools::load_all(".", quiet = TRUE)
  } else {
    library(multivarious)
  }
})

simulate_mixed_family <- function(
  n_subject = 24,
  p = 50,
  signal_term = c("interaction", "within", "between"),
  signal_rank = 0,
  signal_strength = 0,
  random_slope = TRUE,
  missing_mid_prob = 0,
  seed = NULL
) {
  signal_term <- match.arg(signal_term)
  if (!is.null(seed)) {
    set.seed(seed)
  }
  levels_within <- c("low", "mid", "high")
  design <- expand.grid(
    subject = factor(seq_len(n_subject)),
    level = factor(levels_within, levels = levels_within),
    KEEP.OUT.ATTRS = FALSE
  )
  subject_group <- rep(c("A", "B"), length.out = n_subject)
  design$group <- factor(subject_group[as.integer(design$subject)])

  if (missing_mid_prob > 0) {
    drop_subject <- runif(n_subject) < missing_mid_prob
    keep <- !(design$level == "mid" & drop_subject[as.integer(design$subject)])
    design <- design[keep, , drop = FALSE]
  }

  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  level_quad <- c(low = 1, mid = -2, high = 1)[as.character(design$level)]
  group_num <- ifelse(design$group == "B", 1, 0)
  subj_idx <- as.integer(design$subject)
  n <- nrow(design)

  b0 <- rnorm(n_subject, sd = 0.7)
  b1 <- if (random_slope) rnorm(n_subject, sd = 0.3) else rep(0, n_subject)

  make_signal_scores <- function(signal_term, signal_rank, signal_strength) {
    if (signal_rank < 1) {
      return(matrix(0, nrow = n, ncol = 0))
    }

    if (signal_term == "interaction") {
      base_scores <- cbind(
        group_num * level_num,
        group_num * level_quad
      )
    } else if (signal_term == "within") {
      base_scores <- cbind(level_num, level_quad)
    } else if (signal_term == "between") {
      base_scores <- cbind(group_num, as.numeric(as.integer(design$subject) %% 2 == 0))
    } else {
      stop("Unsupported signal term: ", signal_term)
    }

    base_scores <- scale(base_scores, center = TRUE, scale = FALSE)
    signal_strength * base_scores[, seq_len(min(signal_rank, ncol(base_scores))), drop = FALSE]
  }

  if (signal_rank < 1) {
    effect_mat <- matrix(0, nrow = n, ncol = p)
  } else {
    V <- matrix(rnorm(p * signal_rank), nrow = p, ncol = signal_rank)
    V <- qr.Q(qr(V))[, seq_len(signal_rank), drop = FALSE]
    scores <- make_signal_scores(signal_term, signal_rank, signal_strength)
    effect_mat <- scores %*% t(V)
  }

  Y <- effect_mat +
    cbind(
      b0[subj_idx] + b1[subj_idx] * level_num,
      matrix(0, nrow = n, ncol = p - 1)
    ) +
    matrix(rnorm(n * p, sd = 0.3), nrow = n, ncol = p)

  list(Y = Y, design = design)
}

run_null_calibration <- function(
  nsim = 50,
  nperm = 99,
  alpha = 0.05,
  p = 100,
  test_term = "group:level",
  random_slope = TRUE,
  missing_mid_prob = 0
) {
  out <- vector("list", nsim)
  for (i in seq_len(nsim)) {
    dat <- simulate_mixed_family(
      p = p,
      signal_rank = 0,
      signal_strength = 0,
      random_slope = random_slope,
      missing_mid_prob = missing_mid_prob,
      seed = 1000 + i
    )
    fit <- mixed_regress(
      dat$Y,
      design = dat$design,
      fixed = ~ group * level,
      random = if (random_slope) ~ 1 + level | subject else ~ 1 | subject,
      basis = shared_pca(min(10, p)),
      preproc = center()
    )
    E <- effect(fit, test_term)
    pt <- perm_test(E, nperm = nperm, alpha = alpha)
    out[[i]] <- data.frame(
      sim = i,
      term = test_term,
      omnibus_p = pt$omnibus_p_value,
      selected_rank = ncomp(pt)
    )
  }
  do.call(rbind, out)
}

run_rank_recovery <- function(
  nsim = 50,
  nperm = 99,
  signal_term = "interaction",
  test_term = "group:level",
  signal_rank = 1,
  signal_strength = 1.0,
  p = 100,
  random_slope = TRUE
) {
  out <- vector("list", nsim)
  for (i in seq_len(nsim)) {
    dat <- simulate_mixed_family(
      p = p,
      signal_term = signal_term,
      signal_rank = signal_rank,
      signal_strength = signal_strength,
      random_slope = random_slope,
      seed = 2000 + i
    )
    fit <- mixed_regress(
      dat$Y,
      design = dat$design,
      fixed = ~ group * level,
      random = if (random_slope) ~ 1 + level | subject else ~ 1 | subject,
      basis = shared_pca(min(10, p)),
      preproc = center()
    )
    E <- effect(fit, test_term)
    pt <- perm_test(E, nperm = nperm, alpha = 0.05)
    out[[i]] <- data.frame(
      sim = i,
      term = test_term,
      target_rank = signal_rank,
      omnibus_p = pt$omnibus_p_value,
      selected_rank = ncomp(pt),
      leading_sv2 = if (nrow(pt$component_results)) pt$component_results$lead_sv2[1] else 0
    )
  }
  do.call(rbind, out)
}

summarize_null_calibration <- function(df, alpha = 0.05) {
  data.frame(
    n = nrow(df),
    type1_or_power = mean(df$omnibus_p <= alpha, na.rm = TRUE),
    mean_selected_rank = mean(df$selected_rank, na.rm = TRUE),
    rank_zero_rate = mean(df$selected_rank == 0, na.rm = TRUE)
  )
}

summarize_rank_recovery <- function(df, alpha = 0.05) {
  data.frame(
    n = nrow(df),
    power = mean(df$omnibus_p <= alpha, na.rm = TRUE),
    mean_selected_rank = mean(df$selected_rank, na.rm = TRUE),
    rank_match_rate = mean(df$selected_rank == df$target_rank, na.rm = TRUE),
    rank_at_least_target = mean(df$selected_rank >= df$target_rank, na.rm = TRUE)
  )
}

run_benchmark_grid <- function(
  n_subject_vec = c(20, 50),
  p_vec = c(100, 1000),
  k_vec = c(5, 10),
  nperm_vec = c(49, 99),
  signal_term = "interaction",
  test_term = "group:level"
) {
  grid <- expand.grid(
    n_subject = n_subject_vec,
    p = p_vec,
    k = k_vec,
    nperm = nperm_vec,
    stringsAsFactors = FALSE
  )

  out <- vector("list", nrow(grid))
  for (i in seq_len(nrow(grid))) {
    g <- grid[i, , drop = FALSE]
    dat <- simulate_mixed_family(
      n_subject = g$n_subject,
      p = g$p,
      signal_term = signal_term,
      signal_rank = 1,
      signal_strength = 1.0,
      random_slope = TRUE,
      seed = 3000 + i
    )

    t0 <- proc.time()[["elapsed"]]
    fit <- mixed_regress(
      dat$Y,
      design = dat$design,
      fixed = ~ group * level,
      random = ~ 1 + level | subject,
      basis = shared_pca(min(g$k, g$p)),
      preproc = center()
    )
    E <- effect(fit, test_term)
    pt <- perm_test(E, nperm = g$nperm, alpha = 0.05)
    elapsed <- proc.time()[["elapsed"]] - t0

    out[[i]] <- data.frame(
      term = test_term,
      n_subject = g$n_subject,
      p = g$p,
      basis_rank = g$k,
      nperm = g$nperm,
      elapsed_sec = elapsed,
      selected_rank = ncomp(pt),
      omnibus_p = pt$omnibus_p_value
    )
  }

  do.call(rbind, out)
}

parse_cli_kv <- function(args) {
  out <- list()
  if (length(args) <= 2) {
    return(out)
  }
  for (arg in args[-c(1, 2)]) {
    parts <- strsplit(arg, "=", fixed = TRUE)[[1]]
    if (length(parts) == 2) {
      out[[parts[[1]]]] <- parts[[2]]
    }
  }
  out
}

as_num_or <- function(x, default) {
  if (is.null(x)) default else as.numeric(x)
}

as_int_or <- function(x, default) {
  if (is.null(x)) default else as.integer(x)
}

args <- commandArgs(trailingOnly = TRUE)
mode <- if (length(args)) args[[1]] else "null"
outfile <- if (length(args) >= 2) args[[2]] else NULL
kv <- parse_cli_kv(args)

result <- switch(
  mode,
  null = {
    df <- run_null_calibration(
      nsim = as_int_or(kv$nsim, 50),
      nperm = as_int_or(kv$nperm, 99),
      p = as_int_or(kv$p, 100),
      test_term = if (is.null(kv$term)) "group:level" else kv$term,
      random_slope = if (is.null(kv$random_slope)) TRUE else as.logical(kv$random_slope)
    )
    list(raw = df, summary = summarize_null_calibration(df))
  },
  missing = {
    df <- run_null_calibration(
      nsim = as_int_or(kv$nsim, 50),
      nperm = as_int_or(kv$nperm, 99),
      p = as_int_or(kv$p, 100),
      test_term = if (is.null(kv$term)) "group:level" else kv$term,
      random_slope = if (is.null(kv$random_slope)) TRUE else as.logical(kv$random_slope),
      missing_mid_prob = as_num_or(kv$missing_mid_prob, 0.25)
    )
    list(raw = df, summary = summarize_null_calibration(df))
  },
  within_null = {
    df <- run_null_calibration(
      nsim = as_int_or(kv$nsim, 50),
      nperm = as_int_or(kv$nperm, 99),
      p = as_int_or(kv$p, 100),
      test_term = "level",
      random_slope = if (is.null(kv$random_slope)) TRUE else as.logical(kv$random_slope),
      missing_mid_prob = as_num_or(kv$missing_mid_prob, 0)
    )
    list(raw = df, summary = summarize_null_calibration(df))
  },
  between_null = {
    df <- run_null_calibration(
      nsim = as_int_or(kv$nsim, 50),
      nperm = as_int_or(kv$nperm, 99),
      p = as_int_or(kv$p, 100),
      test_term = "group",
      random_slope = if (is.null(kv$random_slope)) TRUE else as.logical(kv$random_slope),
      missing_mid_prob = as_num_or(kv$missing_mid_prob, 0)
    )
    list(raw = df, summary = summarize_null_calibration(df))
  },
  rank1 = {
    df <- run_rank_recovery(
      nsim = as_int_or(kv$nsim, 50),
      nperm = as_int_or(kv$nperm, 99),
      signal_term = "interaction",
      test_term = "group:level",
      signal_rank = 1,
      signal_strength = as_num_or(kv$signal_strength, 1.0),
      p = as_int_or(kv$p, 100),
      random_slope = if (is.null(kv$random_slope)) TRUE else as.logical(kv$random_slope)
    )
    list(raw = df, summary = summarize_rank_recovery(df))
  },
  rank2 = {
    df <- run_rank_recovery(
      nsim = as_int_or(kv$nsim, 50),
      nperm = as_int_or(kv$nperm, 99),
      signal_term = "interaction",
      test_term = "group:level",
      signal_rank = 2,
      signal_strength = as_num_or(kv$signal_strength, 1.1),
      p = as_int_or(kv$p, 100),
      random_slope = if (is.null(kv$random_slope)) TRUE else as.logical(kv$random_slope)
    )
    list(raw = df, summary = summarize_rank_recovery(df))
  },
  within_rank1 = {
    df <- run_rank_recovery(
      nsim = as_int_or(kv$nsim, 50),
      nperm = as_int_or(kv$nperm, 99),
      signal_term = "within",
      test_term = "level",
      signal_rank = 1,
      signal_strength = as_num_or(kv$signal_strength, 1.0),
      p = as_int_or(kv$p, 100),
      random_slope = if (is.null(kv$random_slope)) TRUE else as.logical(kv$random_slope)
    )
    list(raw = df, summary = summarize_rank_recovery(df))
  },
  within_rank2 = {
    df <- run_rank_recovery(
      nsim = as_int_or(kv$nsim, 50),
      nperm = as_int_or(kv$nperm, 99),
      signal_term = "within",
      test_term = "level",
      signal_rank = 2,
      signal_strength = as_num_or(kv$signal_strength, 1.1),
      p = as_int_or(kv$p, 100),
      random_slope = if (is.null(kv$random_slope)) TRUE else as.logical(kv$random_slope)
    )
    list(raw = df, summary = summarize_rank_recovery(df))
  },
  between_rank1 = {
    df <- run_rank_recovery(
      nsim = as_int_or(kv$nsim, 50),
      nperm = as_int_or(kv$nperm, 99),
      signal_term = "between",
      test_term = "group",
      signal_rank = 1,
      signal_strength = as_num_or(kv$signal_strength, 1.0),
      p = as_int_or(kv$p, 100),
      random_slope = if (is.null(kv$random_slope)) TRUE else as.logical(kv$random_slope)
    )
    list(raw = df, summary = summarize_rank_recovery(df))
  },
  benchmark = {
    df <- run_benchmark_grid(
      n_subject_vec = as.integer(strsplit(if (is.null(kv$n_subject)) "20,50" else kv$n_subject, ",")[[1]]),
      p_vec = as.integer(strsplit(if (is.null(kv$p)) "100,1000" else kv$p, ",")[[1]]),
      k_vec = as.integer(strsplit(if (is.null(kv$k)) "5,10" else kv$k, ",")[[1]]),
      nperm_vec = as.integer(strsplit(if (is.null(kv$nperm)) "49,99" else kv$nperm, ",")[[1]]),
      signal_term = if (is.null(kv$signal_term)) "interaction" else kv$signal_term,
      test_term = if (is.null(kv$term)) "group:level" else kv$term
    )
    list(raw = df, summary = df)
  },
  stop("Unknown mode: ", mode)
)

print(result$summary)

if (!is.null(outfile)) {
  utils::write.csv(result$raw, outfile, row.names = FALSE)
  summary_file <- sub("([.]csv)?$", "_summary.csv", outfile)
  utils::write.csv(result$summary, summary_file, row.names = FALSE)
}
