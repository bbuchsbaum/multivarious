manual_fixed_effect <- function(Y, design, fixed, term) {
  Terms <- stats::terms(fixed, data = design)
  X <- stats::model.matrix(Terms, data = design)
  assign_vec <- attr(X, "assign")
  term_labels <- attr(Terms, "term.labels")
  term_index <- match(term, term_labels)
  effect_cols <- which(assign_vec == term_index)
  nuisance_cols <- setdiff(seq_len(ncol(X)), effect_cols)

  X_nuis <- X[, nuisance_cols, drop = FALSE]
  P_nuis <- multivarious:::orthogonal_projector(X_nuis)
  P_full <- multivarious:::orthogonal_projector(cbind(X_nuis, X[, effect_cols, drop = FALSE]))
  P_term <- P_full - P_nuis

  list(
    M = P_term %*% Y,
    df = qr(cbind(X_nuis, X[, effect_cols, drop = FALSE]))$rank - qr(X_nuis)$rank,
    P_term = P_term
  )
}

expect_projection_matrix <- function(P, tolerance = 1e-7) {
  expect_equal(P, t(P), tolerance = tolerance)
  expect_equal(P %*% P, P, tolerance = tolerance)
}

make_grouped_stress_data <- function() {
  set.seed(4101)
  design <- expand.grid(
    subject = factor(seq_len(9)),
    visit = factor(paste0("v", seq_len(4)), levels = paste0("v", seq_len(4))),
    KEEP.OUT.ATTRS = FALSE
  )
  design$time <- as.integer(design$visit) - 1

  keep <- !(design$subject %in% c("2", "5") & design$visit == "v2") &
    !(design$subject == "8" & design$visit == "v4")
  design <- design[keep, , drop = FALSE]

  subject_group <- c("A", "A", "B", "B", "A", "B", "A", "B", "A")
  design$group <- factor(subject_group[as.integer(design$subject)])

  subj_idx <- as.integer(design$subject)
  group_num <- ifelse(design$group == "B", 1, -1)
  time_num <- design$time
  n_subject <- nlevels(design$subject)
  n <- nrow(design)

  b0 <- rnorm(n_subject, sd = 0.7)
  b1 <- 0.35 * b0 + rnorm(n_subject, sd = 0.04)
  raw <- cbind(
    b0[subj_idx] + 0.8 * time_num + 0.4 * group_num + rnorm(n, sd = 0.08),
    b1[subj_idx] * time_num - 0.5 * time_num + rnorm(n, sd = 0.05),
    group_num * time_num + rnorm(n, sd = 0.08),
    matrix(rnorm(n * 4, sd = 0.1), nrow = n, ncol = 4)
  )
  Y <- sweep(raw, 2, c(1e-3, 1e-1, 1, 10, 100, 1e3, 1e2), `*`)

  list(Y = Y, design = design)
}

test_that("fixed-effect path matches projector algebra under randomized scale stress", {
  fixed <- ~ group * level
  terms_to_check <- c("group", "level", "group:level")

  for (seed in 3001:3005) {
    set.seed(seed)
    design <- expand.grid(
      group = factor(c("A", "B")),
      level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
      rep = seq_len(4),
      KEEP.OUT.ATTRS = FALSE
    )
    n <- nrow(design)
    level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
    group_num <- ifelse(design$group == "B", 1, -1)

    Y_signal <- cbind(
      1.6 * group_num + rnorm(n, sd = 0.04),
      -1.2 * level_num + rnorm(n, sd = 0.04),
      0.9 * group_num * level_num + rnorm(n, sd = 0.04),
      matrix(rnorm(n * 3, sd = 0.08), nrow = n, ncol = 3)
    )
    R <- qr.Q(qr(matrix(rnorm(ncol(Y_signal)^2), ncol(Y_signal))))
    Y <- Y_signal %*% R

    scale_factor <- 10^runif(1, min = -4, max = 4)
    offset <- runif(ncol(Y), min = -1e3, max = 1e3)
    Y_scaled <- scale_factor * Y + matrix(offset, nrow = n, ncol = ncol(Y), byrow = TRUE)

    fit <- mixed_regress(
      Y,
      design = design,
      fixed = fixed,
      random = NULL,
      basis = identity_basis(),
      preproc = pass()
    )
    fit_scaled <- mixed_regress(
      Y_scaled,
      design = design,
      fixed = fixed,
      random = NULL,
      basis = identity_basis(),
      preproc = pass()
    )

    for (term in terms_to_check) {
      manual <- manual_fixed_effect(Y, design, fixed, term)
      eff <- effect(fit, term)
      eff_scaled <- effect(fit_scaled, term)
      rank_use <- min(manual$df, qr(manual$M)$rank, ncol(Y))
      d_manual <- if (rank_use > 0L) {
        svd(manual$M, nu = 0, nv = 0)$d[seq_len(rank_use)]
      } else {
        numeric(0)
      }

      expect_equal(fit$effects_meta[[term]]$df_term, manual$df)
      expect_equal(eff$effect_matrix, manual$M, tolerance = 1e-8)
      expect_equal(eff$sdev, d_manual, tolerance = 1e-8)
      expect_equal(eff_scaled$effect_matrix, scale_factor * eff$effect_matrix, tolerance = 1e-7)
      expect_equal(eff_scaled$sdev, abs(scale_factor) * eff$sdev, tolerance = 1e-7)
      expect_true(all(is.finite(eff_scaled$effect_matrix)))
      expect_true(all(is.finite(eff_scaled$sdev)))
    }
  }
})

test_that("aliased fixed terms produce zero-rank effect operators", {
  set.seed(4102)
  design <- expand.grid(
    subject = factor(seq_len(4)),
    level = factor(c("low", "high"), levels = c("low", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design$group <- factor(c("A", "A", "B", "B"))[as.integer(design$subject)]
  Y <- matrix(rnorm(nrow(design) * 3), nrow = nrow(design), ncol = 3)

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ subject + group + level,
    random = NULL,
    basis = identity_basis(),
    preproc = pass()
  )
  eff <- effect(fit, "group")
  rec <- reconstruct(eff, scale = "processed")

  expect_equal(fit$effects_meta[["group"]]$df_term, 0)
  expect_equal(ncomp(eff), 0)
  expect_equal(dim(scores(eff)), c(nrow(Y), 0))
  expect_equal(dim(stats::coef(eff)), c(ncol(Y), 0))
  expect_equal(eff$effect_matrix, matrix(0, nrow = nrow(Y), ncol = ncol(Y)), tolerance = 1e-10)
  expect_equal(rec, matrix(0, nrow = nrow(Y), ncol = ncol(Y)), tolerance = 1e-10)
})

test_that("mixed_regress rejects non-finite responses before deferred effect algebra", {
  design <- expand.grid(
    subject = factor(seq_len(3)),
    level = factor(c("low", "high"), levels = c("low", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  Y <- matrix(rnorm(nrow(design) * 2), nrow = nrow(design), ncol = 2)
  Y_na <- Y
  Y_na[1, 1] <- NA_real_
  Y_inf <- Y
  Y_inf[2, 2] <- Inf

  expect_error(
    mixed_regress(Y_na, design = design, fixed = ~ level, random = NULL),
    "`Y` must contain only finite values"
  )
  expect_error(
    mixed_regress(Y_inf, design = design, fixed = ~ level, random = NULL),
    "`Y` must contain only finite values"
  )

  Y_array <- array(Y, dim = c(3, 2, 2))
  Y_array[1, 1, 1] <- NaN
  expect_error(
    mixed_regress(Y_array, design = design, fixed = ~ level, random = NULL),
    "`Y` must contain only finite values"
  )
})

test_that("grouped random-slope stress preserves row-metric and projector invariants", {
  dat <- make_grouped_stress_data()
  Y <- dat$Y
  design <- dat$design

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * visit,
    random = ~ 1 + time | subject,
    basis = identity_basis(),
    preproc = pass()
  )

  expect_equal(fit$row_metric$mode, "grouped_lmm")
  expect_true(all(is.finite(fit$row_metric$G)))
  expect_true(all(is.finite(fit$row_metric$theta)))
  expect_gt(fit$row_metric$sigma2, 0)
  expect_true(all(eigen(fit$row_metric$G, symmetric = TRUE, only.values = TRUE)$values > -1e-8))

  set.seed(4103)
  A <- matrix(rnorm(nrow(Y) * 5), nrow = nrow(Y), ncol = 5)
  B <- matrix(rnorm(nrow(Y) * 4), nrow = nrow(Y), ncol = 4)
  expect_equal(fit$row_metric$unwhiten(fit$row_metric$whiten(A)), A, tolerance = 1e-7)
  expect_equal(
    crossprod(fit$row_metric$whiten(A), fit$row_metric$whiten(B)),
    crossprod(A, fit$row_metric$solve(B)),
    tolerance = 1e-7
  )

  for (term in names(fit$effects_meta)) {
    meta <- fit$effects_meta[[term]]
    expect_projection_matrix(meta$P_nuis)
    expect_projection_matrix(meta$P_full)
    expect_projection_matrix(meta$P_term, tolerance = 1e-6)
    expect_equal(meta$P_full %*% meta$P_nuis, meta$P_nuis, tolerance = 1e-6)
    expect_equal(sum(diag(meta$P_term)), meta$df_term, tolerance = 1e-6)

    eff <- effect(fit, term)
    rec_processed <- reconstruct(eff, scale = "processed")
    rec_whitened <- reconstruct(eff, scale = "whitened")
    expect_true(all(is.finite(eff$sdev)))
    expect_true(all(is.finite(eff$effect_matrix)))
    expect_true(all(is.finite(rec_processed)))
    expect_equal(fit$row_engine$whiten(rec_processed), rec_whitened, tolerance = 1e-6)
  }
})
