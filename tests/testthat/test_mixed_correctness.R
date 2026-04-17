test_that("fixed-effect identity-basis path matches direct projector algebra", {
  set.seed(21)

  design <- expand.grid(
    group = factor(c("A", "B")),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    rep = seq_len(4),
    KEEP.OUT.ATTRS = FALSE
  )

  n <- nrow(design)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  group_num <- ifelse(design$group == "B", 1, 0)
  Y <- cbind(
    1.5 * level_num + rnorm(n, sd = 0.05),
    0.8 * group_num + rnorm(n, sd = 0.05),
    1.2 * level_num * group_num + rnorm(n, sd = 0.05),
    rnorm(n, sd = 0.05)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = NULL,
    basis = identity_basis(),
    preproc = pass()
  )

  eff <- effect(fit, "group:level")

  X <- model.matrix(~ group * level, data = design)
  assign_vec <- attr(X, "assign")
  effect_cols <- which(assign_vec == 3L)
  nuisance_cols <- setdiff(seq_len(ncol(X)), effect_cols)
  P_nuis <- multivarious:::orthogonal_projector(X[, nuisance_cols, drop = FALSE])
  P_full <- multivarious:::orthogonal_projector(X)
  P_term <- P_full - P_nuis
  M_manual <- P_term %*% Y
  d_manual <- svd(M_manual, nu = 0, nv = 0)$d
  d_manual <- d_manual[seq_len(qr(M_manual)$rank)]

  expect_equal(eff$effect_matrix, M_manual, tolerance = 1e-8)
  expect_equal(eff$sdev, d_manual, tolerance = 1e-8)
  expect_equal(reconstruct(eff, scale = "processed"), M_manual, tolerance = 1e-8)
})

test_that("reordering grouped observations preserves effect singular values", {
  set.seed(22)

  design <- expand.grid(
    subject = factor(seq_len(10)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design$group <- factor(rep(c("A", "B"), each = 15))

  subj_idx <- as.integer(design$subject)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  group_num <- ifelse(design$group == "B", 1, 0)
  b0 <- rnorm(nlevels(design$subject), sd = 0.5)
  b1 <- rnorm(nlevels(design$subject), sd = 0.2)
  Y <- cbind(
    b0[subj_idx] + b1[subj_idx] * level_num + rnorm(nrow(design), sd = 0.1),
    1.0 * group_num + rnorm(nrow(design), sd = 0.1),
    0.8 * level_num * group_num + rnorm(nrow(design), sd = 0.1),
    rnorm(nrow(design), sd = 0.1)
  )

  fit1 <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 + level | subject,
    basis = shared_pca(3),
    preproc = center()
  )

  perm <- sample.int(nrow(design))
  fit2 <- mixed_regress(
    Y[perm, , drop = FALSE],
    design = design[perm, , drop = FALSE],
    fixed = ~ group * level,
    random = ~ 1 + level | subject,
    basis = shared_pca(3),
    preproc = center()
  )

  eff1 <- effect(fit1, "group:level")
  eff2 <- effect(fit2, "group:level")

  expect_equal(eff1$sdev, eff2$sdev, tolerance = 1e-8)
  expect_equal(sort(eigen(fit1$row_metric$G, symmetric = TRUE, only.values = TRUE)$values),
               sort(eigen(fit2$row_metric$G, symmetric = TRUE, only.values = TRUE)$values),
               tolerance = 1e-8)
})

test_that("shared_pca fit targets are distinct and residual-based targets stay finite", {
  set.seed(23)

  design <- expand.grid(
    subject = factor(seq_len(8)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design$group <- factor(rep(c("A", "B"), each = 12))

  subj_idx <- as.integer(design$subject)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  Y <- cbind(
    rnorm(nrow(design)) + level_num,
    rnorm(nrow(design)) + subj_idx / 10,
    matrix(rnorm(nrow(design) * 18), nrow = nrow(design), ncol = 18)
  )

  fit_full <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 + level | subject,
    basis = shared_pca(5, fit_on = "full"),
    preproc = center()
  )
  fit_nuis <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 + level | subject,
    basis = shared_pca(5, fit_on = "nuisance_residual"),
    preproc = center()
  )
  fit_white <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 + level | subject,
    basis = shared_pca(5, fit_on = "whitened_residual"),
    preproc = center()
  )

  expect_equal(dim(fit_full$basis_matrix), c(ncol(Y), 5))
  expect_true(all(is.finite(fit_nuis$basis_matrix)))
  expect_true(all(is.finite(fit_white$basis_matrix)))
  expect_gt(sum(abs(fit_full$basis_matrix - fit_nuis$basis_matrix)), 1e-6)
  expect_gt(sum(abs(fit_full$basis_matrix - fit_white$basis_matrix)), 1e-6)
})

test_that("omnibus permutation result exposes studentized trace metadata", {
  set.seed(24)

  design <- expand.grid(
    subject = factor(seq_len(6)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design$group <- factor(rep(c("A", "B"), each = 9))

  subj_idx <- as.integer(design$subject)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  group_num <- ifelse(design$group == "B", 1, 0)
  b0 <- rnorm(nlevels(design$subject), sd = 0.5)
  b1 <- rnorm(nlevels(design$subject), sd = 0.2)
  Y <- cbind(
    b0[subj_idx] + b1[subj_idx] * level_num + rnorm(nrow(design), sd = 0.1),
    group_num + rnorm(nrow(design), sd = 0.1),
    level_num * group_num + rnorm(nrow(design), sd = 0.1),
    rnorm(nrow(design), sd = 0.1)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 + level | subject,
    basis = shared_pca(3),
    preproc = center()
  )
  pt <- perm_test(effect(fit, "group:level"), nperm = 19, alpha = 0.05)

  ## Regression guard for the old pivot bug: the full-model residual energy
  ## must be computed from the whitened basis-space response, not from the
  ## effect matrix itself. Otherwise the denominator collapses to zero and the
  ## omnibus path silently falls back to the raw trace statistic.
  expect_identical(pt$omnibus_statistic_type, "trace_ratio")
  expect_true(is.numeric(pt$omnibus_statistic_raw))
  expect_true(is.numeric(pt$omnibus_statistic_residual_energy))
  expect_gt(pt$omnibus_statistic_residual_energy, 0)
  expect_equal(
    pt$omnibus_statistic * pt$omnibus_statistic_residual_energy,
    pt$omnibus_statistic_raw,
    tolerance = 1e-8
  )
  expect_true(all(c("omnibus", "omnibus_raw", "omnibus_residual_energy") %in% names(pt$perm_values)))
  expect_true(all(is.finite(pt$perm_values$omnibus)))
  expect_true(all(is.finite(pt$perm_values$omnibus_residual_energy)))
  expect_true(all(pt$perm_values$omnibus_residual_energy > 0))
})

test_that("near-degenerate grouped designs still return finite effect decompositions", {
  set.seed(25)

  design <- expand.grid(
    subject = factor(seq_len(7)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design <- subset(design, !(subject %in% c("2", "6") & level == "mid"))
  design$group <- factor(rep(c("A", "B"), length.out = 7)[as.integer(design$subject)])

  subj_idx <- as.integer(design$subject)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  b0 <- rnorm(nlevels(design$subject), sd = 0.6)
  b1 <- rnorm(nlevels(design$subject), sd = 0.05)
  Y <- cbind(
    b0[subj_idx] + b1[subj_idx] * level_num + rnorm(nrow(design), sd = 0.05),
    rnorm(nrow(design), sd = 0.05),
    rnorm(nrow(design), sd = 0.05)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ (1 | subject) + (0 + level | subject),
    basis = identity_basis(),
    preproc = pass()
  )
  eff <- effect(fit, "level")
  pt <- perm_test(eff, nperm = 19, alpha = 0.05)

  expect_true(all(is.finite(eff$sdev)))
  expect_true(all(is.finite(reconstruct(eff, scale = "processed"))))
  expect_true(is.finite(pt$omnibus_statistic))
  expect_true(is.finite(pt$omnibus_p_value))
})
