test_that("effect_operator reconstruction matches fitted term contribution", {
  set.seed(3)

  design <- expand.grid(
    group = factor(c("A", "B")),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    rep = seq_len(3),
    KEEP.OUT.ATTRS = FALSE
  )

  n <- nrow(design)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  group_num <- ifelse(design$group == "B", 1, 0)

  Y <- cbind(
    1.2 * level_num + rnorm(n, sd = 0.05),
    0.8 * level_num * group_num + rnorm(n, sd = 0.05),
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

  eff <- effect(fit, "level")

  expect_equal(
    reconstruct(eff, scale = "processed"),
    eff$fitted_contribution,
    tolerance = 1e-8
  )

  trunc1 <- truncate(eff, 1)
  expect_equal(ncomp(trunc1), 1)
  expect_equal(ncol(components(trunc1)), 1)
  expect_equal(ncol(scores(trunc1)), 1)
})

test_that("perm_test.effect_operator returns sequential rank summary", {
  set.seed(4)

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
    2 * level_num + rnorm(n, sd = 0.05),
    1.5 * level_num * group_num + rnorm(n, sd = 0.05),
    rnorm(n, sd = 0.05),
    -level_num + rnorm(n, sd = 0.05)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = NULL,
    basis = shared_pca(3),
    preproc = pass()
  )

  eff <- effect(fit, "group:level")
  ptest <- perm_test(eff, nperm = 39, alpha = 0.2, parallel = FALSE)

  expect_s3_class(ptest, c("perm_test_effect_operator", "perm_test"))
  expect_true(is.numeric(ptest$omnibus_p_value))
  expect_true(ncomp(ptest) >= 0)
  expect_true(ncomp(ptest) <= ncomp(eff))
  expect_true(all(c("comp", "observed", "pval") %in% names(ptest$component_results)))
  expect_length(ptest$observed_projected_residuals, ncomp(eff))
  expect_true(all(vapply(ptest$observed_projected_residuals, inherits, logical(1), "projected_effect_residual")))
  expect_equal(
    vapply(ptest$observed_projected_residuals, `[[`, integer(1), "rank_correction"),
    seq_len(ncomp(eff)) - 1L
  )
  expect_true(all(c("statistic", "effective_rank", "rel") %in% names(ptest$component_results)))
})

test_that("one-df between-subject effects use a non-degenerate sequential statistic", {
  set.seed(2001)

  design <- expand.grid(
    subject = factor(seq_len(24)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design$group <- factor(rep(c("A", "B"), length.out = 24)[as.integer(design$subject)])

  subj_idx <- as.integer(design$subject)
  group_num <- ifelse(design$group == "B", 1, 0)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  b0 <- rnorm(nlevels(design$subject), sd = 0.7)
  b1 <- rnorm(nlevels(design$subject), sd = 0.3)
  V <- qr.Q(qr(matrix(rnorm(100), nrow = 100, ncol = 1)))[, 1, drop = FALSE]
  signal <- 1.5 * scale(cbind(group_num), center = TRUE, scale = FALSE) %*% t(V)

  Y <- signal +
    cbind(b0[subj_idx] + b1[subj_idx] * level_num, matrix(0, nrow = nrow(design), ncol = 99)) +
    matrix(rnorm(nrow(design) * 100, sd = 0.3), nrow = nrow(design), ncol = 100)

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 + level | subject,
    basis = shared_pca(10),
    preproc = center()
  )

  pt <- perm_test(effect(fit, "group"), nperm = 39, alpha = 0.05)

  expect_equal(pt$component_results$statistic[1], "lead_sv2")
  expect_equal(ncomp(pt), 1)
})

test_that("rank-2 interaction effects remain testable after first-axis deflation", {
  set.seed(2001)

  design <- expand.grid(
    subject = factor(seq_len(24)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design$group <- factor(rep(c("A", "B"), length.out = 24)[as.integer(design$subject)])

  subj_idx <- as.integer(design$subject)
  group_num <- ifelse(design$group == "B", 1, 0)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  level_quad <- c(low = 1, mid = -2, high = 1)[as.character(design$level)]
  b0 <- rnorm(nlevels(design$subject), sd = 0.7)
  b1 <- rnorm(nlevels(design$subject), sd = 0.3)

  V <- qr.Q(qr(matrix(rnorm(100 * 2), nrow = 100, ncol = 2)))[, 1:2, drop = FALSE]
  S <- scale(cbind(group_num * level_num, group_num * level_quad), center = TRUE, scale = FALSE)
  signal <- 1.1 * S %*% t(V)

  Y <- signal +
    cbind(b0[subj_idx] + b1[subj_idx] * level_num, matrix(0, nrow = nrow(design), ncol = 99)) +
    matrix(rnorm(nrow(design) * 100, sd = 0.3), nrow = nrow(design), ncol = 100)

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 + level | subject,
    basis = shared_pca(10),
    preproc = center()
  )

  pt <- perm_test(effect(fit, "group:level"), nperm = 39, alpha = 0.05)

  expect_true(all(pt$component_results$statistic == "lead_sv2"))
  expect_equal(ncomp(pt), 2)
})

test_that("between-subject effect uses subject-block permutation when grouping is available", {
  set.seed(7)

  design <- expand.grid(
    subject = factor(seq_len(6)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  subject_group <- rep(c("A", "B"), each = 3)
  design$group <- factor(subject_group[as.integer(design$subject)])

  n <- nrow(design)
  group_num <- ifelse(design$group == "B", 1, 0)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]

  Y <- cbind(
    2 * group_num + rnorm(n, sd = 0.05),
    level_num + rnorm(n, sd = 0.05),
    rnorm(n, sd = 0.05)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 | subject,
    basis = identity_basis(),
    preproc = pass()
  )

  eff <- effect(fit, "group")
  ptest <- perm_test(eff, nperm = 19)

  expect_match(ptest$exchangeability, "subject-mean permutation", fixed = TRUE)
})

test_that("within-subject effect uses within-subject permutation when grouping is available", {
  set.seed(11)

  design <- expand.grid(
    subject = factor(seq_len(6)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  subject_group <- rep(c("A", "B"), each = 3)
  design$group <- factor(subject_group[as.integer(design$subject)])

  n <- nrow(design)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  Y <- cbind(
    level_num + rnorm(n, sd = 0.05),
    -level_num + rnorm(n, sd = 0.05),
    rnorm(n, sd = 0.05)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 | subject,
    basis = identity_basis(),
    preproc = pass()
  )

  eff <- effect(fit, "level")
  ptest <- perm_test(eff, nperm = 19)

  expect_match(ptest$exchangeability, "within-subject contrast sign flips", fixed = TRUE)
})

test_that("bootstrap.effect_operator returns stability summaries", {
  set.seed(5)

  design <- expand.grid(
    group = factor(c("A", "B")),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    rep = seq_len(3),
    KEEP.OUT.ATTRS = FALSE
  )

  n <- nrow(design)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  group_num <- ifelse(design$group == "B", 1, 0)

  Y <- cbind(
    level_num + rnorm(n, sd = 0.05),
    1.2 * level_num * group_num + rnorm(n, sd = 0.05),
    rnorm(n, sd = 0.05)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = NULL,
    basis = shared_pca(2),
    preproc = pass()
  )

  eff <- effect(fit, "group:level")
  bres <- bootstrap(eff, nboot = 9)

  expect_s3_class(bres, "bootstrap_effect_operator_result")
  expect_equal(dim(bres$loadings_mean), dim(components(eff)))
  expect_equal(length(bres$singular_values_mean), ncomp(eff))
})

test_that("bootstrap.effect_operator resamples subjects when grouping metadata are available", {
  set.seed(8)

  design <- expand.grid(
    subject = factor(seq_len(5)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  subject_group <- c("A", "B", "A", "B", "A")
  design$group <- factor(subject_group[as.integer(design$subject)])

  n <- nrow(design)
  group_num <- ifelse(design$group == "B", 1, 0)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  Y <- cbind(
    group_num + rnorm(n, sd = 0.05),
    level_num + rnorm(n, sd = 0.05),
    rnorm(n, sd = 0.05)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 | subject,
    basis = shared_pca(2),
    preproc = pass()
  )

  eff <- effect(fit, "group")
  bres <- bootstrap(eff, nboot = 7, resample = "subject")

  expect_equal(bres$resample, "subject")
})
