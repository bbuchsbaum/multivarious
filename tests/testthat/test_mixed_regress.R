test_that("mixed_regress fits fixed-effect operator model and extracts named effects", {
  set.seed(1)

  design <- expand.grid(
    subject = factor(seq_len(8)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design$group <- factor(rep(c("A", "B"), each = 12))

  n <- nrow(design)
  p <- 5

  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  group_num <- ifelse(design$group == "B", 1, 0)

  Y <- cbind(
    2 * level_num + rnorm(n, sd = 0.1),
    group_num + rnorm(n, sd = 0.1),
    1.5 * level_num * group_num + rnorm(n, sd = 0.1),
    rnorm(n, sd = 0.1),
    -level_num + rnorm(n, sd = 0.1)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = NULL,
    basis = identity_basis(),
    preproc = pass()
  )

  expect_s3_class(fit, "mixed_fit")
  expect_true(all(c("group", "level", "group:level") %in% names(fit$effects_meta)))

  eff <- effect(fit, "group:level")
  expect_s3_class(eff, c("effect_operator", "bi_projector", "projector"))
  expect_equal(eff$term, "group:level")
  expect_lte(ncomp(eff), min(eff$df_term, eff$basis_rank))
})

test_that("array input normalizes to the same effect decomposition as matrix input", {
  set.seed(2)

  n_subject <- 4
  n_within <- 3
  p <- 4

  design <- data.frame(
    subject = factor(rep(seq_len(n_subject), each = n_within)),
    level = factor(rep(c("low", "mid", "high"), times = n_subject),
                   levels = c("low", "mid", "high"))
  )

  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  Y_mat <- cbind(
    level_num + rnorm(nrow(design), sd = 0.05),
    -level_num + rnorm(nrow(design), sd = 0.05),
    rnorm(nrow(design), sd = 0.05),
    0.5 * level_num + rnorm(nrow(design), sd = 0.05)
  )

  Y_arr <- array(NA_real_, dim = c(n_subject, n_within, p))
  idx <- 1
  for (i in seq_len(n_subject)) {
    for (j in seq_len(n_within)) {
      Y_arr[i, j, ] <- Y_mat[idx, ]
      idx <- idx + 1
    }
  }

  fit_mat <- mixed_regress(
    Y_mat,
    design = design,
    fixed = ~ level,
    random = NULL,
    basis = identity_basis(),
    preproc = pass()
  )

  fit_arr <- mixed_regress(
    Y_arr,
    design = design,
    fixed = ~ level,
    random = NULL,
    basis = identity_basis(),
    preproc = pass()
  )

  eff_mat <- effect(fit_mat, "level")
  eff_arr <- effect(fit_arr, "level")

  expect_equal(eff_arr$sdev, eff_mat$sdev, tolerance = 1e-8)
  expect_equal(
    reconstruct(eff_arr, scale = "processed"),
    reconstruct(eff_mat, scale = "processed"),
    tolerance = 1e-8
  )
})

test_that("mixed_regress accepts a simple random formula and stores grouping metadata", {
  set.seed(6)

  design <- expand.grid(
    subject = factor(seq_len(5)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  subject_group <- c("A", "B", "A", "B", "A")
  design$group <- factor(subject_group[as.integer(design$subject)])

  n <- nrow(design)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  Y <- cbind(level_num + rnorm(n, sd = 0.1), rnorm(n, sd = 0.1))

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 | subject,
    basis = identity_basis(),
    preproc = pass()
  )

  expect_equal(fit$grouping_var, "subject")
  expect_length(fit$subject_blocks, nlevels(design$subject))
  expect_equal(fit$effects_meta[["group"]]$term_scope, "between")
  expect_true(fit$effects_meta[["level"]]$term_scope %in% c("within", "mixed"))
})

test_that("mixed_regress estimates a grouped row metric for random intercept models", {
  set.seed(9)

  design <- expand.grid(
    subject = factor(seq_len(10)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  subject_group <- rep(c("A", "B"), each = 5)
  design$group <- factor(subject_group[as.integer(design$subject)])

  subj_eff <- rnorm(nlevels(design$subject), sd = 0.8)
  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  subj_idx <- as.integer(design$subject)
  n <- nrow(design)

  Y <- cbind(
    subj_eff[subj_idx] + level_num + rnorm(n, sd = 0.2),
    0.5 * subj_eff[subj_idx] - level_num + rnorm(n, sd = 0.2),
    rnorm(n, sd = 0.2)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 | subject,
    basis = identity_basis(),
    preproc = pass()
  )

  expect_equal(fit$row_metric$mode, "grouped_lmm")
  expect_true(is.matrix(fit$row_metric$G))
  expect_gt(fit$row_metric$sigma2, 0)

  eff <- effect(fit, "level")
  expect_s3_class(eff, "effect_operator")
  expect_equal(nrow(reconstruct(eff, scale = "processed")), nrow(Y))
})

test_that("mixed_regress handles random slopes and unbalanced visits", {
  set.seed(10)

  design <- expand.grid(
    subject = factor(seq_len(8)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design$group <- factor(rep(c("A", "B"), each = 12))

  keep <- !(design$subject %in% c("3", "7") & design$level == "mid")
  design <- design[keep, , drop = FALSE]

  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  subj_idx <- as.integer(design$subject)
  b0 <- rnorm(nlevels(design$subject), sd = 0.6)
  b1 <- rnorm(nlevels(design$subject), sd = 0.3)
  n <- nrow(design)

  Y <- cbind(
    b0[subj_idx] + b1[subj_idx] * level_num + rnorm(n, sd = 0.2),
    -b1[subj_idx] * level_num + rnorm(n, sd = 0.2),
    rnorm(n, sd = 0.2)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ 1 + level | subject,
    basis = shared_pca(2),
    preproc = pass()
  )

  expect_equal(fit$row_metric$mode, "grouped_lmm")
  expect_equal(fit$grouping_var, "subject")
  expect_true(length(fit$subject_blocks) == nlevels(design$subject))

  eff <- effect(fit, "group:level")
  expect_s3_class(eff, "effect_operator")
  expect_true(all(is.finite(eff$sdev)))
})

test_that("mixed_regress combines multiple same-group random terms into one row metric", {
  set.seed(12)

  design <- expand.grid(
    subject = factor(seq_len(6)),
    level = factor(c("low", "mid", "high"), levels = c("low", "mid", "high")),
    KEEP.OUT.ATTRS = FALSE
  )
  design$group <- factor(rep(c("A", "B"), each = 9))

  level_num <- c(low = -1, mid = 0, high = 1)[as.character(design$level)]
  subj_idx <- as.integer(design$subject)
  b0 <- rnorm(nlevels(design$subject), sd = 0.5)
  b1 <- rnorm(nlevels(design$subject), sd = 0.2)
  n <- nrow(design)

  Y <- cbind(
    b0[subj_idx] + b1[subj_idx] * level_num + rnorm(n, sd = 0.1),
    rnorm(n, sd = 0.1)
  )

  fit <- mixed_regress(
    Y,
    design = design,
    fixed = ~ group * level,
    random = ~ (1 | subject) + (0 + level | subject),
    basis = identity_basis(),
    preproc = pass()
  )

  expect_equal(fit$grouping_var, "subject")
  expect_equal(fit$random_spec$n_bars, 2L)
  expect_true(all(c("level") %in% fit$random_spec$random_terms))
  expect_equal(fit$row_metric$mode, "grouped_lmm")
})
