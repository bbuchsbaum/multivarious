context("plsc core and inference")

library(multivarious)

make_signal_blocks <- function(n = 60L, pX = 6L, pY = 5L, d = 2L, noise = 0.05) {
  set.seed(123)
  Vx <- qr.Q(qr(matrix(rnorm(pX * d), pX, d)))
  Vy <- qr.Q(qr(matrix(rnorm(pY * d), pY, d)))
  F  <- matrix(rnorm(n * d), n, d)
  X  <- F %*% t(Vx) + noise * matrix(rnorm(n * pX), n, pX)
  Y  <- F %*% t(Vy) + noise * matrix(rnorm(n * pY), n, pY)
  list(X = X, Y = Y, Vx = Vx, Vy = Vy, F = F)
}

make_null_blocks <- function(n = 60L, pX = 6L, pY = 5L) {
  set.seed(999)
  X <- matrix(rnorm(n * pX), n, pX)
  Y <- matrix(rnorm(n * pY), n, pY) # independent
  list(X = X, Y = Y)
}

test_that("plsc returns expected shapes and metadata", {
  dat <- make_signal_blocks()
  fit <- plsc(dat$X, dat$Y, ncomp = 3)
  expect_s3_class(fit, c("plsc", "cross_projector", "projector"))
  expect_equal(shape(fit, "X"), c(ncol(dat$X), 3))
  expect_equal(shape(fit, "Y"), c(ncol(dat$Y), 3))
  expect_true(length(fit$singvals) == 3)
  expect_true(abs(sum(fit$explained_cov) - 1) < 1e-8)
  # scores accessor works
  expect_equal(scores(fit, "X"), fit$sx)
  expect_equal(scores(fit, "Y"), fit$sy)
})

test_that("perm_test.plsc detects true LVs and stops on nulls", {
  dat <- make_signal_blocks(noise = 0.1)
  fit <- plsc(dat$X, dat$Y, ncomp = 3)
  set.seed(2024)
  ptest <- perm_test(fit, dat$X, dat$Y, nperm = 199, comps = 3, parallel = FALSE)
  expect_s3_class(ptest, "perm_test_plsc")
  expect_lt(ptest$component_results$pval[1], 0.05)
  expect_lt(ptest$component_results$pval[2], 0.05)
  expect_gt(ptest$component_results$pval[3], 0.05)
  expect_equal(ptest$n_significant, 2)
})

test_that("perm_test.plsc controls type I error under independence", {
  dat <- make_null_blocks()
  fit <- plsc(dat$X, dat$Y, ncomp = 2)
  set.seed(42)
  ptest <- perm_test(fit, dat$X, dat$Y, nperm = 199, comps = 2, parallel = FALSE)
  expect_gt(ptest$component_results$pval[1], 0.05)
})

test_that("perm_test.plsc returns well-formed permutation matrix and sequential count", {
  dat <- make_signal_blocks(noise = 0.15, d = 1L)
  fit <- plsc(dat$X, dat$Y, ncomp = 2)
  set.seed(111)
  ptest <- perm_test(fit, dat$X, dat$Y, nperm = 99, comps = 2, alpha = 0.2)
  # perm_values has nperm rows and comps columns
  expect_equal(dim(ptest$perm_values), c(99, 2))
  # all successful permutations counted
  expect_equal(ptest$nperm, rep(99, 2))
  # n_significant cannot exceed first non-significant component
  if (ptest$component_results$pval[2] > 0.2) {
    expect_equal(ptest$n_significant, 1)
  }
})

test_that("bootstrap.plsc dispatch works and stabilizes signal loadings", {
  dat <- make_signal_blocks(noise = 0.05, pX = 4L, pY = 4L, d = 1L)
  fit <- plsc(dat$X, dat$Y, ncomp = 1)
  set.seed(7)
  bres <- bootstrap(fit, nboot = 40, X = dat$X, Y = dat$Y, comps = 1, parallel = FALSE)
  expect_s3_class(bres, "bootstrap_plsc_result")
  # Signal variables (rows 1:2) should have larger |z| than noise rows (3:4)
  zsig <- colMeans(abs(bres$z_vx[1:2, , drop = FALSE]))
  znoise <- colMeans(abs(bres$z_vx[3:4, , drop = FALSE]))
  expect_gt(zsig, znoise + 1)
})

test_that("bootstrap.plsc sign-alignment keeps loadings oriented and reproducible", {
  dat <- make_signal_blocks(noise = 0.02, pX = 5L, pY = 5L, d = 1L)
  fit <- plsc(dat$X, dat$Y, ncomp = 1)
  set.seed(99)
  b1 <- bootstrap(fit, nboot = 25, X = dat$X, Y = dat$Y, comps = 1, parallel = FALSE)
  set.seed(99)
  b2 <- bootstrap(fit, nboot = 25, X = dat$X, Y = dat$Y, comps = 1, parallel = FALSE)
  # reproducible under seed
  expect_equal(b1$z_vx, b2$z_vx)
  # bootstrap means align in sign with fitted loadings
  corr_sign <- sign(crossprod(b1$E_vx, coef(fit, "X")[, 1, drop = FALSE]))
  expect_true(all(corr_sign >= 0))
})
