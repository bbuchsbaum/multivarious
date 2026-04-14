context("cca reference implementation")

library(multivarious)

make_cca_signal <- function(n = 80L, pX = 6L, pY = 5L, d = 2L, noise = 0.05) {
  Vx <- qr.Q(qr(matrix(rnorm(pX * d), pX, d)))
  Vy <- qr.Q(qr(matrix(rnorm(pY * d), pY, d)))
  F <- matrix(rnorm(n * d), n, d)

  list(
    X = F %*% t(Vx) + noise * matrix(rnorm(n * pX), n, pX),
    Y = F %*% t(Vy) + noise * matrix(rnorm(n * pY), n, pY),
    F = F
  )
}

test_that("cca matches classical cancor in the full-rank setting", {
  set.seed(11)
  dat <- make_cca_signal(n = 120, pX = 4, pY = 3, d = 2, noise = 0.02)

  fit <- cca(dat$X, dat$Y, ncomp = 3, lambda = 0)
  ref <- stats::cancor(dat$X, dat$Y)

  expect_s3_class(fit, c("cca", "cross_projector", "projector"))
  expect_equal(fit$cor, ref$cor[1:3], tolerance = 1e-6)
  expect_equal(shape(fit, "X"), c(ncol(dat$X), 3))
  expect_equal(shape(fit, "Y"), c(ncol(dat$Y), 3))
  expect_equal(project(fit, dat$X, source = "X"), fit$sx, tolerance = 1e-8)
  expect_equal(project(fit, dat$Y, source = "Y"), fit$sy, tolerance = 1e-8)
})

test_that("cca supports p > n via ridge-regularized whitening", {
  set.seed(22)
  dat <- make_cca_signal(n = 24, pX = 40, pY = 30, d = 2, noise = 0.08)

  fit <- cca(dat$X, dat$Y, ncomp = 2, lambda = 0.1)

  expect_s3_class(fit, c("cca", "cross_projector", "projector"))
  expect_equal(shape(fit, "X"), c(40, 2))
  expect_equal(shape(fit, "Y"), c(30, 2))
  expect_true(all(is.finite(fit$cor)))
  expect_true(all(is.finite(fit$sx)))
  expect_true(all(is.finite(fit$sy)))
  expect_equal(fit$ridge$lambda_x, 0.1)
  expect_equal(fit$ridge$lambda_y, 0.1)

  score_cor <- diag(stats::cor(scores(fit, "X"), scores(fit, "Y")))
  expect_gt(score_cor[1], 0.7)
  expect_gt(score_cor[2], 0.5)

  Y_hat <- transfer(fit, dat$X, from = "X", to = "Y",
                    opts = list(ls_rr = TRUE, comps = 1:2))
  expect_equal(dim(Y_hat), dim(dat$Y))
  expect_true(all(is.finite(Y_hat)))
})

test_that("cca supports block-specific ridge shrinkage", {
  set.seed(33)
  dat <- make_cca_signal(n = 30, pX = 20, pY = 18, d = 2, noise = 0.05)

  fit <- cca(dat$X, dat$Y, ncomp = 2, lambda_x = 0.2, lambda_y = 0.05)

  expect_equal(fit$ridge$lambda_x, 0.2)
  expect_equal(fit$ridge$lambda_y, 0.05)
  expect_true(fit$ridge$penalty_x > fit$ridge$penalty_y)
  expect_equal(dim(scores(fit, "X")), c(nrow(dat$X), 2))
  expect_equal(dim(scores(fit, "Y")), c(nrow(dat$Y), 2))
})
