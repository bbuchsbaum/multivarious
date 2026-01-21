context("reconstruct_new.bi_projector")

library(testthat)
library(multivarious)

set.seed(123)

X <- matrix(rnorm(20 * 6), 20, 6)
fit <- pca(X, preproc = standardize())

# full reconstruction via reconstruct_new on all columns
full_rec <- reconstruct_new(fit, X, comp = 1:ncomp(fit))

# partial reconstruction using only subset of columns
cols <- 1:3
partial_rec <- reconstruct_new(fit, X[, cols], colind = cols, comp = 1:ncomp(fit))

# the partial reconstruction should match the corresponding subset
# of the full reconstruction

test_that("partial reconstruction matches subset of full reconstruction", {
  expect_equal(partial_rec, full_rec[, cols], tolerance = 1e-6, ignore_attr = TRUE)
})

# Regression test: reconstruct_new on held-out data should match manual computation
# This tests for the double-preprocessing bug fixed in 2026-01
test_that("reconstruct_new on held-out data matches manual reconstruction", {
  set.seed(456)
  X_full <- matrix(rnorm(100 * 5), 100, 5)
  train_idx <- 1:70
  test_idx <- 71:100

  model <- pca(X_full[train_idx, ], ncomp = 3, preproc = center())
  test_data <- X_full[test_idx, ]

  # Manual reconstruction (known correct)
  scores <- project(model, test_data)
  recon_manual <- scores %*% t(components(model))
  recon_manual <- inverse_transform(model$preproc, recon_manual)

  # reconstruct_new should match
 recon_new <- reconstruct_new(model, test_data)

  expect_equal(dim(recon_new), dim(test_data))
  expect_equal(recon_new, recon_manual, tolerance = 1e-10)
})

test_that("reconstruct_new with subset of components works on held-out data", {
  set.seed(789)
  X_full <- matrix(rnorm(80 * 4), 80, 4)
  train_idx <- 1:60
  test_idx <- 61:80

  model <- pca(X_full[train_idx, ], ncomp = 4, preproc = center())
  test_data <- X_full[test_idx, ]

  # Use only first 2 components
  comp <- 1:2
  scores <- project(model, test_data)[, comp, drop = FALSE]
  v_sub <- components(model)[, comp, drop = FALSE]
  recon_manual <- scores %*% t(v_sub)
  recon_manual <- inverse_transform(model$preproc, recon_manual)

  recon_new <- reconstruct_new(model, test_data, comp = comp)

  expect_equal(dim(recon_new), dim(test_data))
  expect_equal(recon_new, recon_manual, tolerance = 1e-10)
})
