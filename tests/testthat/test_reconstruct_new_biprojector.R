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
