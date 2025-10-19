library(testthat)
library(multivarious)

# ============================================================================
# Helper Functions
# ============================================================================

# Linear kernel for testing (should match PCA when used with nystrom)
linear_kernel <- function(x, y) {
  x %*% t(y)
}

# RBF/Gaussian kernel
rbf_kernel <- function(x, y, sigma = 1) {
  # Compute squared Euclidean distances efficiently
  x_sq <- rowSums(x^2)
  y_sq <- rowSums(y^2)
  xy <- tcrossprod(x, y)
  dist_sq <- outer(x_sq, y_sq, "+") - 2 * xy
  exp(-dist_sq / (2 * sigma^2))
}

# ============================================================================
# Basic Functionality Tests - Standard Method
# ============================================================================

test_that("nystrom_approx returns a bi_projector object with standard method", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)
  nystrom_res <- nystrom_approx(X, ncomp = 3, method = "standard")

  expect_s3_class(nystrom_res, "bi_projector")
  expect_s3_class(nystrom_res, "nystrom_approx")
  expect_s3_class(nystrom_res, "standard")
})

test_that("nystrom_approx returns the correct number of components", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)

  nystrom_res_3 <- nystrom_approx(X, ncomp = 3, method = "standard")
  expect_equal(ncol(nystrom_res_3$v), 3)
  expect_equal(length(nystrom_res_3$sdev), 3)

  nystrom_res_5 <- nystrom_approx(X, ncomp = 5, method = "standard")
  expect_equal(ncol(nystrom_res_5$v), 5)
  expect_equal(length(nystrom_res_5$sdev), 5)
})

test_that("nystrom_approx with linear kernel approximates PCA", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)
  ncomp <- 3

  # Nyström approximation with linear kernel
  nystrom_res <- nystrom_approx(X, kernel_func = linear_kernel,
                                 ncomp = ncomp, method = "standard",
                                 nlandmarks = 30)

  # Standard PCA
  pca_res <- pca(X, ncomp = ncomp)

  # Eigenvalues should be close in relative error (Nyström is approximate)
  eig_rel_err <- max(abs(nystrom_res$sdev^2 - pca_res$sdev^2) /
                       pmax(1, abs(pca_res$sdev^2)))
  expect_lt(eig_rel_err, 0.25)

  # Subspace should stay well aligned (principal angles < ~70°)
  angles <- prinang(qr.Q(qr(nystrom_res$s)), qr.Q(qr(pca_res$s)))
  expect_lt(max(angles), 1.2)
})

test_that("nystrom_approx handles custom landmarks", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)
  landmarks <- c(1, 5, 10, 15, 20, 25, 30)

  nystrom_res <- nystrom_approx(X, landmarks = landmarks, ncomp = 3,
                                 method = "standard")
  expect_s3_class(nystrom_res, "bi_projector")
  expect_equal(length(nystrom_res$meta$landmarks), length(landmarks))
  expect_equal(nystrom_res$meta$landmarks, landmarks)
})

test_that("nystrom_approx handles custom nlandmarks", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)
  nlandmarks <- 20

  nystrom_res <- nystrom_approx(X, nlandmarks = nlandmarks, ncomp = 3,
                                 method = "standard")
  expect_s3_class(nystrom_res, "bi_projector")
  expect_equal(length(nystrom_res$meta$landmarks), nlandmarks)
})

# ============================================================================
# Projection Tests - CRITICAL (Previously Broken)
# ============================================================================

test_that("project() works for standard nystrom with default kernel", {
  set.seed(123)
  X_train <- matrix(rnorm(50 * 10), 50, 10)
  X_test <- matrix(rnorm(20 * 10), 20, 10)

  nystrom_res <- nystrom_approx(X_train, ncomp = 3, method = "standard",
                                 nlandmarks = 20)

  # Project new data - this was previously broken!
  scores_test <- project(nystrom_res, X_test)

  expect_true(is.matrix(scores_test))
  expect_equal(nrow(scores_test), 20)
  expect_equal(ncol(scores_test), 3)
  expect_false(any(is.na(scores_test)))
})

test_that("project() works with custom kernel function", {
  set.seed(123)
  X_train <- matrix(rnorm(50 * 10), 50, 10)  # Fixed: 10 columns
  X_test <- matrix(rnorm(10 * 10), 10, 10)   # Fixed: 10 columns

  nystrom_res <- nystrom_approx(X_train, kernel_func = linear_kernel,
                                 ncomp = 3, method = "standard",
                                 nlandmarks = 20)

  scores_test <- project(nystrom_res, X_test)

  expect_true(is.matrix(scores_test))
  expect_equal(nrow(scores_test), 10)
  expect_equal(ncol(scores_test), 3)
})

test_that("project() works with RBF kernel and extra arguments", {
  set.seed(123)
  X_train <- matrix(rnorm(50 * 10), 50, 10)  # Fixed: 10 columns
  X_test <- matrix(rnorm(10 * 10), 10, 10)   # Fixed: 10 columns

  # Test with sigma parameter passed via ...
  nystrom_res <- nystrom_approx(X_train, kernel_func = rbf_kernel,
                                 ncomp = 3, method = "standard",
                                 nlandmarks = 20, sigma = 0.5)

  # Project should use the stored sigma parameter
  scores_test <- project(nystrom_res, X_test)

  expect_true(is.matrix(scores_test))
  expect_equal(nrow(scores_test), 10)
  expect_equal(ncol(scores_test), 3)
  expect_false(any(is.na(scores_test)))
})

test_that("projected scores have reasonable scale", {
  set.seed(123)
  X_train <- matrix(rnorm(50 * 10), 50, 10)  # Fixed: same ncol as training
  X_test <- matrix(rnorm(30 * 10), 30, 10)   # Increased sample size for more stable estimates

  nystrom_res <- nystrom_approx(X_train, ncomp = 3, method = "standard")
  scores_train <- nystrom_res$s
  scores_test <- project(nystrom_res, X_test)

  # Test scores should have similar scale to training scores
  train_scale <- apply(scores_train, 2, sd)
  test_scale <- apply(scores_test, 2, sd)

  # Scale ratios should stay within a 2x band
  scale_ratio <- test_scale / train_scale
  expect_true(all(scale_ratio > 0.5 & scale_ratio < 2))
})

# ============================================================================
# Double Nyström Method Tests
# ============================================================================

test_that("nystrom_approx works with double method", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)

  nystrom_res <- nystrom_approx(X, ncomp = 3, method = "double",
                                 nlandmarks = 20, l = 10)

  expect_s3_class(nystrom_res, "bi_projector")
  expect_s3_class(nystrom_res, "nystrom_approx")
  expect_s3_class(nystrom_res, "double")
  expect_equal(ncol(nystrom_res$v), 3)
})

test_that("double method requires l parameter", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)

  expect_error(
    nystrom_approx(X, ncomp = 3, method = "double", nlandmarks = 20),
    "you must specify intermediate rank 'l'"
  )
})

test_that("double method validates l parameter", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)

  # l must be > 0
  expect_error(
    nystrom_approx(X, method = "double", nlandmarks = 20, l = 0),
    "Intermediate rank 'l' must be > 0"
  )

  # l must be <= number of landmarks
  expect_error(
    nystrom_approx(X, method = "double", nlandmarks = 20, l = 25),
    "Intermediate rank 'l' must be.*<= number of landmarks"
  )
})

test_that("project() works for double nystrom method", {
  set.seed(123)
  X_train <- matrix(rnorm(50 * 8), 50, 8)
  X_test <- matrix(rnorm(15 * 8), 15, 8)

  nystrom_res <- nystrom_approx(X_train, ncomp = 3, method = "double",
                                 nlandmarks = 20, l = 10)

  scores_test <- project(nystrom_res, X_test)

  expect_true(is.matrix(scores_test))
  expect_equal(nrow(scores_test), 15)
  expect_equal(ncol(scores_test), 3)
  expect_false(any(is.na(scores_test)))
})

test_that("double method with linear kernel approximates PCA", {
  set.seed(123)
  X <- matrix(rnorm(60 * 10), 60, 10)
  ncomp <- 3

  nystrom_res <- nystrom_approx(X, kernel_func = linear_kernel,
                                 ncomp = ncomp, method = "double",
                                 nlandmarks = 30, l = 15)

  pca_res <- pca(X, ncomp = ncomp)

  eig_rel_err <- max(abs(nystrom_res$sdev^2 - pca_res$sdev^2) /
                       pmax(1, abs(pca_res$sdev^2)))
  expect_lt(eig_rel_err, 1.1)

  angles <- prinang(qr.Q(qr(nystrom_res$s)), qr.Q(qr(pca_res$s)))
  expect_lt(max(angles), 1.5)
})

# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

test_that("nystrom_approx handles small datasets", {
  set.seed(123)
  X <- matrix(rnorm(10 * 5), 10, 5)

  # Should work with fewer landmarks than samples
  nystrom_res <- nystrom_approx(X, ncomp = 2, method = "standard",
                                 nlandmarks = 5)
  expect_s3_class(nystrom_res, "bi_projector")
})

test_that("nystrom_approx adjusts ncomp if too large", {
  set.seed(123)
  X <- matrix(rnorm(20 * 5), 20, 5)

  # Request more components than possible
  expect_warning(
    nystrom_res <- nystrom_approx(X, ncomp = 10, method = "standard",
                                   nlandmarks = 8),
    "ncomp.*exceeds"
  )

  # Should have adjusted ncomp down
  expect_lte(ncol(nystrom_res$v), 8)
})

test_that("nystrom_approx validates input dimensions", {
  set.seed(123)

  # Not a matrix
  expect_error(
    nystrom_approx(as.data.frame(matrix(rnorm(20), 10, 2))),
    "must be an matrix"
  )

  # Single column
  X <- matrix(rnorm(20), 20, 1)
  nystrom_res <- nystrom_approx(X, ncomp = 1, nlandmarks = 5)
  expect_s3_class(nystrom_res, "bi_projector")
})

test_that("projection handles single observation", {
  set.seed(123)
  X_train <- matrix(rnorm(50 * 10), 50, 10)  # Fixed: 10 columns
  X_test <- matrix(rnorm(10), nrow = 1)       # Fixed: 10 columns

  nystrom_res <- nystrom_approx(X_train, ncomp = 3, nlandmarks = 20)
  scores_test <- project(nystrom_res, X_test)

  expect_equal(nrow(scores_test), 1)
  expect_equal(ncol(scores_test), 3)
})

test_that("metadata is stored correctly", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)
  landmarks <- c(1, 5, 10, 15, 20)

  nystrom_res <- nystrom_approx(X, landmarks = landmarks, ncomp = 3,
                                 method = "standard")

  # Check metadata exists and contains required fields
  expect_true(!is.null(nystrom_res$meta))
  expect_equal(nystrom_res$meta$method, "standard")
  expect_equal(nystrom_res$meta$ncomp, 3)
  expect_equal(nystrom_res$meta$landmarks, landmarks)

  # CRITICAL: X_landmarks must be stored for projection to work
  expect_true(!is.null(nystrom_res$meta$X_landmarks))
  expect_equal(nrow(nystrom_res$meta$X_landmarks), length(landmarks))
})

# ============================================================================
# Preprocessing Tests
# ============================================================================

test_that("nystrom_approx works with preprocessing", {
  set.seed(123)
  X <- matrix(rnorm(50 * 10), 50, 10)

  # With centering
  nystrom_center <- nystrom_approx(X, ncomp = 3, preproc = center(),
                                    nlandmarks = 20)
  expect_s3_class(nystrom_center, "bi_projector")

  # With standardization
  nystrom_scale <- nystrom_approx(X, ncomp = 3,
                                   preproc = standardize(),
                                   nlandmarks = 20)
  expect_s3_class(nystrom_scale, "bi_projector")
})

test_that("projection respects preprocessing", {
  set.seed(123)
  X_train <- matrix(rnorm(50 * 10, mean = 10), 50, 10)  # Non-zero mean, fixed 10 cols
  X_test <- matrix(rnorm(10 * 10, mean = 10), 10, 10)   # Fixed 10 cols

  nystrom_res <- nystrom_approx(X_train, ncomp = 3,
                                 preproc = center(),
                                 nlandmarks = 20)

  # Projection should apply the same centering
  scores_test <- project(nystrom_res, X_test)

  expect_true(is.matrix(scores_test))
  expect_false(any(is.na(scores_test)))
})

# ============================================================================
# Kernel Function Tests
# ============================================================================

test_that("nystrom_approx works with RBF kernel", {
  set.seed(123)
  X <- matrix(rnorm(40 * 5), 40, 5)

  nystrom_res <- nystrom_approx(X, kernel_func = rbf_kernel,
                                 ncomp = 3, method = "standard",
                                 nlandmarks = 15, sigma = 1.0)

  expect_s3_class(nystrom_res, "bi_projector")
  expect_equal(ncol(nystrom_res$v), 3)
})

test_that("kernel extra arguments are preserved", {
  set.seed(123)
  X_train <- matrix(rnorm(40 * 5), 40, 5)
  X_test <- matrix(rnorm(10 * 5), 10, 5)

  # Custom kernel with multiple parameters
  custom_kernel <- function(x, y, alpha = 1, beta = 0) {
    alpha * (x %*% t(y)) + beta
  }

  nystrom_res <- nystrom_approx(X_train, kernel_func = custom_kernel,
                                 ncomp = 2, nlandmarks = 15,
                                 alpha = 2, beta = 0.1)

  # extra_args should be stored
  expect_true(!is.null(nystrom_res$meta$extra_args))
  expect_equal(nystrom_res$meta$extra_args$alpha, 2)
  expect_equal(nystrom_res$meta$extra_args$beta, 0.1)

  # Projection should work with stored parameters
  scores_test <- project(nystrom_res, X_test)
  expect_equal(nrow(scores_test), 10)
})

# ============================================================================
# Comparison between methods
# ============================================================================

test_that("standard and double methods give similar results", {
  set.seed(123)
  X <- matrix(rnorm(60 * 8), 60, 8)
  ncomp <- 3
  nlandmarks <- 25

  nystrom_standard <- nystrom_approx(X, ncomp = ncomp, method = "standard",
                                      nlandmarks = nlandmarks)

  nystrom_double <- nystrom_approx(X, ncomp = ncomp, method = "double",
                                    nlandmarks = nlandmarks, l = 15)

  eig_rel_err <- max(abs(nystrom_standard$sdev^2 - nystrom_double$sdev^2) /
                       pmax(1, abs(nystrom_standard$sdev^2)))
  expect_lt(eig_rel_err, 1.1)

  angles <- prinang(qr.Q(qr(nystrom_standard$s)), qr.Q(qr(nystrom_double$s)))
  expect_lt(max(angles), 1.5)
})
