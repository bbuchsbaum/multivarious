library(testthat)
library(MASS)
library(RSpectra)
# Assuming relative_eigen and other related functions are already loaded in the environment

# Test 1: Relative Eigenanalysis works correctly for identity covariance matrices
test_that("Relative Eigenanalysis works correctly for identity covariance matrices", {
  p <- 3
  n_A <- 2000
  n_B <- 2000
  
  # Generate data with identity covariance matrices
  Sigma_identity <- diag(p)
  XA <- MASS::mvrnorm(n = n_A, mu = rep(0, p), Sigma = Sigma_identity)
  XB <- MASS::mvrnorm(n = n_B, mu = rep(0, p), Sigma = Sigma_identity)
  
  # Perform relative eigenanalysis
  res <- relative_eigen(XA, XB, ncomp = p)
  
  # Eigenvalues should be close to 1
  expect_true(all(abs(res$values - 1) < 0.1))  # Adjust tolerance as needed
})

# Test 2: Relative Eigenanalysis with SigmaA = 2 * SigmaB yields eigenvalues ~2
test_that("Relative Eigenanalysis with SigmaA = 2 * SigmaB yields eigenvalues ~2", {
  p <- 2
  n_A <- 3000
  n_B <- 3000
  
  # Define SigmaB as identity, and SigmaA as 2 * identity
  SigmaB <- diag(p)
  SigmaA <- 2 * diag(p)
  
  # Generate data with these covariance matrices
  XA <- MASS::mvrnorm(n = n_A, mu = rep(0, p), Sigma = SigmaA)
  XB <- MASS::mvrnorm(n = n_B, mu = rep(0, p), Sigma = SigmaB)
  
  # Perform relative eigenanalysis
  res <- relative_eigen(XA, XB, ncomp = p)
  
  # Eigenvalues should be close to 2
  expect_true(all(abs(res$values - 2) < 0.2))  # Adjust tolerance as needed
})

# Test 3: Relative Eigenanalysis with different diagonal covariances
test_that("Relative Eigenanalysis with different diagonal covariances", {
  p <- 2
  n_A <- 1000
  n_B <- 1000
  
  # Define SigmaA and SigmaB as diagonal matrices with different variances
  SigmaA <- diag(c(2, 3))
  SigmaB <- diag(c(1, 1))
  
  # Generate data with these covariance matrices
  XA <- MASS::mvrnorm(n = n_A, mu = rep(0, p), Sigma = SigmaA)
  XB <- MASS::mvrnorm(n = n_B, mu = rep(0, p), Sigma = SigmaB)
  
  # Perform relative eigenanalysis
  res <- relative_eigen(XA, XB, ncomp = p)
  
  # Compute theoretical generalized eigenvalues and eigenvectors
  geigen_res <- eigen(solve(SigmaB) %*% SigmaA)
  expected_values <- geigen_res$values
  expected_vectors <- geigen_res$vectors
  
  # Compare eigenvalues
  expect_equal(sort(res$values), sort(expected_values), tolerance = 0.1)
  
  # Compare eigenvectors (up to sign)
  for(i in 1:p){
    # Adjust for sign differences
    corr <- cor(res$v[,i], expected_vectors[,i])
    expect_true(abs(abs(corr) - 1) < 0.1)
  }
})

# Test 4: Relative Eigenanalysis with known eigenvectors
test_that("Relative Eigenanalysis with known eigenvectors", {
  p <- 2
  n_A <- 100
  n_B <- 100
  
  # Define SigmaA and SigmaB with off-diagonal elements
  SigmaA <- matrix(c(2, 1, 1, 3), nrow = 2)
  SigmaB <- diag(2)
  
  # Generate data with these covariance matrices
  XA <- MASS::mvrnorm(n = n_A, mu = rep(0, p), Sigma = SigmaA)
  XB <- MASS::mvrnorm(n = n_B, mu = rep(0, p), Sigma = SigmaB)
  
  # Perform relative eigenanalysis
  res <- relative_eigen(XA, XB, ncomp = p)
  
  # Expected generalized eigenvalues and eigenvectors
  gen_eigen <- eigen(solve(SigmaB) %*% SigmaA)
  expected_values <- gen_eigen$values
  expected_vectors <- gen_eigen$vectors
  
  # Compare eigenvalues
  expect_equal(sort(res$values), sort(expected_values), tolerance = 0.2)
  
  # Compare eigenvectors (up to sign)
  for(i in 1:p){
    expect_true(any(abs(res$v[,i] - expected_vectors[,i]) < 0.1) ||
                  any(abs(res$v[,i] + expected_vectors[,i]) < 0.1))
  }
})

# Test 5: Relative Eigenanalysis handles high-dimensional data correctly
test_that("Relative Eigenanalysis handles high-dimensional data correctly", {
  p <- 5000
  n_A <- 100
  n_B <- 100
  XA <- matrix(rnorm(n_A * p), nrow = n_A, ncol = p)
  XB <- matrix(rnorm(n_B * p), nrow = n_B, ncol = p)
  
  expect_error(
    res <- relative_eigen(XA, XB, ncomp = 5),
    NA  # Expect no error
  )
  
  expect_true(length(res$values) == 5)
  expect_true(ncol(res$v) == 5)
  expect_true(nrow(res$v) == p)
})

# Test 6: Relative Eigenanalysis produces consistent results for multiple runs
test_that("Relative Eigenanalysis produces consistent results for multiple runs", {
  p <- 100
  n_A <- 100
  n_B <- 100
  set.seed(123)
  XA <- matrix(rnorm(n_A * p), nrow = n_A, ncol = p)
  XB <- matrix(rnorm(n_B * p), nrow = n_B, ncol = p)
  
  res1 <- relative_eigen(XA, XB, ncomp = 5)
  res2 <- relative_eigen(XA, XB, ncomp = 5)
  
  # Eigenvalues should be similar
  expect_equal(res1$values, res2$values, tolerance = 1e-6)
  
  # Eigenvectors should be similar up to sign
  for(i in 1:5){
    expect_true(all(abs(res1$v[,i] - res2$v[,i]) < 1e-6) ||
                  all(abs(res1$v[,i] + res2$v[,i]) < 1e-6))
  }
})

# Test 7: Relative Eigenanalysis handles singular SigmaB with regularization
test_that("Relative Eigenanalysis handles singular SigmaB with regularization", {
  p <- 2
  n_A <- 100
  n_B <- 1  # Very small n_B leading to singular SigmaB without regularization
  XA <- matrix(rnorm(n_A * p), nrow = n_A, ncol = p)
  XB <- matrix(rnorm(n_B * p), nrow = n_B, ncol = p)
  
  expect_error(
    res <- relative_eigen(XA, XB, ncomp = 2),
    NA  # Expect no error due to regularization
  )
  
  # Eigenvalues should be finite
  expect_true(all(is.finite(res$values)))
})

# Test 8: Relative Eigenanalysis respects pre-processing parameters
test_that("Relative Eigenanalysis respects pre-processing parameters", {
  p <- 3
  n_A <- 100
  n_B <- 100
  XA <- matrix(rnorm(n_A * p, mean = 5, sd = 2), nrow = n_A, ncol = p)
  XB <- matrix(rnorm(n_B * p, mean = -3, sd = 4), nrow = n_B, ncol = p)
  
  res_centered <- relative_eigen(XA, XB, ncomp = p, preproc = center())
  res_not_centered <- relative_eigen(XA, XB, ncomp = p, preproc = pass())
  
  # Eigenvalues should differ when not centered
  expect_false(all(abs(res_centered$values - res_not_centered$values) < 1e-3))
  
  # Now test scaling
  res_standardized <- relative_eigen(XA, XB, ncomp = p, preproc = standardize())
  res_not_standardized <- relative_eigen(XA, XB, ncomp = p, preproc = center())
  
  # Eigenvalues should differ when standardized
  expect_false(all(abs(res_standardized$values - res_not_standardized$values) < 1e-3))
})

# Test 9: Relative Eigenanalysis handles non-square data matrices gracefully
test_that("Relative Eigenanalysis handles non-square data matrices gracefully", {
  p <- 100
  n_A <- 100
  n_B <- 120
  XA <- matrix(rnorm(n_A * p), nrow = n_A, ncol = p)
  XB <- matrix(rnorm(n_B * p), nrow = n_B, ncol = p)
  
  res <- relative_eigen(XA, XB, ncomp = 10)
  
  expect_true(length(res$values) == 10)
  expect_true(ncol(res$v) == 10)
  expect_true(nrow(res$v) == p)
})

# Test 10: Relative Eigenanalysis throws error for mismatched column numbers
test_that("Relative Eigenanalysis throws error for mismatched column numbers", {
  p_A <- 100
  p_B <- 101
  n_A <- 100
  n_B <- 100
  XA <- matrix(rnorm(n_A * p_A), nrow = n_A, ncol = p_A)
  XB <- matrix(rnorm(n_B * p_B), nrow = n_B, ncol = p_B)
  
  expect_error(
    res <- relative_eigen(XA, XB, ncomp = 5),
    "XA and XB must have the same number of columns"
  )
})