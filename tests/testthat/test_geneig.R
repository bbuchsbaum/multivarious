library(testthat)
library(Matrix) # For diagonal matrix operations and checks

# Define a known symmetric matrix A and a positive definite matrix B
A <- matrix(c(4, 1, 1, 2), nrow=2, byrow=TRUE)
B <- matrix(c(6, 2, 2, 5), nrow=2, byrow=TRUE)  # B must be symmetric and positive definite


test_that("geigen method returns correct results", {
  result <- geneig(A, B, ncomp=2, method="geigen")
  expect_equal(dim(result$v), c(2, 2))
  expect_equal(length(result$values), 2)
  expect_true(all(result$values > 0))  # Eigenvalues should be positive for positive definite B
})

test_that("robust method returns correct results", {
  result <- geneig(A, B, ncomp=2, method="robust")
  expect_equal(dim(result$v), c(2, 2))
  expect_equal(length(result$values), 2)
  expect_true(all(result$values > 0))
})

test_that("sdiag method returns correct results", {
  result <- geneig(A, B, ncomp=2, method="sdiag")
  expect_equal(dim(result$v), c(2, 2))
  expect_equal(length(result$values), 2)
  expect_true(all(result$values > 0))
})

test_that("primme method returns correct results", {
  skip_if_not_installed("PRIMME")  # Skip if PRIMME is not available
  result <- geneig(A, B, ncomp=2, method="primme")
  expect_equal(dim(result$vectors), c(2, 2))
  expect_equal(length(result$values), 2)
  expect_true(all(result$values > 0))
})

test_that("non-square matrices are handled", {
  non_square_A <- matrix(1:6, nrow=2)
  non_square_B <- matrix(1:6, nrow=2)
  
  expect_error(geneig(non_square_A, non_square_B, ncomp=2, method="geigen"))
})

test_that("negative and very small eigenvalues in B are handled in sdiag", {
  B_with_negative <- matrix(c(4, 1, 1, -2), nrow=2, byrow=TRUE)
  result <- geneig(A, B_with_negative, ncomp=2, method="sdiag")
  expect_equal(dim(result$v), c(2, 2))
  expect_true(all(result$values > 0))
})