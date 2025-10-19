library(testthat)

make_spd <- function(n) {
  M <- matrix(rnorm(n * n), n)
  crossprod(M) + diag(n) * 0.5
}

reference_gep <- function(S1, S2) {
  R <- chol(S2)
  R_inv <- solve(R)
  M <- t(R_inv) %*% S1 %*% R_inv
  M <- (M + t(M)) / 2
  eigen(M, symmetric = TRUE)
}

test_that("solve_gep_subspace matches dense reference for largest and smallest", {
  set.seed(123)
  n <- 6
  q <- 3
  S1 <- make_spd(n)
  S2 <- make_spd(n)

  ref <- reference_gep(S1, S2)
  ref_vals <- ref$values
  ref_largest <- sort(ref_vals, decreasing = TRUE)[seq_len(q)]
  ref_smallest <- sort(ref_vals, decreasing = FALSE)[seq_len(q)]

  largest <- multivarious:::solve_gep_subspace(S1, S2, q = q, which = "largest", max_iter = 200, tol = 1e-8, seed = 99)
  expect_true(largest$converged)
  expect_equal(largest$values, ref_largest, tolerance = 1e-5)

  smallest <- multivarious:::solve_gep_subspace(S1, S2, q = q, which = "smallest", max_iter = 200, tol = 1e-8, seed = 99)
  expect_true(smallest$converged)
  expect_equal(smallest$values, ref_smallest, tolerance = 1e-5)
})
