library(testthat)
library(multivarious)

test_that("bi_projector_union concatenates bi_projector instances", {
  X1 <- matrix(rnorm(5 * 5), 5, 5)
  X2 <- matrix(rnorm(5 * 5), 5, 5)
  
  fit1 <- pca(X1)
  fit2 <- pca(X2)
  
  combined_fit <- bi_projector_union(list(fit1, fit2))
  
  expect_equal(dim(combined_fit$v), c(5, ncomp(fit1) + ncomp(fit2)))
  expect_equal(dim(combined_fit$s), c(5, ncomp(fit1) + ncomp(fit2)))
  expect_equal(length(combined_fit$sdev), ncomp(fit1) + ncomp(fit2))
  expect_s3_class(combined_fit, "bi_projector_union")
})

test_that("bi_projector_union with custom outer_block_indices", {
  X1 <- matrix(rnorm(5 * 5), 5, 5)
  X2 <- matrix(rnorm(5 * 5), 5, 5)
  
  fit1 <- pca(X1)
  fit2 <- pca(X2)
  
  outer_block_indices <- list(1:5, 6:10)
  
  combined_fit <- bi_projector_union(list(fit1, fit2), outer_block_indices = outer_block_indices)
  
  expect_equal(dim(combined_fit$v), c(5, ncomp(fit1) + ncomp(fit2)))
  expect_equal(dim(combined_fit$s), c(5, ncomp(fit1) + ncomp(fit2)))
  expect_equal(length(combined_fit$sdev), ncomp(fit1) + ncomp(fit2))
  expect_equal(combined_fit$outer_block_indices, outer_block_indices)
  expect_s3_class(combined_fit, "bi_projector_union")
})

test_that("bi_projector_union fails with non-bi_projector instances", {
  X1 <- matrix(rnorm(5 * 5), 5, 5)
  X2 <- matrix(rnorm(5 * 5), 5, 5)
  
  fit1 <- pca(X1)
  non_fit <- list(v = matrix(rnorm(5 * 5), 5, 5), s = matrix(rnorm(5 * 5), 5, 5))
  
  expect_error(bi_projector_union(list(fit1, non_fit)))
})

test_that("bi_projector_union fails with incorrect outer_block_indices length", {
  X1 <- matrix(rnorm(5 * 5), 5, 5)
  X2 <- matrix(rnorm(5 * 5), 5, 5)
  
  fit1 <- pca(X1)
  fit2 <- pca(X2)
  
  outer_block_indices <- list(1:4, 5:9) # Incorrect length
  
  expect_error(bi_projector_union(list(fit1, fit2), outer_block_indices = outer_block_indices))
})
