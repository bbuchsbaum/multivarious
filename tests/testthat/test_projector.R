context("projector")
library(testthat)
library(multivarious)

test_that("can construct a projector object", {
  v <- matrix(rnorm(10*5), 10, 5)
  proj <- projector(v)
  expect_s3_class(proj, "projector")
  expect_equal(ncomp(proj), 5)
  expect_equal(shape(proj), c(10, 5))
})

test_that("can project data onto subspace", {
  mat1 <- matrix(rnorm(10*10), 10, 10)
  v <- matrix(rnorm(10*5), 10, 5)

  preproc <- prep(pass())
  Xp <- init_transform(preproc, mat1)

  proj <- projector(v, preproc=preproc)
  pdat <- project(proj, mat1)
  expect_equal(dim(pdat), c(10, 5))
})

test_that("can partially project data onto subspace", {
  mat1 <- matrix(rnorm(10*10), 10, 10)
  v <- matrix(rnorm(10*5), 10, 5)

  preproc <- prep(pass())
  Xp <- init_transform(preproc, mat1)
  proj <- projector(v, preproc=preproc)
  
  
  pdat <- partial_project(proj, mat1[, 1:5], 1:5) 
  expect_equal(dim(pdat), c(10, 5))
})

test_that("can compute inverse projection", {
  v <- matrix(rnorm(10*5), 10, 5)
  proj <- projector(v)
  inv_proj <- inverse_projection(proj)
  expect_equal(dim(inv_proj), c(5, 10))
})

test_that("can compute partial inverse projection", {
  v <- matrix(rnorm(10*5), 10, 5)
  proj <- projector(v)
  inv_proj <- partial_inverse_projection(proj, 1:5)
  expect_equal(dim(inv_proj), c(5, 10))
})

test_that("can truncate a projector", {
  v <- matrix(rnorm(10*5), 10, 5)
  proj <- projector(v)
  proj_trunc <- truncate(proj, 3)
  expect_equal(ncomp(proj_trunc), 3)
  expect_equal(shape(proj_trunc), c(10, 3))
})

test_that("can create and use a partial projector", {
  v <- matrix(rnorm(10*5), 10, 5)
  proj <- projector(v)
  
  placeholder_orig_data <- matrix(0, nrow=2, ncol=10)
  Xp <- init_transform(proj$preproc, placeholder_orig_data)
  
  partial_proj <- partial_projector(proj, 1:7)
  expect_s3_class(partial_proj, "partial_projector")
  expect_equal(shape(partial_proj), c(7, 5))
  
  mat1_partial <- matrix(rnorm(10*7), 10, 7)
  pdat <- project(partial_proj, mat1_partial)
  expect_equal(dim(pdat), c(10, 5))
})


