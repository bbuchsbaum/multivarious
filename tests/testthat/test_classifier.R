testthat::context("classifier")

test_that("can construct a pca classifier", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1)
  y <- rep(letters[1:4], length.out=10)
  cl <- classifier(pres, labels=y, new_data=mat1)
 
  p <- predict(cl, mat1)
  expect_true(ncol(p$prob) == 4)
  expect_true(nrow(p$prob) == 10)
})


test_that("can construct a pca classifier with colind", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1)
  y <- rep(letters[1:4], length.out=10)
  cl <- classifier(pres, labels=y, new_data=mat1[,1:5], colind=1:5)
  
  p <- predict(cl, mat1[,1:5])
  expect_true(ncol(p$prob) == 4)
  expect_true(nrow(p$prob) == 10)
  
  p2 <- predict(cl, mat1[1,1:5])
  expect_true(ncol(p2$prob) == 4)
  expect_true(nrow(p2$prob) == 1)
  
  p2 <- predict(cl, mat1[1,1:5], colind=1:5, metric="euclidean")
  expect_true(ncol(p2$prob) == 4)
  expect_true(nrow(p2$prob) == 1)
  
  p2 <- project(cl, mat1[1,1:5])
  expect_true(ncol(p2) == ncomp(pres))
  
})

test_that("can split a matrix", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  Y <- rep(letters[1:3], length.out=nrow(mat1))
  sm <- split_matrix(mat1,Y)
  expect_equal(3, length(sm))
})

test_that("can compute group means of rows of matrix", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  Y <- rep(letters[1:3], length.out=nrow(mat1))
  gm <- group_means(Y, mat1)
  expect_equal(nrow(gm), 3)
  
  Y <- letters[1:nrow(mat1)]
  gm <- group_means(Y, mat1)
  expect_equal(nrow(gm), length(Y))
})

test_that("can compute principle angles of a set of pca fits", {
  
  fits <- lapply(1:4, function(i) {
    m <- matrix(rnorm(10*15), 10, 15)
    pca(m, ncomp=10)
  })
  
  pa <- prinang(fits)
  expect_true(length(pa) == 10)
  
})


