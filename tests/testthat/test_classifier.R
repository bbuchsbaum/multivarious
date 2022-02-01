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