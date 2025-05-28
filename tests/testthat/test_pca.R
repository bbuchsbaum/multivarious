test_that("can run a simple pca analysis", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1)
  
  proj <- project(pres, mat1)
  s <- scores(pres)
  
  expect_equal(proj ,s)
  expect_equal(sdev(pres)[1:length(pres$d)], svd(scale(mat1,center=TRUE, scale=FALSE))$d[1:length(pres$d)])
})

test_that("can project variables using pca result", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1)
  
  pv <- project_vars(pres, mat1)
  expect_equal(pv * (nrow(mat1) - 1), coefficients(pres), tolerance = 1e-6)
})

test_that("can reconstruct a PCA and recover X", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, preproc = pass())
  
  recon <- reconstruct(pres)
  expect_equal(recon, mat1, tolerance = 1e-3, ignore_attr = TRUE)
})

test_that("can reconstruct a PCA and recover X after centering", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, preproc = center())
  
  recon <- reconstruct(pres)
  expect_equal(recon, mat1, tolerance = 1e-3, ignore_attr = TRUE)
})

test_that("can reconstruct a PCA and recover X after standardizing", {
  mat1 <- matrix(rnorm(10*15, mean=5, sd=3), 10, 15) # Add mean/sd variation
  # Ensure no zero variance columns
  mat1[,1] <- mat1[,1] + 1:10 
  pres <- pca(mat1, preproc = standardize())
  
  recon <- reconstruct(pres)
  expect_equal(recon, mat1, tolerance = 1e-3, ignore_attr = TRUE)
})

test_that("can reconstruct a PCA and recover X with pass() preproc", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, preproc = pass())
  
  recon <- reconstruct(pres)
  expect_equal(recon, mat1, tolerance = 1e-3, ignore_attr = TRUE)
})

test_that("can reconstruct a PCA and recover X after colscale(type='z')", {
  mat1 <- matrix(rnorm(10*15, mean=5, sd=3), 10, 15) # Add mean/sd variation
  # Ensure no zero variance columns
  mat1[,1] <- mat1[,1] + 1:10 
  
  # colscale(type='z') scales by 1/sd, does not center
  pres <- pca(mat1, preproc = colscale(type='z'))
  
  recon <- reconstruct(pres)
  expect_equal(recon, mat1, tolerance = 1e-3, ignore_attr = TRUE)
})

test_that("can reconstruct a PCA and recover X after colscale(type='unit')", {
  mat1 <- matrix(rnorm(10*15, mean=2, sd=2), 10, 15) # Add mean/sd variation
  # Ensure no zero variance columns
  mat1[,1] <- mat1[,1] + 1:10 
  
  # colscale(type='unit') scales columns to unit variance
  pres <- pca(mat1, preproc = colscale(type='unit'))
  
  recon <- reconstruct(pres)
  expect_equal(recon, mat1, tolerance = 1e-3, ignore_attr = TRUE)
})

test_that("can compute pca residuals", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, ncomp=2)
  
  mat1_centered <- scale(mat1, center=TRUE, scale=FALSE)
  resid_vals <- residuals(pres, ncomp=2, xorig=mat1_centered)
  expect_true(all.equal(dim(resid_vals), dim(mat1)))
})

test_that("can truncate a pca", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, ncomp=4)
  pres2 <- truncate(pres, 2)

  expect_true(ncomp(pres2) == 2)
  expect_equal(length(pres2$sdev), 2)
  expect_equal(ncol(coef(pres2)), 2)
  expect_equal(ncol(scores(pres2)), 2)
})

test_that("can compute permutation test statistics", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, ncomp=4)
  ptest <- perm_test(pres, mat1, nperm = 50, parallel = FALSE)
  expect_s3_class(ptest, "perm_test")
  expect_s3_class(ptest, "perm_test_pca")
  expect_true(!is.null(ptest$component_results))
  expect_true(nrow(ptest$component_results) > 0)
})

