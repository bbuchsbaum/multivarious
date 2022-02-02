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
  expect_equal(pv, coefficients(pres))
})

test_that("can run bootstrap analysis with 100 bootstraps", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1)
  
  bres <- bootstrap(pres, nboot=100)
  expect_true(length(bres) == 4)
})

test_that("can reconstruct a PCA and recover X", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1)
  
  recon <- reconstruct(pres)
  expect_true(all.equal(recon, mat1))
})

test_that("can compute pca residuals", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, ncomp=2)
  
  recon <- residuals(pres, 2, mat1)
  expect_true(all.equal(dim(recon), dim(mat1)))
})


test_that("can truncate a pca", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, ncomp=4)
  pres2 <- truncate(pres, 2)

  expect_true(ncomp(pres2) == 2)
})

test_that("can compute permutation confidence intervals", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, ncomp=4)
  pci <- perm_ci(pres, mat1, 100)
  expect_true(!is.null(pci))
})

