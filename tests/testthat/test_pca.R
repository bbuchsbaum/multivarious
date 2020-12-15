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
  expect_equal(pv, components(pres))
})

test_that("can run bootstrap analysis with 100 bootstraps", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1)
  
  bres <- bootstrap(pres, nboot=100)
  expect_true(length(bres) == 4)
})


