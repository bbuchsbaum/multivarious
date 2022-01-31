context("pre-processing")
library(magrittr)
library(testthat)

test_that("can preprocess a matrix no center, no scale", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pp <- pass() %>% prep()
  X <- pp$init(mat1)
  x2 <- pp$reverse_transform(X)
  expect_equal(mat1,x2)
  expect_equal(X, mat1)
})

test_that("can preprocess a matrix center only", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pp <- center() %>% prep()
  Xp <- pp$init(mat1)
  x2 <- pp$reverse_transform(Xp)
  expect_equal(mat1,x2)
  expect_true(all(mat1 != Xp))
})

test_that("can apply a centering transform", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pp <- center()
  x <- prep(pp)
  x2 <- init_transform(x,mat1)
  x3 <- apply_transform(x,mat1)
  expect_equal(x2,x3)
})

test_that("can apply a scaling transform", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pp <- standardize()
  x <- prep(pp)
  x2 <- init_transform(x, mat1)
  x3 <- apply_transform(x, mat1)
  expect_equal(x2,x3)
 
})

test_that("can preprocess a matrix with column scaling", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  wts <- 2:16
  pp <- colscale(type="weights", weights=wts)
  x <- prep(pp)
  xinit <- init_transform(x, mat1)
  xrev <- reverse_transform(x, xinit)
  expect_equal(mat1,xrev)
})

# test_that("can reset a prepper with `fresh`", {
#   mat1 <- matrix(rnorm(10*15), 10, 15)
#   pp <- center()
#   x <- prep(pp, mat1)
#   
# })



test_that("can reverse transform a matrix after standardization", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pp <- standardize()
  x <- prep(pp)
  x1 <- init_transform(x,mat1)
  x2 <- reverse_transform(x, x1)
  expect_equal(mat1,x2)
})



test_that("can compose two pre-processors", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  x <- center() %>% colscale(type="z") %>% prep()
  
  x1 <- init_transform(x,mat1)
  x2 <- reverse_transform(x, x1)
  expect_equal(mat1,x2)

})



test_that("can preprocess a matrix with a colind", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pp <- center() %>% prep()
  
  x <- init_transform(pp,mat1)
  ret <- apply_transform(pp, mat1[,1:2], colind=1:2)
  
  expect_equal(ret, x[,1:2])
})

test_that("can concatenate two pre-processors", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  mat2 <- matrix(rnorm(10*15), 10, 15)
  p <- center()
  
  proclist <- lapply(1:2, function(i) {
    fresh(p) %>% prep()
  })
  
  m1 <- init_transform(proclist[[1]], mat1)
  m2 <- init_transform(proclist[[2]], mat2)
  proc <- concat_pre_processors(proclist, list(1:15, 16:30))
  
  a1 <- apply_transform(proc, cbind(mat1,mat2))
  a2 <- apply_transform(proc, mat1, colind=1:15)
  a3 <- apply_transform(proc, mat2, colind=16:30)
  
  pres<- pca(cbind(mat1,mat2), ncomp=12)
  proj <- multiblock_biprojector(pres$v, s=pres$s, sdev=pres$sdev, proc, block_indices=list(1:15, 16:30))
  p1 <- project_block(proj, m1, 1)
  p2 <- project_block(proj, m2, 2)
  
  expect_true(!is.null(p1))
  expect_true(!is.null(p2))
  
  proc$transform(mat1, colind=1:15)
  proc$transform(mat2, colind=16:30)
  
  proj <- multiblock_projector(pres$v, proc, block_indices=list(1:15, 16:30))
  p1 <- project_block(proj, m1, 1)
  p2 <- project_block(proj, m2, 2)
  
  expect_true(!is.null(p1))
  expect_true(!is.null(p2))
  
})


# 
# test_that("can preprocess a block projector", {
#   mat1 <- matrix(rnorm(10*15), 10, 15)
#   mat2 <-  matrix(rnorm(10*10), 10, 10)
#   pca1 <- pca(mat1, ncomp=4)
#   pca2 <- pca(mat2, ncomp=2)
#   
#   bm <- block_projector(list(pca1,pca2))
#   pp <- pre_processor(bm,center=FALSE, scale=FALSE)
#   pdat <- pre_process(pp)
#   expect_equal(ncol(pdat), 6)
#   expect_equal(project(bm), pdat)
# })
# 
# test_that("can preprocess a block projector with newdata", {
#   mat1 <- matrix(rnorm(10*15), 10, 15)
#   mat2 <-  matrix(rnorm(10*10), 10, 10)
#   pca1 <- pca(mat1, ncomp=4)
#   pca2 <- pca(mat2, ncomp=2)
#   
#   bm <- block_projector(list(pca1,pca2))
#   pp <- pre_processor(bm,center=FALSE, scale=FALSE)
#   
#   mat3 <- cbind(mat1,mat2)
#   pdat <- pre_process(pp,mat3)
#   
#   expect_equal(ncol(pdat), 6)
#   expect_equal(project(bm), pdat)
# })
# 
# test_that("can preprocess a block projector with newdata from a sub-block", {
#   mat1 <- matrix(rnorm(10*15), 10, 15)
#   mat2 <-  matrix(rnorm(10*10), 10, 10)
#   pca1 <- pca(mat1, ncomp=4)
#   pca2 <- pca(mat2, ncomp=2)
#   
#   bm <- block_projector(list(pca1,pca2))
#   pp <- pre_processor(bm,center=FALSE, scale=FALSE)
#   
#   mat3 <- cbind(mat2)
#   pdat <- pre_process(pp,mat3, block_index=2)
#   
#   expect_equivalent(project(bm, block_index=2), unclass(pdat))
# })




