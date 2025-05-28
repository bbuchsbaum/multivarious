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

test_that("concat_pre_processors handles complex colind across different block types", {
  set.seed(123) # for reproducibility
  mat1 <- matrix(rnorm(10*5, mean=10), 10, 5)
  mat2 <- matrix(rnorm(10*7, mean=20, sd=5), 10, 7)
  mat3 <- matrix(rnorm(10*3, mean=0, sd=1), 10, 3)
  
  p1 <- center() %>% prep()       # Center only
  p2 <- standardize() %>% prep()  # Center and scale
  p3 <- pass() %>% prep()         # No-op
  
  # Initialize individual processors
  m1_init <- init_transform(p1, mat1)
  m2_init <- init_transform(p2, mat2)
  m3_init <- init_transform(p3, mat3)
  
  proclist <- list(p1, p2, p3)
  block_indices <- list(1:5, 6:12, 13:15)
  
  proc_concat <- concat_pre_processors(proclist, block_indices)
  
  # --- Test apply_transform --- 
  # Select columns: 2nd and 4th from block 1, 2nd and 3rd from block 2, 2nd from block 3
  colind_global <- c(2, 4, 7, 8, 14)
  
  # Create the input matrix corresponding to these global columns
  test_mat_apply <- cbind(mat1[, c(2, 4)], mat2[, c(2, 3)], mat3[, 2, drop=FALSE])
  
  # Apply the concatenated transform
  result_apply <- apply_transform(proc_concat, test_mat_apply, colind = colind_global)
  
  # Manually apply transforms to corresponding original blocks and columns
  manual_res1 <- apply_transform(p1, mat1[, c(2, 4), drop = FALSE], colind = c(2, 4)) # Local indices: 2, 4
  manual_res2 <- apply_transform(p2, mat2[, c(2, 3), drop = FALSE], colind = c(2, 3)) # Local indices: 2, 3
  manual_res3 <- apply_transform(p3, mat3[, 2, drop = FALSE], colind = 2)        # Local index: 2
  
  # Combine manual results in the order defined by colind_global
  expected_apply <- cbind(manual_res1[, 1, drop=FALSE],  # Corresponds to global col 2
                        manual_res1[, 2, drop=FALSE],  # Corresponds to global col 4
                        manual_res2[, 1, drop=FALSE],  # Corresponds to global col 7
                        manual_res2[, 2, drop=FALSE],  # Corresponds to global col 8
                        manual_res3[, 1, drop=FALSE]) # Corresponds to global col 14
  
  expect_equal(result_apply, expected_apply, tolerance = 1e-7)
  
  # --- Test reverse_transform --- 
  # Input for reverse is the result from apply_transform
  test_mat_reverse <- result_apply
  
  # Reverse using concatenated processor
  result_reverse <- reverse_transform(proc_concat, test_mat_reverse, colind = colind_global)
  
  # Manually reverse transforms using the outputs from the manual forward transforms
  manual_rev1 <- reverse_transform(p1, manual_res1, colind = c(2, 4))
  manual_rev2 <- reverse_transform(p2, manual_res2, colind = c(2, 3))
  manual_rev3 <- reverse_transform(p3, manual_res3, colind = 2)
  
  # Combine manual reverse results in the order defined by colind_global
  expected_reverse <- cbind(manual_rev1[, 1, drop=FALSE], 
                            manual_rev1[, 2, drop=FALSE], 
                            manual_rev2[, 1, drop=FALSE], 
                            manual_rev2[, 2, drop=FALSE], 
                            manual_rev3[, 1, drop=FALSE])
                            
  # The result of reverse should match the original input subset 'test_mat_apply'
  expect_equal(result_reverse, test_mat_apply, tolerance = 1e-7)
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




