test_that("create a projector and project new data", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  s <- svd(mat1)
  proj <- projector(s$v)
  p <- project(proj, mat1)
  expect_equal(nrow(p), nrow(mat1))
  expect_equal(ncomp(proj), length(s$d))
  
  vec <- mat1[1,]
  p <- project(proj, vec)
  expect_equal(nrow(p), 1)
  expect_equal(ncol(p), length(s$d))
})

test_that("can partially project data onto subspace", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  s <- svd(mat1)
  proj <- projector(s$v)
  p <- partial_project(proj, mat1[,1:5], 1:5)
  expect_equal(nrow(p), nrow(mat1))
  expect_equal(ncomp(proj), length(s$d))
  
  vec <- mat1[1,1:5]
  p <- partial_project(proj, vec, 1:5)
  expect_equal(nrow(p), 1)
  expect_equal(ncol(p), length(s$d))
  
})

test_that("can test for orthogonality", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  s <- svd(mat1)
  proj <- projector(s$v)

  expect_true(is_orthogonal(proj))
  
  proj2 <- projector(mat1)
  expect_false(is_orthogonal(proj2))
  
})


test_that("can compose two projectors", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  s <- svd(mat1)
  proj1 <- projector(s$v)
  proj2 <- projector(s$v[1:10,1:3])
  proj3 <- compose_projectors(proj1,proj2)
})


