test_that("can run a simple svd with 1 component", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- svd_wrapper(mat1, 1)
  
  expect_equal(ncomp(pres),1)
})

test_that("can run a simple svd with 1 rsvd", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- svd_wrapper(mat1, 1, method="rsvd")
  
  expect_equal(ncomp(pres),1)
})
