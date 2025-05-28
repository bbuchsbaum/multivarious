test_that("can run a regress analysis with one y variable", {
  mat1 <- matrix(rnorm(100*15), 100, 15)
  y <- rnorm(100)
  reg <- regress(mat1, y, preproc=pass(), method="lm")
  expect_true(!is.null(reg))
  
  recon <- reconstruct(reg)
  expect_true(!is.null(recon))
})

test_that("can run a regress analysis with one y variable and intercept", {
  mat1 <- matrix(rnorm(100*15), 100, 15)
  y <- rnorm(100)
  reg <- regress(mat1, y, preproc=pass(), method="lm", intercept=TRUE)
  expect_true(!is.null(reg))
  
  recon <- reconstruct(reg)
  expect_true(!is.null(recon))
  
  expect_true(ncol(reg$v) == ncol(mat1)+1)
  
})

test_that("can run a regress analysis with multiple y variables", {
  mat1 <- matrix(rnorm(100*15), 100, 15)
  y <- cbind(rnorm(100), rnorm(100), rnorm(100))
  reg <- regress(mat1, y, preproc=pass(), method="lm")
  recon <- reconstruct(reg)
  expect_true(!is.null(reg))
  expect_true(!is.null(recon))
})

test_that("can run a regress analysis with multiple y variables and ridge", {
  mat1 <- matrix(rnorm(100*15), 100, 15)
  y <- cbind(rnorm(100), rnorm(100), rnorm(100))
  reg <- regress(mat1, y, preproc=pass(), method="mridge")
  recon <- reconstruct(reg)
  expect_true(!is.null(reg))
  expect_true(!is.null(recon))
})

test_that("mridge works with single response", {
  mat1 <- matrix(rnorm(100*15), 100, 15)
  y_vec <- rnorm(100)
  reg_vec <- regress(mat1, y_vec, preproc=pass(), method="mridge")
  recon_vec <- reconstruct(reg_vec)
  expect_true(!is.null(reg_vec))
  expect_true(!is.null(recon_vec))

  y_mat1 <- matrix(rnorm(100), 100, 1)
  reg_mat <- regress(mat1, y_mat1, preproc=pass(), method="mridge")
  recon_mat <- reconstruct(reg_mat)
  expect_true(!is.null(reg_mat))
  expect_true(!is.null(recon_mat))
})

test_that("can run a regress analysis with multiple y variables and enet", {
  mat1 <- matrix(rnorm(100*15), 100, 15)
  y <- cbind(rnorm(100), rnorm(100), rnorm(100))
  reg <- regress(mat1, y, preproc=pass(), method="enet", alpha=.3, lambda=.01)
  recon <- reconstruct(reg)
  expect_true(!is.null(reg))
  expect_true(!is.null(recon))
})


test_that("enet works with single response", {
  mat1 <- matrix(rnorm(100*15), 100, 15)
  y_vec <- rnorm(100)
  reg_vec <- regress(mat1, y_vec, preproc=pass(), method="enet", alpha=.3, lambda=.01)
  recon_vec <- reconstruct(reg_vec)
  expect_true(!is.null(reg_vec))
  expect_true(!is.null(recon_vec))

  y_mat1 <- matrix(rnorm(100), 100, 1)
  reg_mat <- regress(mat1, y_mat1, preproc=pass(), method="enet", alpha=.3, lambda=.01)
  recon_mat <- reconstruct(reg_mat)
  expect_true(!is.null(reg_mat))
  expect_true(!is.null(recon_mat))
})

test_that("ridge and enet handle intercept correctly", {
  mat1 <- matrix(rnorm(100*15), 100, 15)
  y <- cbind(rnorm(100), rnorm(100))

  r_ridge <- regress(mat1, y, preproc=pass(), method="mridge", intercept=TRUE)
  r_enet  <- regress(mat1, y, preproc=pass(), method="enet", intercept=TRUE)

  expect_equal(ncol(r_ridge$v), ncol(mat1) + 1)
  expect_equal(ncol(r_enet$v), ncol(mat1) + 1)

})