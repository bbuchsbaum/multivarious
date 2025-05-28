# tests/testthat/test_cross_projector.R
context("cross_projector core operators")

library(multivarious)
library(MASS)          # for ginv()

set.seed(1)

# helper that makes an *arbitrary but exact* latent relationship
make_blocks <- function(n  = 40L,
                        pX = 6L,
                        pY = 5L,
                        d  = 3L) {

  Vx <- qr.Q(qr(matrix(rnorm(pX * d), pX, d)))          # orthonormal pX x d
  Vy <- qr.Q(qr(matrix(rnorm(pY * d), pY, d)))          # orthonormal pY x d
  F  <- matrix(rnorm(n * d), n, d)                      # latent scores

  list(
    X  = F %*% t(Vx),                                   # exact factor model
    Y  = F %*% t(Vy),
    Vx = Vx,
    Vy = Vy,
    F  = F
  )
}

# Non-orthonormal variant for ridge tests
make_blocks_nonortho <- function(n = 40L,
                                 pX = 6L,
                                 pY = 5L,
                                 d  = 3L) {
  Vx <- matrix(rnorm(pX * d), pX, d)         # not orthonormal
  Vy <- matrix(rnorm(pY * d), pY, d)
  F  <- matrix(rnorm(n * d), n, d)

  list(
    X  = F %*% t(Vx),
    Y  = F %*% t(Vy),
    Vx = Vx,
    Vy = Vy,
    F  = F
  )
}


# =========================================================================
# 1. ‑‑ Constructor + accessor sanity -------------------------------------
# =========================================================================
test_that("cross_projector stores and reports correct shapes & coefficients", {

  dims <- make_blocks()
  preproc <- prep(pass())
  Xp <- init_transform(preproc, dims$X)
  cp   <- cross_projector(dims$Vx, dims$Vy, preproc_x = preproc, preproc_y = preproc)

  ## classes
  expect_s3_class(cp, c("cross_projector", "projector"))

  ## shape() respects block argument
  expect_equal(shape(cp, "X"), c(nrow(dims$Vx), ncol(dims$Vx)))
  expect_equal(shape(cp, "Y"), c(nrow(dims$Vy), ncol(dims$Vy)))

  ## coef() returns identical matrices
  expect_equal(coef(cp, "X"), dims$Vx)
  expect_equal(coef(cp, "Y"), dims$Vy)
})


# =========================================================================
# 2. ‑‑ project() and partial_project() -----------------------------------
# =========================================================================
test_that("project and partial_project yield expected factor scores", {

  dims <- make_blocks()
  preproc <- prep(pass())
  Xp <- init_transform(preproc, dims$X)
  cp   <- cross_projector(dims$Vx, dims$Vy, preproc_x = preproc, preproc_y = preproc)

  ## ----------   full projection on X   ----------
  F_hat <- project(cp, dims$X, source = "X")            # n x d
  expect_equal(dim(F_hat), c(nrow(dims$X), ncol(dims$Vx)))
  expect_true(max(abs(F_hat - dims$F)) < 1e-12)         # exact because Vx orthonormal

  ## ----------   vector input automatically reshapes   ----------
  one_vec <- dims$X[1, ]
  sc1     <- project(cp, one_vec, source = "X")          # 1 x d
  expect_equal(as.numeric(sc1), as.numeric(F_hat[1, ]))

  ## ----------   partial projection with LS inverse   ----------
  cols      <- c(2, 4, 5)                                # pick 3 out of 6 features
  X_sub     <- dims$X[, cols, drop = FALSE]
  v_sub     <- dims$Vx[cols, , drop = FALSE]
  ls_expect <- X_sub %*% v_sub %*% ginv(crossprod(v_sub))  # analytical expectation

  sc_sub <- partial_project(cp, X_sub, colind = cols,
                            source = "X", least_squares = TRUE)

  expect_equal(sc_sub, ls_expect, tolerance = 1e-3)
})


# =========================================================================
# 3. ‑‑ transfer() reconstructs other block with low error ----------------
# =========================================================================
test_that("transfer converts X‑>Y (and Y‑>X) with low reconstruction error", {

  dims <- make_blocks()
  preproc <- prep(pass())
  Xp <- init_transform(preproc, dims$X)
  cp   <- cross_projector(dims$Vx, dims$Vy, preproc_x = preproc, preproc_y = preproc)

  ## X  →  latent  →  Y
  Y_hat <- transfer(cp, dims$X, from = "X", to = "Y",
                    opts = list(ls_rr = TRUE))

  mse_xy <- mean((Y_hat - dims$Y)^2)
  expect_lt(mse_xy, 1e-10)                               # essentially exact

  ## Y  →  latent  →  X
  X_hat <- transfer(cp, dims$Y, from = "Y", to = "X",
                    opts = list(ls_rr = TRUE))

  mse_yx <- mean((X_hat - dims$X)^2)
  expect_lt(mse_yx, 1e-4)
})


# -------------------------------------------------------------------------
# 4. -- transfer with ridge uses lambda argument --------------------------
# -------------------------------------------------------------------------
test_that("transfer uses ridge regularized projection when opts$ls_rr", {
  dims <- make_blocks_nonortho()
  preproc <- prep(pass())
  cp   <- cross_projector(dims$Vx, dims$Vy, preproc_x = preproc, preproc_y = preproc)
  lambda <- 0.5

  manual_scores <- dims$X %*% dims$Vx %*%
    multivarious:::robust_inv_vTv(dims$Vx, lambda = lambda)
  inv_y <- MASS::ginv(dims$Vy)
  manual <- manual_scores %*% inv_y

  Y_hat <- transfer(cp, dims$X, from = "X", to = "Y",
                    opts = list(ls_rr = TRUE, lambda = lambda))

  expect_equal(Y_hat, manual, tolerance = 1e-6)
})

# =========================================================================
# 4. -- reprocess() validates column indices --------------------------------
# =========================================================================
test_that("reprocess.cross_projector validates column indices", {

  dims <- make_blocks()
  preproc <- prep(pass())
  Xp <- init_transform(preproc, dims$X)
  cp   <- cross_projector(dims$Vx, dims$Vy, preproc_x = preproc, preproc_y = preproc)

  # Valid subset works
  expect_silent(reprocess(cp, dims$X[, 1:2], colind = c(1, 2), source = "X"))

  # Invalid index should error
  expect_error(reprocess(cp, dims$X[, 1:2], colind = c(1, 10), source = "X"))
})

