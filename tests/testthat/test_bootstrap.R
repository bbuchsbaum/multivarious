# tests/testthat/test_bootstrap_pca.R
context("bootstrap_pca")

set.seed(123)                                       # reproducible randomness
library(multivarious)

# --- helper to build a tiny pca object quickly ------------------------------
make_toy_pca <- function(n = 25L, p = 12L, noise = .15, k = 2L) {
  s1     <- rnorm(n, sd = 3)                        # strong latent factor
  load1  <- runif(p, -1, 1)
  X      <- outer(s1, load1) + matrix(rnorm(n * p, 0, noise), n, p)
  pca(X, ncomp = k, preproc = center(), method = "fast")
}

# reuse in all tests ---------------------------------------------------------
toy_pca  <- make_toy_pca()
p        <- nrow(toy_pca$v); n <- nrow(toy_pca$s); k <- ncol(toy_pca$v)
nboot    <- 40L                                    # keep tests snappy
boot_res <- bootstrap_pca(toy_pca, nboot = nboot, k = k, seed = 999)

# -------------  1. structural integrity  ------------------------------------
test_that("bootstrap_pca returns object of correct class and shape", {

  expect_s3_class(boot_res, "bootstrap_pca_result")

  # core slots must exist
  expect_true(all(c("E_Vb", "sd_Vb", "z_loadings",
                    "E_Scores", "sd_Scores", "z_scores",
                    "Ab_array", "Scores_array") %in% names(boot_res)))

  # dimensions
  expect_equal(dim(boot_res$E_Vb),       c(p, k))
  expect_equal(dim(boot_res$sd_Vb),      c(p, k))
  expect_equal(dim(boot_res$z_loadings), c(p, k))

  expect_equal(dim(boot_res$E_Scores),   c(n, k))
  expect_equal(dim(boot_res$sd_Scores),  c(n, k))
  expect_equal(dim(boot_res$z_scores),   c(n, k))

  expect_equal(dim(boot_res$Ab_array),     c(k, k, nboot))
  expect_equal(dim(boot_res$Scores_array), c(n, k, nboot))

  # bookkeeping scalars
  expect_equal(boot_res$nboot, nboot)
  expect_equal(boot_res$k,     k)
})

# -------------  2. Z‑score identity check  ----------------------------------
test_that("Z‑scores equal mean divided by SD", {

  # helper that checks element‑wise equality up to 1e‑12 (machine precision wiggle)
  identical_Z <- function(E, SD, Z) {
    delta <- abs(E / SD - Z)
    all(delta[is.finite(delta)] < 1e-12)
  }

  expect_true(identical_Z(boot_res$E_Vb,    boot_res$sd_Vb,    boot_res$z_loadings))
  expect_true(identical_Z(boot_res$E_Scores,boot_res$sd_Scores,boot_res$z_scores))
})

# -------------  3. statistical faithfulness ---------------------------------
test_that("bootstrap means track dominant component and SD hierarchy is sensible", {

  # correlation between original and bootstrap‑mean loadings for PC1
  cor1 <- cor(toy_pca$v[, 1], boot_res$E_Vb[, 1])
  expect_true(abs(cor1) >= 0.9)

  # dominant component should be estimated more precisely
  expect_true(median(boot_res$sd_Vb[, 1]) < median(boot_res$sd_Vb[, 2]))
})

# -------------  4. input validation -----------------------------------------
test_that("nboot must be a positive integer", {
  expect_error(bootstrap_pca(toy_pca, nboot = 0, k = k),
               "nboot must be a positive integer")
  expect_error(bootstrap_pca(toy_pca, nboot = 2.5, k = k),
               "nboot must be a positive integer")
})