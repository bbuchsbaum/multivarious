context("Nyström approximation: correctness and projection")

test_that("standard Nyström equals exact kernel eigendecomposition when m = N (linear kernel, centered)", {
  skip_if_not_installed("RSpectra")
  set.seed(123)
  N <- 60
  p <- 15
  X <- matrix(rnorm(N * p), N, p)

  # Build model with all points as landmarks and centered linear kernel
  ncomp <- 5
  fit <- nystrom_approx(
    X,
    ncomp = ncomp,
    landmarks = 1:N,
    preproc = center(),
    method = "standard",
    use_RSpectra = TRUE
  )

  # Recreate centered X used internally
  Xc <- multivarious::transform(fit$preproc, X)
  K <- Xc %*% t(Xc)

  ee <- eigen(K, symmetric = TRUE)
  lam <- ee$values[1:ncomp]
  U   <- ee$vectors[, 1:ncomp, drop = FALSE]

  # Check eigenvalues (sdev^2)
  expect_equal(sort(fit$sdev^2, decreasing = TRUE), sort(lam, decreasing = TRUE), tolerance = 1e-6)

  # Check eigen-equation residual: K v ≈ v diag(λ)
  Kv <- K %*% fit$v
  vLam <- fit$v %*% diag(fit$sdev^2, ncomp)
  resid <- sqrt(sum((Kv - vLam)^2)) / (sqrt(sum(Kv^2)) + 1e-12)
  expect_lt(resid, 1e-6)

  # Check that projecting training points reproduces stored scores
  sc_proj <- project(fit, X)
  expect_equal(sc_proj[, 1:ncomp, drop = FALSE], fit$s[, 1:ncomp, drop = FALSE], tolerance = 1e-6)
})

test_that("double Nyström matches standard when l = m = N (linear kernel, centered)", {
  set.seed(42)
  N <- 40
  p <- 10
  X <- matrix(rnorm(N * p), N, p)
  ncomp <- 4

  fit_std <- nystrom_approx(
    X,
    ncomp = ncomp,
    landmarks = 1:N,
    preproc = center(),
    method = "standard",
    use_RSpectra = FALSE
  )
  fit_dbl <- nystrom_approx(
    X,
    ncomp = ncomp,
    landmarks = 1:N,
    preproc = center(),
    method = "double",
    l = N,
    use_RSpectra = FALSE
  )

  # Compare eigenvalues and eigen-equation residuals
  expect_equal(sort(fit_dbl$sdev^2, decreasing = TRUE), sort(fit_std$sdev^2, decreasing = TRUE), tolerance = 1e-6)

  # Same eigen-equation residual quality
  Xc <- multivarious::transform(fit_std$preproc, X)
  K <- Xc %*% t(Xc)
  Kv_d <- K %*% fit_dbl$v
  vLam_d <- fit_dbl$v %*% diag(fit_dbl$sdev^2, ncomp)
  resid_d <- sqrt(sum((Kv_d - vLam_d)^2)) / (sqrt(sum(Kv_d^2)) + 1e-12)
  expect_lt(resid_d, 1e-6)
})

## TODO: Add an RSpectra-specific double Nyström test once scaling nuances are finalized.

test_that("project() matches manual formulas for standard and double Nyström on new data", {
  set.seed(99)
  N <- 30
  p <- 8
  X <- matrix(rnorm(N * p), N, p)
  ncomp <- 4
  new_idx <- 1:5
  X_new <- X[new_idx, , drop = FALSE]

  # Standard
  fit_std <- nystrom_approx(
    X, ncomp = ncomp, landmarks = 1:N, preproc = center(), method = "standard", use_RSpectra = FALSE
  )
  # Manual projection
  X_l <- fit_std$meta$X_landmarks
  K_new_landmark <- X_new %*% t(X_l) # linear kernel after centering inside reprocess(project)
  # Use the package's preproc to ensure identical preprocessing
  X_new_p <- reprocess(fit_std, X_new)
  K_new_landmark <- X_new_p %*% t(X_l)
  lambda_mm <- fit_std$meta$lambda_mm
  U_mm <- fit_std$meta$U_mm
  m <- length(fit_std$meta$landmarks)
  Ntr <- nrow(fit_std$v)
  scaling <- sqrt(m / Ntr)
  proj_vec <- (scaling * fit_std$sdev) / lambda_mm
  proj_mat <- U_mm %*% diag(proj_vec, nrow = length(proj_vec))
  scores_std_manual <- K_new_landmark %*% proj_mat
  scores_std <- project(fit_std, X_new)
  expect_equal(scores_std, scores_std_manual, tolerance = 1e-6)

  # Double
  fit_dbl <- nystrom_approx(
    X, ncomp = ncomp, landmarks = 1:N, preproc = center(), method = "double", l = N, use_RSpectra = FALSE
  )
  V_S_l <- fit_dbl$meta$V_S_l
  inv_sqrt_lambda_l <- fit_dbl$meta$inv_sqrt_lambda_l
  V_k <- fit_dbl$meta$V_k
  inv_sqrt_lambda_k <- fit_dbl$meta$inv_sqrt_lambda_k
  X_new_p <- reprocess(fit_dbl, X_new)
  K_new_landmark <- X_new_p %*% t(fit_dbl$meta$X_landmarks)
  proj_mat_d <- V_S_l %*% inv_sqrt_lambda_l %*% V_k %*% inv_sqrt_lambda_k
  scores_dbl_manual <- K_new_landmark %*% (proj_mat_d %*% diag(fit_dbl$sdev, nrow = length(fit_dbl$sdev)))
  scores_dbl <- project(fit_dbl, X_new)
  expect_equal(scores_dbl, scores_dbl_manual, tolerance = 1e-6)
})

test_that("reprocess.nystrom_approx rejects wrong column counts", {
  set.seed(202)
  X <- matrix(rnorm(40), 10, 4)
  fit <- nystrom_approx(X, ncomp = 2, landmarks = 1:10, preproc = pass(), method = "standard")
  bad <- matrix(rnorm(15), 5, 3)
  expect_error(reprocess(fit, bad))
})

## Landmark validation behavior depends on package build; not asserting here.
