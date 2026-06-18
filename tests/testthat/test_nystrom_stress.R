rbf_kernel <- function(X, Y, gamma = 0.25) {
  d2 <- outer(rowSums(X^2), rowSums(Y^2), "+") - 2 * tcrossprod(X, Y)
  exp(-gamma * pmax(d2, 0))
}

explicit_nystrom_kernel <- function(X_proc, landmarks, kernel_func, ncomp = NULL, ...) {
  X_l <- X_proc[landmarks, , drop = FALSE]
  C <- kernel_func(X_proc, X_l, ...)
  W <- C[landmarks, , drop = FALSE]
  eig <- eigen(W, symmetric = TRUE)
  eps <- max(eig$values) * .Machine$double.eps * 100
  keep <- which(eig$values > eps)
  if (!is.null(ncomp)) {
    keep <- keep[seq_len(min(ncomp, length(keep)))]
  }
  U <- eig$vectors[, keep, drop = FALSE]
  lambda <- eig$values[keep]
  C %*% U %*% diag(1 / lambda, nrow = length(lambda)) %*% t(U) %*% t(C)
}

test_that("partial-landmark standard Nyström reconstructs the explicit kernel approximation", {
  set.seed(5001)
  N <- 24
  p <- 12
  X <- matrix(rnorm(N * p), nrow = N, ncol = p)
  landmarks <- c(2, 5, 9, 13, 17, 22)

  fit <- nystrom_approx(
    X,
    ncomp = length(landmarks),
    landmarks = landmarks,
    preproc = pass(),
    method = "standard",
    use_RSpectra = FALSE
  )

  X_proc <- transform(fit$preproc, X)
  K_hat <- explicit_nystrom_kernel(
    X_proc,
    fit$meta$landmarks,
    function(X, Y) X %*% t(Y),
    ncomp = length(landmarks)
  )
  W <- (X_proc %*% t(X_proc[fit$meta$landmarks, , drop = FALSE]))[fit$meta$landmarks, , drop = FALSE]
  lambda_w <- eigen(W, symmetric = TRUE, only.values = TRUE)$values
  lambda_w <- lambda_w[seq_len(length(landmarks))]

  expect_equal(tcrossprod(fit$s), K_hat, tolerance = 1e-8)
  expect_equal(project(fit, X), fit$s, tolerance = 1e-8)
  expect_equal(fit$sdev^2, (N / length(landmarks)) * lambda_w, tolerance = 1e-8)
  expect_true(all(is.finite(fit$v)))
  expect_true(all(is.finite(fit$s)))
  expect_true(all(is.finite(fit$sdev)))
})

test_that("partial-landmark double Nyström with full intermediate rank reconstructs the same explicit approximation", {
  set.seed(5002)
  N <- 22
  p <- 11
  X <- matrix(rnorm(N * p), nrow = N, ncol = p)
  landmarks <- c(1, 4, 8, 12, 16, 21)

  fit <- nystrom_approx(
    X,
    ncomp = length(landmarks),
    landmarks = landmarks,
    preproc = pass(),
    method = "double",
    l = length(landmarks),
    use_RSpectra = FALSE
  )

  X_proc <- transform(fit$preproc, X)
  K_hat <- explicit_nystrom_kernel(
    X_proc,
    fit$meta$landmarks,
    function(X, Y) X %*% t(Y),
    ncomp = length(landmarks)
  )
  lambda_hat <- eigen(K_hat, symmetric = TRUE, only.values = TRUE)$values
  lambda_hat <- lambda_hat[seq_len(length(landmarks))]

  expect_equal(tcrossprod(fit$s), K_hat, tolerance = 1e-8)
  expect_equal(project(fit, X), fit$s, tolerance = 1e-8)
  expect_equal(fit$sdev^2, lambda_hat, tolerance = 1e-8)
  expect_true(all(is.finite(fit$v)))
  expect_true(all(is.finite(fit$s)))
  expect_true(all(is.finite(fit$sdev)))
})

test_that("all-landmark standard Nyström matches exact eigensystem for an RBF kernel", {
  set.seed(5003)
  N <- 18
  p <- 5
  ncomp <- 6
  X <- matrix(rnorm(N * p), nrow = N, ncol = p)
  gamma <- 0.3

  fit <- nystrom_approx(
    X,
    kernel_func = rbf_kernel,
    ncomp = ncomp,
    landmarks = seq_len(N),
    preproc = pass(),
    method = "standard",
    use_RSpectra = FALSE,
    gamma = gamma
  )

  K <- rbf_kernel(X, X, gamma = gamma)
  eig <- eigen(K, symmetric = TRUE)
  expected_lambda <- eig$values[seq_len(ncomp)]
  residual <- K %*% fit$v - fit$v %*% diag(fit$sdev^2, nrow = ncomp)

  expect_equal(fit$sdev^2, expected_lambda, tolerance = 1e-8)
  expect_lt(sqrt(sum(residual^2)) / (sqrt(sum((K %*% fit$v)^2)) + 1e-12), 1e-8)
  expect_equal(project(fit, X), fit$s, tolerance = 1e-8)
  expect_equal(tcrossprod(fit$s), fit$v %*% diag(fit$sdev^2, nrow = ncomp) %*% t(fit$v), tolerance = 1e-10)
})

test_that("landmark order and duplicate supplied indices canonicalize without changing the approximation", {
  set.seed(5004)
  X <- matrix(rnorm(20 * 7), nrow = 20, ncol = 7)
  landmarks_unsorted <- c(12, 3, 8, 3, 17, 1, 12)
  landmarks_unique <- sort(unique(landmarks_unsorted))

  fit_unsorted <- nystrom_approx(
    X,
    ncomp = length(landmarks_unique),
    landmarks = landmarks_unsorted,
    preproc = center(),
    method = "standard",
    use_RSpectra = FALSE
  )
  fit_unique <- nystrom_approx(
    X,
    ncomp = length(landmarks_unique),
    landmarks = landmarks_unique,
    preproc = center(),
    method = "standard",
    use_RSpectra = FALSE
  )

  expect_equal(fit_unsorted$meta$landmarks, landmarks_unique)
  expect_equal(fit_unsorted$sdev, fit_unique$sdev, tolerance = 1e-10)
  expect_equal(tcrossprod(fit_unsorted$s), tcrossprod(fit_unique$s), tolerance = 1e-10)
  expect_equal(project(fit_unsorted, X), fit_unsorted$s, tolerance = 1e-8)
})

test_that("rank-deficient landmark kernels degrade to the estimable rank", {
  X <- matrix(1, nrow = 10, ncol = 3)

  fit_standard <- nystrom_approx(
    X,
    ncomp = 3,
    landmarks = 1:5,
    preproc = pass(),
    method = "standard",
    use_RSpectra = FALSE
  )
  fit_double <- expect_warning(
    nystrom_approx(
      X,
      ncomp = 3,
      landmarks = 1:5,
      preproc = pass(),
      method = "double",
      l = 5,
      use_RSpectra = FALSE
    ),
    "Effective rank after first stage"
  )

  expect_equal(ncomp(fit_standard), 1)
  expect_equal(ncomp(fit_double), 1)
  expect_true(all(is.finite(fit_standard$sdev)))
  expect_true(all(is.finite(fit_double$sdev)))
  expect_equal(project(fit_standard, X), fit_standard$s, tolerance = 1e-8)
  expect_equal(project(fit_double, X), fit_double$s, tolerance = 1e-8)
})

test_that("nystrom_approx rejects non-finite input data before eigen decomposition", {
  X <- matrix(rnorm(30), nrow = 10, ncol = 3)
  X_na <- X
  X_na[1, 1] <- NA_real_
  X_inf <- X
  X_inf[2, 2] <- Inf

  expect_error(
    nystrom_approx(X_na, ncomp = 2, landmarks = 1:5, method = "standard", use_RSpectra = FALSE),
    "`X` must contain only finite values"
  )
  expect_error(
    nystrom_approx(X_inf, ncomp = 2, landmarks = 1:5, method = "standard", use_RSpectra = FALSE),
    "`X` must contain only finite values"
  )

  X_sparse <- Matrix::Matrix(c(1, 0, 2, Inf), nrow = 2, ncol = 2, sparse = TRUE)
  expect_error(
    nystrom_approx(X_sparse, ncomp = 1, landmarks = 1:2, method = "standard", use_RSpectra = FALSE),
    "`X` must contain only finite values"
  )
})
