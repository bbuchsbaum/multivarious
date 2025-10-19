context("geneig backends consistency")

test_that("geneig methods agree on symmetric SPD (largest eigenpairs)", {
  skip_if_not_installed("Matrix")
  set.seed(42)
  n <- 30; k <- 5
  A0 <- matrix(rnorm(n*n), n, n)
  B0 <- matrix(rnorm(n*n), n, n)
  A  <- crossprod(A0) + diag(n) * 1e-3     # symmetric PD-ish
  B  <- crossprod(B0) + diag(n) * 1e-3     # symmetric PD

  base <- geneig(A, B, ncomp = k, method = "sdiag", preproc = NULL)
  rob  <- geneig(A, B, ncomp = k, method = "robust", preproc = NULL)
  sub  <- geneig(A, B, ncomp = k, method = "subspace", which = "LA",
                 preproc = NULL, tol = 1e-8, max_iter = 200)

  # Eigenvalues close
  expect_equal(rob$values, base$values, tolerance = 1e-6)
  expect_equal(sub$values, base$values, tolerance = 1e-5)

  # B-orthonormality
  M_base <- t(base$vectors) %*% B %*% base$vectors
  M_rob  <- t(rob$vectors)  %*% B %*% rob$vectors
  M_sub  <- t(sub$vectors)  %*% B %*% sub$vectors
  expect_lt(max(abs(M_base - diag(k))), 1e-6)
  expect_lt(max(abs(M_rob  - diag(k))), 1e-6)
  expect_lt(max(abs(M_sub  - diag(k))), 1e-6)

  # Subspace agreement (principal angles near 0)
  # Align signs by B-inner product, then compare columns
  align_col <- function(V1, V2, B) {
    for (j in seq_len(ncol(V2))) {
      sgn <- sign(drop(t(V1[,j,drop=FALSE]) %*% B %*% V2[,j,drop=FALSE]))
      if (!is.finite(sgn) || sgn == 0) sgn <- 1
      V2[,j] <- V2[,j] * sgn
    }
    V2
  }
  rob_aligned <- align_col(base$vectors, rob$vectors, B)
  sub_aligned <- align_col(base$vectors, sub$vectors, B)
  expect_lt(max(abs(base$vectors - rob_aligned)), 1e-3)
  expect_lt(max(abs(base$vectors - sub_aligned)), 2e-3)

  # geigen backend (if available)
  if (requireNamespace("geigen", quietly = TRUE)) {
    gei <- geneig(A, B, ncomp = k, method = "geigen", preproc = NULL)
    expect_equal(sort(gei$values, decreasing = TRUE), sort(base$values, decreasing = TRUE), tolerance = 1e-5)
  }

  # primme backend (if available)
  if (requireNamespace("PRIMME", quietly = TRUE)) {
    pri <- geneig(A, B, ncomp = k, method = "primme", which = "LA", preproc = NULL)
    expect_equal(sort(pri$values, decreasing = TRUE), sort(base$values, decreasing = TRUE), tolerance = 1e-5)
  }

  # rspectra backend (smallest λ via which = "SA")
  if (requireNamespace("RSpectra", quietly = TRUE)) {
    base_full <- geneig(A, B, ncomp = nrow(A), method = "sdiag", preproc = NULL)
    base_small <- sort(base_full$values, decreasing = FALSE)[1:k]
    rsp <- geneig(A, B, ncomp = k, method = "rspectra", which = "SA", preproc = NULL,
                   opts = list(tol = 1e-10, ncv = max(4 * k, 20)))
    expect_equal(sort(rsp$values, decreasing = FALSE), base_small, tolerance = 2e-2)
    M_rsp <- t(rsp$vectors) %*% B %*% rsp$vectors
    expect_lt(max(abs(M_rsp - diag(k))), 1e-5)
  }
})

test_that("subspace method recovers smallest eigenpairs on SPD", {
  skip_if_not_installed("Matrix")
  set.seed(19)
  n <- 28; k <- 4
  A0 <- matrix(rnorm(n * n), n, n)
  B0 <- matrix(rnorm(n * n), n, n)
  A  <- crossprod(A0) + diag(n) * 1e-3
  B  <- crossprod(B0) + diag(n) * 1e-3

  base_full <- geneig(A, B, ncomp = n, method = "sdiag", preproc = NULL)
  base_small <- sort(base_full$values, decreasing = FALSE)[1:k]

  sub_sa <- geneig(A, B, ncomp = k, method = "subspace", which = "SA",
                    preproc = NULL, tol = 1e-8, max_iter = 400)

  expect_equal(sort(sub_sa$values), base_small, tolerance = 1e-5)
  M_sub <- t(sub_sa$vectors) %*% B %*% sub_sa$vectors
  expect_lt(max(abs(M_sub - diag(k))), 1e-6)
})


test_that("rspectra which='SA' recovers smallest eigenpairs on SPD", {
  skip_if_not_installed("Matrix")
  skip_if_not_installed("RSpectra")
  set.seed(1)
  n <- 25; k <- 4
  A0 <- matrix(rnorm(n*n), n, n)
  B0 <- matrix(rnorm(n*n), n, n)
  A  <- crossprod(A0) + diag(n) * 1e-3
  B  <- crossprod(B0) + diag(n) * 1e-3

  # Baseline via dense sdiag for the full spectrum
  full <- geneig(A, B, ncomp = n, method = "sdiag", preproc = NULL)
  smallest_baseline <- sort(full$values, decreasing = FALSE)[1:k]

  rsp <- geneig(A, B, ncomp = k, method = "rspectra", which = "SA", preproc = NULL,
                 opts = list(tol = 1e-10, ncv = max(4 * k, 20)))
  expect_equal(sort(rsp$values, decreasing = FALSE), smallest_baseline, tolerance = 1e-2)
})


test_that("rspectra works on sparse SPD and agrees with dense baseline", {
  skip_if_not_installed("Matrix")
  skip_if_not_installed("RSpectra")
  set.seed(7)
  n <- 60; k <- 6
  A <- Matrix::rsparsematrix(n, n, 0.05); A <- Matrix::crossprod(A) + Matrix::Diagonal(n) * 1e-3
  B <- Matrix::rsparsematrix(n, n, 0.05); B <- Matrix::crossprod(B) + Matrix::Diagonal(n) * 1e-3

  base <- geneig(as.matrix(A), as.matrix(B), ncomp = n, method = "sdiag", preproc = NULL)
  base_small <- sort(base$values, decreasing = FALSE)[1:k]
  rsp  <- geneig(A, B, ncomp = k, method = "rspectra", which = "SA", preproc = NULL,
                  opts = list(tol = 1e-10, ncv = max(4 * k, 20)))
  expect_equal(sort(rsp$values, decreasing = FALSE), base_small, tolerance = 1e-2)
  M_rsp <- t(rsp$vectors) %*% B %*% rsp$vectors
  expect_lt(max(abs(M_rsp - diag(k))), 1e-4)
})


test_that("subspace method rejects unsupported which targets", {
  A <- diag(5)
  B <- diag(5)
  expect_error(
    geneig(A, B, ncomp = 2, method = "subspace", which = "LM", preproc = NULL),
    "supports which='LA' or 'SA'",
    fixed = FALSE
  )
})
