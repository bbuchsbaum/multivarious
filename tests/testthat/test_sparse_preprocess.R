context("sparse preprocessing")

sparse_fixture <- function(n = 30, p = 12) {
  set.seed(1001)
  X <- matrix(rnorm(n * p), nrow = n, ncol = p)
  X[sample(length(X), size = floor(0.72 * length(X)))] <- 0
  X[cbind(seq_len(min(n, p)), seq_len(min(n, p)))] <- seq_len(min(n, p)) / 10
  Matrix::Matrix(X, sparse = TRUE)
}

test_that("sparse-safe preprocessing steps preserve sparse storage", {
  skip_if_not_installed("Matrix")
  X_sparse <- sparse_fixture()
  X_dense <- as.matrix(X_sparse)

  pass_res <- fit_transform(pass(), X_sparse)
  expect_true(inherits(pass_res$transformed, "sparseMatrix"))
  expect_equal(as.matrix(pass_res$transformed), X_dense)

  weights <- seq_len(ncol(X_sparse)) / 3
  scale_res <- fit_transform(colscale(type = "weights", weights = weights), X_sparse)
  expect_true(inherits(scale_res$transformed, "sparseMatrix"))
  expect_equal(as.matrix(scale_res$transformed), sweep(X_dense, 2, weights, "*"))
})

test_that("sparse centering uses an affine operator view by default", {
  skip_if_not_installed("Matrix")
  X_sparse <- sparse_fixture()
  X_dense <- as.matrix(X_sparse)
  expected <- sweep(X_dense, 2, colMeans(X_dense), "-")

  res <- fit_transform(center(), X_sparse)
  expect_true(methods::is(res$transformed, "sparse_affine_matrix"))
  expect_equal(as.matrix(res$transformed), expected, tolerance = 1e-12)

  subset_view <- res$transformed[1:5, 2:4, drop = FALSE]
  expect_true(methods::is(subset_view, "sparse_affine_matrix"))
  expect_equal(as.matrix(subset_view), expected[1:5, 2:4, drop = FALSE], tolerance = 1e-12)

  transformed_subset <- multivarious::transform(
    res$preproc,
    X_sparse[, 2:4, drop = FALSE],
    colind = 2:4
  )
  expect_true(methods::is(transformed_subset, "sparse_affine_matrix"))
  expect_equal(as.matrix(transformed_subset), expected[, 2:4, drop = FALSE], tolerance = 1e-12)
})

test_that("sparse densification policy is explicit", {
  skip_if_not_installed("Matrix")
  X_sparse <- sparse_fixture()
  X_dense <- as.matrix(X_sparse)
  expected <- sweep(X_dense, 2, colMeans(X_dense), "-")

  expect_error(
    fit_transform(center(), X_sparse, sparse = "preserve"),
    class = "multivarious_sparse_densify_error"
  )

  dense_res <- fit_transform(center(), X_sparse, sparse = "allow_dense")
  expect_true(is.matrix(dense_res$transformed))
  expect_equal(dense_res$transformed, expected, tolerance = 1e-12)
})

test_that("standardizing sparse matrices stays lazy and matches dense arithmetic", {
  skip_if_not_installed("Matrix")
  X_sparse <- sparse_fixture()
  X_dense <- as.matrix(X_sparse)
  sds <- apply(X_dense, 2, stats::sd)
  expected <- sweep(sweep(X_dense, 2, colMeans(X_dense), "-"), 2, sds, "/")

  res <- fit_transform(standardize(), X_sparse)
  expect_true(methods::is(res$transformed, "sparse_affine_matrix"))
  expect_equal(as.matrix(res$transformed), expected, tolerance = 1e-10)
})

test_that("sparse affine matrix supports public algebra without densifying first", {
  skip_if_not_installed("Matrix")
  X_sparse <- sparse_fixture(n = 18, p = 7)
  Xp <- fit_transform(center(), X_sparse)$transformed
  X_dense <- as.matrix(Xp)

  V <- matrix(rnorm(ncol(Xp) * 3), nrow = ncol(Xp), ncol = 3)
  left <- matrix(rnorm(4 * nrow(Xp)), nrow = 4, ncol = nrow(Xp))
  Y_rows <- matrix(rnorm(nrow(Xp) * 2), nrow = nrow(Xp), ncol = 2)
  Y_cols <- matrix(rnorm(5 * ncol(Xp)), nrow = 5, ncol = ncol(Xp))
  left_t <- matrix(rnorm(4 * ncol(Xp)), nrow = 4, ncol = ncol(Xp))

  expect_equal(Xp %*% V, X_dense %*% V, tolerance = 1e-12)
  expect_equal(left %*% Xp, left %*% X_dense, tolerance = 1e-12)
  expect_equal(crossprod(Xp), crossprod(X_dense), tolerance = 1e-12)
  expect_equal(crossprod(Xp, Y_rows), crossprod(X_dense, Y_rows), tolerance = 1e-12)
  expect_equal(crossprod(Y_rows, Xp), crossprod(Y_rows, X_dense), tolerance = 1e-12)
  expect_equal(tcrossprod(Xp), tcrossprod(X_dense), tolerance = 1e-12)
  expect_equal(tcrossprod(Xp, Y_cols), tcrossprod(X_dense, Y_cols), tolerance = 1e-12)
  expect_equal(tcrossprod(Y_cols, Xp), tcrossprod(Y_cols, X_dense), tolerance = 1e-12)

  Xpt <- t(Xp)
  expect_true(methods::is(Xpt, "sparse_affine_transpose"))
  expect_equal(t(Xpt), Xp)
  expect_equal(Xpt %*% Y_rows, t(X_dense) %*% Y_rows, tolerance = 1e-12)
  expect_equal(left_t %*% Xpt, left_t %*% t(X_dense), tolerance = 1e-12)
})

test_that("sparse-centered PCA uses implicit SVD and projects without densifying preprocessing", {
  skip_if_not_installed("Matrix")
  skip_if_not_installed("irlba")
  X_sparse <- sparse_fixture(n = 36, p = 14)
  X_dense <- as.matrix(X_sparse)

  set.seed(2002)
  sparse_fit <- pca(X_sparse, ncomp = 3, preproc = center(), method = "fast")
  dense_fit <- pca(X_dense, ncomp = 3, preproc = center(), method = "base")

  expect_true(methods::is(multivarious::transform(sparse_fit$preproc, X_sparse), "sparse_affine_matrix"))
  expect_equal(sparse_fit$sdev, dense_fit$sdev[seq_along(sparse_fit$sdev)], tolerance = 1e-5)

  projected <- project(sparse_fit, X_sparse)
  centered_dense <- sweep(X_dense, 2, colMeans(X_dense), "-")
  expect_equal(projected, centered_dense %*% coef(sparse_fit), tolerance = 1e-8)
})

test_that("sparse uncentered PCA stays on sparse storage", {
  skip_if_not_installed("Matrix")
  skip_if_not_installed("irlba")
  X_sparse <- sparse_fixture(n = 36, p = 14)
  X_dense <- as.matrix(X_sparse)

  set.seed(3003)
  sparse_fit <- pca(X_sparse, ncomp = 3, preproc = pass(), method = "fast")
  dense_svd <- svd(X_dense, nu = 3, nv = 3)

  expect_true(inherits(multivarious::transform(sparse_fit$preproc, X_sparse), "sparseMatrix"))
  expect_equal(sparse_fit$sdev, dense_svd$d[seq_along(sparse_fit$sdev)], tolerance = 1e-5)
  expect_equal(project(sparse_fit, X_sparse), X_dense %*% coef(sparse_fit), tolerance = 1e-8)
})
