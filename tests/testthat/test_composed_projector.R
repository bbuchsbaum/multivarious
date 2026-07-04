context("composed projectors")

make_composed_pca_fixture <- function() {
  set.seed(1)
  X <- matrix(rnorm(30 * 15), nrow = 30, ncol = 15)
  p1 <- pca(X, ncomp = 8)
  p2 <- pca(scores(p1), ncomp = 7)

  list(
    X = X,
    p1 = p1,
    p2 = p2,
    pipe = compose_partial_projector(first = p1, second = p2)
  )
}

test_that("composed projector matches explicit sequential projection", {
  fixture <- make_composed_pca_fixture()

  composed_scores <- project(fixture$pipe, fixture$X)
  sequential_scores <- project(
    fixture$p2,
    project(fixture$p1, fixture$X)
  )

  expect_equal(composed_scores, sequential_scores, tolerance = 1e-8)
  expect_equal(dim(composed_scores), c(30L, 7L))
  expect_equal(dim(coef(fixture$pipe)), c(15L, 7L))
  expect_equal(dim(inverse_projection(fixture$pipe)), c(7L, 15L))
})

test_that("composed partial projection subsets only the first stage", {
  fixture <- make_composed_pca_fixture()
  cols <- 1:5

  partial_scores <- partial_project(
    fixture$pipe,
    fixture$X[, cols, drop = FALSE],
    colind = cols
  )

  expect_equal(dim(partial_scores), c(30L, 7L))
  expect_warning(
    partial_project(
      fixture$pipe,
      fixture$X[, cols, drop = FALSE],
      colind = list(cols, 1:2)
    ),
    "beyond the first stage"
  )
})

test_that("truncate.composed_projector truncates the final stage", {
  fixture <- make_composed_pca_fixture()

  truncated <- truncate(fixture$pipe, ncomp = 4)

  expect_s3_class(truncated, "composed_projector")
  expect_equal(unname(summary(truncated)$out_dim), c(8L, 4L))
  expect_equal(dim(project(truncated, fixture$X)), c(30L, 4L))
  expect_equal(dim(coef(truncated)), c(15L, 4L))
})
