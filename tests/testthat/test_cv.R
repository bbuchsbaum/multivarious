library(testthat)
suppressWarnings(library(tibble)) # Needed for dummy_eval (suppress version mismatch warning)
# Assuming measure_reconstruction_error and cv_generic/cv.bi_projector are loaded
# May need to load the package being tested, e.g., using devtools::load_all()

set.seed(1)

# ------------------------------------------------------------------
# helper: 5x identity "model" guarantees perfect reconstruction
perfect_fit  <- function(train_data, ...)   "identity"
perfect_eval <- function(model, test_data, ...) {
  # Reconstruct identically
  Xrec <- test_data   # perfect
  # Assuming measure_reconstruction_error is available in the test env
  measure_reconstruction_error(test_data, Xrec, metrics = c("mse","r2"))
}

# Helper k-fold function (as defined in the test)
kfold_split <- function(n, k = 5) {
  idx <- sample(rep(1:k, length.out = n))
  lapply(1:k, function(j) list(train = which(idx != j),
                               test  = which(idx == j)))
}

test_that("cv_generic yields zero-error on identity model", {
  X      <- matrix(rnorm(100 * 4), 100, 4)
  folds  <- kfold_split(nrow(X))
  
  # Assuming cv_generic is available in the test env
  res <- cv_generic(
    data          = X,
    folds         = folds,
    .fit_fun      = perfect_fit,
    .measure_fun  = perfect_eval
  )
  
  # every fold must have zero mse and r2 == 1
  metrics <- do.call(rbind, lapply(res$metrics, as.data.frame))
  expect_true(all(metrics$mse < 1e-12))
  expect_true(all(abs(metrics$r2 - 1) < 1e-12))
})

# ------------------------------------------------------------------

error_fit <- function(train_data, ...) {
  stop("boom")             # deliberate failure
}

dummy_eval <- function(model, test_data, ...) {
  tibble::tibble(dummy = 0) # won't be reached
}

test_that("cv_generic gracefully records fit errors", {
  X     <- matrix(rnorm(20), 10, 2)
  folds <- list(list(train = 1:8, test = 9:10))

  # Assuming cv_generic is available (suppress expected warning about fit failure)
  res <- suppressWarnings(cv_generic(X, folds, error_fit, dummy_eval))
  
  expect_true(grepl("Fit failed", res$metrics[[1]]$error[1]))
  expect_null(res$model[[1]])
})

# ------------------------------------------------------------------

# Need prep, center, bi_projector, cv.bi_projector, reconstruct_new, truncate.bi_projector
# Also assumes required dplyr/tidyr functions are available

# test_that("cv.bi_projector produces monotonically improving MSE", {
#   # Requires functions from the package (prep, center, bi_projector, etc.)
#   # Requires dplyr and tidyr
#   skip_if_not_installed("dplyr")
#   skip_if_not_installed("tidyr")
#   
#   set.seed(1)
#   X     <- scale(matrix(rnorm(60 * 6), 60, 6))   # centred, unit-var
#   folds <- list(
#     list(train = 1:40, test = 41:60),
#     list(train = 21:60, test = 1:20)
#   )
#   
#   # Assuming cv.bi_projector is available
#   res <- cv.bi_projector(
#     x            = X,
#     folds        = folds,
#     max_comp     = 4,
#     measure      = "mse",
#     return_models= FALSE
#   )
#   
#   # Unnest and average MSE across folds per comp
#   # Ensure necessary packages are loaded for pipe operators
#   mses <- tidyr::unnest(res$results, component_metrics) |>
#           dplyr::group_by(comp) |>
#           dplyr::summarise(mse = mean(mse, na.rm = TRUE)) |>
#           dplyr::arrange(comp)
#           
#   # Check if 'mse' column exists (might be comp_error if truncation failed)
#   if (!"mse" %in% names(mses)) {
#     fail("MSE column not found in results, check for errors during CV.")
#   }
#   
#   # Expect strictly decreasing (or at least non-increasing) curve
#   # Allow for small tolerance due to numerical precision
#   expect_true(all(diff(mses$mse) <= 1e-8))
# })
