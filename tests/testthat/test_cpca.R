library(testthat)
library(multivarious) # Load the package

# Helper function to prepare Iris data for cPCA context
# X_f: Between-group means (Foreground, 3 samples x 4 features)
# X_b: Original centered data (Background, 150 samples x 4 features)
prepare_iris_data <- function() {
  data(iris)
  X_full <- iris[, 1:4] # Numeric features only

  # Calculate between-group means (X_f)
  species <- iris$Species
  levels_species <- levels(species)
  X_f <- vapply(levels_species, function(sp) {
    colMeans(X_full[species == sp, , drop = FALSE])
  }, FUN.VALUE = numeric(ncol(X_full)))
  X_f <- t(X_f)
  rownames(X_f) <- levels_species

  # Use original data centered by overall mean as X_b (common use case)
  overall_means <- colMeans(X_full)
  # Note: cPCAplus internally centers by background mean if center_background=TRUE
  # Here we just return the raw data for X_b
  X_b <- as.matrix(X_full)

  return(list(X_f = X_f, X_b = X_b, n_f = nrow(X_f), n_b = nrow(X_b), p = ncol(X_f)))
}

test_that("cPCAplus basic structure and methods ('geigen' family)", {
  iris_data <- prepare_iris_data()
  X_f <- iris_data$X_f
  X_b <- iris_data$X_b
  n_components <- 2 # Test fewer components than features/samples

  # Test a geigen family method
  method <- "geigen" # Can also test "primme", "sdiag" if available/desired

  # --- Test with lambda = 0 ---
  result_lambda_0 <- cPCAplus(X_f, X_b, ncomp = n_components, lambda = 0, method = method)

  # Check classes
  expect_s3_class(result_lambda_0, "cPCAplus")
  expect_s3_class(result_lambda_0, "bi_projector")

  # Check dimensions using accessors
  expect_equal(ncomp(result_lambda_0), n_components)
  expect_equal(nrow(coef(result_lambda_0)), iris_data$p) # features x ncomp
  expect_equal(ncol(coef(result_lambda_0)), n_components)
  expect_equal(nrow(scores(result_lambda_0)), iris_data$n_f) # n_f_samples x ncomp
  expect_equal(ncol(scores(result_lambda_0)), n_components)

  # Check values/sdev length
  expect_length(result_lambda_0$values, n_components)
  expect_length(result_lambda_0$sdev, n_components)
  expect_equal(result_lambda_0$sdev, sqrt(pmax(result_lambda_0$values, 0)))

  # Check preprocessor class (use correct name)
  expect_s3_class(result_lambda_0$preproc, "pre_processor")

  # Check stored method info
  expect_equal(result_lambda_0$method_used$method, method)
  expect_equal(result_lambda_0$method_used$ncomp, n_components)

  # --- Test with lambda > 0 ---
  result_lambda_pos <- cPCAplus(X_f, X_b, ncomp = n_components, lambda = 0.1, method = method)

  # Basic structure checks again
  expect_s3_class(result_lambda_pos, "cPCAplus")
  expect_equal(ncomp(result_lambda_pos), n_components)
  expect_length(result_lambda_pos$values, n_components)

  # Check if results differ due to lambda (eigenvalues should change)
  # Use tolerance due to potential numerical noise
  expect_false(isTRUE(all.equal(result_lambda_0$values, result_lambda_pos$values)))

  # --- Test Strategy Force ---
  # Force "feature" strategy (should work for small iris data)
  result_feat_strat <- cPCAplus(X_f, X_b, ncomp = n_components, lambda = 0, method = method, strategy = "feature")
  expect_equal(result_feat_strat$method_used$strategy, "feature")
  # Compare with auto strategy result (should be identical here as auto chooses feature)
  expect_equal(coef(result_lambda_0), coef(result_feat_strat))
  expect_equal(result_lambda_0$values, result_feat_strat$values)

})

test_that("cPCAplus method 'corpcor'", {
  iris_data <- prepare_iris_data()
  X_f <- iris_data$X_f
  X_b <- iris_data$X_b
  n_components <- 2

  # --- Test corpcor method ---
  result_corpcor <- cPCAplus(X_f, X_b, ncomp = n_components, lambda = 0.1, method = "corpcor")

  # Check classes (includes corpcor_pca)
  expect_s3_class(result_corpcor, "cPCAplus")
  expect_s3_class(result_corpcor, "corpcor_pca")
  expect_s3_class(result_corpcor, "bi_projector")

  # Check dimensions
  expect_equal(ncomp(result_corpcor), n_components)
  expect_equal(nrow(coef(result_corpcor)), iris_data$p)
  expect_equal(nrow(scores(result_corpcor)), iris_data$n_f)

  # Check values/sdev length (these are from PCA on whitened data)
  expect_length(result_corpcor$values, n_components)
  expect_length(result_corpcor$sdev, n_components)
  expect_equal(result_corpcor$values, result_corpcor$sdev^2)

  # Check preprocessor
  expect_s3_class(result_corpcor$preproc, "pre_processor")

  # Check stored method info
  expect_equal(result_corpcor$method_used$method, "corpcor")

})

# Test centering option
test_that("cPCAplus center_background = FALSE works", {
  iris_data <- prepare_iris_data()
  # Pre-center data using background mean
  mean_b <- colMeans(iris_data$X_b)
  X_f_cent <- sweep(iris_data$X_f, 2, mean_b, "-")
  X_b_cent <- sweep(iris_data$X_b, 2, mean_b, "-")

  res_no_center <- cPCAplus(X_f_cent, X_b_cent, ncomp = 2, center_background = FALSE, method = "geigen")

  expect_s3_class(res_no_center, "cPCAplus")
  expect_equal(ncomp(res_no_center), 2)
  # Check preprocessor class
  expect_s3_class(res_no_center$preproc, "pre_processor")

  # Compare with default centering (should be numerically close)
  res_center <- cPCAplus(iris_data$X_f, iris_data$X_b, ncomp = 2, center_background = TRUE, method = "geigen")
  expect_equal(res_no_center$values, res_center$values, tolerance = 1e-6)
  # Check cosine similarity of eigenvectors (more robust than direct comparison)
  v1 <- coef(res_no_center)
  v2 <- coef(res_center)
  cos_sim <- abs(diag(crossprod(v1, v2))) # Absolute cosine similarity
  expect_true(all(cos_sim > 0.999)) # Expect vectors to point in same/opposite direction
})


# TODO: Add tests for strategy="sample" if feasible (requires p >> n data)
# TODO: Add tests for edge cases (e.g., ncomp=1, singular Rb)
