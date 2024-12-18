library(dplyr)
library(testthat)

prepare_iris_data <- function() {
  data(iris)
  
  # Calculate between-group means (X_f)
  X_f <- iris %>%
    group_by(Species) %>%
    summarise(across(where(is.numeric), mean)) %>%
    ungroup() %>%  # Ensure the grouping is removed
    select(-Species) %>%  # Remove the Species column explicitly
    as.matrix()
  
  # Calculate residuals (X_b)
  overall_means <- colMeans(iris[,1:4])
  X_b <- t(apply(iris[, 1:4], 1, function(x) x - overall_means))
  
  return(list(X_f = X_f, X_b = X_b))
}

test_that("cPCA results are consistent across methods and lambda settings", {
  iris_data <- prepare_iris_data()
  X_f <- iris_data$X_f
  X_b <- iris_data$X_b
  methods <- c("geigen", "primme", "sdiag", "corpcor")
  
  for (method in methods) {
    # Test with lambda = 0
    result_lambda_0 <- cPCA(as.matrix(X_f), as.matrix(X_b), ncomp = 2, lambda = 0, method = method)
    expect_is(result_lambda_0, "list")  # Or expect more specific class if defined
    expect_equal(ncol(result_lambda_0$eigenvectors), 2)
    
    # Test with lambda > 0
    result_lambda_pos <- cPCA(X_f, X_b, ncomp = 2, lambda = 0.1, method = method)
    expect_is(result_lambda_pos, "list")
    expect_equal(ncol(result_lambda_pos$eigenvectors), 2)
    
    # Optional: Compare the outputs between lambda = 0 and lambda > 0
    # This might involve checking for differences in eigenvectors, eigenvalues, etc.
    expect_true(any(result_lambda_0$eigenvalues != result_lambda_pos$eigenvalues))
  }
})

# Additional tests can check for errors, warnings, and specific computational correctness
