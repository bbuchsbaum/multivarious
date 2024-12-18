library(testthat)
test_that("Sign-flipping improves consistency for multiple PCA components", {
  set.seed(123)
  
  # Simulate data with two principal components:
  # We create two known directions:
  # dir1: strong primary direction
  # dir2: a secondary direction orthogonal to dir1
  
  N <- 100
  D <- 10
  
  dir1 <- c(1,2,3,rep(0,D-3))
  dir1 <- dir1 / sqrt(sum(dir1^2))
  
  dir2 <- c(0,0,1,1,1,rep(0,D-5))
  dir2 <- dir2 / sqrt(sum(dir2^2))
  
  # Ensure orthogonality (not strictly necessary but good practice)
  # If they are not perfectly orthogonal due to normalization, it's fine for illustration.
  
  # Scores for each direction:
  scores1 <- rnorm(N, mean=0, sd=10)
  scores2 <- rnorm(N, mean=0, sd=5)
  
  # Construct data: X = scores1*dir1 + scores2*dir2 + noise
  X <- (outer(scores1, dir1) + outer(scores2, dir2)) + matrix(rnorm(N*D,0,1), N, D)
  
  # Fit PCA with 2 components
  pca_fit <- pca(X, ncomp=2)
  
  # Bootstrap without sign flipping
  boot_none <- bootstrap(pca_fit, nboot=50, k=2, sign_method="none")
  
  # Bootstrap with max_abs sign flipping
  boot_max_abs <- bootstrap(pca_fit, nboot=50, k=2, sign_method="max_abs")
  
  # We have true_dir1 and true_dir2 as our "ground truth" directions.
  # After PCA and bootstrap, we get zboot_loadings for each component.
  # Let's attempt to identify which component corresponds to dir1 and which to dir2 in the PCA solution.
  #
  # For simplicity, we can assume that the first PCA component is closer to dir1 and 
  # the second PCA component is closer to dir2. This might not always be guaranteed, but 
  # for this test scenario, we design data so that dir1 is the stronger direction and should 
  # appear as the first component, and dir2 as the second.
  
  comp1_none <- boot_none$zboot_loadings[,1,drop=FALSE]
  comp2_none <- boot_none$zboot_loadings[,2,drop=FALSE]
  
  comp1_max <- boot_max_abs$zboot_loadings[,1,drop=FALSE]
  comp2_max <- boot_max_abs$zboot_loadings[,2,drop=FALSE]
  
  # Compute correlations:
  cor_none_1 <- cor(comp1_none, dir1)
  cor_none_2 <- cor(comp2_none, dir2)
  
  cor_max_1 <- cor(comp1_max, dir1)
  cor_max_2 <- cor(comp2_max, dir2)
  
  # With no sign flipping, the components might flip signs arbitrarily across bootstraps,
  # reducing the correlation with the true directions. With max_abs, we expect better alignment.
  
  # Check that using max_abs increases correlation with the correct directions:
  expect_true(abs(cor_max_1) >= abs(cor_none_1),
              info = "max_abs sign method should improve or maintain alignment for component 1 compared to none")
  
  expect_true(abs(cor_max_2) >= abs(cor_none_2),
              info = "max_abs sign method should improve or maintain alignment for component 2 compared to none")
  
  # Also expect that correlation with true_dir1 for the first component and true_dir2 for the second component is reasonably high:
  expect_true(abs(cor_max_1) > 0.5, 
              info = "Component 1 with max_abs should have moderate alignment with dir1")
  expect_true(abs(cor_max_2) > 0.5, 
              info = "Component 2 with max_abs should have moderate alignment with dir2")
})