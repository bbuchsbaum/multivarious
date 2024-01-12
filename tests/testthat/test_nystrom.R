library(testthat)
library(multivarious)

# test_that("nystrom_approx returns a bi_projector object", {
#   X <- matrix(rnorm(50 * 10), 50, 10)
#   nystrom_res <- nystrom_approx(X)
#   expect_s3_class(nystrom_res, "bi_projector")
# })
# 
# test_that("nystrom_approx returns the correct number of components", {
#   X <- matrix(rnorm(50 * 10), 50, 10)
#   nystrom_res_3 <- nystrom_approx(X, ncomp=3)
#   expect_equal(ncol(nystrom_res_3$v), 3)
#   
#   nystrom_res_5 <- nystrom_approx(X, ncomp=5)
#   expect_equal(ncol(nystrom_res_5$v), 5)
# })
# 
# test_that("nystrom_approx handles custom kernel functions", {
#   X <- matrix(rnorm(50 * 10), 50, 10)
#   
#   linear_kernel <- function(x, y) {
#     x %*% t(y)
#   }
#   
#   nystrom_res_linear <- nystrom_approx(X, kernel_func=linear_kernel)
#   expect_s3_class(nystrom_res_linear, "bi_projector")
#   
#   rbf_kernel <- function(x, y, sigma=1) {
#     size_x <- nrow(x)
#     size_y <- nrow(y)
#     
#     dist_matrix <- matrix(0, nrow=size_x, ncol=size_y)
#     for (i in 1:size_x) {
#       for (j in 1:size_y) {
#         dist_matrix[i, j] <- sum((x[i, ] - y[j, ])^2)
#       }
#     }
#     
#     exp(-dist_matrix / (2 * sigma^2))
#   }
#   
#   nystrom_res_rbf <- nystrom_approx(X, kernel_func=rbf_kernel)
#   expect_s3_class(nystrom_res_rbf, "bi_projector")
# })
# 
# test_that("nystrom_approx handles custom landmarks", {
#   X <- matrix(rnorm(50 * 10), 50, 10)
#   landmarks <- c(1, 5, 10, 15, 20,25,27)
#   
#   nystrom_res_landmarks <- nystrom_approx(X, landmarks=landmarks)
#   expect_s3_class(nystrom_res_landmarks, "bi_projector")
# })
# 
# test_that("nystrom_approx handles custom nlandmarks", {
#   X <- matrix(rnorm(50 * 10), 50, 10)
#   
#   nystrom_res_nlandmarks <- nystrom_approx(X, nlandmarks=5)
#   expect_s3_class(nystrom_res_nlandmarks, "bi_projector")
# })
