#' Nystrom method for out-of-sample embedding
#'
#' Approximate the embedding of a new data point using the Nystrom method, which is particularly useful
#' for large datasets and data-dependent embedding spaces, such as multidimensional scaling (MDS).
#'
#' @param new_data A matrix or data frame containing the new data points to be projected.
#' @param landmark_data A matrix or data frame containing the landmark data points used for approximation.
#' @param kernel_function A function used to compute the kernel matrix (e.g., a distance function for MDS).
#' @param eigenvectors A matrix containing the eigenvectors obtained from the eigendecomposition of the
#'   kernel matrix between the landmark points.
#' @param eigenvalues A vector containing the eigenvalues obtained from the eigendecomposition of the
#'   kernel matrix between the landmark points.
#' @param ... Additional arguments passed to the kernel_function.
#' @return A matrix containing the approximate embedding of the new_data in the data-dependent space.
#' @export
nystrom_embedding <- function(new_data, landmark_data, kernel_function, eigenvectors, eigenvalues, ...) {
  # Compute the kernel matrix between the new data and the landmark data
  K_new_landmark <- kernel_function(new_data, landmark_data, ...)
  
  # Compute the approximate eigendecomposition of the kernel matrix for the new data
  approx_eigenvectors <- K_new_landmark %*% eigenvectors / sqrt(eigenvalues)
  
  # Return the approximate embedding of the new data
  return(approx_eigenvectors)
}



# Nyström Method for Eigendecomposition Approximation
#
# Approximate the eigendecomposition of a kernel matrix using the Nyström method.
#
# @param X The data matrix.
# @param kernel_func A kernel function to compute the kernel matrix (e.g., radial basis function, linear kernel, etc.). Default is NULL, which uses the covariance matrix as in standard PCA.
# @param ncomp The number of requested components to estimate (default is the minimum dimension of the data matrix).
# @param landmarks A vector of indices specifying the columns of the matrix to use as landmark points (default is NULL).
# @param nlandmarks The number of landmark points to use if the "landmarks" argument is not provided (default is 10).
# @param preproc The pre-processing function to apply to the data matrix (default is centering).
# @return A `bi_projector` object containing the Nyström approximation results.
# @export
# nystrom_approx <- function(X, kernel_func=NULL, ncomp=min(dim(X)), landmarks=NULL, nlandmarks=10, preproc=center()) {
#   chk::chkor(chk::chk_matrix(X), chk::chk_s4_class("Matrix"))
#   
#   if (is.null(landmarks)) {
#     landmarks <- sample(nrow(X), nlandmarks)
#   } else {
#     chk::chk_numeric(landmarks)
#     chk::chk_true(all(landmarks > 0 & landmarks <= nrow(X)))
#   }
#   
#   proc <- prep(preproc)
#   X_preprocessed <- init_transform(proc, X)
#   
#   if (is.null(kernel_func)) {
#     K_mm <- cov(X_preprocessed[landmarks,])
#     K_nm <- X_preprocessed[-landmarks,] %*% t(X_preprocessed[landmarks,])
#   } else {
#     K_mm <- kernel_func(X_preprocessed[landmarks, ])
#     K_nm <- kernel_func(X_preprocessed, X_preprocessed[landmarks, ])
#   }
#   
#   eig_mm <- eigen(K_mm)
#   
#   keep <- which(abs(eig_mm$values) > 1e-8)
#   U_mm <- eig_mm$vectors
#   lambda_mm <- eig_mm$values
#   lambda_mm[-keep] <- 0
#   
#   U_nm <- K_nm %*% (U_mm %*% Matrix::Diagonal(x=(1 / sqrt(lambda_mm))))
#   
#   v <- U_nm
#   s <- U_nm * sqrt(lambda_mm)
#   sdev <- sqrt(lambda_mm)
#   
#   bi_projector(v, s=s, sdev=sdev, preproc=proc, classes="nystrom_approx")
# }



# # Compute the centered squared Euclidean distance kernel
# double_centering_kernel <- function(D) {
#   n <- nrow(D)
#   ones_n <- matrix(1, n, n)
#   K <- -0.5 * (D - (1/n) * (D %*% ones_n + ones_n %*% D - ones_n %*% D %*% ones_n))
#   return(K)
# }
# 
# # Nystrom method for MDS projection
# nystrom_mds_embedding <- function(X, Y, mds_embedding, n_components) {
#   D_XY <- as.matrix(dist(rbind(X, Y)))[1:nrow(X), (nrow(X) + 1):(nrow(X) + nrow(Y))]
#   D_XX <- as.matrix(dist(X))
#   D_YY <- as.matrix(dist(Y))
#   
#   K_XX <- double_centering_kernel(D_XX)
#   K_XY <- -0.5 * (D_XY^2 - rowMeans(D_XX^2) - colMeans(D_YY^2) + mean(D_XX^2))
#   
#   # Compute the eigenvectors and eigenvalues of K_XX
#   eigen_decomposition <- eigen(K_XX)
#   eigenvectors <- eigen_decomposition$vectors[, 1:n_components]
#   eigenvalues <- eigen_decomposition$values[1:n_components]
#   
#   # Nystrom method for out-of-sample MDS projection
#   Y_embedding <- K_XY %*% eigenvectors / sqrt(eigenvalues)
#   return(Y_embedding)
# }

