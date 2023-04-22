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

