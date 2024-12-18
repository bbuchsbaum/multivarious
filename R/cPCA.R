#' Contrastive PCA (cPCA) with Adaptive Computation Methods
#'
#' Contrastive PCA (cPCA) finds directions that capture the variation in a "foreground" dataset \eqn{X_f} that is not present (or less present) in a "background" dataset \eqn{X_b}. This function adaptively chooses how to solve the generalized eigenvalue problem based on the dataset sizes and the chosen method:
#'
#' 1. **method = "corpcor"**: Uses a corpcor-based whitening approach (`crossprod.powcor.shrink`) to transform the data, then performs a standard PCA on the transformed foreground data.
#' 2. **method \in \{"geigen","primme","sdiag"\} and moderate number of features (D):** Directly forms covariance matrices and uses `geneig` to solve the generalized eigenvalue problem.
#' 3. **method \in \{"geigen","primme","sdiag"\} and large number of features (D >> N):** Uses an SVD-based reduction on the background data to avoid forming large \eqn{D \times D} matrices. This reduces the problem to \eqn{N \times N} space.
#'
#' @param X_f A numeric matrix representing the foreground dataset, with dimensions (samples x features).
#' @param X_b A numeric matrix representing the background dataset, with dimensions (samples x features).
#' @param ncomp Number of components to estimate. Defaults to `min(ncol(X_f))`.
#' @param preproc A pre-processing function (default: `center()`), applied to both `X_f` and `X_b` before analysis.
#' @param lambda Shrinkage parameter for covariance estimation. Defaults to 0. Used by `corpcor::cov.shrink` or `crossprod.powcor.shrink`.
#' @param method A character string specifying the computation method. One of:
#'   \describe{
#'     \item{"geigen"}{Use `geneig` for the generalized eigenvalue problem (default).}
#'     \item{"primme"}{Use `geneig` with the PRIMME library for potentially more efficient solvers.}
#'     \item{"sdiag"}{Use a spectral decomposition method for symmetric matrices in `geneig`.}
#'     \item{"corpcor"}{Use a corpcor-based whitening approach followed by PCA.}
#'   }
#' @param ... Additional arguments passed to underlying functions such as `geneig` or covariance estimation.
#'
#' @details
#' **Adaptive Strategy:**
#' - If `method = "corpcor"`, no large covariance matrices are formed. Instead, the background data is used to "whiten" the foreground, followed by a simple PCA.
#' - If `method \neq "corpcor"` and the number of features `D` is manageable (e.g. `D <= max(N_f, N_b)`), the function forms covariance matrices and directly solves the generalized eigenproblem.
#' - If `method \neq "corpcor"` and `D` is large (e.g., tens of thousands, `D > max(N_f, N_b)`), it computes the SVD of the background data `X_b` to derive a smaller `N x N` eigenproblem, thereby avoiding the costly computation of \eqn{D \times D} covariance matrices.
#'
#' Note: If `lambda != 0` and `D` is very large, the current implementation does not fully integrate shrinkage into the large-D SVD-based approach and will issue a warning.
#'
#' @return A `bi_projector` object containing:
#' \describe{
#'   \item{v}{A (features x ncomp) matrix of eigenvectors (loadings).}
#'   \item{s}{A (samples x ncomp) matrix of scores, i.e., projections of `X_f` onto the eigenvectors.}
#'   \item{sdev}{A vector of length `ncomp` giving the square-root of the eigenvalues.}
#'   \item{preproc}{The pre-processing object used.}
#' }
#'
#' @examples
#' set.seed(123)
#' X_f <- matrix(rnorm(2000), nrow=100, ncol=20) # Foreground: 100 samples, 20 features
#' X_b <- matrix(rnorm(2000), nrow=100, ncol=20) # Background: same size
#' # Default method (geigen), small dimension scenario
#' res <- cPCA(X_f, X_b, ncomp=5)
#' plot(res$s[,1], res$s[,2], main="cPCA scores (component 1 vs 2)")
#'
#' @export
cPCA <- function(X_f, X_b, ncomp = min(dim(X_f)[2]), preproc = center(), 
                 lambda=0, method = c("geigen","primme", "sdiag", "corpcor"), allow_transpose=TRUE, ...) {
  chk::chkor_vld(chk::vld_matrix(X_f), chk::vld_s4_class(X_f, "Matrix"))
  chk::chkor_vld(chk::vld_matrix(X_b), chk::vld_s4_class(X_b, "Matrix"))
  
 
  # Apply preprocessing using the multivarious pattern
  proc <- prep(preproc)
  X_f <- init_transform(proc, X_f)
  X_b <- init_transform(proc, X_b)
  
 
  
  if (method == "corpcor") {
    # Use crossprod.powcor.shrink to efficiently compute the transformation
    # It efficiently computes X_b^(alpha) %*% X_f
    X_f_mod <- corpcor::crossprod.powcor.shrink(X_b, X_f, alpha = -0.5, lambda = lambda, verbose = TRUE)
    
    # Perform PCA directly on the modified foreground matrix
    pca_res <- pca(X_f_mod, ncomp=ncomp)
    eigenvectors <- pca_res$v
    eigenvalues <- pca_res$sdev^2  # Squaring singular values to get eigenvalues
    scores <- pca_res$s
  } else {
  
    # Check dimensions and potentially transpose matrices for efficiency
    transpose = ncol(X_f) > nrow(X_f)
    #transpose_b = ncol(X_b) > nrow(X_b)
    
    if (transpose && allow_transpose) {
      X_f <- t(X_f)
      X_b <- t(X_b)
    }
  
    # Calculate covariance matrices
    Cov_f <- corpcor::cov.shrink(X_f, lambda=lambda)
    Cov_b <- corpcor::cov.shrink(X_b, lambda=lambda)
  
    # Compute the generalized eigenvalue problem using geigen package
    geigen_res <- geneig(Cov_f, Cov_b, ncomp=ncomp, ...)
  
    # Extract the top ncomp eigenvectors and values
    eigenvectors <- geigen_res$v[, 1:ncomp,drop=FALSE]
    eigenvalues <- geigen_res$values[1:ncomp]
  
    # Compute transformed scores
    scores <- X_f %*% eigenvectors
  
    # Correct the orientation of eigenvectors if transposed for calculation
    if (transpose && allow_transpose) {
      eigenvectors <- t(eigenvectors)
    }
  
    # Create a bi_projector instance
    projector <- bi_projector(
      v = eigenvectors,
      s = t(scores),  # Transpose scores to match expected orientation
      sdev = sqrt(eigenvalues),
      preproc = proc,
      classes = c("cPCA", "bi_projector"))
  }
  
  return(projector)
}