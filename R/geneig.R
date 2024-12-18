#' Generalized Eigenvalue Decomposition
#'
#' Computes the generalized eigenvalues and eigenvectors for the problem: A x = λ B x.
#' Various methods are available and differ in their assumptions about A and B.
#'
#' @param A The left-hand side square matrix.
#' @param B The right-hand side square matrix, same dimension as A.
#' @param ncomp Number of eigenpairs to return.
#' @param method Method to compute the eigenvalues and eigenvectors:
#'   - "robust": Uses a stable decomposition via a whitening transform (requires B to be symmetric positive-definite).
#'   - "sdiag": Uses a spectral decomposition of B and transforms the problem, works when B is symmetric positive-definite.
#'   - "geigen": Uses the `geigen` package for a general solution.
#'   - "primme": Uses the `PRIMME` package for large sparse matrices.
#' @param ... Additional arguments passed to the underlying methods.
#' @return An object of class `projector` with eigenvalues stored in `values` and standard deviations in `sdev = sqrt(values)`.
#' @importFrom Matrix isSymmetric
#' @importFrom chk chk_equal
#' @export
#'
#' @examples
#' if (requireNamespace("geigen", quietly = TRUE)) {
#'   A <- matrix(c(14, 10, 12, 10, 12, 13, 12, 13, 14), nrow=3, byrow=TRUE)
#'   B <- matrix(c(48, 17, 26, 17, 33, 32, 26, 32, 34), nrow=3, byrow=TRUE)
#'   res <- geneig(A, B, ncomp=3, method="geigen")
#'   # res$values and coefficients(res)
#' }
geneig <- function(A, B, ncomp, method = c("robust", "sdiag", "geigen", "primme"), ...) {
  method <- match.arg(method)
  
  # Validate inputs
  chk::chk_equal(nrow(A), ncol(A))
  chk::chk_equal(nrow(B), ncol(B))
  chk::chk_equal(nrow(A), nrow(B))
  
  if (!is.numeric(ncomp) || ncomp <= 0) {
    stop("ncomp must be a positive integer.")
  }
  
  # Perform generalized eigen decomposition
  ret <- switch(method,
                robust = {
                  # Robust method: assumes B is symmetric positive definite
                  # Approach:
                  #   1. Compute B^{1/2} and its inverse: B = Q Λ Q^T, B^{1/2} = Q Λ^{1/2} Q^T
                  #   2. Define W = B^{-1/2} A B^{-1/2}. Then solve eigen(W).
                  #   3. Eigenvectors in original space: V = B^{-1/2} eigenvectors(W)
                  if (!isSymmetric(B)) {
                    stop("For the robust method, B must be symmetric positive definite.")
                  }
                  B_chol <- chol(B)
                  B_sqrt_inv <- solve(B_chol)
                  
                  W <- B_sqrt_inv %*% A %*% B_sqrt_inv
                  decomp <- eigen(W)
                  
                  # Extract ncomp
                  vectors <- B_sqrt_inv %*% decomp$vectors[, 1:ncomp, drop=FALSE]
                  values <- decomp$values[1:ncomp]
                  list(vectors = vectors, values = values)
                },
                sdiag = {
                  # sdiag method:
                  #   1. B must be symmetric. Eigen-decompose B.
                  #   2. Adjust small/negative eigenvalues if necessary.
                  #   3. Define B^{-1/2} using these eigenvectors and eigenvalues.
                  #   4. Transform A: A' = B^{-1/2} A B^{-1/2}.
                  #   5. Solve eigen(A'), then map back eigenvectors = B^{-1/2} V
                  if (!isSymmetric(B)) {
                    stop("Matrix B must be symmetric for the sdiag method.")
                  }
                  min_eigenvalue <- 1e-6
                  B_eig <- eigen(B)
                  valsB <- B_eig$values
                  # Ensure no too-small or negative eigenvalues
                  valsB[valsB < abs(min_eigenvalue)] <- min_eigenvalue
                  
                  B_sqrt_inv <- B_eig$vectors %*% diag(1 / sqrt(valsB)) %*% t(B_eig$vectors)
                  
                  A_transformed <- t(B_sqrt_inv) %*% A %*% B_sqrt_inv
                  A_eig <- eigen(A_transformed)
                  
                  vectors <- B_sqrt_inv %*% A_eig$vectors[, 1:ncomp, drop=FALSE]
                  values <- A_eig$values[1:ncomp]
                  list(vectors = vectors, values = values)
                },
                geigen = {
                  if (!requireNamespace("geigen", quietly = TRUE)) {
                    stop("The 'geigen' package is required for method='geigen'. Please install it.")
                  }
                  res <- geigen::geigen(A, B)
                  # Extract ncomp
                  vectors <- res$vectors[, 1:ncomp, drop = FALSE]
                  values <- res$values[1:ncomp]
                  list(vectors = vectors, values = values)
                },
                primme = {
                  if (!requireNamespace("PRIMME", quietly = TRUE)) {
                    stop("The 'PRIMME' package is required for method='primme'. Please install it.")
                  }
                  # PRIMME::eigs_sym can handle generalized eigen problems if specified
                  # But PRIMME typically handles symmetric problems, ensure A,B are suitable
                  # For a general problem, you may need a different call or ensure A,B are symmetric.
                  # Check PRIMME docs for generalized eigenproblems.
                  res <- PRIMME::eigs_sym(A = A, B = B, NEig = ncomp, ...)
                  vectors <- res$vectors
                  values <- res$values
                  list(vectors = vectors, values = values)
                }
  )
  
  # sdev is often sqrt of eigenvalues (especially if they represent variances)
  # However, generalized eigenvalues can be negative or complex. Here we assume they are real and non-negative.
  # If negative/complex eigenvalues appear, handle accordingly.
  # For now, let's assume they are real and non-negative:
  ev <- ret$values
  if (any(ev < 0)) {
    warning("Some eigenvalues are negative. 'sdev' may not be meaningful.")
  }
  
  sdev <- sqrt(pmax(ev, 0)) # Prevent sqrt of negative
  
  # Construct the projector object.
  # The projector class doesn't have a `values` parameter by default, but we can pass them in `...`:
  out <- projector(
    v = ret$vectors,
    preproc = prep(pass()),
    classes = "geneig",
    values = ev,
    sdev = sdev
  )
  
  out
}

