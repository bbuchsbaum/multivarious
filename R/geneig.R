#' Generalized Eigenvalue Decomposition
#'
#' Computes the generalized eigenvalues and eigenvectors for the problem: A x = λ B x.
#' Various methods differ in assumptions about A and B.
#'
#' @param A The left-hand side square matrix.
#' @param B The right-hand side square matrix, same dimension as A.
#' @param ncomp Number of eigenpairs to return.
#' @param preproc A preprocessing function to apply to the matrices before solving the generalized eigenvalue problem.
#' @param method One of:
#'   - "robust": Uses a stable decomposition via a whitening transform (B must be symmetric PD).
#'   - "sdiag":  Uses a spectral decomposition of B (must be symmetric PD). Requires A to be symmetric for meaningful results.
#'   - "geigen": Uses the \pkg{geigen} package for a general solution (A and B can be non-symmetric).
#'   - "primme": Uses the \pkg{PRIMME} package for large/sparse symmetric problems (A and B must be symmetric).
#' @param which Only used for `method = "primme"`. Specifies which eigenvalues to compute ("LM", "SM", "LA", "SA", etc.). Default is "LM" (Largest Magnitude). See \code{\link[PRIMME]{eigs_sym}}.
#' @param ... Additional arguments to pass to the underlying solver.
#' @return A `projector` object with generalized eigenvectors and eigenvalues.
#' @seealso \code{\link{projector}} for the base class structure.
#' 
#' #' @references
#' Golub, G. H. & Van Loan, C. F. (2013) *Matrix Computations*,
#'   4th ed., § 8.7 – textbook derivation for the "robust" (Cholesky)
#'   and "sdiag" (spectral) transforms.
#'
#' Moler, C. & Stewart, G. (1973) "An Algorithm for Generalized Matrix
#'   Eigenvalue Problems". *SIAM J. Numer. Anal.*, 10 (2): 241‑256 –
#'   the QZ algorithm behind the \code{geigen} backend.
#'
#' Stathopoulos, A. & McCombs, J. R. (2010) "PRIMME: PReconditioned
#'   Iterative Multi‑Method Eigensolver". *ACM TOMS* 37 (2): 21:1‑21:30 –
#'   the algorithmic core of the \code{primme} backend.
#'
#' See also the \pkg{geigen} (CRAN) and \pkg{PRIMME} documentation.
#'
#' @importFrom PRIMME eigs_sym
#' @importFrom geigen geigen
#' @export
#' @examples
#' # Simulate two matrices
#' set.seed(123)
#' A <- matrix(rnorm(50 * 50), 50, 50)
#' B <- matrix(rnorm(50 * 50), 50, 50)
#' A <- A %*% t(A) # Make A symmetric
#' B <- B %*% t(B) + diag(50) * 0.1 # Make B symmetric positive definite
#'
#' # Solve generalized eigenvalue problem
#' result <- geneig(A = A, B = B, ncomp = 3)
#'
geneig <- function(A = NULL,
                   B = NULL,
                   ncomp = 2,
                   preproc = prep(pass()),
                   method = c("robust", "sdiag", "geigen", "primme"),
                   which = "LR", ...) {
  method <- match.arg(method)
  
  # Validate inputs
  # Restore original check
  stopifnot(is.numeric(A), is.numeric(B)) 
  chk::chk_equal(nrow(A), ncol(A))
  chk::chk_equal(nrow(B), ncol(B))
  chk::chk_equal(nrow(A), nrow(B))
  
  if (!is.numeric(ncomp) || ncomp <= 0 || !chk::vld_whole_number(ncomp)) {
    stop("'ncomp' must be a positive integer.")
  }
  
  # Truncate ncomp if it exceeds matrix dimensions
  if (ncomp > nrow(A)) {
    warning(sprintf("'ncomp' (%d) exceeds matrix dimensions (%d), truncating.", ncomp, nrow(A)))
    ncomp <- nrow(A)
  }
  
  # Dispatch to chosen method
  res_raw <- switch(
    method,
    robust = {
      if (!isSymmetric(B)) {
        stop("For method='robust', B must be symmetric.")
      }
      # Check for Positive Definiteness
      B_chol_try <- try(chol(B), silent = TRUE)
      if (inherits(B_chol_try, "try-error")) {
        stop("For method='robust', B must be positive definite (Cholesky failed).")
      }
      B_chol <- B_chol_try 
      
      # B_sqrt_inv <- solve(B_chol) # Avoid forming full inverse if possible
      # W <- t(B_sqrt_inv) %*% A %*% B_sqrt_inv # More stable calculation for W
      # Use solve(chol(B), X) for B^{-1/2} X and solve(t(chol(B)), X) for B^{-T/2} X
      tmp <- solve(B_chol, A) # tmp = R^-1 A where B=R^T R
      W <- solve(t(B_chol), t(tmp)) # W = R^-T (R^-1 A)^T = R^-T A^T R^-1
      # W should be symmetric if A is symmetric. Let's ensure it is treated as such.
      W <- (W + t(W)) / 2
      
      decomp <- eigen(W, symmetric = TRUE) 
      
      # Back-transform eigenvectors: vectors = B^{-1/2} * eigenvecs(W)
      # where B = R^T R, B^{-1/2} = solve(R)
      vectors_raw <- solve(B_chol, decomp$vectors[, 1:ncomp, drop = FALSE])
      values <- decomp$values[1:ncomp]
      
      # Explicitly B-orthonormalize
      norm_factor <- sqrt(diag(t(vectors_raw) %*% B %*% vectors_raw))
      # Avoid division by zero/NaN if norm_factor is very small
      norm_factor[norm_factor < .Machine$double.eps] <- 1 
      vectors <- sweep(vectors_raw, 2, norm_factor, "/")
      
      list(vectors = vectors, values = values)
    },
    sdiag = {
      if (!isSymmetric(B)) {
        stop("For method='sdiag', B must be symmetric.")
      }
      if (!isSymmetric(A)) {
        warning("For method='sdiag', A is not symmetric. Results may be inaccurate or complex.")
      }
      # Check for Positive Definiteness needed for B_sqrt_inv
      min_eigenvalue <- sqrt(.Machine$double.eps) # Use machine epsilon based threshold
      B_eig <- eigen(B, symmetric = TRUE)
      valsB <- B_eig$values
      
      if(any(valsB < min_eigenvalue)){
        warning(sprintf("B has %d eigenvalues close to zero or negative (min eigenvalue=%.2e). Clamping for inversion.", 
                        sum(valsB < min_eigenvalue), min(valsB)))
        valsB[valsB < min_eigenvalue] <- min_eigenvalue
      } 
      
      B_sqrt_inv_diag <- diag(1 / sqrt(valsB), nrow=length(valsB), ncol=length(valsB))
      B_sqrt_inv <- B_eig$vectors %*% B_sqrt_inv_diag %*% t(B_eig$vectors)
      
      A_transformed <- B_sqrt_inv %*% A %*% B_sqrt_inv
      # Ensure symmetry for eigen
      A_transformed <- (A_transformed + t(A_transformed)) / 2
      A_eig <- eigen(A_transformed, symmetric = TRUE)
      
      vectors_raw <- B_sqrt_inv %*% A_eig$vectors[, 1:ncomp, drop=FALSE]
      values  <- A_eig$values[1:ncomp]
      
      # Explicitly B-orthonormalize
      norm_factor <- sqrt(diag(t(vectors_raw) %*% B %*% vectors_raw))
      norm_factor[norm_factor < .Machine$double.eps] <- 1 
      vectors <- sweep(vectors_raw, 2, norm_factor, "/")
      
      list(vectors = vectors, values = values)
    },
    geigen = {
      if (!requireNamespace("geigen", quietly = TRUE)) {
        stop("Package 'geigen' not installed. Please install it for method='geigen'.")
      }
      res <- geigen::geigen(A, B, symmetric = FALSE) # Assume potentially non-symmetric
      
      # Sort by decreasing absolute value of eigenvalues
      ord <- order(abs(res$values), decreasing = TRUE)
      values_sorted  <- res$values[ord]
      vectors_sorted <- res$vectors[, ord, drop=FALSE]
      
      vectors <- vectors_sorted[, 1:ncomp, drop=FALSE]
      values  <- values_sorted[1:ncomp]
      list(vectors = vectors, values = values)
    },
    primme = {
      if (!requireNamespace("PRIMME", quietly = TRUE)) {
        stop("Package 'PRIMME' not installed. Please install it for method='primme'.")
      }
      if (!isSymmetric(A) || !isSymmetric(B)) {
        stop("For method='primme' using eigs_sym, both A and B must be symmetric.")
      }
      # Use the provided 'which' argument, pass others via ...
      res <- PRIMME::eigs_sym(A = A, B = B, NEig = ncomp, which = which, ...)
      vectors <- res$vectors
      values  <- res$values
      list(vectors = vectors, values = values)
    }
  )
  
  ev <- res_raw$values
  vec <- res_raw$vectors
  
  # Check for complex eigenvalues and handle
  if (is.complex(ev)) {
    ev_im <- Im(ev)
    if (any(abs(ev_im) > sqrt(.Machine$double.eps))) {
       warning("Complex eigenvalues found. Taking the real part.")
    }
    ev <- Re(ev)
    # If eigenvalues were complex, eigenvectors might be too. Take real part.
    if(is.complex(vec)){
      warning("Complex eigenvectors found. Taking the real part.")
      vec <- Re(vec)
    }
  }

  # Check for negative eigenvalues (after taking real part)
  if (any(ev < 0)) {
    warning("Some real eigenvalues are negative. 'sdev' computed using absolute values.")
  }
  
  sdev <- sqrt(abs(ev)) # Use abs(Re(ev))
  
  # Return a simple list with a class attribute
  out <- list(
    values  = ev,
    vectors = vec,
    sdev    = sdev,
    ncomp   = ncomp,
    method  = method
  )
  
  class(out) <- c("geneig", "list")
  out
}



#' Factor a matrix with regularization
#'
#' Attempts a Cholesky factorization with a diagonal `reg` until it succeeds.
#'
#' @param M A symmetric matrix to factor.
#' @param reg Initial regularization term.
#' @param max_tries Number of times to multiply reg by 10 if factorization fails.
#' @return A list with `ch` (the Cholesky factor) and `reg` (the final reg used).
#' @keywords internal
#' @importFrom Matrix Diagonal Cholesky
#' @noRd
factor_mat <- function(M, reg = 1e-3, max_tries = 5) {
  d <- nrow(M)
  M_reg <- M # Start with original M
  current_reg <- 0 # Keep track of added regularization
  
  for (i in seq_len(max_tries + 1)) { # Try initial M first, then add reg
    if (i > 1) { 
      # Add regularization for attempts 2 onwards
      if (i == 2) {
        current_reg <- reg
      } else {
        current_reg <- current_reg * 10
      }
      # Add Diagonal efficiently
      diag(M_reg) <- diag(M) + current_reg
    } else {
      # First attempt with M as is (or previous M_reg if loop continues)
      M_reg <- M
    }
    
    # Attempt Cholesky
    ch <- try(Matrix::Cholesky(M_reg, LDL = FALSE, super = TRUE), silent = TRUE) # Use Matrix::Cholesky
    
    if (!inherits(ch, "try-error")) {
      # Return the successful factor and the *added* regularization amount
      return(list(ch = ch, reg_added = if(i==1) 0 else current_reg)) 
    }
    
    # If first attempt failed, prepare M_reg for the next iteration with base reg
    if (i == 1) {
       M_reg <- M # Ensure we add reg to original M next time
    }
  }
  stop(sprintf("Unable to factor matrix even after adding regularization up to %.2e.", current_reg))
}

#' Solve using a precomputed Cholesky factor
#'
#' @param ch A Cholesky factor object from `Matrix::Cholesky()`.
#' @param RHS A right-hand-side matrix/vector compatible with `Matrix::solve`.
#' @keywords internal
#' @importFrom Matrix solve
#' @noRd
solve_chol <- function(ch, RHS) {
  Matrix::solve(ch, RHS) # Use Matrix::solve method
}

#' Orthonormalize columns via QR
#'
#' @param X A numeric matrix whose columns we want to orthonormalize.
#' @return A matrix of the same dimension with orthonormal columns. Handles potential rank deficiency by returning fewer columns.
#' @keywords internal
#' @importFrom methods as is
#' @noRd
orthonormalize <- function(X) {
  if (ncol(X) == 0) return(X) # Handle empty matrix
  
  # Use Matrix::qr for potential sparse input
  # Convert to dense first if it's not already suitable for base qr
  if (!methods::is(X, "matrix")) {
      X_dense <- try(methods::as(X, "matrix"), silent = TRUE)
      if (inherits(X_dense, "try-error")){
          stop("orthonormalize: Input matrix cannot be coerced to dense matrix for QR.")
      } 
      X <- X_dense
  }
  
  QR <- qr(X)
  rank <- QR$rank
  if (rank == 0) {
      warning("orthonormalize: Input matrix has rank 0.")
      return(matrix(0.0, nrow = nrow(X), ncol = 0)) # Return empty matrix
  }
  
  Q <- qr.Q(QR)
  
  # Return only the first 'rank' columns corresponding to the independent basis
  if (rank < ncol(X)) {
    warning(sprintf("orthonormalize: Input matrix rank (%d) is less than number of columns (%d). Returning orthonormal basis for the column space.", rank, ncol(X)))
  } 
  Q[, 1:rank, drop = FALSE]
}

#' Subspace Iteration for Generalized Eigenproblem
#'
#' Iteratively solves S1 x = λ S2 x using a subspace approach.
#' Assumes S1, S2 are symmetric. S2 (or S1 if which="smallest") must be PD.
#'
#' @param S1 A square symmetric matrix (e.g., n x n).
#' @param S2 A square symmetric positive definite matrix of the same dimension.
#' @param q Number of eigenpairs to approximate.
#' @param which "largest" or "smallest" eigenvalues to seek.
#' @param max_iter Maximum iteration count.
#' @param tol Convergence tolerance on relative change in eigenvalues.
#' @param V0 Optional initial guess matrix (n x q). If NULL, uses random.
#' @param reg_S Regularization added to S1 or S2 during factorization attempts. Default 1e-6.
#' @param reg_T Regularization for the small T matrix. Default 1e-9.
#' @param seed Optional seed for random V0 initialization. If NULL, uses current RNG state.
#' @return A list with `values` = the approximate eigenvalues, `vectors` = the approximate eigenvectors (n x q).
#' @keywords internal
#' @importFrom Matrix Diagonal solve t
#' @noRd
solve_gep_subspace <- function(S1, S2, q = 2,
                               which = c("largest", "smallest"),
                               max_iter = 100, tol = 1e-6,
                               V0 = NULL, reg_S = 1e-6, reg_T = 1e-9,
                               seed = NULL) { 
  
  which <- match.arg(which)
  d <- nrow(S1)
  
  # Factor either S2 or S1 once, depending on which eigenvalues we want
  if (which == "largest") {
    # Factor S2 => solve S2 V_hat = S1 V
    s_fact <- factor_mat(S2, reg = reg_S)
    ch <- s_fact$ch
    
    solve_step <- function(V) {
      RHS <- S1 %*% V
      solve_chol(ch, RHS)  # V_hat = S2^-1 (S1 V)
    }
  } else { # smallest
    # Factor S1 => solve S1 V_hat = S2 V
    s_fact <- factor_mat(S1, reg = reg_S)
    ch <- s_fact$ch
    
    solve_step <- function(V) {
      RHS <- S2 %*% V
      solve_chol(ch, RHS)  # V_hat = S1^-1 (S2 V)
    }
  }
  
  # Initialize subspace V
  if (is.null(V0)) {
    if (!is.null(seed)) set.seed(seed) # Set seed only if provided
    V <- matrix(rnorm(d * q), d, q)
  } else {
    if (ncol(V0) != q || nrow(V0) != d) {
       stop(sprintf("V0 dimensions (%d x %d) do not match expected (%d x %d).", 
                 nrow(V0), ncol(V0), d, q))
    }
    V <- V0
  }
  V <- orthonormalize(V)
  # Handle case where initial V is rank deficient
  if (ncol(V) < q) {
      warning(sprintf("Initial subspace V has rank %d, less than requested q=%d. Proceeding with reduced rank.", ncol(V), q))
      q <- ncol(V)
      if (q == 0) stop("Initial subspace V has rank 0.")
  }
  
  lambda_old <- rep(NA, q)
  
  for (iter in seq_len(max_iter)) {
    # 1) Expand subspace: V_hat = Op(V) where Op = S2^-1 S1 or S1^-1 S2
    V_hat <- solve_step(V)
    
    # 2) Orthonormalize V_hat using QR decomposition
    V_new <- orthonormalize(V_hat)
    
    # Check rank after orthonormalization
    q_new <- ncol(V_new)
    if (q_new < q) {
        warning(sprintf("Subspace rank reduced to %d during iteration %d. Stopping early.", q_new, iter))
        # Might need deflation or other strategy here, for now stop/return current
        q <- q_new
        if (q == 0) stop("Subspace iteration collapsed to rank 0.")
        # Trim lambda_old if needed
        lambda_old <- lambda_old[1:q]
        V <- V_new[, 1:q, drop=FALSE] # Update V to the reduced rank version
        break # Stop iteration as rank changed
    }
    
    # 3) Form Rayleigh quotient matrices using V_new (the orthonormal basis)
    # T_mat = V_new^T S2 V_new 
    # S_mat = V_new^T S1 V_new
    T_mat <- Matrix::t(V_new) %*% S2 %*% V_new
    S_mat <- Matrix::t(V_new) %*% S1 %*% V_new
    
    # Ensure symmetry (numerical precision might cause slight asymmetry)
    T_mat <- (T_mat + Matrix::t(T_mat)) / 2
    S_mat <- (S_mat + Matrix::t(S_mat)) / 2
    
    # 4) Solve the small (q x q) generalized eigenproblem: S_mat w = lambda T_mat w
    # Regularize T_mat before inversion if needed, although it should be PD if S2 is
    T_mat_reg <- T_mat + Matrix::Diagonal(q, reg_T)
    
    eig_res <- tryCatch({
        # Use base eigen for the small dense problem
        # Solve S_mat w = lambda T_mat w <=> T_mat^-1 S_mat w = lambda w
        # Ensure matrices are dense for base::eigen
        eigen(solve(as.matrix(T_mat_reg), as.matrix(S_mat)), symmetric = TRUE)
      },
      error = function(e) {
        warning(sprintf("Small eigenproblem failed at iter %d: %s. Trying geigen.", iter, e$message))
        # Fallback to geigen for the small problem if solve/eigen fails
        try(geigen::geigen(as.matrix(S_mat), as.matrix(T_mat_reg), symmetric=TRUE), silent=TRUE)
      }
    )
    
    if (inherits(eig_res, "try-error")) {
        stop(sprintf("Unable to solve small (%d x %d) eigenproblem at iteration %d even with fallback.", q, q, iter))
    }
    
    # Order eigenvalues (largest abs for convergence check stability, but use actual values)
    ord <- order(abs(eig_res$values), decreasing = TRUE)
    lambda <- eig_res$values[ord]
    W <- eig_res$vectors[, ord, drop = FALSE]
        
    # Adjust target eigenvalues based on 'which' argument
    if (which == "smallest") {
        # If we factored S1 (Op = S1^-1 S2), then eigenvalues of Op are 1/lambda_orig
        # We solved S_mat w = lambda_op T_mat w
        # where lambda_op corresponds to eigenvalues of Op = S1^-1 S2
        # We want eigenvalues of S1 x = lambda_orig S2 x
        # Need to sort 1/lambda_op to find the largest, which correspond to smallest lambda_orig
        ord_smallest_orig <- order(1/lambda, decreasing = TRUE) # Order by decreasing 1/lambda
        lambda <- lambda[ord_smallest_orig] # Get the lambda_op corresponding to smallest lambda_orig
        W <- W[, ord_smallest_orig, drop=FALSE]
    } else {
         # If we factored S2 (Op = S2^-1 S1), eigenvalues lambda_op are the ones we want (lambda_orig)
         # We already sorted by magnitude, which is standard for 'largest'
         ord_largest_orig <- order(lambda, decreasing = TRUE) # Order by value
         lambda <- lambda[ord_largest_orig]
         W <- W[, ord_largest_orig, drop=FALSE]
    }
    
    # 5) Update V using the eigenvectors W of the small problem
    # V_k+1 = V_new %*% W
    V <- V_new %*% W 
    # V should already be orthonormal if W is orthonormal and V_new is orthonormal
    # Re-orthonormalize just in case of numerical drift?
    V <- orthonormalize(V) 
    q_check <- ncol(V)
    if (q_check < q) {
        warning(sprintf("Subspace rank reduced to %d after update at iteration %d. Stopping early.", q_check, iter))
        q <- q_check
        if (q == 0) stop("Subspace iteration collapsed to rank 0 after update.")
        lambda_old <- lambda_old[1:q]
        lambda <- lambda[1:q]
        break
    }

    # 6) Check convergence
    if (iter > 1) { # Compare with previous iteration's eigenvalues
      # Use lambda_old from the *previous* iteration (corresponding to the same subspace V)
      valid_idx <- !is.na(lambda_old) & !is.na(lambda) & abs(lambda_old) > 1e-12
      if (all(!valid_idx)) {
         rel_change <- Inf # Cannot compare if no valid old values
      } else {
         rel_change <- max(abs(lambda[valid_idx] - lambda_old[valid_idx]) / pmax(abs(lambda_old[valid_idx]), 1e-12))
      }
      
      if (rel_change < tol) {
        # message(sprintf("Subspace iteration converged at iteration %d with rel_change=%.2e", iter, rel_change))
        break
      }
    }
    
    lambda_old <- lambda # Store current eigenvalues for next iteration's comparison
    
    if (iter == max_iter) {
       warning(sprintf("Subspace iteration did not converge within %d iterations (tol=%.1e, last rel_change=%.2e).", max_iter, tol, rel_change))
    }
  }
  
  # Return final approximate eigenpairs
  list(values = lambda_old, vectors = V[, 1:q, drop=FALSE])
}

