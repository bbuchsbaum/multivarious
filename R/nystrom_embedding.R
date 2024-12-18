#' Nyström approximation for kernel-based decomposition (Unified Version)
#'
#' Approximate the eigen-decomposition of a large kernel matrix using either
#' the standard Nyström method or the Double Nyström method.
#'
#' The Double Nyström method introduces an intermediate step that reduces the
#' size of the decomposition problem, potentially improving efficiency and scalability.
#'
#' @param X A numeric matrix or data frame of size (N x D), where N is number of samples.
#' @param kernel_func A kernel function with signature `kernel_func(X, Y, ...)`.
#'   If NULL, defaults to a linear kernel: X %*% t(Y).
#' @param ncomp Number of components (eigenvectors/eigenvalues) to return.
#' @param landmarks A vector of row indices (of X) specifying the landmark points.
#'   If NULL, `nlandmarks` points are sampled uniformly at random.
#' @param nlandmarks The number of landmark points to sample if `landmarks` is NULL. Default is 10.
#' @param preproc A pre-processing pipeline (default `prep(pass())`) to apply before computing the kernel.
#' @param method Either "standard" (the classic single-stage Nyström) or "double" (the two-stage Double Nyström method).
#' @param l Intermediate rank for the double Nyström method. Ignored if `method="standard"`.
#'   Typically, `l < length(landmarks)` to reduce complexity.
#' @param use_RSpectra Logical. If TRUE, use `RSpectra::svds` for partial SVD. Recommended for large problems.
#' @param ... Additional arguments passed to `kernel_func`.
#'
#' @return A `bi_projector` object with fields:
#' \describe{
#'   \item{\code{v}}{The eigenvectors (N x ncomp) approximating the kernel eigenbasis.}
#'   \item{\code{s}}{The scores (N x ncomp) = v * diag(sdev), analogous to principal component scores.}
#'   \item{\code{sdev}}{The square roots of the eigenvalues.}
#'   \item{\code{preproc}}{The pre-processing pipeline used.}
#' }
#'
#' @importFrom RSpectra svds
#'
#' @export
#'
#' @examples
#' set.seed(123)
#' X <- matrix(rnorm(1000*1000), 1000, 1000)
#' # Standard Nyström
#' res_std <- nystrom_approx(X, ncomp=5, nlandmarks=20, method="standard")
#' # Double Nyström
#' res_db <- nystrom_approx(X, ncomp=5, nlandmarks=20, method="double", l=10)
nystrom_approx <- function(X, kernel_func=NULL, ncomp=min(dim(X)), 
                           landmarks=NULL, nlandmarks=10, preproc=pass(), 
                           method=c("standard","double"), 
                           l=NULL, use_RSpectra=TRUE, ...) {
  
  method <- match.arg(method)
  
  # Basic checks
  chk::chkor(chk::vld_matrix(X), chk::vld_s4_class(X, "Matrix"))
  N <- nrow(X)
  
  # If no landmarks given, sample them
  if (is.null(landmarks)) {
    if (nlandmarks > N) {
      stop("Number of landmarks cannot exceed the number of samples.")
    }
    landmarks <- sort(sample(N, nlandmarks))
  }
  
  # Ensure kernel_func is valid
  if (!is.null(kernel_func) && !is.function(kernel_func)) {
    stop("kernel_func must be a function or NULL.")
  }
  
  # Default to linear kernel if none provided
  if (is.null(kernel_func)) {
    kernel_func <- function(X, Y, ...) X %*% t(Y)
  }
  
  # Preprocess data
  proc <- prep(preproc)
  X_preprocessed <- init_transform(proc, X)
  
  # Determine sets
  non_landmarks <- setdiff(seq_len(N), landmarks)
  
  X_l <- X_preprocessed[landmarks, , drop=FALSE]         
  X_nl <- if (length(non_landmarks) > 0) X_preprocessed[non_landmarks, , drop=FALSE] else matrix(0, 0, ncol(X_preprocessed))
  
  # Compute kernel submatrices
  K_mm <- kernel_func(X_l, X_l, ...)
  K_nm <- if (length(non_landmarks) > 0) kernel_func(X_nl, X_l, ...) else matrix(0,0,length(landmarks))
  
  # Function for partial SVD or full eigen if needed
  low_rank_decomp <- function(M, k) {
    # For stability, if k >= nrow(M), we just do eigen
    if (!use_RSpectra || k >= nrow(M)) {
      # full eigen decomposition
      eig <- eigen(M, symmetric=TRUE)
      return(list(d=eig$values[1:k], v=eig$vectors[,1:k,drop=FALSE]))
    } else {
      # partial SVD using RSpectra
      sv <- RSpectra::svds(M, k=k)
      # svds returns U,D,V with M = U diag(d) V^T
      # for symmetric M, we want eigen decomposition
      # sv$v are the principal directions
      return(list(d=sv$d^2, v=sv$v)) # d returned by svds are singular values, so lambda = d^2
    }
  }
  
  if (method == "standard") {
    # Standard Nyström as before
    eig_mm <- eigen(K_mm, symmetric=TRUE)
    lambda_mm <- eig_mm$values
    U_mm <- eig_mm$vectors
    
    # Filter out near-zero eigenvalues
    eps <- 1e-8
    keep <- which(lambda_mm > eps)
    if (length(keep) == 0) stop("No significant eigenvalues found in K_mm.")
    
    keep <- keep[seq_len(min(ncomp, length(keep)))]
    lambda_mm <- lambda_mm[keep]
    U_mm <- U_mm[, keep, drop=FALSE]
    
    # Compute U_nm
    inv_sqrt_lambda <- diag(1 / sqrt(lambda_mm), nrow=length(lambda_mm))
    U_nm <- K_nm %*% (U_mm %*% inv_sqrt_lambda)
    
    U_full <- matrix(0, N, length(lambda_mm))
    U_full[landmarks, ] <- U_mm
    if (length(non_landmarks) > 0) {
      U_full[non_landmarks, ] <- U_nm
    }
    
    sdev <- sqrt(lambda_mm)
    s <- U_full %*% diag(sdev, nrow=length(sdev))
    
    out <- bi_projector(
      v = U_full, 
      s = s, 
      sdev = sdev, 
      preproc = proc, 
      classes = "nystrom_approx",
      kernel_func = kernel_func,
      landmarks = landmarks,
      X_landmarks = X_l,
      lambda_mm = lambda_mm,
      U_mm = U_mm,
      ...
    )
    return(out)
    
  } else {
    # Double Nyström Method
    if (is.null(l)) stop("For method='double', you must specify 'l' < length(landmarks).")
    s <- length(landmarks)
    if (l > s) stop("l must be less than or equal to number of landmarks.")
    
    # 1. Approximate principal subspace of K_mm to rank l
    #    Use partial SVD or eigen if l >= s
    approx_l <- low_rank_decomp(K_mm, l)
    # approx_l$d are eigenvalues, approx_l$v are eigenvectors
    lambda_l <- approx_l$d
    V_S_l <- approx_l$v
    
    # Construct W = Phi * V_S_l
    # We have: W[i,j] = sum_p kernel_func(X[i], X_l[p]) * V_S_l[p,j]
    # Instead of a direct sum, we can do: W = [K_nm_full; K_mm_full]^T * V_S_l^(-1) kind of approach.
    # But we have only landmark sets. Let's do a direct approach:
    # Compute K_(X,L) for all X once:
    # This is N x s
    K_all_landmarks <- kernel_func(X_preprocessed, X_l, ...)
    # W = K_all_landmarks * V_S_l, but we must scale appropriately.
    # Actually, V_S_l are eigenvectors of K_mm. 
    # We don't need inverse scaling here (that's for final step), W is just as defined in theory.
    # W = Phi * V_S_l with Phi implicitly represented by kernel_func.
    # From the derivation in double Nyström, W ~ C V_W Sigma_W^{-1} form,
    # but here we treat W simply as: W = K_all_landmarks * (V_S_l * Lambda_l^{-1/2}) for correct scaling.
    # Actually, to follow theory precisely:
    # In double Nyström, first step obtains a subspace V_S_l that approximates principal subspace of K_mm.
    # Those V_S_l are eigenvectors of K_mm, so to map them back to feature space we need sqrt-lambda scaling.
    # Actually the construction differs slightly from the basic form in your previous code.
    # For double Nyström final formula:
    #   We want W = Phi * V_S_l. Given K_mm = V_S_l diag(lambda_l) V_S_l^T,
    #   V_S_l are eigenvectors in landmark space. The mapping Phi is not explicitly known.
    # However, for consistency with the published double Nyström approach:
    # We can approximate feature space direction U_S_l = X_l^phi * V_S_l * diag(1/sqrt(lambda_l))
    # Then W = Phi V_S_l = (K_all_landmarks * V_S_l * diag(1/sqrt(lambda_l)))
    sqrt_lambda_l <- sqrt(lambda_l)
    inv_sqrt_lambda_l <- diag(1/sqrt_lambda_l, nrow=length(lambda_l))
    W <- K_all_landmarks %*% (V_S_l %*% inv_sqrt_lambda_l)
    
    # 2. Compute K_W = W^T W (l x l) and do final low-rank approximation to ncomp
    K_W <- crossprod(W)
    approx_k <- low_rank_decomp(K_W, ncomp)
    lambda_k <- approx_k$d
    V_k <- approx_k$v
    sqrt_lambda_k <- sqrt(lambda_k)
    inv_sqrt_lambda_k <- diag(1/sqrt_lambda_k, nrow=ncomp)
    
    # U_k = W * V_k * inv_sqrt_lambda_k (final eigenvectors in feature space)
    U_k <- W %*% (V_k %*% inv_sqrt_lambda_k)
    
    # Eigenvalues = lambda_k (top ncomp)
    # sdev = sqrt(lambda_k)
    sdev <- sqrt_lambda_k
    
    # Scores = U_k * diag(sdev)
    s <- U_k %*% diag(sdev, nrow=ncomp)
    
    # Store necessary info for out-of-sample extension:
    # For out-of-sample projection, we need the original landmarks and
    # a way to reconstruct projections. Here, the final embedding:
    # approx eigenvectors = U_k, 
    # but to project out-of-sample points, we replicate logic from project.nystrom_approx:
    # approx_eigenvectors(new) = K(new, X_l) * V_S_l * inv_sqrt_lambda_l * V_k * inv_sqrt_lambda_k
    
    # We store intermediate results:
    out <- bi_projector(
      v = U_k, 
      s = s,
      sdev = sdev,
      preproc = proc,
      classes = "nystrom_approx",
      kernel_func = kernel_func,
      landmarks = landmarks,
      X_landmarks = X_l,
      lambda_mm = lambda_l,    # intermediate eigenvalues from first step
      U_mm = V_S_l,            # intermediate eigenvectors from first step
      lambda_final = lambda_k, # final eigenvalues
      V_k = V_k,
      inv_sqrt_lambda_l = inv_sqrt_lambda_l,
      inv_sqrt_lambda_k = inv_sqrt_lambda_k,
      method = "double",
      ...
    )
    return(out)
  }
}


#' @export
project.nystrom_approx <- function(x, new_data, ...) {
  # Project new_data using the (standard or double) Nyström approximation
  kernel_func <- x$kernel_func
  landmarks <- x$landmarks
  X_l <- x$X_landmarks
  
  new_data_p <- reprocess(x$preproc, new_data)
  K_new_landmark <- kernel_func(new_data_p, X_l, ...)
  
  if (!is.null(x$method) && x$method == "double") {
    # Double Nyström projection:
    # approx_eigenvectors(new) = K_new_landmark * U_mm * inv_sqrt_lambda_l * V_k * inv_sqrt_lambda_k
    # where:
    # U_mm = x$U_mm (from first step)
    # inv_sqrt_lambda_l = x$inv_sqrt_lambda_l
    # V_k = x$V_k
    # inv_sqrt_lambda_k = x$inv_sqrt_lambda_k
    approx_eigenvectors <- K_new_landmark %*% (x$U_mm %*% x$inv_sqrt_lambda_l %*% x$V_k %*% x$inv_sqrt_lambda_k)
    
  } else {
    # Standard Nyström projection:
    # approx_eigenvectors(new) = K_new_landmark * U_mm * Lambda_mm^{-1/2}, from the original code
    lambda_mm <- x$lambda_mm
    U_mm <- x$U_mm
    inv_sqrt_lambda <- diag(1 / sqrt(lambda_mm), length(lambda_mm))
    approx_eigenvectors <- K_new_landmark %*% (U_mm %*% inv_sqrt_lambda)
  }
  
  sdev <- x$sdev
  approx_scores <- approx_eigenvectors %*% diag(sdev, length(sdev))
  approx_scores
}
