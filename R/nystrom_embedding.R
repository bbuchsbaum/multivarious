#' Nyström approximation for kernel-based decomposition (Unified Version)
#'
#' Approximate the eigen-decomposition of a large kernel matrix K using either
#' the standard Nyström method (Williams & Seeger, 2001) or the Double Nyström method 
#' (Lim et al., 2015, Algorithm 3). 
#'
#' The Double Nyström method introduces an intermediate step that reduces the
#' size of the decomposition problem, potentially improving efficiency and scalability.
#'
#' @param X A numeric matrix or data frame of size (N x D), where N is number of samples.
#' @param kernel_func A kernel function with signature `kernel_func(X, Y, ...)`.
#'   If NULL, defaults to a linear kernel: `X %*% t(Y)`.
#' @param ncomp Number of components (eigenvectors/eigenvalues) to return. 
#'   Cannot exceed the number of landmarks. Default capped at `length(landmarks)`.
#' @param landmarks A vector of row indices (1-based, from X) specifying the landmark points.
#'   If NULL, `nlandmarks` points are sampled uniformly at random.
#' @param nlandmarks The number of landmark points to sample if `landmarks` is NULL. Default is 10.
#' @param preproc A pre-processing pipeline object (e.g., from `prep()`) or a pre-processing function 
#'   (default `pass()`) to apply before computing the kernel.
#' @param method Either "standard" (the classic single-stage Nyström) or "double" (the two-stage Double Nyström method).
#' @param center Logical. If TRUE, attempts kernel centering. Default FALSE. 
#'   **Note:** True kernel centering (required for equivalence to Kernel PCA) is 
#'   computationally expensive and **not fully implemented**. Setting `center=TRUE` currently only 
#'   issues a warning. For results equivalent to standard PCA, use a linear kernel 
#'   and center the input data `X` (e.g., via `preproc`). See Details.
#' @param l Intermediate rank for the double Nyström method. Ignored if `method="standard"`.
#'   Typically, `l < length(landmarks)` to reduce complexity.
#' @param use_RSpectra Logical. If TRUE, use `RSpectra::svds` for partial SVD. Recommended for large problems.
#' @param ... Additional arguments passed to `kernel_func`.
#'
#' @return A `bi_projector` object with class "nystrom_approx" and additional fields:
#' \describe{
#'   \item{\code{v}}{The eigenvectors (N x ncomp) approximating the kernel eigenbasis.}
#'   \item{\code{s}}{The scores (N x ncomp) = v * diag(sdev), analogous to principal component scores.}
#'   \item{\code{sdev}}{The square roots of the eigenvalues.}
#'   \item{\code{preproc}}{The pre-processing pipeline used.}
#'   \item{\code{meta}}{A list containing parameters and intermediate results used (method, landmarks, kernel_func, etc.).}
#' }
#'
#' @details 
#' **Kernel Centering:** Standard Kernel PCA requires the kernel matrix K to be centered 
#' in the feature space (Schölkopf et al., 1998). This implementation currently 
#' **does not perform kernel centering** by default (`center=FALSE`) due to computational complexity. 
#' Consequently, with non-linear kernels, the results approximate the eigen-decomposition 
#' of the *uncentered* kernel matrix, and are not strictly equivalent to Kernel PCA. 
#' If using a linear kernel, centering the input data `X` (e.g., using `preproc=prep(center())`) 
#' yields results equivalent to standard PCA, which is often sufficient.
#' 
#' **Standard Nyström:** Uses the method from Williams & Seeger (2001), including the 
#' `sqrt(m/N)` scaling for eigenvectors and `N/m` for eigenvalues (`m` landmarks, `N` samples).
#' 
#' **Double Nyström:** Implements Algorithm 3 from Lim et al. (2015).
#'
#' @references
#' Schölkopf, B., Smola, A., & Müller, K. R. (1998). Nonlinear component analysis as a kernel eigenvalue problem. 
#' *Neural computation*, 10(5), 1299-1319.
#' 
#' Williams, C. K. I., & Seeger, M. (2001). Using the Nyström Method to Speed Up Kernel Machines. 
#' In *Advances in Neural Information Processing Systems 13* (pp. 682-688).
#' 
#' Lim, D., Jin, R., & Zhang, L. (2015). An Efficient and Accurate Nystrom Scheme for Large-Scale Data Sets. 
#' *Proceedings of the Twenty-Ninth AAAI Conference on Artificial Intelligence* (pp. 2765-2771).
#'
#' @importFrom RSpectra svds
#' @importFrom stats rnorm
#' @export
#'
#' @examples
#' set.seed(123)
#' # Smaller example matrix
#' X <- matrix(rnorm(1000*300), 1000, 300)
#' 
#' # Standard Nyström
#' res_std <- nystrom_approx(X, ncomp=5, nlandmarks=50, method="standard")
#' print(res_std)
#' 
#' # Double Nyström
#' res_db <- nystrom_approx(X, ncomp=5, nlandmarks=50, method="double", l=20)
#' print(res_db)
#' 
#' # Projection (using standard result as example)
#' scores_new <- project(res_std, X[1:10,])
#' head(scores_new)
nystrom_approx <- function(X, kernel_func=NULL, ncomp=NULL,
                           landmarks=NULL, nlandmarks=10, preproc=pass(),
                           method=c("standard","double"),
                           center=FALSE, # Added center argument
                           l=NULL, use_RSpectra=TRUE, ...) {
  
  method <- match.arg(method)
  
  # Basic checks
  chk::chkor_vld(chk::vld_matrix(X), chk::vld_s4_class(X, "Matrix"))
  N <- nrow(X)
  
  # If no landmarks given, sample them; otherwise validate input
  if (is.null(landmarks)) {
    if (nlandmarks > N) {
      warning("Number of landmarks requested exceeds number of samples. Using N landmarks.")
      nlandmarks <- N
    }
    if (nlandmarks <= 0) {
        stop("'nlandmarks' must be positive.")
    }
    landmarks <- sort(sample(N, nlandmarks))
  } else {
    # Validate user-supplied landmarks
    if (any(!is.finite(landmarks))) {
      stop("'landmarks' must be finite indices.")
    }
    if (any(landmarks != as.integer(landmarks))) {
      stop("'landmarks' must be integer indices.")
    }
    landmarks <- as.integer(landmarks)
    if (any(landmarks < 1L | landmarks > N)) {
      stop("'landmarks' indices out of range [1..N].")
    }
    landmarks <- sort(unique(landmarks))
    if (length(landmarks) == 0) stop("'landmarks' cannot be empty after validation.")
  }
  m <- length(landmarks)
  
  # Validate ncomp
  if (is.null(ncomp)) {
      ncomp <- min(m, ncol(X)) # Default ncomp capped by landmarks
  } else {
      chk::chk_whole_number(ncomp)
      chk::chk_gt(ncomp, 0)
      if (ncomp > m) {
          warning(sprintf("Requested ncomp (%d) exceeds number of landmarks (%d). Setting ncomp = %d.", ncomp, m, m))
          ncomp <- m
      }
  }
  
  # Ensure kernel_func is valid
  if (!is.null(kernel_func) && !is.function(kernel_func)) {
    stop("kernel_func must be a function or NULL.")
  }
  
  # Default to linear kernel if none provided
  if (is.null(kernel_func)) {
    kernel_func <- function(X, Y, ...) X %*% t(Y)
  }
  
  # Placeholder check for centering - not implemented yet
  if (center) {
      warning("Kernel centering (center=TRUE) is requested but not yet implemented. Proceeding with uncentered kernel.")
      # Future: implement kernel centering logic here or within kernel calls
  }

  # Store original input dimension before preprocessing
  original_ncol <- ncol(X)

  # Fit and transform preprocessing pipeline
  result <- fit_transform(preproc, X)
  proc <- result$preproc  # Fitted preprocessor
  X_preprocessed <- result$transformed

  # Determine sets
  non_landmarks <- setdiff(seq_len(N), landmarks)

  X_l <- X_preprocessed[landmarks, , drop=FALSE]
  X_nl <- if (length(non_landmarks) > 0) X_preprocessed[non_landmarks, , drop=FALSE] else matrix(0, 0, ncol(X_preprocessed))

  # Compute kernel submatrices
  K_mm <- kernel_func(X_l, X_l, ...)
  K_nm <- if (length(non_landmarks) > 0) kernel_func(X_nl, X_l, ...) else matrix(0, 0, m)

  # Function for partial SVD or full eigen if needed
  low_rank_decomp <- function(M, k) {
    # Check if k exceeds dimensions or if RSpectra should not be used
    if (k >= nrow(M) || !use_RSpectra) {
      # full eigen decomposition
      eig <- eigen(M, symmetric=TRUE)
      # Eigenvectors from eigen() are already orthonormal, no need to re-orthogonalize
      return(list(d = eig$values[1:k], v = eig$vectors[, 1:k, drop=FALSE]))
    } else {
      # partial SVD using RSpectra
      sv <- tryCatch(RSpectra::svds(M, k=k, nu=0, nv=k), # only need V
                     error = function(e) {
                         warning(sprintf("RSpectra::svds failed: %s. Falling back to eigen().", e$message))
                         NULL
                     })

      if (is.null(sv)) {
         # Fallback to eigen if svds fails
         eig <- eigen(M, symmetric=TRUE)
         # Eigenvectors from eigen() are already orthonormal
         return(list(d = eig$values[1:k], v = eig$vectors[, 1:k, drop=FALSE]))
      } else {
          # svds returns U,D,V with M = U D V^T
          # For symmetric PSD M, singular values equal eigenvalues (not squared).
          # sv$v are right singular vectors = eigenvectors for symmetric M.
          return(list(d = sv$d, v = sv$v))
      }
    }
  }

  meta_info <- list(method = method,
                    landmarks = landmarks,
                    kernel_func = kernel_func,
                    ncomp = ncomp,
                    center = center,
                    X_landmarks = X_l,  # Store landmark data for projection
                    original_ncol = original_ncol,  # Store original input dimension
                    extra_args = list(...)) # Store extra args passed to kernel
  
  if (method == "standard") {
    # Standard Nyström with proper scaling
    eig_mm <- eigen(K_mm, symmetric=TRUE)
    lambda_mm_raw <- eig_mm$values
    U_mm_raw <- eig_mm$vectors
    
    # Filter out near-zero eigenvalues using scaled epsilon
    eps <- max(lambda_mm_raw) * .Machine$double.eps * 100 
    keep <- which(lambda_mm_raw > eps)
    if (length(keep) == 0) stop("No significant eigenvalues found in K_mm.")
    
    # Keep only up to ncomp components
    keep <- keep[seq_len(min(ncomp, length(keep)))]
    lambda_mm <- lambda_mm_raw[keep]
    U_mm <- U_mm_raw[, keep, drop=FALSE]
    k_eff <- length(lambda_mm) # Effective number of components
    
    # Approximate eigenvalues of full K: lambda_hat = (N/m) * lambda_mm
    lambda_hat <- (N / m) * lambda_mm
    sdev <- sqrt(lambda_hat)
    
    # Approximate eigenvectors U_hat = sqrt(m/N) * [U_mm ; K_nm %*% U_mm %*% diag(1/lambda_mm)]
    scaling_factor_u <- sqrt(m / N)
    inv_lambda_mm_mat <- diag(1 / lambda_mm, nrow = k_eff, ncol = k_eff)
    U_mm_scaled <- U_mm * scaling_factor_u
    U_nm_scaled <- (K_nm %*% (U_mm %*% inv_lambda_mm_mat)) * scaling_factor_u
    
    U_full <- matrix(0, N, k_eff)
    U_full[landmarks, ] <- U_mm_scaled
    if (length(non_landmarks) > 0) {
      U_full[non_landmarks, ] <- U_nm_scaled
    }
    
    # Scores s = U_hat * diag(sdev_hat)
    # s <- U_full %*% diag(sdev, nrow=k_eff) # Old way
    s <- sweep(U_full, 2, sdev, "*") # Efficient way
    
    # Store results
    out <- bi_projector(
      v = U_full, 
      s = s, 
      sdev = sdev, 
      preproc = proc, 
      classes = c("nystrom_approx", "standard"),
      meta = c(meta_info, list( 
        lambda_mm = lambda_mm, # Eigenvalues of K_mm used
        U_mm = U_mm           # Eigenvectors of K_mm used
      )),
      ...
    )
    return(out)
    
  } else { # method == "double"
    # Double Nyström Method
    if (is.null(l)) stop("For method='double', you must specify intermediate rank 'l'.")
    if (l <= 0 || l > m) stop("Intermediate rank 'l' must be > 0 and <= number of landmarks.")
    if (l < ncomp) {
        warning(sprintf("Intermediate rank l (%d) is less than final ncomp (%d). Setting l = ncomp.", l, ncomp))
        l <- ncomp
    }
    
    # 1. Approximate principal subspace of K_mm to rank l
    approx_l <- low_rank_decomp(K_mm, l)
    lambda_l_raw <- approx_l$d
    V_S_l_raw <- approx_l$v
    
    # Filter near-zero eigenvalues
    eps_l <- max(lambda_l_raw) * .Machine$double.eps * 100
    keep_l <- which(lambda_l_raw > eps_l)
    if (length(keep_l) == 0) stop("No significant eigenvalues found in K_mm first stage (rank l). Reduce l?")
    
    l_eff <- length(keep_l)
    if (l_eff < ncomp) {
        warning(sprintf("Effective rank after first stage (%d) is less than requested ncomp (%d). Proceeding with %d components.", l_eff, ncomp, l_eff))
        ncomp <- l_eff # Adjust final ncomp down
    }
    lambda_l <- lambda_l_raw[keep_l]
    V_S_l <- V_S_l_raw[, keep_l, drop=FALSE]
    
    # Compute K_(X,L) for all X once - potentially large N x m matrix
    K_all_landmarks <- kernel_func(X_preprocessed, X_l, ...)
    
    # Construct W = K_all_landmarks * (V_S_l * Lambda_l^{-1/2}) 
    # Using robust diag for the inverse sqrt lambda matrix
    inv_sqrt_lambda_l_mat <- diag(1 / sqrt(lambda_l), nrow = l_eff, ncol = l_eff)
    W <- K_all_landmarks %*% (V_S_l %*% inv_sqrt_lambda_l_mat)
    
    # 2. Compute K_W = W^T W (l x l) and do final low-rank approximation to ncomp
    K_W <- crossprod(W)
    approx_k <- low_rank_decomp(K_W, ncomp) # ncomp might have been adjusted down
    lambda_k_raw <- approx_k$d
    V_k_raw <- approx_k$v
    
    # Filter near-zero eigenvalues for final step
    eps_k <- max(lambda_k_raw) * .Machine$double.eps * 100
    keep_k <- which(lambda_k_raw > eps_k)
    if (length(keep_k) == 0) stop("No significant eigenvalues found in second stage (K_W).")
    
    k_eff <- length(keep_k)
    if (k_eff < ncomp) {
        warning(sprintf("Effective rank after second stage (%d) is less than requested ncomp (%d). Final components: %d.", k_eff, ncomp, k_eff))
    }
    lambda_k <- lambda_k_raw[keep_k]
    V_k <- V_k_raw[, keep_k, drop=FALSE]
    
    # Final eigenvalues = lambda_k 
    sdev <- sqrt(lambda_k)
    
    # Final eigenvectors U_k = W * V_k * Lambda_k^{-1/2}
    inv_sqrt_lambda_k_mat <- diag(1 / sqrt(lambda_k), nrow = k_eff, ncol = k_eff)
    U_k <- W %*% (V_k %*% inv_sqrt_lambda_k_mat)
    
    # Scores = U_k * diag(sdev)
    # s <- U_k %*% diag(sdev, nrow=k_eff) # Old way
    s <- sweep(U_k, 2, sdev, "*") # Efficient way
    
    # Store intermediate results needed for projection
    out <- bi_projector(
      v = U_k, 
      s = s,
      sdev = sdev,
      preproc = proc,
      classes = c("nystrom_approx", "double"),
      meta = c(meta_info, list(
          # Intermediate results needed for projection:
          V_S_l = V_S_l, # U_mm in standard Nystrom context (eigenvectors of K_mm subset)
          inv_sqrt_lambda_l = inv_sqrt_lambda_l_mat, # Diagonal matrix
          V_k = V_k, # Eigenvectors of K_W
          inv_sqrt_lambda_k = inv_sqrt_lambda_k_mat, # Diagonal matrix
          # Store original eigenvalues for reference if needed?
          lambda_l_raw = lambda_l_raw,
          lambda_k_raw = lambda_k_raw
        )
      ),
      ...
    )
    return(out)
  }
}


#' Project new data using a Nyström approximation model
#'
#' @param x A `nystrom_approx` object (inheriting from `bi_projector`).
#' @param new_data New data matrix to project.
#' @param ... Additional arguments (currently ignored).
#' @return A matrix of projected scores.
#' @export
project.nystrom_approx <- function(x, new_data, ...) {
  # Extract necessary info from the meta slot
  meta <- x$meta
  if (is.null(meta) || is.null(meta$method)) {
      stop("Nystrom model object 'x' appears incomplete or corrupted (missing meta information).")
  }
  
  kernel_func <- meta$kernel_func
  landmarks <- meta$landmarks
  # X_l <- x$X_landmarks # This might not be stored directly anymore
  # Need to recompute X_l from original X if not stored, or store it.
  # Let's assume X_l IS stored or reconstructable via preproc if needed
  # We need X_l for K(new, X_l). It should be retrievable if preproc is identity or stored.
  # For now, assume it was stored (though it wasn't explicitly in constructor output). 
  # It is better to store X_l in the object.
  # Let's modify the constructor to store X_l in meta.
  if (is.null(meta$X_landmarks)) {
     stop("Cannot project: Landmark data (X_l) not found in the model object.")
  }
  X_l <- meta$X_landmarks
  
  # Use the *projector* object for reprocess to get consistent preprocessing
  new_data_p <- reprocess(x, new_data)
  K_new_landmark <- do.call(kernel_func, c(list(new_data_p, X_l), meta$extra_args))
  
  if (meta$method == "double") {
    # Double Nyström projection:
    # approx_scores(new) = K_new_landmark * V_S_l * inv_sqrt_lambda_l * V_k * inv_sqrt_lambda_k * diag(sdev)
    # where sdev = sqrt(lambda_k)
    
    if (is.null(meta$V_S_l) || is.null(meta$inv_sqrt_lambda_l) || is.null(meta$V_k) || is.null(meta$inv_sqrt_lambda_k)){
        stop("Double Nystrom model object 'x' is missing necessary components for projection.")
    }
    
    # Combine projection matrices
    proj_matrix <- meta$V_S_l %*% meta$inv_sqrt_lambda_l %*% meta$V_k %*% meta$inv_sqrt_lambda_k
    # approx_eigenvectors <- K_new_landmark %*% proj_matrix
    # approx_scores <- sweep(approx_eigenvectors, 2, x$sdev, "*")
    
    # Or, more directly: Scores = K(new,L) * ProjMatrix * diag(sdev)
    scores <- K_new_landmark %*% (proj_matrix %*% diag(x$sdev, nrow=length(x$sdev), ncol=length(x$sdev)))
    
  } else { # method == "standard"
    # Standard Nyström projection:
    # For training: U_full = sqrt(m/N) * [U_mm; K_nm * U_mm * inv(Lambda_mm)]
    # Scores = U_full * diag(sdev)
    # For test: U_test = sqrt(m/N) * K_test_landmark * U_mm * inv(Lambda_mm)
    # Scores_test = U_test * diag(sdev)

    if (is.null(meta$U_mm) || is.null(meta$lambda_mm)) {
        stop("Standard Nystrom model object 'x' is missing necessary components for projection.")
    }

    lambda_mm <- meta$lambda_mm
    U_mm <- meta$U_mm
    k_eff <- length(lambda_mm)

    # Ensure lambda_mm > 0 before inversion
    lambda_mm[lambda_mm <= 0] <- .Machine$double.eps

    # Get dimensions
    N <- nrow(x$v) # Training sample size
    m <- length(meta$landmarks)

    # Projection formula: K_new_landmark * U_mm * diag(sqrt(m/N) * sdev / lambda_mm)
    # This matches the training formula for non-landmarks
    scaling_factor <- sqrt(m / N)
    proj_vector <- (scaling_factor * x$sdev) / lambda_mm

    # Handle potential Inf/NaN
    proj_vector[!is.finite(proj_vector)] <- 0

    proj_matrix <- U_mm %*% diag(proj_vector, nrow=k_eff, ncol=k_eff)

    scores <- K_new_landmark %*% proj_matrix
  }
  
  scores
}

#' Reprocess data for Nyström approximation
#'
#' Apply preprocessing to new data for projection using a Nyström approximation.
#' This method overrides the default `reprocess.projector` to handle the fact that
#' Nyström components are in kernel space (not feature space).
#'
#' @param x A `nystrom_approx` object
#' @param new_data A matrix with the same number of columns as the original training data
#' @param colind Optional column indices (not typically used for Nyström)
#' @param ... Additional arguments (ignored)
#'
#' @return Preprocessed data matrix
#' @export
reprocess.nystrom_approx <- function(x, new_data, colind = NULL, ...) {
  # For Nyström, we need to check against the original input dimension,
  # not the component dimension (which is N x ncomp in kernel space)
  original_ncol <- x$meta$original_ncol

  if (is.null(original_ncol)) {
    stop("Cannot reprocess: original input dimension not found in model metadata. ",
         "This may be an older nystrom_approx object that needs to be retrained.")
  }

  if (is.null(colind)) {
    # Full dimension check
    chk::chk_equal(ncol(new_data), original_ncol)
    transform(x$preproc, new_data)
  } else {
    chk::chk_equal(length(colind), ncol(new_data))
    transform(x$preproc, new_data, colind)
  }
}
