#' Singular Value Decomposition (SVD) Wrapper
#'
#' Computes the singular value decomposition of a matrix using one of the specified methods.
#' It is designed to be an easy-to-use wrapper for various SVD methods available in R.
#'
#' @param X the input matrix
#' @param ncomp the number of components to estimate (default: min(dim(X)))
#' @param preproc the pre-processor to apply on the input matrix (e.g., `center()`, `standardize()`, `pass()`) 
#'              Can be a `prepper` object or a pre-processing function.
#' @param method the SVD method to use: 'base', 'fast', 'irlba', 'propack', 'rsvd', or 'svds'
#' @param q parameter passed to method `rsvd` (default: 2)
#' @param p parameter passed to method `rsvd` (default: 10)
#' @param tol minimum relative tolerance for dropping singular values (compared to the largest). Default: `.Machine$double.eps`.
#' @param ... extra arguments passed to the selected SVD function
#' @return an SVD object that extends `bi_projector`
#' @export
#' @importFrom RSpectra svds
#' @importFrom rsvd rsvd
#' @importFrom irlba irlba
#' @importFrom corpcor fast.svd
#' @importFrom svd propack.svd
#' @examples
#' # Load iris dataset and select the first four columns
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#'
#' # Compute SVD using the base method and 3 components
#' fit <- svd_wrapper(X, ncomp = 3, preproc = center(), method = "base")
svd_wrapper <- function(X, ncomp=min(dim(X)), 
                        preproc=pass(),
                        method=c("fast", "base", "irlba", 
                                 "propack", "rsvd", "svds"), 
                        q=2,
                        p=10,
                        tol=.Machine$double.eps,
                        ...) {
  method <- match.arg(method)
  
  # FIXME: remove old API usage in v1.0
  # proc <- prep(preproc)
  # Xp <- init_transform(proc, X)
  
  # Use new API
  result <- fit_transform(preproc, X)
  proc <- result$preproc
  Xp <- result$transformed
  
  # Cap ncomp based on dimensions and method requirements
  max_k <- min(dim(Xp))
  # RSpectra and irlba require k < min(dim), base/fast/propack/rsvd work up to min(dim)
  safe_k <- if (method %in% c("svds", "irlba")) max_k - 1 else max_k
  if (safe_k < 1) stop("Matrix dimensions too small for requested SVD method.")
  
  k <- min(ncomp, safe_k) # Final number of components to request
  if (k < ncomp) {
      warning(sprintf("Requested ncomp=%d reduced to k=%d due to matrix dimensions or method constraints.", ncomp, k))
  }
  if (k <= 0) stop("Cannot request 0 or negative components (k).", call. = FALSE)
  
  res <- tryCatch({ # Wrap the switch in tryCatch for robustness
    switch(method,
           base = svd(Xp, nu = k, nv = k, ...),
           fast = {
               if (!requireNamespace("corpcor", quietly = TRUE)) stop("Package 'corpcor' needed for method='fast'.", call.=FALSE)
               corpcor::fast.svd(Xp, tol = 0) # Use tol later for relative filtering
           },
           rsvd = {
               if (!requireNamespace("rsvd", quietly = TRUE)) stop("Package 'rsvd' needed for method='rsvd'.", call.=FALSE)
               rsvd::rsvd(Xp, k = k, q = q, p = p, ...)
           },
           svds = {
               if (!requireNamespace("RSpectra", quietly = TRUE)) stop("Package 'RSpectra' needed for method='svds'.", call.=FALSE)
               RSpectra::svds(Xp, k = k, ...)
            },
           propack = {
               if (!requireNamespace("svd", quietly = TRUE)) stop("Package 'svd' needed for method='propack'.", call.=FALSE)
               # propack needs neig = number of eigenvalues = k
               svd::propack.svd(Xp, neig = k, ...)
           },
           irlba = {
               if (!requireNamespace("irlba", quietly = TRUE)) stop("Package 'irlba' needed for method='irlba'.", call.=FALSE)
               irlba::irlba(Xp, nu = k, nv = k, ...)
           },
           stop("Unknown SVD method specified: ", method) # Default stop
    )
   }, error = function(e) {
       stop(sprintf("SVD computation failed for method '%s': %s", method, e$message), call. = FALSE)
   })
  
  # Filter based on relative tolerance
  if (length(res$d) == 0) stop("SVD returned zero singular values.")
  keep <- which(res$d > tol * res$d[1])
  
  if (length(keep) == 0) {
    stop("All singular values are below the relative tolerance.")
  }
  
  # Final ncomp is the minimum of originally requested (k) and kept singular values
  ncomp_final <- min(k, length(keep))
  
  d <- res$d[keep[1:ncomp_final]]
  u <- res$u[, keep[1:ncomp_final], drop=FALSE]
  v <- res$v[, keep[1:ncomp_final], drop=FALSE]
  
  bi_projector(v, s=u %*% diag(d, nrow=ncomp_final, ncol=ncomp_final), 
               sdev=d, u=u, preproc=proc, 
               classes="svd", method=method)
}

#' Calculate Standardized Scores for SVD results
#'
#' Computes standardized scores from an SVD result performed by `svd_wrapper`.
#' These scores are scaled to have approximately unit variance, assuming the original
#' data used for SVD was centered. They differ from the `s` component of the 
#' `svd` object, which contains scores scaled by singular values.
#'
#' @param x An object of class `svd`, typically from `svd_wrapper`.
#' @param ... Extra arguments (ignored).
#' @return A matrix of standardized scores (N x k) with columns having variance close to 1.
#' @export
std_scores.svd <- function(x,...) {
  N <- nrow(x$u)
  if (N <= 1) {
      warning("Cannot compute standardized scores with N <= 1. Returning raw U vectors.")
      return(x$u)
  }
  # Scale U vectors to have variance ~1
  sqrt(N - 1) * x$u 
}




