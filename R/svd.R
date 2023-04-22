#' Singular Value Decomposition (SVD) Wrapper
#'
#' Computes the singular value decomposition of a matrix using one of the specified methods.
#' It is designed to be an easy-to-use wrapper for various SVD methods available in R.
#'
#' @param X the input matrix
#' @param ncomp the number of components to estimate (default: min(dim(X)))
#' @param preproc the pre-processor to apply on the input matrix (e.g., `center()`, `standardize()`, `pass()`)
#' @param method the SVD method to use: 'base', 'fast', 'irlba', 'propack', 'rsvd', or 'svds'
#' @param q parameter passed to method `rsvd` (default: 2)
#' @param p parameter passed to method `rsvd` (default: 10)
#' @param tol minimum eigenvalue magnitude, otherwise component is dropped (default: .Machine$double.eps)
#' @param ... extra arguments passed to the selected SVD function
#' @return an SVD object that extends `projector`
#' @export
#' @importFrom RSpectra svds
#' @importFrom rsvd rsvd
#' @importFrom irlba irlba
#' @importFrom corpcor fast.svd
#' @importFrom svd propack.svd
#' @examples
#' # Load iris dataset and select the first four columns
#' data(iris)
#' X <- iris[, 1:4]
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
  
  chk::chk_s3_class(preproc, "prepper")
  
  proc <- prep(preproc)
  X <- init_transform(proc, X)
  
  res <- switch(method,
                base=svd(X,...),
                fast=corpcor::fast.svd(X, tol),
                rsvd=rsvd::rsvd(X, k=ncomp, q=q, p=p, ...),
                svds=RSpectra::svds(X,k=ncomp),
                propack=svd::propack.svd(X, neig=ncomp,...),
                irlba=irlba::irlba(X, nu=min(ncomp, min(dim(X)) -3), nv=min(ncomp, min(dim(X)) -3)), ...)
  
  keep <- which(res$d^2 > tol)
  
  if (length(keep) == 0) {
    stop("error: all singular values are zero")
  }
  
  ncomp <- min(ncomp,length(keep))
  
  d <- res$d[1:ncomp]
  u <- res$u[,1:ncomp, drop=FALSE]
  v <- res$v[,1:ncomp, drop=FALSE]
  ncomp <- length(1:ncomp)
  
  rm(X)
  rm(res)
  bi_projector(v, s=u %*% diag(d, nrow=ncomp, ncol=ncomp), 
               sdev=d, u=u, preproc=proc, 
               classes="svd", method=method)
}

#' @export
std_scores.svd <- function(x,...) {
  sqrt(nrow(x$u)-1) * x$u 
}




