#' svd_wrapper
#' 
#' @param X the \code{matrix}
#' @param ncomp number of components to estimate
#' @param method the svd method to use. One of: 'base', 'fast', 'irlba', 'propack', 'rsvd', 'svds'
#' @param q parameter passed to method `rsvd`
#' @param p parameter passed to method `rsvd`
#' @param tol minimum eigenvalue magnitude, otherwise component is dropped
#' @export
#' @importFrom RSpectra svds
#' @importFrom rsvd rsvd
#' @importFrom irlba irlba
#' @importFrom corpcor fast.svd
#' @importFrom svd propack.svd
#' 
#' @return 
#' 
#' @examples 
#' 
#' data(iris)
#' X <- iris[,1:4]
#' 
#' fit <- svd_wrapper(X, ncomp=3, preproc=center(), method="base")
#' 
#' 
#' an `svd` object that extends `projector`
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
                fast=corpcor:::fast.svd(X, tol),
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
std_scores.svd <- function(x) {
  sqrt(nrow(x$u)-1) * x$u 
}




