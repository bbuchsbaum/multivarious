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
#' @importFrom svd propack
#' 
#' @return 
#' 
#' a `projector` object
svd_wrapper <- function(X, ncomp=min(dim(X)), 
                        method=c("base", "fast", "irlba", 
                                 "propack", "rsvd", "svds"), 
                        q=2,
                        p=10,
                        
                        tol=.Machine$double.eps,
                        ...) {
  method <- match.arg(method)
  
  res <- switch(method,
                base=svd(X,...),
                fast=corpcor:::fast.svd(X, tol),
                rsvd=rsvd::rsvd(X, k=ncomp, q=q, p=p, ...),
                svds=RSpectra::svds(X,k=ncomp),
                propack=svd::propack.svd(X, neig=ncomp,...),
                irlba=irlba::irlba(X, nu=min(ncomp, min(dim(X)) -3), nv=min(ncomp, min(dim(X)) -3)), ...)
  
  
  keep <- which(res$d^2 > tol)
  ncomp <- min(ncomp,length(keep))
  
  res$d <- res$d[keep]
  res$u <- res$u[,keep, drop=FALSE]
  res$v <- res$v[,keep, drop=FALSE]
  res$ncomp <- length(keep)
  
  bi_projector(res$v, s=res$u %*% diag(res$d), sdev=res$d, preproc=NULL, classes="svd", method=method)
}

