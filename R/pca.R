#' principal components analysis
#' 
#' Compute the directions of maximal variance in a matrix via the singula value decomposition
#' 
#' @param X the data matrix
#' @param ncomp the number of requested components to estimate
#' @param preproc the pre_processor
#' @param method th svd method (passed to `svd_wrapper`)
#' @param extra arguments
#' @export
#' 
#' @examples 
#' 
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' res <- pca(X, ncomp=4)
#' tres <- truncate(res, 3)
pca <- function(X, ncomp=min(dim(X)), preproc=center(), method = c("fast", "base", "irlba", "propack", "rsvd", "svds"), ...) {
  chk::chkor(chk::chk_matrix(X), chk::chk_s4_class("Matrix"))
  
  method <- match.arg(method)
  svdres <- svd_wrapper(X, ncomp, preproc, method=method, ...)
  
  ## todo add rownames slot to `bi_projector`?
  if (!is.null(row.names(scores))) {
    row.names(scores) <- row.names(X)[seq_along(svdres$d)]
  }
  

  attr(svdres, "class") <- c("pca", attr(svdres, "class"))
  svdres
}

#' @export
truncate.pca <- function(x, ncomp) {
  chk_range(ncomp, c(1, ncomp(x)))
  x$v <- x$v[,1:ncomp, drop=FALSE]
  x$sdev <- x$sdev[1:ncomp]
  x$s <- x$s[,1:ncomp,drop=FALSE]
  x$u <- x$u[, 1:ncomp, drop=FALSE]
  x
}