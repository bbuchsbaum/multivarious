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
#' X <- matrix(rnorm(10*20), 10, 20)
#' res <- pca(X, ncomp=10, preproc=center())
pca <- function(X, ncomp=min(dim(X)), preproc=center(), method = c("base", "fast", "irlba","propack", "rsvd", "svds"), ...) {
  chk::chkor(chk::chk_matrix(X), chk::chk_s4_class("Matrix"))
  
  method <- match.arg(method)
  svdres <- svd_wrapper(X, ncomp, preproc, method=method, ...)
  
  ## todo add rownames slot to `bi_projector`?
  if (!is.null(row.names(scores))) {
    row.names(scores) <- row.names(X)[seq_along(svdres$d)]
  }
  

  attr(svdres, "class") <- c("pca", attr(res, "class"))
  svdres
  
  
}