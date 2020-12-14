#' construct a `bi_projector` instance
#' 
#' 
#' A `bi_projector` offers a two-way mapping from from samples (rows) to scores and from variables (columns) to components.
#' Thus, one can project from D-dimensional input space to d-dimensional subspace. And one can project (`project_vars`) from n-dimensional 
#' variable space to the d-dimensional component space.
#' 
#' 
#' @export
#' @inheritParams projector
#' @param preproc
#' @param ncomp
#' @param s the score matrix
#' @param sdev the standard deviations of th score matrix
#' @export
#' 
#' X <- matrix(rnorm(10*20), 10, 20)
#' svdfit <- svd(X)
#' 
#' p <- bi_projector(svdfit$v, s = svdfit$u %*% diag(svdfit$d), sdev=svdfit$d)
#' pcres <- prcomp(X, center=FALSE, scale=FALSE)
bi_projector <- function(v, s, sdev, preproc=NULL, classes=NULL, ...) {
  chk::vld_matrix(v)
  chk::vld_matrix(s)
  chk::vld_numeric(sdev)
  chk::chk_equal(length(sdev), ncol(s))
  chk::chk_equal(ncol(v), length(sdev))
  
  out <- projector(v, preproc=preproc, s=s, sdev=sdev, classes=c(classes, "bi_projector"), ...)
}

scores.bi_projector <- function(x) {
  x$s
}

sdev.bi_projector <- function(x) {
  x$sdev
}


project_vars <- function(x, new_data) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data)
  }
  
  chk::chk_equal(nrow(new_data), nrow(scores(x)))
  
  variance <- sdev(x)^2
  t(new_data) %*% (scores(x)) %*% diag(1/variance, nrow=length(variaance), ncol=length(variance))
}


genreconstruct <- function(x, comp, rowind, colind) {
  scores(x)[rowind,comp] %*% t(components(x)[,comp,drop=FALSE])[,colind]
}

#' @export
reconstruct.bi_projector <- function(x, comp=1:ncomp(x), rowind=1:nrow(scores(x)), colind=1:nrow(components(x))) {
  chk_numeric(comp)
  chk_true(max(comp) <= ncomp(x))
  chk_numeric(rowind)
  chk_numeric(colind)
  chk_range(comp, c(1,ncomp(x)))
  chk_range(rowind, c(1,nrow(scores(x))))
  chk_range(colind, c(1,nrow(components(x))))
  genreconstruct(x,comp, rowind, colind)
}

  

