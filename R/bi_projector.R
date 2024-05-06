#' Construct a bi_projector instance
#'
#' A bi_projector offers a two-way mapping from samples (rows) to scores and from variables (columns) to components.
#' Thus, one can project from D-dimensional input space to d-dimensional subspace. And one can project (project_vars) from n-dimensional
#' variable space to the d-dimensional component space. The singular value decomposition is a canonical example of such a two-way mapping.
#'
#' @inheritParams projector
#' @param s The score matrix
#' @param sdev The standard deviations of the score matrix
#' @param preproc (optional) A pre-processing pipeline, default is prep(pass())
#' @param classes (optional) A character vector specifying the class attributes of the object, default is NULL
#' @return A bi_projector object
#' @examples
#' X <- matrix(rnorm(200), 10, 20)
#' svdfit <- svd(X)
#'
#' p <- bi_projector(svdfit$v, s = svdfit$u %% diag(svdfit$d), sdev=svdfit$d)
#' @export
bi_projector <- function(v, s, sdev, preproc=prep(pass()), classes=NULL, ...) {
  chk::vld_matrix(v)
  chk::vld_matrix(s)
  chk::vld_numeric(sdev)
  chk::chk_equal(length(sdev), ncol(s))
  chk::chk_equal(ncol(v), length(sdev))
  
  out <- projector(v, preproc=preproc, s=s, sdev=sdev, classes=c(classes, "bi_projector"), ...)
}



#' @export
scores.bi_projector <- function(x,...) {
  x$s
}

#' @export
sdev.bi_projector <- function(x) {
  x$sdev
}

#' @export
project_vars.bi_projector <- function(x, new_data,...) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data)
  }
  
  sc <- scores(x)
  chk::chk_equal(nrow(new_data), nrow(sc))
  
  variance <- sdev(x)^2
  t(new_data) %*% (sc) %*% diag(1/variance, nrow=length(variance), ncol=length(variance))
}

#' @keywords internal
#' @noRd
genreconstruct <- function(x, comp, rowind, colind) {
  ip <- inverse_projection(x)
  out <- scores(x)[rowind,comp,drop=FALSE] %*% ip[comp,,drop=FALSE][,colind,drop=FALSE]
  reverse_transform(x$preproc, out)
}

#' @export
reconstruct.bi_projector <- function(x, comp=1:ncomp(x), rowind=1:nrow(scores(x)), 
                                     colind=1:nrow(coefficients(x)), ...) {
  chk::chk_numeric(comp)
  chk::chk_true(max(comp) <= ncomp(x))
  chk::chk_numeric(rowind)
  chk::chk_numeric(colind)
  chk::chk_range(comp, c(1,ncomp(x)))
  chk::chk_range(rowind, c(1,nrow(scores(x))))
  chk::chk_range(colind, c(1,nrow(coefficients(x))))
  genreconstruct(x,comp, rowind, colind)
}

#' @export
residuals.bi_projector <- function(x, ncomp=ncomp(x), xorig,...) {
  recon <- reconstruct(x, comp=1:ncomp)
  xorig - recon
}



#' Pretty Print S3 Method for bi_projector Class
#'
#' @param x A `bi_projector` object
#' @param ... Additional arguments passed to the print function
#' @return Invisible `bi_projector` object
#' @export
print.bi_projector <- function(x, ...) {
  cat("A bi_projector object with the following properties:\n\n")
  
  cat("Dimensions of the weights (v) matrix:\n")
  cat("  Rows: ", nrow(x$v), " Columns: ", ncol(x$v), "\n")
  
  cat("\nDimensions of the scores (s) matrix:\n")
  cat("  Rows: ", nrow(x$s), " Columns: ", ncol(x$s), "\n")
  
  cat("\nLength of the standard deviations (sdev) vector:\n")
  cat("  Length: ", length(x$sdev), "\n")
  
  cat("\nPreprocessing information:\n")
  print(x$preproc, ...)
  
  invisible(x)
}
