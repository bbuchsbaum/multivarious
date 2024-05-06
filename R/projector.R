#' Construct a `projector` instance
#'
#' A `projector` maps a matrix from an N-dimensional space to d-dimensional space, where `d` may be less than `N`.
#' The projection matrix, `v`, is not necessarily orthogonal. This function constructs a `projector` instance which can be
#' used for various dimensionality reduction techniques like PCA, LDA, etc.
#'
#' @param v A matrix of coefficients with dimensions `nrow(v)` by `ncol(v)` (number of columns = number of components)
#' @param preproc A prepped pre-processing object. Default is the no-processing `pass()` preprocessor.
#' @param classes Additional class information used for creating subtypes of `projector`. Default is NULL.
#' @param ... Extra arguments to be stored in the `projector` object.
#'
#' @return An instance of type `projector`.
#'
#' @examples
#' X <- matrix(rnorm(10*10), 10, 10)
#' svdfit <- svd(X)
#' p <- projector(svdfit$v)
#' proj <- project(p, X)
#'
#' @export
projector <- function(v, preproc=prep(pass()), ..., classes=NULL) {
  chk::chkor(chk::chk_matrix(v), chk::chk_s4_class(v, "Matrix"))
  chk::chk_s3_class(preproc, "pre_processor")
  
  out <- structure(
    list(
      v=v,
      preproc=preproc,
      ...),
    class= c(classes, "projector")
  )
  
  out
}

#' @export
components.projector <- function(x,...) {
  x$v
}


#' @export
coef.projector <- function(object,...) {
  object$v
}

#' @export
ncomp.projector <- function(x) {
  ncol(coefficients(x))
}




#' @export
#' @importFrom stats coefficients
project.projector <- function(x, new_data,...) {
  if (is.vector(new_data)) {
    chk::chk_equal(length(new_data), shape(x)[1])
    new_data <- matrix(new_data, byrow=TRUE, ncol=length(new_data))
  }
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, values=nrow(coefficients(x)))
  
  reprocess(x, new_data) %*% coefficients(x)
}

#' @export
partial_project.projector <- function(x, new_data, colind,...) {
  if (is.vector(new_data) && length(colind) > 1) {
    new_data <- matrix(new_data, byrow=TRUE, ncol=length(new_data))
  } else if (is.vector(new_data) && length(colind) == 1) {
    new_data <- matrix(new_data, ncol=1)
  }
  
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, length(colind))
  comp <- components(x)
  
  #reprocess(x,new_data, colind) %*% comp[colind,] * sqrt(ncol(comp)/length(colind))
  reprocess(x,new_data, colind) %*% comp[colind,] * nrow(comp)/length(colind)
}




#' @export
is_orthogonal.projector <- function(x) {
  comp <- coefficients(x)
  
  z <- if (nrow(comp) > ncol(comp)) {
    crossprod(comp)
  } else {
    tcrossprod(comp)
  }
  
  Matrix::isDiagonal(zapsmall(z))
}

# compose_projector.projector <- function(x,y) {
#   chk::chk_s3_class(y, "projector")
#   ## functional projector?
# }

#' @export
inverse_projection.projector <- function(x,...) {
  ## assume orthogonal
  t(coefficients(x))
}

#' @export
partial_inverse_projection.projector <- function(x, colind,...) {
  chk::chk_range(max(colind), c(1, nrow(coefficients(x))))
  chk::chk_range(min(colind), c(1, nrow(coefficients(x))))
  cx <- coefficients(x)
  corpcor::pseudoinverse(cx[colind,,drop=FALSE])
}

#' @export
truncate.projector <- function(x, ncomp) {
  chk_range(ncomp, c(1, ncomp(x)))
  projector(coefficients(x)[,1:ncomp,drop=FALSE], ncomp=ncomp, preprox=x$preproc)
}

#' @export
reprocess.projector <- function(x, new_data, colind=NULL,...) {
  if (is.null(colind)) {
    chk::chk_equal(ncol(new_data), nrow(coefficients(x)))
    apply_transform(x$preproc, new_data)
  } else {
    chk::chk_equal(length(colind), ncol(new_data)) 
    apply_transform(x$preproc, new_data, colind)
  }
  
}

#' @export
shape.projector <- function(x,...) {
  c(nrow(x$v), ncol(x$v))
}


#' @export
#' @return the `projector` object
print.projector <- function(x,...) {
  cat("projector: ", paste0(class(x)), "\n")
  cat("input dim: ", shape(x)[1], "\n")
  cat("output dim: ", shape(x)[2], "\n")
  Invisible(x)
}


#' construct a partial_projector from a `projector` instance
#' 
#' @export
#' @inheritParams partial_projector
#' @return A `partial_projector` instance
#' @examples 
#' 
#' X <- matrix(rnorm(10*10), 10, 10)
#' pfit <- pca(X, ncomp=9)
#' proj <- project(pfit, X)
#' 
#' pp <- partial_projector(pfit, 1:5)
partial_projector.projector <- function(x, colind, ...) {
  projector(x$v[colind,], preproc=x$preproc, colind=colind, porig=x, classes="partial_projector")
}

#' @export
reprocess.partial_projector <- function(x, new_data, colind=NULL,...) {
  if (is.null(colind)) {
    chk::chk_equal(ncol(new_data), nrow(coefficients(x)))
    apply_transform(x$preproc, new_data, colind)
  } else {
    chk::chk_equal(length(colind), ncol(new_data)) 
    apply_transform(x$preproc, new_data, x$colind[colind])
  }
  
}

#' @export
#' @importFrom stats coefficients
project.partial_projector <- function(x, new_data,...) {
  partial_project(x$porig, new_data, x$colind, ...)
}

#' @export
truncate.partial_projector <- function(x, ncomp) {
  chk_range(ncomp, c(1, ncomp(x)))
  
  porig <- truncate(x,ncomp)
  projector(porig$v, preproc=x$preproc, colind=x$colind, porig=x$x, 
            classes="partial_projector")
  
}

#' @export
partial_project.partial_projector <- function(x, new_data, colind, ...) {
  stop("not implemented")
}

#' Pretty Print Method for `projector` Objects
#'
#' Display a human-readable summary of a `projector` object, including information about the dimensions of the projection matrix and the pre-processing pipeline.
#'
#' @param x A `projector` object.
#' @param ... Additional arguments passed to `print()`.
#'
#' @examples
#' X <- matrix(rnorm(10*10), 10, 10)
#' svdfit <- svd(X)
#' p <- projector(svdfit$v)
#' print(p)
#' @export
print.projector <- function(x, ...) {
  cat("projector object:\n")
  cat("  Projection matrix dimensions: ", nrow(x$v), "x", ncol(x$v), "\n")
  invisible(x)
}




