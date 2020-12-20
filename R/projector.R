#' construct a `projector` instance
#' 
#' A `projector` maps a matrix from D-dimensional space to d-dimensional space, where `d` may be less than `D`.
#' The projection matrix, `v` is not necessarily orthogonal.
#' 
#' @export
#' @param v a matrix of coefficients with dimension `nrow(v)` by `ncol(v)` (number of columns = number of components)
#' @param preproc a prepped pre-processing object (default is the no-processing `pass()` pre_processor)
#' @param classes additional class information used for creating subtypes of `projector`
#' @param ... extra args
#' 
#' 
#' @return 
#' 
#' a instance of type `projector`
#' 
#' @export
#' 
#' @example 
#' 
#' X <- matrix(rnorm(10*10), 10, 10)
#' svdfit <- svd(X)
#' 
#' p <- projector(svdfit$v)
#' 
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
components.projector <- function(x) {
  x$v
}

#' @export
coefficients.projector <- function(object) {
  object$v
}

#' @export
ncomp.projector <- function(x) {
  ncol(coefficients(x))
}

#' @export
project.projector <- function(x, new_data) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, byrow=TRUE, ncol=length(new_data))
  }
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, values=nrow(coefficients(x)))
  
  reprocess(x, new_data) %*% coefficients(x)
}

#' @export
partial_project.projector <- function(x, new_data, colind) {
  if (is.vector(new_data) && length(colind) > 1) {
    new_data <- matrix(new_data, byrow=TRUE, ncol=length(new_data))
  } 
  
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, length(colind))
  comp <- coefficients(x)
  
  reprocess(x,new_data, colind) %*% comp[colind,] * sqrt(ncol(comp)/length(colind))
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
inverse_projection.projector <- function(x) {
  ## assume orthogonal
  t(coefficients(x))
}

#' @export
partial_inverse_projection.projector <- function(x, colind) {
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
reprocess.projector <- function(x, new_data, colind=NULL) {
  if (is.null(colind)) {
    chk::chk_equal(ncol(new_data), nrow(coefficients(x)))
    apply_transform(x$preproc, new_data)
  } else {
    chk::chk_equal(length(colind), ncol(new_data)) 
    apply_transform(x$preproc, new_data, colind)
  }
  
}

#' @export
shape.projector <- function(x) {
  c(nrow(x$v), ncol(x$v))
}


#' @export
print.projector <- function(x) {
  cat("projector: ", paste0(class(x)), "\n")
  cat("input dim: ", nrow(x$v), "\n")
  cat("output dim: ", ncol(x$v), "\n")
}


