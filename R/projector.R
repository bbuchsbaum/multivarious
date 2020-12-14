#' construct a `projector` instance
#' 
#' A `projector` maps a matrix from D-dimensional space to d-dimensional space, where `d` may be less than `D`.
#' The projection matrix, `v` is not necessarily orthogonal.
#' 
#' @export
#' @param v a matrix of coefficients with dimension `nrow(v)` by `ncol(v)` (number of columns = number of components)
#' @param preproc a pre-processing object
#' @param classes additional class information used for creating subtypes of `projector`
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
projector <- function(v, preproc=NULL, ..., classes=NULL) {
  out <- structure(
    list(
      preproc=preproc,
      ncomp=ncomp,
      v=v,
      ...),
    class= c(classes, "projector")
  )
  
  out
}

components.projector <- function(x) {
  x$v
}

ncomp.projector <- function(x) {
  ncol(components(x))
}

project.projector <- function(x, new_data) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, byrow=TRUE)
  }
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, values=nrow(components(x)))
  
  new_data %*% components(x)
}

partial_project.projector <- function(x, new_data, colind) {
  if (is.vector(new_data) && length(colind) > 1) {
    new_data <- matrix(new_data, byrow=TRUE)
  } 
  
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, length(colind))
  comp <- components(x)
  
  new_data %*% comp[colind,] * sqrt(ncol(comp)/length(colind))
}


is_orthogonal.projector <- function(x) {
  comp <- components(x)
  
  z <- if (nrow(comp) > ncol(comp)) {
    crosssprod(comp)
  } else {
    tcrossprod(comp)
  }
  
  Matrix::isDiagonal(zapsmall(z))
}


truncate.projector <- function(x, ncomp) {
  chk_range(ncomp, c(1, ncomp(x)))
  projector(components(x)[,1:ncomp,drop=FALSE], ncomp=ncomp, preprox=x$preproc)
}


  

print.projector <- function(x) {
  cat("projector: ", paste0(class(x)), "\n")
  cat("input dim: ", nrow(x$v), "\n")
  cat("output dim: ", ncol(x$v), "\n")
}


