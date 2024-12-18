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

#' @export
inverse_projection.projector <- function(x,...) {
  # assume orthogonal
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
  projector(coefficients(x)[,1:ncomp, drop=FALSE], preproc=x$preproc)
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

# Removed the first print.projector definition as requested. Keep only one.

#' Pretty Print Method for `projector` Objects
#'
#' Display a human-readable summary of a `projector` object using crayon formatting, including information 
#' about the dimensions of the projection matrix and the pre-processing pipeline.
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
  # Using crayon for a more appealing output
  cat(crayon::bold(crayon::green("Projector object:\n")))
  cat(crayon::yellow("  Input dimension: "), shape(x)[1], "\n", sep="")
  cat(crayon::yellow("  Output dimension: "), shape(x)[2], "\n", sep="")
  
  if (!is.null(x$preproc)) {
    cat(crayon::cyan("  With pre-processing:\n"))
    # We can attempt to print the preproc summary if it has a print method.
    if (inherits(x$preproc, "pre_processor")) {
      # pre_processor can be printed directly
      print(x$preproc)
    } else {
      cat(crayon::cyan("    (pre-processing pipeline not fully available)\n"))
    }
  } else {
    cat(crayon::cyan("  No pre-processing pipeline.\n"))
  }
  
  invisible(x)
}

#' construct a partial_projector from a `projector` instance
#' 
#' @export
#' @inheritParams partial_projector
#' @return A `partial_projector` instance
#' @examples 
#' 
#' # Assuming pfit is a projector with many components:
#' # pp <- partial_projector(pfit, 1:5)
partial_projector.projector <- function(x, colind, ...) {
  # colind and porig stored in ... for future reference
  projector(x$v[colind,], preproc=x$preproc, colind=colind, porig=x, classes="partial_projector")
}

#' @export
reprocess.partial_projector <- function(x, new_data, colind=NULL,...) {
  if (is.null(colind)) {
    chk::chk_equal(ncol(new_data), nrow(coefficients(x)))
    apply_transform(x$preproc, new_data)
  } else {
    chk::chk_equal(length(colind), ncol(new_data)) 
    # x$colind was stored at creation. Ensure it exists
    base_colind <- x$colind
    chk::chk_not_null(base_colind)
    # remap colind if needed
    apply_transform(x$preproc, new_data, base_colind[colind])
  }
}

#' @export
#' @importFrom stats coefficients
project.partial_projector <- function(x, new_data,...) {
  # use partial_project on original projector (x$porig) with stored colind
  chk::chk_not_null(x$porig)
  chk::chk_not_null(x$colind)
  partial_project(x$porig, new_data, x$colind, ...)
}

#' @export
truncate.partial_projector <- function(x, ncomp) {
  chk_range(ncomp, c(1, ncomp(x)))
  # Truncate original projector
  chk::chk_not_null(x$porig)
  porig <- truncate(x$porig, ncomp)
  
  # Recreate partial_projector with truncated projector's v
  projector(porig$v, preproc=x$preproc, colind=x$colind, porig=x$porig, 
            classes="partial_projector")
}

#' @export
partial_project.partial_projector <- function(x, new_data, colind, ...) {
  stop("not implemented")
}



