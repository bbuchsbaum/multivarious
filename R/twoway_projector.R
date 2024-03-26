#' Two-way (cross) projection to latent components
#'
#' A projector that reduces two blocks of data, X and Y, yielding a pair of weights for each component.
#' This structure can be used, for example, to store weights derived from canonical correlation analysis.
#'
#' @details This class extends `projector` and therefore basic operations such as `project`, `shape`, `reprocess`,
#' and `coef` work, but by default, it is assumed that the `X` block is primary. To access `Y` block operations, an
#' additional argument `source` must be supplied to the relevant functions, e.g., `coef(fit, source = "Y")`
#'
#' @param vx the X coefficients
#' @param vy the Y coefficients
#' @param preproc_x the X pre-processor
#' @param preproc_y the Y pre-processor
#' @param ... extra parameters or results to store
#' @param classes additional class names
#' @return a cross_projector object
#' @export
#' @examples
#' # Create two scaled matrices X and Y
#' X <- scale(matrix(rnorm(10 * 5), 10, 5))
#' Y <- scale(matrix(rnorm(10 * 5), 10, 5))
#'
#' # Perform canonical correlation analysis on X and Y
#' cres <- cancor(X, Y)
#' sx <- X %*% cres$xcoef
#' sy <- Y %*% cres$ycoef
#'
#' # Create a cross_projector object using the canonical correlation analysis results
#' canfit <- cross_projector(cres$xcoef, cres$ycoef, cor = cres$cor,
#'                           sx = sx, sy = sy, classes = "cancor")
cross_projector <- function(vx, vy, preproc_x=prep(pass()), preproc_y=prep(pass()), 
                             ..., classes=NULL) {
  
  chk::chkor(chk::chk_matrix(vx), chk::chk_s4_class(vx, "Matrix"))
  chk::chkor(chk::chk_matrix(vy), chk::chk_s4_class(vy, "Matrix"))
  chk::chk_s3_class(preproc_x, "pre_processor")
  chk::chk_s3_class(preproc_y, "pre_processor")
  
  
  out <- structure(
    list(
      v=vx,
      vx=vx,
      vy=vy,
      preproc=preproc_x,
      preproc_x=preproc_x,
      preproc_y=preproc_y,
      ...),
    class= c(classes, "cross_projector", "projector")
  )
  
  out
}

#' project a cross_projector instance
#' 
#' @inheritParams project
#' @param source the source of the data (X or Y block)
#' @return the projected data
#' @export
#' @family project
project.cross_projector <- function(x, new_data, source=c("X", "Y"),...) {
  source <- match.arg(source)
  chk::vld_matrix(new_data)
  
  if (is.vector(new_data)) {
    chk::chk_equal(length(new_data), shape(x, source=source)[1])
    new_data <- matrix(new_data, byrow=TRUE, ncol=length(new_data))
  }
  
  chk::check_dim(new_data, ncol, values=nrow(coefficients(x, source=source)))
  reprocess(x, new_data, source=source) %*% coefficients(x, source=source)

}  

#' Extract coefficients from a cross_projector object
#'
#' @param object the model fit
#' @param source the source of the data (X or Y block), either "X" or "Y"
#' @param ... extra args
#' @return the coefficients
#' @export
coef.cross_projector <- function(object, source=c("X", "Y"),...) {
  source <- match.arg(source)
  if (source == "X") {
    object$vx
  } else {
    object$vy
  }
}

#' reprocess a cross_projector instance
#' 
#' @inheritParams reprocess
#' @param source the source of the data (X or Y block)
#' @return the re(pre-)processed data
#' @export
#' @family reprocess
reprocess.cross_projector <- function(x, new_data, colind=NULL, source=c("X", "Y"), ...) {
  source <- match.arg(source)
  if (is.null(colind)) {
    chk::chk_equal(ncol(new_data), nrow(coefficients(x, source=source)))
    colind <- 1:ncol(new_data)
  } else {
    chk::chk_equal(length(colind), ncol(new_data)) 
  }
  
  if (source == "X") {
    apply_transform(x$preproc_x, new_data, colind)
  } else {
    apply_transform(x$preproc_y, new_data, colind)
  }
  
}

#' shape of a cross_projector instance
#' 
#' @param source the source of the data (X or Y block)
#' @return the shape of the data
#' @export
#' @family shape
#' @inheritParams shape
shape.cross_projector <- function(x, source=c("X", "Y"), ...) {
  source <- match.arg(source)
  if (source == "X") {
    c(nrow(x$vx), ncol(x$vx))
  } else {
    c(nrow(x$vy), ncol(x$vy))
  }
}

#' @export
print.cross_projector <- function(x,...) {
  cat("cross projector: ", paste0(class(x)), "\n")
  cat("input dim (X): ", shape(x, source="X")[1], "\n")
  cat("output dim (X): ", shape(x, source="X")[2], "\n")
  cat("input dim (Y): ", shape(x, source="Y")[1], "\n")
  cat("output dim (Y): ", shape(x, source="Y")[2], "\n")
}


  
  
  
  