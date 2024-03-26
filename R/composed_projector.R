#' Projector Composition
#'
#' Compose a sequence of `projector` objects in forward order.
#' This function allows the composition of multiple projectors, applying them sequentially to the input data.
#'
#' @param ... The sequence of `projector` objects to be composed.
#'
#' @return A `composed_projector` object that extends the `function` class, allowing the composed projectors to be 
#' applied to input data.
#' @export
#' @seealso \code{\link{projector}}, \code{\link{project}}
#'
#' @examples
#' # Create two PCA projectors and compose them
#' X <- matrix(rnorm(20*20), 20, 20)
#' pca1 <- pca(X, ncomp=10)
#' X2 <- scores(pca1)
#' pca2 <- pca(X2, ncomp=4)
#'
#' # Compose the PCA projectors
#' cproj <- compose_projectors(pca1, pca2)
#'
#' # Ensure the output of the composed projectors has the expected dimensions
#' stopifnot(ncol(cproj(X)) == 4)
#' # Check that the composed projectors work as expected
#' all.equal(project(cproj, X), cproj(X))
#' @export
compose_projectors <- function(...) {
  args <- list(...)
  sapply(args, function(p) chk::chk_s3_class(p, "projector"))
  if (length(args) == 1) {
    return(args[[1]])
  }
  
  shapelist <- lapply(args, shape)
  for (i in 2:length(args)) {
    chk::chk_equal(shapelist[[i-1]][2], shapelist[[i]][1])
  }
  
  out <- lapply(args, function(arg) {
    f <- function(new_data) {
      project(arg, new_data)
    }
  })

  f <- do.call(purrr::compose, c(out,.dir="forward"))
  
  out <- structure(f,
    class=c("composed_projector", "function")
  )
  attr(out, "length") <- length(args)
  out
}

# compose_partial_projector <- function(...) {
#   args <- list(...)
#   out <- lapply(1:length(args), function(i) {
#     arg <- args[[i]]
#     chk::chk_s3_class(arg, "projector")
#     f <- function(new_data, colind) {
#       partial_project(arg, new_data, colind=1:nrow(coefficients(x)))
#     }
#   })
#   
#   f <- do.call(purrr::compose, c(out,.dir="forward"))
#   
#   out <- structure(f,
#                    class=c("composed_partial_projector", "composed_projector", "function")
#   )
# }

#' @export
project.composed_projector <- function(x, new_data,...) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, byrow=TRUE)
  }
  chk::vld_matrix(new_data)
  
  x(new_data)
}

# partial_project.composed_partial_projector <- function(x, new_data, colind) {
#   if (is.vector(new_data) && length(colind) > 1) {
#     new_data <- matrix(new_data, byrow=TRUE)
#   } 
#   chk::vld_matrix(new_data)
#   chk::check_dim(new_data, ncol, length(colind))
#   x(new_data, colind)
# }

#' Pretty Print Method for `composed_projector` Objects
#'
#' Display a human-readable summary of a `composed_projector` object, including information about the number and order of projectors.
#'
#' @param x A `composed_projector` object.
#' @param ... Additional arguments passed to `print()`.
#' @return The `composed_projector` object.
#' @examples
#' # Create two PCA projectors and compose them
#' X <- matrix(rnorm(20*20), 20, 20)
#' pca1 <- pca(X, ncomp=10)
#' X2 <- scores(pca1)
#' pca2 <- pca(X2, ncomp=4)
#' cproj <- compose_projectors(pca1, pca2)
#' @export
print.composed_projector <- function(x, ...) {
  n_proj <- attr(x, "length")
  cat("Composed projector object:\n")
  cat("  Number of projectors: ", n_proj, "\n")
  #cat("  Projector order:\n")
  #for (i in seq_len(n_proj)) {
  #  cat("    ", i, ": ", class(unclass(x)[[i]]$projector)[1], "\n")
  #}
  invisible(x)
}



