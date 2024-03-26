

#' Create a Multiblock Projector
#'
#' Constructs a multiblock projector using the given component matrix (`v`), a preprocessing function, and a list of block indices. 
#' This allows for the projection of multiblock data, where each block represents a different set of variables or features.
#'
#' @param v A matrix of components with dimensions `nrow(v)` by `ncol(v)` (number of columns = number of components).
#' @param preproc A pre-processing function for the data (default is a pass-through with `prep(pass())`).
#' @param block_indices A list of numeric vectors specifying the indices of each data block.
#' @param classes (optional) A character vector specifying the class attributes of the object, default is NULL.
#' @param ... Extra arguments.
#' @return A `multiblock_projector` object.
#'
#' @seealso projector
#' @export
#' @examples
#' # Generate some example data
#' X1 <- matrix(rnorm(10 * 5), 10, 5)
#' X2 <- matrix(rnorm(10 * 5), 10, 5)
#' X <- cbind(X1, X2)
#'
#' # Compute PCA on the combined data
#' pc <- pca(X, ncomp = 8)
#'
#' # Create a multiblock projector using PCA components and block indices
#' mb_proj <- multiblock_projector(pc$v, block_indices = list(1:5, 6:10))
#'
#' # Project the multiblock data using the multiblock projector
#' mb_scores <- project(mb_proj, X)
multiblock_projector <- function(v, preproc=prep(pass()), ..., block_indices, classes=NULL) {
  chk::chk_list(block_indices)
  sumind <- sum(sapply(block_indices, length))
  chk::chk_equal(sumind, nrow(v))
  
  projector(v, preproc, block_indices=block_indices, ..., classes=c(classes, "multiblock_projector"))
}


#' Create a Multiblock Bi-Projector
#'
#' Constructs a multiblock bi-projector using the given component matrix (`v`), score matrix (`s`), singular values (`sdev`),
#' a preprocessing function, and a list of block indices. This allows for the projection of multiblock data, where each block 
#' represents a different set of variables or features, with two-way mapping from samples to scores and from variables to components.
#'
#' @param v A matrix of components with dimensions `nrow(v)` by `ncol(v)` (number of columns = number of components).
#' @param s A matrix of scores.
#' @param sdev A numeric vector of singular values.
#' @param preproc A pre-processing function for the data (default is a pass-through with `prep(pass())`).
#' @param block_indices A list of numeric vectors specifying the indices of each data block.
#' @param classes (optional) A character vector specifying the class attributes of the object, default is NULL.
#' @param ... Extra arguments.
#' @return A `multiblock_biprojector` object.
#'
#' @seealso bi_projector, multiblock_projector
#' @export
multiblock_biprojector <- function(v, s, sdev, preproc=prep(pass()), ..., block_indices, classes=NULL) {
  sumind <- sum(sapply(block_indices, length))
  chk::chk_equal(sumind, nrow(v))
  bi_projector(v, s=s, sdev=sdev, preproc=preproc, block_indices=block_indices, ..., classes=c(classes, "multiblock_biprojector", "multiblock_projector"))
}


#' @export
block_indices.multiblock_projector <- function(x,i,...) {
  x$block_indices
}

#' @export
block_lengths.multiblock_projector <- function(x) {
  sapply(block_indices(x), length)
}

#' @export
nblocks.multiblock_projector <- function(x) {
  length(block_indices(x))
}

#' @export
project_block.multiblock_projector <- function(x, new_data, block,...) {
  ind <- block_indices(x)[[block]]
  partial_project(x, new_data, colind=ind )
}

#' @export
coef.multiblock_projector <- function(object, block,...) {
  if (missing(block)) {
    NextMethod(object)
  } else {
    ind <- object$block_indices[[block]]
    object$v[ind,,drop=FALSE]
  }
}

#' Pretty Print Method for `multiblock_biprojector` Objects
#'
#' Display a human-readable summary of a `multiblock_biprojector` object, including information about the dimensions of the projection matrix, the pre-processing pipeline, and block indices.
#'
#' @param x A `multiblock_biprojector` object.
#' @param ... Additional arguments passed to `print()`.
#' @return Invisible `multiblock_biprojector` object.
#' @examples
#' # Generate some example data
#' X1 <- matrix(rnorm(10 * 5), 10, 5)
#' X2 <- matrix(rnorm(10 * 5), 10, 5)
#' X <- cbind(X1, X2)
#' # Compute PCA on the combined data
#' pc <- pca(X, ncomp = 8)
#' # Create a multiblock bi-projector using PCA components and block indices
#' mb_biproj <- multiblock_biprojector(pc$v, s = pc$u %*% diag(sdev(pc)), sdev = sdev(pc), 
#' block_indices = list(1:5, 6:10))
#' # Pretty print the multiblock bi-projector object
#' print(mb_biproj)
#' @export
print.multiblock_biprojector <- function(x, ...) {
  cat("Multiblock Bi-Projector object:\n")
  cat("  Projection matrix dimensions: ", nrow(x$v), "x", ncol(x$v), "\n")
  cat("  Block indices: ", toString(x$block_indices), "\n")
  invisible(x)   
}



