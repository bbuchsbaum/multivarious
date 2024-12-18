#' Create a Multiblock Projector
#'
#' Constructs a multiblock projector using the given component matrix (`v`), a preprocessing function, and a list of block indices. 
#' This allows for the projection of multiblock data, where each block represents a different set of variables or features.
#'
#' @param v A matrix of components with dimensions `nrow(v)` by `ncol(v)` (columns = number of components).
#' @param preproc A pre-processing function for the data (default: `prep(pass())`).
#' @param block_indices A list of numeric vectors specifying the indices of each data block.
#' @param classes (optional) A character vector specifying additional class attributes of the object, default is NULL.
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
#' # Project multiblock data using the multiblock projector
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
#' a preprocessing function, and a list of block indices. This allows for two-way mapping with multiblock data.
#'
#' @param v A matrix of components (nrow = number of variables, ncol = number of components).
#' @param s A matrix of scores (nrow = samples, ncol = components).
#' @param sdev A numeric vector of singular values or standard deviations.
#' @param preproc A pre-processing object (default: `prep(pass())`).
#' @param block_indices A list of numeric vectors specifying data block variable indices.
#' @param classes Additional class attributes (default NULL).
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


#' Extract the Block Indices from a Multiblock Projector
#'
#' @param x A `multiblock_projector` object.
#' @param i Ignored.
#' @param ... Ignored.
#' @return The list of block indices.
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

#' Project Data onto a Specific Block
#'
#' Projects the new data onto the subspace defined by a specific block of variables.
#'
#' @param x A `multiblock_projector` object.
#' @param new_data The new data to be projected.
#' @param block The block index (1-based) to project onto.
#' @param ... Additional arguments passed to `partial_project`.
#' @return The projected scores for the specified block.
#' @export
project_block.multiblock_projector <- function(x, new_data, block,...) {
  # Check block validity
  nb <- nblocks(x)
  if (block < 1 || block > nb) {
    stop("Block index out of range.")
  }
  
  ind <- block_indices(x)[[block]]
  partial_project(x, new_data, colind=ind, ...)
}

#' Coefficients for a Multiblock Projector
#'
#' Extracts the components (loadings) for a given block or the entire projector.
#'
#' @param object A `multiblock_projector` object.
#' @param block Optional block index. If missing, returns loadings for all variables.
#' @param ... Additional arguments.
#' @return A matrix of loadings.
#' @export
coef.multiblock_projector <- function(object, block,...) {
  if (missing(block)) {
    # Instead of NextMethod(object), just use NextMethod() to call coef.projector
    NextMethod()
  } else {
    nb <- nblocks(object)
    if (block < 1 || block > nb) {
      stop("Block index out of range.")
    }
    ind <- object$block_indices[[block]]
    object$v[ind,,drop=FALSE]
  }
}

#' Pretty Print Method for `multiblock_biprojector` Objects
#'
#' Display a summary of a `multiblock_biprojector` object.
#'
#' @param x A `multiblock_biprojector` object.
#' @param ... Additional arguments passed to `print()`.
#' @return Invisible `multiblock_biprojector` object.
#' @export
print.multiblock_biprojector <- function(x, ...) {
  cat("Multiblock Bi-Projector object:\n")
  cat("  Projection matrix dimensions: ", nrow(x$v), "x", ncol(x$v), "\n")
  
  # Print block indices in a nicer format
  cat("  Block indices:\n")
  lapply(seq_along(x$block_indices), function(i) {
    cat("    Block ", i, ": ", paste(x$block_indices[[i]], collapse = ","), "\n", sep="")
  })
  
  invisible(x)
}


