
#' @export
multiblock_projector <- function(v, preproc=prep(pass()), ..., block_indices, classes=NULL) {
  chk::chk_list(block_indices)
  sumind <- sum(sapply(block_indices, length))
  chk::chk_equal(sumind, nrow(v))
  
  projector(v, preproc, block_indices=block_indices, classes=c(classes, "multiblock_projector"))
}

#' @export
multiblock_biprojector <- function(v, s, sdev, preproc=prep(pass()), ..., block_indices, classes=NULL) {
  sumind <- sum(sapply(block_indices, length))
  chk::chk_equal(sumind, nrow(v))
  bi_projector(v, s=s, sdev=sdev, preproc=preproc, block_indices=block_indices, classes=c(classes, "multiblock_biprojector", "multiblock_projector"))
}

#' @export
block_indices.multiblock_projector <- function(x,i) {
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
project_block.multiblock_projector <- function(x, new_data, block) {
  ind <- block_indices(x)[[block]]
  partial_project(x, new_data, colind=ind )
}

#' @export
coef.multiblock_projector <- function(object, block) {
  if (missing(block)) {
    NextMethod(object)
  } else {
    ind <- x$block_indices[[block]]
    object$v[ind,drop=FALSE]
  }
}

#' @export
print.multiblock_projector <- function(x) {
  NextMethod(x)
  cat("number of blocks: ", length(x$block_indices))
}



