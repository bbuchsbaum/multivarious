

multiblock_projector <- function(v, preproc=prep(pass()), ..., block_indices, classes=NULL) {
  chk::chk_list(block_indices)
  sumind <- sum(sapply(block_indices, length))
  chk::chk_equal(sumind, nrow(v))
  
  projector(v, preproc, block_indices=block_indices, classes=c(classes, "multiblock_projector"))
}


multiblock_biprojector <- function(v, preproc=prep(pass()), ..., block_indices, classes=NULL) {
  sumind <- sum(sapply(block_indices, length))
  chk::chk_equal(sumind, nrow(v))
  bi_projector(v, preproc, block_indices=block_indices, classes=c(classes, "multiblock_biprojector", "multiblock_projector"))
}


block_indices.multiblock_projector <- function(x,i) {
  x$block_indices
}

block_lengths.multiblock_projector <- function(x) {
  length(x$block_indices)
}

project_block.multiblock_projector <- function(x, new_data, block) {
  ind <- block_indices(x)[[block]]
  partial_project(x, new_data, colind=ind )
}



