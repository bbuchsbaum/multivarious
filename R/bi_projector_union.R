

#' a union of concatenated `bi_projector` fits
#' 
#' given a set of `bi_projector`` fits, join the together to create a new `bi_projector`` instance.
#' The new weights and associated scores will simply be concatenated.
#' 
#' @param fits a list of `bi_projector` instances with the same row space.
#' @param outer_block_indices list of indices for the outer blocks
#' 
#' @examples 
#' 
#' X1 <- matrix(rnorm(5*5), 5,5)
#' X2 <- matrix(rnorm(5*5), 5,5)
#' 
#' bpu <- bi_projector_union(list(pca(X1), pca(X2)))
#' 
#' @export
bi_projector_union <- function(fits, outer_block_indices=NULL) {
  chk::chk_all(fits, chk_s3_class, "bi_projector")
  
  if (is.null(outer_block_indices)) {
    nv <- sapply(fits, function(f) shape(f)[1])
    offsets <- cumsum(c(1, nv))
    outer_block_indices <- lapply(1:length(nv), function(i) {
      seq(offsets[i], offsets[i] + nv[i]-1)
    })
  } else {
    nv <- sapply(fits, function(f) shape(f)[1])
    chk::chk_equal(nv, sapply(outer_block_indices, length))
  }
  
  v <- do.call(cbind, lapply(fits, coef))
  s <- do.call(cbind, lapply(fits, scores))
  sdev <- sapply(fits, sdev)
  
  cpreproc <- concat_pre_processors(lapply(fits, "[[", "preproc"), outer_block_indices)
    
  ret <- bi_projector(
    v=v,
    s=s,
    sdev=sdev,
    preproc=cpreproc,
    fits=fits,
    outer_block_indices=outer_block_indices,
    classes="bi_projector_union"
  )
    
}