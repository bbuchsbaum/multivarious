

#' A Union of Concatenated `bi_projector` Fits
#'
#' This function combines a set of `bi_projector` fits into a single `bi_projector` instance.
#' The new instance's weights and associated scores are obtained by concatenating the weights
#' and scores of the input fits.
#'
#' @param fits A list of `bi_projector` instances with the same row space. These instances
#'   will be combined to create a new `bi_projector` instance.
#' @param outer_block_indices An optional list of indices for the outer blocks. If not provided,
#'   the function will compute the indices based on the dimensions of the input fits.
#'
#' @examples
#'
#' X1 <- matrix(rnorm(5*5), 5, 5)
#' X2 <- matrix(rnorm(5*5), 5, 5)
#'
#' bpu <- bi_projector_union(list(pca(X1), pca(X2)))
#'
#' @return A new `bi_projector` instance with concatenated weights, scores, and other
#'   properties from the input `bi_projector` instances.
#' @export
#' @import chk
bi_projector_union <- function(fits, outer_block_indices=NULL) {
  chk::chk_all(fits, chk::chk_s3_class, "bi_projector")
  
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


#' @export
print.bi_projector_union <- function(x, ...) {
  cat("A bi_projector_union object with the following properties:\n\n")
  
  cat("Combined bi_projector instances:\n")
  num_instances <- length(x$fits)
  cat("  Number of instances: ", num_instances, "\n")
  
  cat("\nDimensions of the weights (v) matrix:\n")
  cat("  Rows: ", nrow(x$v), " Columns: ", ncol(x$v), "\n")
  
  cat("\nDimensions of the scores (s) matrix:\n")
  cat("  Rows: ", nrow(x$s), " Columns: ", ncol(x$s), "\n")
  
  cat("\nLength of the standard deviations (sdev) vector:\n")
  cat("  Length: ", length(x$sdev), "\n")
  
  cat("\nPreprocessing information:\n")
  print(x$preproc, ...)
  
  cat("\nOuter block indices:\n")
  print(x$outer_block_indices, ...)
  
  invisible(x)
}
