

#' compute column-wise mean in \code{X} for each factor level of \code{Y}
#' 
#' compute group means
#' 
#' @param Y a vector of labels to compute means over disjoint sets
#' @param X a data matrix from which to compute means
#' @export
group_means <- function (Y, X) {
  chk::chk_equal(nrow(X), length(Y))
  
  if (all(table(Y) == 1)) {
    warnings("`Y` does not contain more than one replicate of any level")
    row.names(X) <- names(table(Y))
    X
  }
  else {
    if (any(is.na(X))) {
      xspl <- split_matrix(X, Y)
      ret <- do.call(rbind, lapply(xspl, function(x) matrixStats::colMeans2(x, 
                                                                            na.rm = TRUE)))
      row.names(ret) <- names(xspl)
      ret
    }
    else {
      Rs <- rowsum(X, Y, na.rm = TRUE)
      yt <- table(Y)
      ret <- sweep(Rs, 1, yt, "/")
      row.names(ret) <- names(yt)
      ret
    }
  }
}
