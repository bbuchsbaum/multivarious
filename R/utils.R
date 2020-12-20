

#' compute column-wise mean in \code{X} for each factor level of \code{Y}
#' 
#' compute group means
#' 
#' @param Y
#' @param X
#' @export
group_means <- function (Y, X) {
  if (all(table(Y) == 1)) {
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
