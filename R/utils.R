

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

#' principal angles
#' 
#' compute principal angles for a set of subspaces
#' @param fits a list of `bi_projector` instances
#' @return a numeric vector of principle angles with length equal to minimum dimension of input subspaces
prinang <- function(fits) {
  chk::chk_all(fits, chk_fun = chk_s3_class, "bi_projector")
  
  mindim <- min(sapply(fits, function(x) shape(x)[2]))
  sclist <- lapply(fits, function(x) {
    sc <- scores(x)[,1:mindim,drop=FALSE]
    apply(sc,2, function(z) z/sqrt(sum(z^2)))
  })
  
  cmat <- do.call(cbind, sclist)
  sres <- svd(cmat)
  sqrt(sres$d)[1:mindim]/length(fits)
}


