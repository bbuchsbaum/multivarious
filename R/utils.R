

#' @noMd
split_matrix <- function(X, fac) {
  idx <- split(1:nrow(X), fac)
  lapply(idx, function(i) X[i,])
}


#' Compute column-wise mean in X for each factor level of Y
#'
#' This function computes group means for each factor level of Y in the provided data matrix X.
#'
#' @param Y a vector of labels to compute means over disjoint sets
#' @param X a data matrix from which to compute means
#' @return a matrix with row names corresponding to factor levels of Y and column-wise means for each factor level
#' @export
#' @examples
#' # Example data
#' X <- matrix(rnorm(50), 10, 5)
#' Y <- factor(rep(1:2, each = 5))
#'
#' # Compute group means
#' gm <- group_means(Y, X)
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

#' Compute principal angles for a set of subspaces
#'
#' This function calculates the principal angles between subspaces derived from a list of bi_projector instances.
#'
#' @param fits a list of `bi_projector` instances
#' @return a numeric vector of principal angles with length equal to the minimum dimension of input subspaces
#' @export
#' @examples
#' # Assuming 'fit1', 'fit2', and 'fit3' are bi_projector objects created using the svd_wrapper function
#' fits_list <- list(fit1, fit2, fit3)
#' principal_angles <- prinang(fits_list)
prinang <- function(fits) {
  chk::chk_all(fits, chk_fun = chk_s3_class, "bi_projector")
  
  mindim <- min(sapply(fits, function(x) shape(x)[2]))
  sclist <- lapply(fits, function(x) {
    sc <- scores(x)[,1:mindim,drop=FALSE]
    apply(sc,2, function(z) z/sqrt(sum(z^2)))
  })
  
  cmat <- do.call(cbind, sclist)
  sres <- svd(cmat)
  sqrt(sres$d)/length(fits)
}


