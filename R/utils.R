

#' @noRd
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
#' 
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' res <- pca(X, ncomp = 4)
#' fits_list <- list(res,res,res)
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


#' Compute a regression model for each column in a matrix and return residual matrix
#' 
#' @param form the formula defining the model to fit for residuals
#' @param X the response matrix
#' @param design the \code{data.frame} containing the design variables specified in \code{form} argument.
#' @param intercept add an intercept term (default is FALSE)
#' 
#' @return a \code{matrix} of residuals
#' @examples 
#' 
#' X <- matrix(rnorm(20*10), 20, 10)
#' des <- data.frame(a=rep(letters[1:4], 5), b=factor(rep(1:5, each=4)))
#' xresid <- residualize(~ a+b, X, design=des)
#' 
#' ## design is saturated, residuals should be zero
#' xresid2 <- residualize(~ a*b, X, design=des)
#' sum(xresid2) == 0
#' @export
#' @importFrom stats model.matrix lsfit resid
residualize <- function(form, X, design, intercept=FALSE) {
  #options(contrasts = c("contr.sum", "contr.poly"))
  modmat <- model.matrix(form, data=design)
  stats::resid(lsfit(modmat, X, intercept=intercept))
}

