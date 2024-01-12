
#' Construct a Discriminant Projector
#'
#' A `discriminant_projector` is an instance that extends `bi_projector` with a projection that maximizes class separation.
#' This can be useful for dimensionality reduction techniques that take class labels into account, such as Linear Discriminant Analysis (LDA).
#'
#' @inheritParams bi_projector
#' @param labels A factor or character vector of class labels corresponding to the rows of the score matrix `s`.
#' @return A `discriminant_projector` object.
#'
#' @seealso bi_projector
#' @export
#' @examples
#' # Simulate data and labels
#' set.seed(123)
#' X <- matrix(rnorm(100 * 10), 100, 10)
#' labels <- factor(rep(1:2, each = 50))
#'
#' # Perform LDA and create a discriminant projector
#' lda_fit <- MASS::lda(X, labels)
#'
#' dp <- discriminant_projector(lda_fit$scaling, X %*% lda_fit$scaling, sdev = lda_fit$svd, 
#' labels = labels)
#' @export
discriminant_projector <- function(v, s, sdev, preproc=prep(pass()), labels, classes=NULL, ...) {
  
  chk::vld_matrix(v)
  chk::vld_matrix(s)
  chk::vld_numeric(sdev)
  chk::chk_equal(length(sdev), ncol(s))
  chk::chk_equal(ncol(v), length(sdev))
  chk::chk_equal(length(labels), nrow(s))
  
  out <- bi_projector(v, preproc=preproc, s=s, sdev=sdev, labels=labels, 
                      counts=table(labels), classes=c(classes, "discriminant_projector"), ...)
}

#' @export
print.discriminant_projector <- function(x,...) {
  print.projector(x)
  cat("label counts: ", x$counts)
}
  