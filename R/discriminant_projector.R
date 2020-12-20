
#' construct a discriminant projector
#' 
#' a `bi_projector` that whose projection maximizes class discrimination.
#' @inheritParams bi_projector
#' @param labels the training labels
#' 
#' 
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