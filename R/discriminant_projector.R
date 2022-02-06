
#' construct a discriminant projector
#' 
#' a `discriminant_projector` instance extending `bi_projeector` whose projection maximizes class separation.
#' 
#' @inheritParams bi_projector
#' @param labels the training labels
#' 
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
  