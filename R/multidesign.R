
#' @importFrom dplyr mutate rowwise n
#' @export
#' @examples
#' X <- matrix(rnorm(20*100), 20, 100)
#' Y <- tibble(condition=rep(letters[1:5], 4))
#' 
#' mds <- multidesign(X,Y)
multidesign.matrix <- function(x, y) {
  chk::chk_equal(nrow(x), nrow(y))
  chk::chk_s3_class(y, "data.frame")
  y <- as_tibble(y)
  
  des <- y %>% mutate(.index=1:n())
  structure(list(
    x=x,
    design=tibble::as_tibble(des)
  ),
  class="multidesign")
}

#' @export
subset.multidesign <- function(x, fexpr) {
  des2 <- filter(x$design, !!rlang::enquo(fexpr))
  ind <- des2$.index
  multidesign(x$x[ind,], des2)
}

#' @export
split.multidesign <- function(x, ..., collapse=FALSE) {
  nest.by <- rlang::quos(...)
  ret <- x$design %>% nest_by(!!!nest.by, .keep=TRUE)
  xl <- ret$data %>% purrr::map(~x$x[.x$.index,,drop=FALSE])
  ret <- lapply(1:nrow(ret), function(i) multidesign.matrix(xl[[i]], ret$data[[i]]))
}


#' @export
summarize_by.multidesign <- function(x, ..., sfun=colMeans, extract_data=FALSE) {
  #nested <- split(x, ...)
  nest.by <- rlang::quos(...)
  ret <- x$design %>% nest_by(!!!nest.by)
 
  dsum <- do.call(rbind, ret$data %>% purrr::map( ~ sfun(x$x[.x[[".index"]], drop=FALSE,])))
  ret2 <- ret %>% select(-data)
  multidesign(dsum, ret2)
}

#' @export
print.multidesign <- function(x) {
  cat("a multidesign object. \n")
  cat(nrow(x$design), "rows", "\n")
  cat(ncol(x$des), "design variables", "and", nrow(x$x), "response variables", "\n")
  cat("design variables: ", "\n")
  print(x$design, n=5)
}