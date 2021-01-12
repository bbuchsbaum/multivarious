#' 
#' #' @export
#' obs_group <- function(X, fun=NULL, ind=NULL) {
#'   
#'   chk::chkor(chk::chk_s3_class(X, "matrix"), chk::chk_s3_class(X, "list"), chk::chk_s3_class(X, "deflist"))
#'   
#'   ret <- if (inherits(X, "list") || inherits(X, "deflist")) {
#'     if (is.null(ind)) {
#'       ind <- 1:length(X)
#'     } else {
#'       chk::chk_equal(length(X), length(ind))
#'     }
#'     lapply(1:length(X), function(i) {
#'       observation(X, ind[i])
#'     })
#'   } else {
#'     if (is.null(ind)) {
#'       ind <- 1:nrow(X)
#'     } else {
#'       chk::chk_equal(length(X), length(ind))
#'     }
#'     lapply(1:nrow(X), function(i) {
#'       observation(X, ind[i])
#'     })
#'   }
#'   
#'   structure(ret, class="observation_set")
#' }
#' 
#' #' @export
#' `[[.observation_set` <- function(x,i) {
#'   z <- NextMethod()
#'   z()
#' }
#' 
#' #' @export
#' `[.observation_set` <- function(x, i) {
#'   z <- NextMethod()
#'   lapply(z, function(zi) zi())
#' }
#' 
#' 
#' #' @export
#' observation.deflist <- function(X, i) {
#'   chk::chk_scalar(i)
#'   f <- function() {
#'     X[[i]]
#'   }
#'   
#'   structure(f, i=i, class="observation")
#' }
#' 
#' 
#' #' @export
#' observation.vector <- function(x,i) {
#'   chk::chk_scalar(i)
#'   f <- function() {
#'     x
#'   }
#'   
#'   structure(f, i=i, class="observation")
#' }
#' 
#' #' @export
#' observation.matrix <- function(X, i) {
#'   chk::chk_scalar(i)
#'   f <- function() {
#'     X[i,,drop=FALSE]
#'   }
#'   
#'   structure(f, i=i, class="observation")
#' }
#' 
#' #' @export
#' observation.list <- function(X, i) {
#'   f <- function() {
#'     X[[i]]
#'   }
#'   
#'   structure(f, i=i, class="observation")
#' }
#' 
#' ## observation_set ... a list of observations. this can be "collapsed" to produce a data matrix
#' 
#' 
#' #' @export
#' multiframe.list <- function(x, y) {
#'   chk::chk_equal(length(x), nrow(y))
#'   chk::chk_s3_class(y, "data.frame")
#'   y <- as_tibble(y)
#'   des <- y %>% mutate(.index=1:n()) %>% rowwise() %>% mutate(.obs=list(observation.list(x, .index)))
#'   structure(list(
#'     design=des
#'   ),
#'   class="multiframe")
#'   
#' }
#' 
#' 
#' #' @importFrom dplyr mutate rowwise n
#' #' @export
#' multiframe.matrix <- function(x, y) {
#'   chk::chk_equal(nrow(x), nrow(y))
#'   chk::chk_s3_class(y, "data.frame")
#'   y <- as_tibble(y)
#'   des <- Y %>% mutate(.index=1:n()) %>% rowwise() %>% mutate(.obs=list(observation.matrix(x, .index)))
#'   structure(list(
#'     design=des
#'   ),
#'   class="multiframe")
#' }
#' 
#' 
#' #' @export
#' split.multiframe <- function(x, ..., collapse=FALSE) {
#'   nest.by <- rlang::quos(...)
#'   ret <- x$design %>% nest_by(!!!nest.by)
#'   
#'   if (collapse) {
#'     ret <- ret %>% rowwise() %>% mutate(data={
#'       list(do.call(rbind, lapply(data$.obs, function(o) o())))
#'     })
#'   }
#'   
#'   ret
#' }
#' 
#' 
#' #' @export
#' summarize_by.multiframe <- function(x, sfun=colMeans, extract_data=FALSE, ...) {
#'   #nested <- split(x, ...)
#'   nest.by <- rlang::quos(...)
#'   ret <- x$design %>% nest_by(!!!nest.by)
#'   
#'   ret <- ret %>% rowwise() %>% mutate(data={
#'       list(sfun(do.call(rbind, lapply(data$.obs, function(o) o()))))
#'   })
#'   
#'   if (extract_data) {
#'     ret <- do.call(rbind, ret$data %>% purrr::map( ~ .x))
#'   }
#'   
#'   ret
#' }



  
  

