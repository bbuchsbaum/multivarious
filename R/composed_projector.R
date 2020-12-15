
compose_projector <- function(...) {
  args <- list(...)
  out <- lapply(args, function(arg) {
    chk::chk_s3_class(arg, "projector")
    f <- function(new_data) {
      project(arg, new_data)
    }
  })

  f <- do.call(purrr::combine, out,.dir="forward")
  
  out <- structure(fun,
    class=c("composed_projector", "function")
  )
}

compose_partial_projector <- function(...) {
  args <- list(...)
  out <- lapply(1:length(args), function(i) {
    arg <- args[[i]]
    chk::chk_s3_class(arg, "projector")
    f <- function(new_data, colind) {
      partial_project(arg, new_data, colind=1:nrow(components(x)))
    }
  })
  
  f <- do.call(purrr::combine, out,.dir="forward")
  
  out <- structure(fun,
                   class=c("composed_partial_projector", "composed_projector", "function")
  )
}


project.composed_projector <- function(x, new_data) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, byrow=TRUE)
  }
  chk::vld_matrix(new_data)
  
  x(new_data)
}

partial_project.composed_partial_projector <- function(x, new_data, colind) {
  if (is.vector(new_data) && length(colind) > 1) {
    new_data <- matrix(new_data, byrow=TRUE)
  } 
  chk::vld_matrix(new_data)
  chk::check_dim(new_data, ncol, length(colind))
  x(new_data, colind)
}


