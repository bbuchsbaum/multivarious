
compose_projector <- function(...) {
  args <- list(...)
  sapply(args, function(p) chk::chk_s3_class(p, "projector"))
  if (length(args) == 1) {
    return(args[[1]])
  }
  
  shapelist <- lapply(args, shape)
  for (i in 2:length(args)) {
    chk::chk_equal(shapelist[[i-1]][2], shapelist[[i]][1])
  }
  
  out <- lapply(args, function(arg) {
    f <- function(new_data) {
      project(arg, new_data)
    }
  })

  f <- do.call(purrr::compose, c(out,.dir="forward"))
  
  out <- structure(f,
    class=c("composed_projector", "function")
  )
}

compose_partial_projector <- function(...) {
  args <- list(...)
  out <- lapply(1:length(args), function(i) {
    arg <- args[[i]]
    chk::chk_s3_class(arg, "projector")
    f <- function(new_data, colind) {
      partial_project(arg, new_data, colind=1:nrow(coefficients(x)))
    }
  })
  
  f <- do.call(purrr::compose, c(out,.dir="forward"))
  
  out <- structure(f,
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


