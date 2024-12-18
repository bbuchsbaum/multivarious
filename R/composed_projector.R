#' Compose Multiple Partial Projectors
#'
#' Creates a `composed_partial_projector` object that applies partial projections sequentially.
#' If multiple projectors are composed, the column indices (colind) used at each stage must be considered.
#'
#' @param ... A sequence of projectors that implement `partial_project()`.
#' @return A `composed_partial_projector` object.
#' @export
#'
#' @examples
#' # Suppose pca1 and pca2 support partial_project().
#' # cpartial <- compose_partial_projector(pca1, pca2)
#' # partial_project(cpartial, new_data, colind=1:5)
compose_partial_projector <- function(...) {
  args <- list(...)
  lapply(args, function(p) chk::chk_s3_class(p, "projector"))
  
  if (length(args) == 1) {
    return(args[[1]])
  }
  
  shapelist <- lapply(args, shape)
  for (i in 2:length(args)) {
    chk::chk_equal(shapelist[[i-1]][2], shapelist[[i]][1])
  }
  
  out <- structure(
    list(projectors = args),
    class = c("composed_partial_projector", "composed_projector", "projector")
  )
  out
}


#' Partial Project Through a Composed Partial Projector
#'
#' Applies `partial_project()` through each projector in the composition.
#' If `colind` is a single vector, it applies to the first projector only. Subsequent projectors apply full columns.
#' If `colind` is a list, each element specifies the `colind` for the corresponding projector in the chain.
#'
#' @param x A `composed_partial_projector` object.
#' @param new_data The input data matrix or vector.
#' @param colind A numeric vector or a list of numeric vectors. If a single vector, applies to the first projector. 
#'   If a list, its length must match the number of projectors in `x`.
#' @param ... Additional arguments passed to `partial_project()` methods.
#'
#' @return The partially projected data after all projectors are applied.
#' @export
partial_project.composed_partial_projector <- function(x, new_data, colind, ...) {
  if (is.vector(new_data) && length(colind) > 1) {
    new_data <- matrix(new_data, nrow=1)
  } else if (is.vector(new_data) && length(colind) == 1) {
    new_data <- matrix(new_data, ncol=1)
  }
  
  chk::vld_matrix(new_data)
  
  projs <- x$projectors
  n_proj <- length(projs)
  
  # Determine how to handle colind:
  # If colind is a single vector, only apply it to the first projector.
  # If colind is a list, each element corresponds to a projector.
  
  if (is.list(colind)) {
    # Ensure length matches number of projectors
    if (length(colind) != n_proj) {
      stop("If colind is a list, its length must match the number of projectors.")
    }
    colinds <- colind
  } else {
    # Single vector: first projector uses colind, subsequent projectors use all columns
    colinds <- vector("list", n_proj)
    colinds[[1]] <- colind
    # For subsequent projectors, colinds will be set to NULL indicating "use all columns"
    for (i in 2:n_proj) {
      colinds[[i]] <- NULL
    }
  }
  
  current_data <- new_data
  
  for (i in seq_len(n_proj)) {
    proj <- projs[[i]]
    current_colind <- colinds[[i]]
    
    if (is.null(current_colind)) {
      # Use all columns for this step
      current_colind <- seq_len(ncol(current_data))
    } else {
      # Validate colind
      chk::chk_range(max(current_colind), c(1, ncol(current_data)))
      chk::chk_range(min(current_colind), c(1, ncol(current_data)))
    }
    
    current_data <- partial_project(proj, current_data, current_colind, ...)
    # After this step, current_data changes dimensions.
    # Next step colind interpretation will be handled similarly.
  }
  
  current_data
}


#' @export
project.composed_projector <- function(x, new_data, ...) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, nrow=1)
  }
  
  chk::vld_matrix(new_data)
  
  # Apply each projector in sequence
  for (proj in x$projectors) {
    new_data <- project(proj, new_data, ...)
  }
  
  new_data
}

#' @export
print.composed_projector <- function(x, ...) {
  n_proj <- length(x$projectors)
  cat("Composed projector object:\n")
  cat("  Number of projectors: ", n_proj, "\n")
  invisible(x)
}

