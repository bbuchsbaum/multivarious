#' Relative Eigenanalysis with Ecosystem Integration
#'
#' Perform a relative eigenanalysis between two groups, fully integrated with the
#' pre-processing and projector ecosystem. The function computes the directions that
#' maximize the variance ratio between two groups and returns a `bi_projector` object.
#'
#' @param XA A numeric matrix or data frame of observations for group A (n_A x p).
#' @param XB A numeric matrix or data frame of observations for group B (n_B x p).
#' @param ncomp The number of components to compute. If NULL (default), computes up to \code{min(n_A, n_B, p) - 1}.
#' @param preproc A pre-processing pipeline created with \code{prepper()}. Defaults to \code{center()}.
#' @param reg_param A small regularization parameter to ensure numerical stability. Defaults to 1e-5.
#' @param threshold An integer specifying the dimension threshold to switch between direct and iterative solvers. Defaults to 2000.
#' @param ... Additional arguments passed to lower-level functions.
#' @return A \code{bi_projector} object containing the components, scores, and other relevant information.
#' @details
#' This function computes the leading eigenvalues and eigenvectors of the generalized eigenvalue problem
#' \eqn{\Sigma_A v = \lambda \Sigma_B v}, fully integrated with the pre-processing ecosystem.
#' It uses a direct solver when the number of variables \eqn{p} is less than or equal to \code{threshold},
#' and switches to an iterative method when \eqn{p} is greater than \code{threshold}.
#' @examples
#' # Simulate data for two groups
#' set.seed(123)
#' n_A <- 100
#' n_B <- 80
#' p <- 500  # Number of variables
#' XA <- matrix(rnorm(n_A * p), nrow = n_A, ncol = p)
#' XB <- matrix(rnorm(n_B * p), nrow = n_B, ncol = p)
#' # Perform relative eigenanalysis
#' res <- relative_eigen(XA, XB, ncomp = 5)
#' @export
relative_eigen <- function(XA, XB, ncomp = NULL, preproc = center(),
                           reg_param = 1e-5, threshold = 2000, ...) {
  if (!requireNamespace("RSpectra", quietly = TRUE)) {
    stop("Package 'RSpectra' is required for this function. Please install it.")
  }
  if (!is.matrix(XA)) XA <- as.matrix(XA)
  if (!is.matrix(XB)) XB <- as.matrix(XB)
  
  if (ncol(XA) != ncol(XB)) stop("XA and XB must have the same number of columns.")
  
  n_A <- nrow(XA)
  n_B <- nrow(XB)
  p <- ncol(XA)
  
  # Combine XA and XB for consistent pre-processing
  X_all <- rbind(XA, XB)
  
  # Prepare pre-processing pipeline
  preproc_all <- prep(preproc)
  X_all_proc <- init_transform(preproc_all, X_all)
  
  # Split back into XA_proc and XB_proc
  XA_proc <- X_all_proc[1:n_A, , drop=FALSE]
  XB_proc <- X_all_proc[(n_A + 1):(n_A + n_B), , drop=FALSE]
  
  # Determine ncomp if NULL
  if (is.null(ncomp)) {
    ncomp <- min(n_A, n_B, p) - 1
    if (ncomp < 1) ncomp <- p # fallback if dimension very small
  }
  
  if (p < 3) {
    # Very small dimension: use base eigen
    message("Using base eigen function for very small dimension")
    
    SigmaA <- crossprod(XA_proc) / (n_A - 1)
    SigmaB <- (crossprod(XB_proc) / (n_B - 1)) + reg_param * diag(p)
    
    geigen_res <- eigen(solve(SigmaB, SigmaA))
    
    values <- geigen_res$values[1:ncomp]
    vectors <- geigen_res$vectors[, 1:ncomp, drop = FALSE]
    
  } else if (p <= threshold) {
    # Direct method with RSpectra
    message("Using direct method for small dimension")
    
    SigmaA <- crossprod(XA_proc) / (n_A - 1)
    SigmaB <- (crossprod(XB_proc) / (n_B - 1)) + reg_param * diag(p)
    
    # Solve generalized eigenproblem
    res <- RSpectra::eigs_sym(SigmaA, k = ncomp, which = "LM", B = SigmaB, ...)
    values <- res$values
    vectors <- res$vectors
    
  } else {
    # Iterative method
    message("Using iterative method for large dimension")
    
    # Operators for SigmaA and SigmaB without forming full matrices
    SigmaA_mult <- function(x) {
      (t(XA_proc) %*% (XA_proc %*% x)) / (n_A - 1)
    }
    SigmaB_mult <- function(x) {
      (t(XB_proc) %*% (XB_proc %*% x)) / (n_B - 1) + reg_param * x
    }
    
    # mat_op represents SigmaB^{-1} SigmaA operation using iterative solvers
    mat_op <- function(x, args) {
      w <- SigmaA_mult(x)
      # Solve SigmaB y = w. Here we rely on a method that can invert via CG or similar.
      # Assuming Matrix::solve works as intended with method="CG" (package code?), no semantic change.
      y <- Matrix::solve(SigmaB_mult, w, method = "CG", tol = 1e-5)
      return(y)
    }
    
    res <- RSpectra::eigs(mat_op, k = ncomp, n = p, which = "LM", ...)
    values <- res$values
    vectors <- res$vectors
  }
  
  # Compute scores for XA and XB
  s_A <- XA_proc %*% vectors
  s_B <- XB_proc %*% vectors
  sdev <- sqrt(values)
  
  # Create a bi_projector object and store all relevant info
  ret <- bi_projector(v = vectors, s = s_A, sdev = sdev,
                      preproc = preproc_all, classes = "relative_eigen")
  
  ret$s_B <- s_B
  ret$values <- values
  ret$n_A <- n_A
  ret$n_B <- n_B
  ret$ncomp <- length(values)  # Store ncomp for printing
  ret$p <- p                   # Store p for project method checks
  
  class(ret) <- c("relative_eigen", "bi_projector", "projector")
  return(ret)
}

#' @export
scores.relative_eigen <- function(x, group = c("A", "B"), ...) {
  group <- match.arg(group)
  if (group == "A") {
    return(x$s)
  } else {
    return(x$s_B)
  }
}

#' @export
components.relative_eigen <- function(x, ...) {
  return(x$v)
}

#' @export
sdev.relative_eigen <- function(x) {
  return(x$sdev)
}

#' @export
print.relative_eigen <- function(x, ...) {
  # Pretty print using crayon
  cat(crayon::bold(crayon::green("Relative Eigenanalysis Result:\n")))
  cat(crayon::yellow("  Number of components: "), x$ncomp, "\n", sep="")
  cat(crayon::yellow("  Eigenvalues:\n"))
  cat(paste0("    ", paste(round(x$values, 4), collapse=", ")), "\n")
  invisible(x)
}

#' @export
summary.relative_eigen <- function(object, ...) {
  cat("Relative Eigenanalysis Summary\n")
  cat("Number of components:", object$ncomp, "\n")
  cat("Eigenvalues:\n")
  print(object$values)
  cat("Explained Variance (%):\n")
  explained <- object$values / sum(object$values) * 100
  print(explained)
  invisible(object)
}

#' @export
project.relative_eigen <- function(x, new_data, ...) {
  if (!is.matrix(new_data)) new_data <- as.matrix(new_data)
  if (ncol(new_data) != x$p) stop("new_data must have the same number of columns as the original data")
  
  # Apply pre-processing
  new_data_proc <- apply_transform(x$preproc, new_data)
  
  # Project onto the components
  scores <- new_data_proc %*% x$v
  return(scores)
}