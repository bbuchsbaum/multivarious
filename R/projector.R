#' Construct a `projector` instance
#'
#' A `projector` maps a matrix from an N-dimensional space to d-dimensional space, where `d` may be less than `N`.
#' The projection matrix, `v`, is not necessarily orthogonal. This function constructs a `projector` instance which can be
#' used for various dimensionality reduction techniques like PCA, LDA, etc.
#'
#' @param v A matrix of coefficients with dimensions `nrow(v)` by `ncol(v)` (columns = components)
#' @param preproc A prepped pre-processing object (S3 class `pre_processor`). Default is the no-op `pass()` preprocessor.
#' @param classes Additional class information used for creating subtypes of `projector`. Default is NULL.
#' @param ... Extra arguments to be stored in the `projector` object.
#'
#' @return An instance of type `projector`.
#'
#' @export
projector <- function(v, preproc = prep(pass()), ..., classes = NULL) {
  # B3: Ensure v is matrix
  if (!is.matrix(v)) {
    v <- as.matrix(v)
  }
  
  # chkor_vld(chk::chk_matrix(v), chk::chk_s4_class(v, "Matrix")) # Removed: Redundant after as.matrix and caused error
  chk::chk_s3_class(preproc, "pre_processor")
  
  out <- structure(
    list(
      v       = v,
      preproc = preproc,
      ...), 
    class = c(classes, "projector")
  )
  
  # Add cache environment attribute
  attr(out, ".cache") <- new.env(parent = emptyenv())
  out
}

#' @export
components.projector <- function(x, ...) {
  x$v
}



#' @export
coef.projector <- function(object, ...) {
  components(object)
}


#' @export
#' @importFrom assertthat assert_that
ncomp.projector <- function(x) {
  assertthat::assert_that(inherits(x, "projector"))
  ncol(components(x))
}

#' Stricter check for true orthogonality
#'
#' We test if v^T * v = I (when rows >= cols) or v * v^T = I (when cols > rows).
#'
#' @param tol tolerance for checking orthogonality
#' @param x the projector object
#' @export
#' @importFrom assertthat assert_that
is_orthogonal.projector <- function(x, tol=1e-6) {
  assertthat::assert_that(inherits(x, "projector"))
  v <- components(x)
  
  # If p >= d, check crossprod(v). Otherwise, check tcrossprod(v).
  if (nrow(v) >= ncol(v)) {
    mat <- crossprod(v)  # d x d
  } else {
    mat <- tcrossprod(v) # p x p
  }
  
  # We want it ~ identity. Check diagonal ~ 1 and off-diagonal ~ 0
  diag_elems <- diag(mat)
  off_diag   <- mat - diag(diag_elems)
  all(abs(diag_elems - 1) < tol) && all(abs(off_diag) < tol)
}

#' Possibly use ridge-regularized inversion of crossprod(v)
#' @keywords internal
robust_inv_vTv <- function(v, lambda = 1e-6) {
  vt_v   <- crossprod(v)                # d x d
  vt_v_r <- vt_v + diag(lambda, nrow(vt_v))
  solve(vt_v_r)
}


#' @rdname inverse_projection
#' @export
#' @importFrom assertthat assert_that
inverse_projection.projector <- function(x, ...) {
  assertthat::assert_that(inherits(x, "projector"))
  
  # Robust caching check
  cache_env <- attr(x, ".cache")
  use_caching <- !is.null(cache_env) && is.environment(cache_env)
  key <- "inv_proj"

  if (use_caching && !is.null(cache_env[[key]])) {
    return(cache_env[[key]])
  }

  # Compute if not cached or cache not available
  v <- coef(x)
  if (!requireNamespace("corpcor", quietly=TRUE)) {
    stop("package corpcor required for inverse_projection.")
  }
  inv_p <- corpcor::pseudoinverse(v)

  # Store in cache if available
  if (use_caching) {
    cache_env[[key]] <- inv_p
  }

  inv_p
}


#' @export
partial_inverse_projection.projector <- function(x, colind, ...) {
  assertthat::assert_that(inherits(x, "projector"))
  chk::chk_vector(colind)
  chk::chk_whole_numeric(colind)
  chk::chk_range(max(colind, na.rm=TRUE), c(1, ncol(coef(x))))
  chk::chk_range(min(colind, na.rm=TRUE), c(1, ncol(coef(x))))

  # Robust caching check
  cache_env <- attr(x, ".cache")
  use_caching <- !is.null(cache_env) && is.environment(cache_env)
  key <- paste0(".pinv_v_subset_", paste(sort(unique(colind)), collapse = "_"))

  if (use_caching && !is.null(cache_env[[key]])) {
    return(cache_env[[key]])
  }

  # Compute if not cached or cache not available
  v <- coef(x)[, colind, drop=FALSE]
  if (!requireNamespace("corpcor", quietly=TRUE)) {
      stop("package corpcor required for partial_inverse_projection.")
  }
  pinv_sub <- corpcor::pseudoinverse(v)

  # Store in cache if available
  if (use_caching) {
    cache_env[[key]] <- pinv_sub
  }

  pinv_sub
}


#' @export
truncate.projector <- function(x, ncomp) {
  old_ncomp <- ncomp(x)
  chk::chk_number(ncomp)
  if (ncomp < 1 || ncomp > old_ncomp) {
    stop("Requested ncomp must be between 1 and ", old_ncomp)
  }
  
  v_new       <- components(x)[, seq_len(ncomp), drop = FALSE]
  x$v         <- v_new
  cache_env   <- attr(x, ".cache")
  if (!is.null(cache_env)) {
    rm(list = ls(cache_env), envir = cache_env)  # Clear the cache
  }
  x
}


#' @export
reprocess.projector <- function(x, new_data, colind = NULL, ...) {
  p <- nrow(components(x))
  if (is.null(colind)) {
    # Full dimension
    chk::chk_equal(ncol(new_data), p)
    transform(x$preproc, new_data)
  } else {
    chk::chk_equal(length(colind), ncol(new_data))
    transform(x$preproc, new_data, colind)
  }
}

#' @export
shape.projector <- function(x, ...) {
  dim(components(x))
}


#' @export
project.projector <- function(x, new_data, ...) {
  if (is.vector(new_data)) {
    chk::chk_equal(length(new_data), shape(x)[1])
    new_data <- matrix(new_data, nrow = 1)
  }
  chk::vld_matrix(new_data)
  chk::chk_equal(ncol(new_data), nrow(components(x)))
  
  # Reprocess
  nd_proc <- reprocess(x, new_data)
  nd_proc %*% components(x)
}


#' @export
partial_project.projector <- function(x,
                                      new_data,
                                      colind,
                                      least_squares = TRUE,
                                      lambda = 1e-6,
                                      ...)
{
  # shape checks
  if (is.vector(new_data) && length(colind) > 1) {
    new_data <- matrix(new_data, nrow = 1)
  } else if (is.vector(new_data) && length(colind) == 1) {
    new_data <- matrix(new_data, ncol = 1)
  }
  chk::chk_equal(ncol(new_data), length(colind))
  
  # reprocess partial
  nd_proc <- reprocess(x, new_data, colind)
  
  v_sub <- components(x)[colind, , drop = FALSE]  # subset
  if (least_squares) {
    inv_vt_v   <- robust_inv_vTv(v_sub, lambda = lambda)  # (d x d)
    factor_scr <- nd_proc %*% v_sub %*% inv_vt_v
  } else {
    factor_scr <- nd_proc %*% v_sub
  }
  factor_scr
}


#' @export
reconstruct_new.projector <- function(x,
                                      new_data,
                                      colind         = NULL,
                                      least_squares  = TRUE,
                                      lambda         = 1e-6,
                                      ...)
{
  v_full <- components(x)
  
  if (is.null(colind)) {
    # Reconstruct all columns => new_data is (n x p)
    nd_proc <- reprocess(x, new_data, colind = NULL)
    if (least_squares) {
      inv_vt_v    <- robust_inv_vTv(v_full, lambda = lambda)
      factor_scr  <- nd_proc %*% v_full %*% inv_vt_v
    } else {
      factor_scr  <- nd_proc %*% v_full
    }
    rec_data <- factor_scr %*% t(v_full)
    
    # If you wish, do a reverse transform on rec_data:
    # Apply inverse transform to return to original space
    inverse_transform(x$preproc, rec_data)
  } else {
    # PARTIAL reconstruction => new_data is (n x length(colind))
    factor_scr <- partial_project.projector(
      x, new_data, colind,
      least_squares = least_squares,
      lambda        = lambda
    )
    v_sub    <- v_full[colind, , drop = FALSE]
    rec_data <- factor_scr %*% t(v_sub)
    
    # Apply inverse transform (potentially partial if needed) to return to original space
    inverse_transform(x$preproc, rec_data)
  }
}


#' @export
print.projector <- function(x, ...) {
  cat(crayon::bold(crayon::green("Projector object:\n")))
  cat(crayon::yellow("  Input dimension: "),  shape(x)[1], "\n", sep = "")
  cat(crayon::yellow("  Output dimension: "), shape(x)[2], "\n", sep = "")
  
  if (!is.null(x$preproc)) {
    cat(crayon::cyan("  With pre-processing:\n"))
    if (inherits(x$preproc, "pre_processor")) {
      print(x$preproc)
    } else {
      cat(crayon::cyan("    (pre-processing pipeline not fully available)\n"))
    }
  } else {
    cat(crayon::cyan("  No pre-processing pipeline.\n"))
  }
  invisible(x)
}



#' @export
partial_projector.projector <- function(x, colind, ...) {
  # We'll store a reference to the original (full) projector in x$porig
  # so partial calls can delegate back.
  new_obj <- projector(x$v[colind, ],
                       preproc = x$preproc,
                       colind  = colind,
                       porig   = x,
                       classes = "partial_projector")
  new_obj
}


#' @export
reprocess.partial_projector <- function(x, new_data, colind = NULL, ...) {
  chk::chk_not_null(x$colind)
  
  if (is.null(colind)) {
    # Full dimension = length(x$colind)
    chk::chk_equal(ncol(new_data), nrow(components(x)))
    transform(x$preproc, new_data)
  } else {
    chk::chk_equal(length(colind), ncol(new_data))
    base_colind <- x$colind
    chk::chk_not_null(base_colind)
    # Now map the local colind to the original projector's colind
    # e.g. base_colind[colind]
    transform(x$preproc, new_data, base_colind[colind])
  }
}


#' @export
project.partial_projector <- function(x, new_data, ...) {
  chk::chk_not_null(x$porig)
  chk::chk_not_null(x$colind)
  partial_project(x$porig, new_data, x$colind, ...)
}


#' @export
truncate.partial_projector <- function(x, ncomp) {
  chk::chk_not_null(x$porig)
  old_ncomp <- ncomp(x)
  if (ncomp < 1 || ncomp > old_ncomp) {
    stop("Requested ncomp must be between 1 and ", old_ncomp)
  }
  
  porig_trunc <- truncate(x$porig, ncomp)  # preserve extra fields
  # partial_projector from the truncated original
  partial_projector(porig_trunc, x$colind)
}

#' @export
partial_project.partial_projector <- function(x, new_data, colind, ...) {
  # Ensure we have a reference to the original projector and the stored colind
  if (is.null(x$porig) || is.null(x$colind)) {
    stop("`x` must have `porig` (original projector) and `colind` fields.")
  }
  
  # Local check: colind must be within [1, length(x$colind)]
  n_local_cols <- length(x$colind)
  if (any(colind < 1) || any(colind > n_local_cols)) {
    stop(
      "Requested columns (", paste(colind, collapse=", "), 
      ") exceed local partial projector's range [1..", n_local_cols, "]."
    )
  }
  
  # Map the local colind to the original projector's colind
  global_colind <- x$colind[colind]
  
  # Delegate to the original projector's partial_project
  partial_project(x$porig, new_data, global_colind, ...)
}
