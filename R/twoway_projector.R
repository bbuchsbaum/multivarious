#' Two-way (cross) projection to latent components
#'
#' A projector that reduces two blocks of data, X and Y, yielding a pair of weights for each component.
#' This structure can be used, for example, to store weights derived from canonical correlation analysis.
#'
#' @details This class extends `projector` and therefore basic operations such as `project`, `shape`, `reprocess`,
#' and `coef` work, but by default, it is assumed that the `X` block is primary. To access `Y` block operations, an
#' additional argument `source` must be supplied to the relevant functions, e.g., `coef(fit, source = "Y")`
#'
#' @param vx the X coefficients. Must have the same number of columns as `vy`.
#' @param vy the Y coefficients. Must have the same number of columns as `vx`.
#' @param preproc_x the X pre-processor
#' @param preproc_y the Y pre-processor
#' @param ... extra parameters or results to store
#' @param classes additional class names
#' @return a cross_projector object
#' @export
#' @examples
#' # Create two scaled matrices X and Y
#' X <- scale(matrix(rnorm(10 * 5), 10, 5))
#' Y <- scale(matrix(rnorm(10 * 5), 10, 5))
#'
#' # Perform canonical correlation analysis on X and Y
#' cres <- cancor(X, Y)
#' sx <- X %*% cres$xcoef
#' sy <- Y %*% cres$ycoef
#'
#' # Create a cross_projector object using the canonical correlation analysis results
#' canfit <- cross_projector(cres$xcoef, cres$ycoef, cor = cres$cor,
#'                           sx = sx, sy = sy, classes = "cancor")
cross_projector <- function(vx, vy, preproc_x=prep(pass()), preproc_y=prep(pass()), 
                             ..., classes=NULL) {
  
  chk::chkor_vld(chk::vld_matrix(vx), chk::vld_s4_class(vx, "Matrix"))
  chk::chkor_vld(chk::vld_matrix(vy), chk::vld_s4_class(vy, "Matrix"))
  chk::chk_equal(ncol(vx), ncol(vy))
  chk::chk_s3_class(preproc_x, "pre_processor")
  chk::chk_s3_class(preproc_y, "pre_processor")
  
  
  out <- structure(
    list(
      v=vx,
      vx=vx,
      vy=vy,
      preproc=preproc_x,
      preproc_x=preproc_x,
      preproc_y=preproc_y,
      .cache = new.env(parent = emptyenv()),
      ...),
    class= c(classes, "cross_projector", "projector")
  )
  
  out
}

#' project a cross_projector instance
#' 
#' @inheritParams project
#' @param source the source of the data (X or Y block)
#' @return the projected data
#' @export
#' @family project
project.cross_projector <- function(x, new_data, source=c("X", "Y"),...) {
  source <- match.arg(source)
  
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, nrow = 1)
  }
  chk::chk_matrix(new_data)
  
  # Validate dimensions
  expected_cols <- nrow(coef.cross_projector(x, source=source))
  if (ncol(new_data) != expected_cols) {
    stop(paste0("new_data must have ", expected_cols, " columns for source '", source, "'"))
  }
  
  reprocess(x, new_data, source=source) %*% coef.cross_projector(x, source=source)
}

#' Extract coefficients from a cross_projector object
#'
#' @param object the model fit
#' @param source the source of the data (X or Y block), either "X" or "Y"
#' @param ... extra args
#' @return the coefficients
#' @export
coef.cross_projector <- function(object, source=c("X", "Y"),...) {
  source <- match.arg(source)
  if (source == "X") {
    object$vx
  } else {
    object$vy
  }
}

#' reprocess a cross_projector instance
#' 
#' @inheritParams reprocess
#' @param source the source of the data (X or Y block)
#' @return the re(pre-)processed data
#' @details When `colind` is provided, each index is validated to be within the
#'   available coefficient rows using `chk::chk_subset`.
#' @export
#' @family reprocess
reprocess.cross_projector <- function(x, new_data, colind=NULL, source=c("X", "Y"), ...) {
  source <- match.arg(source)
  if (is.null(colind)) {
    chk::chk_equal(ncol(new_data), nrow(coef.cross_projector(x, source=source)))
    colind <- 1:ncol(new_data)
  } else {
    chk::chk_equal(length(colind), ncol(new_data))
    chk::chk_subset(colind, 1:nrow(coef.cross_projector(x, source=source)))
  }
    
  if (source == "X") {
    transform(x$preproc_x, new_data, colind)
  } else {
    transform(x$preproc_y, new_data, colind)
  }
  
}

#' Partially project data for a cross_projector
#'
#' Projects new data from either the X or Y domain onto the latent subspace,
#' considering only a specified subset of original features (`colind`).
#'
#' @param x A `cross_projector` object.
#' @param new_data A numeric matrix (n x length(colind)) or vector, representing
#'   the observations corresponding to the columns specified by `colind`.
#' @param colind A numeric vector of column indices in the original data space
#'   (either X or Y domain, specified by `source`) that correspond to `new_data`'s columns.
#' @param least_squares Logical; if TRUE (default), use ridge-regularized least squares for projection.
#' @param lambda Numeric; ridge penalty (default 1e-6). Ignored if `least_squares=FALSE`.
#' @param source Character, either "X" or "Y", indicating which domain `new_data` and `colind` belong to.
#' @param ... Additional arguments (currently ignored).
#'
#' @return A numeric matrix (n x d) of factor scores in the latent subspace.
#' @export
partial_project.cross_projector <- function(x, new_data, colind,
                                            least_squares=TRUE,
                                            lambda=1e-6,
                                            source=c("X","Y"),
                                            ...) {
  source <- match.arg(source)
  
  # Validate inputs
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, nrow = 1)
  }
  chk::chk_matrix(new_data)
  chk::chk_numeric(colind)
  
  # Use correct preprocessor based on source
  preproc <- if (source == "X") x$preproc_x else x$preproc_y
  nd_proc <- transform(preproc, new_data, colind)
  
  # subset columns in v
  v_full  <- coef.cross_projector(x, source=source)   # shape (p x d)
  v_sub   <- v_full[colind, , drop=FALSE]     # shape (|colind| x d)
  
  if (least_squares) {
    inv_vtv_sub <- robust_inv_vTv(v_sub, lambda=lambda)
    factor_scores <- nd_proc %*% v_sub %*% inv_vtv_sub
  } else {
    factor_scores <- nd_proc %*% v_sub
  }
  factor_scores
}

#' Transfer from X domain to Y domain (or vice versa) in a cross_projector
#'
#' @inherit transfer description
#' @param x A `cross_projector` object.
#' @param new_data The data to transfer.
#' @param from Source domain ("X" or "Y").
#' @param to Target domain ("X" or "Y").
#' @param opts A list of options (see `transfer` generic).
#' @param ... Ignored.
#' @return Transferred data matrix.
#' @details
#' When `opts$ls_rr` is `TRUE`, the forward projection from the `from`
#' domain is computed using a ridge-regularized least squares approach.
#' The penalty parameter is taken from `opts$lambda`. Component subsetting
#' via `opts$comps` is applied after computing these ridge-based scores.
#' @importFrom utils modifyList
#' @importFrom cli cli_abort
#' @export
transfer.cross_projector <- function(x, new_data,
                                     from, to,
                                     opts = list(),
                                     ...) {

  defaults <- list(cols    = NULL,   # Target columns subset
                   comps   = NULL,   # Latent components subset (currently unused in this method)
                   ls_rr   = FALSE,  # Ridge LS for forward projection?
                   lambda  = 1e-6)
  opts <- utils::modifyList(defaults, opts, keep.null = TRUE)

  # Validate from/to arguments
  # (Ideally use block_names(x) if available, but hardcoding for now)
  from <- match.arg(from, c("X", "Y"))
  to   <- match.arg(to,   c("X", "Y"))
  if (from == to)
    cli::cli_abort("{.arg from} ('{from}') and {.arg to} ('{to}') must differ.")

  # Validate input data
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, nrow = 1)
  }
  chk::chk_matrix(new_data)
  
  # ---------- 1. preprocess & dimension sanity ----------
  px <- if (from == "X") x$preproc_x else x$preproc_y
  
  # Check dimensions before preprocessing
  p_expected <- shape(x, source = from)[1] # Expected number of features in `from` domain
  if (ncol(new_data) != p_expected) {
    stop(paste0("new_data must have ", p_expected, " columns for source '", from, "'"))
  }
  
  # Apply transform
  nd_proc <- transform(px, new_data)

  # ---------- 2. forward projection ----------------------
  # Project the preprocessed data into the latent space
  # Optionally use ridge-regularized LS if opts$ls_rr is TRUE
  v_from <- coef.cross_projector(x, source = from)
  if (isTRUE(opts$ls_rr)) {
      inv_vtv <- robust_inv_vTv(v_from, lambda = opts$lambda)
      scores  <- nd_proc %*% v_from %*% inv_vtv
  } else {
      scores <- nd_proc %*% v_from
  }
                    
  # Subset components if requested
  if (!is.null(opts$comps)) {
      chk::chk_vector(opts$comps)
      chk::chk_subset(opts$comps, 1:ncol(scores))
      scores <- scores[, opts$comps, drop = FALSE]
  }

  # ---------- 3. back-projection to {to} ------------------
  # Get cached inverse for the target domain
  # If opts$comps was used, we need the corresponding columns of the inverse
  inv <- .cache_inv(x, to, colind=opts$comps, lambda = opts$lambda) # Pass opts$comps as colind for cache
  
  rec <- scores %*% inv # Reconstruct using (subsetted) scores and (subsetted) inverse

  # ---------- 4. undo preprocessing for target -----------
  p_to <- if (to == "X") x$preproc_x else x$preproc_y
  # Pass the target columns subset (opts$cols) to inverse_transform
  out  <- inverse_transform(p_to, rec, colind = opts$cols)

  # keep dimnames if sensible
  rownames(out) <- rownames(new_data)
  # Get target coefficient matrix to extract column names if needed
  target_coef_names <- colnames(coef(x, source = to))
  if (!is.null(opts$cols) && !is.null(target_coef_names)) {
      # Ensure opts$cols are valid indices for the target coefficient matrix
      chk::chk_vector(opts$cols)
      chk::chk_subset(opts$cols, 1:length(target_coef_names))
      colnames(out) <- target_coef_names[opts$cols]
  } else if (!is.null(target_coef_names)){
      # If reconstructing all columns, assign all names
      colnames(out) <- target_coef_names
  }

  out
}

#' Default inverse_projection method for cross_projector
#'
#' This function obtains the matrix that maps factor scores in the
#' latent space back into the original domain (X or Y). By default,
#' we assume \code{v_domain} is \emph{not} necessarily orthonormal or invertible,
#' so we use a pseudoinverse approach (e.g. MASS::ginv).
#'
#' @param x A `cross_projector` object.
#' @param domain Either \code{"X"} or \code{"Y"}, indicating which block's inverse 
#'        loading matrix we want (i.e., if you want to reconstruct data in the
#'        X space or Y space).
#' @param ... Additional arguments (currently unused, but may be used by subclasses).
#'
#' @return A matrix that, when multiplied by the factor scores, yields the
#'         reconstruction in the specified domain's original space.
#'
#' @export
#' @examples
#' # Suppose 'cp' is a cross_projector object. If we want the
#' # inverse for the Y domain:
#' #   inv_mat <- inverse_projection(cp, domain="Y")
#' # Then reconstruct:  Yhat <- Fscores %*% inv_mat
inverse_projection.cross_projector <- function(x, domain=c("X","Y"), ...) {
  domain <- match.arg(domain)
  
  # Validate that the cross_projector has the requested domain
  if (domain == "X" && is.null(x$vx)) {
    stop("No X domain coefficients in this cross_projector")
  }
  if (domain == "Y" && is.null(x$vy)) {
    stop("No Y domain coefficients in this cross_projector")
  }
  
  # We'll retrieve the loadings for the target domain.
  v_mat <- coef.cross_projector(x, source=domain)
  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Package 'MASS' is required for the default pseudoinverse approach.")
  }
  
  # Use ginv() for a minimal-norm pseudoinverse.
  inv_mat <- MASS::ginv(v_mat)
  
  inv_mat
}

#' Partial Inverse Projection of a Subset of the Loading Matrix in cross_projector
#'
#' This function obtains the "inverse" mapping for a columnwise subset of the loading
#' matrix in the specified domain. In practice, if \code{v_mat} is not orthonormal
#' or not square, we use a pseudoinverse approach (via \code{MASS::ginv}).
#'
#' By default, this is a minimal-norm solution for partial columns of \code{v_mat}.
#' If you need a different approach (e.g., ridge, direct solve, etc.), you can override
#' this method in your specific class or code.
#'
#' @param x A `cross_projector` object.
#' @param colind A numeric vector specifying the columns (indices) of the \emph{latent factors}
#'        or loadings to invert. Typically these correspond to a subset of canonical components
#'        or principal components, etc.
#' @param domain Either \code{"X"} or \code{"Y"}, indicating which block's partial
#'        loadings we want to invert.
#' @param ... Additional arguments (unused by default, but may be used by subclasses).
#'
#' @return A matrix of shape \code{(length(colind) x p_block)} that, when multiplied
#'         by factor scores restricted to \code{colind} columns, yields an
#'         \code{(n x p_block)} reconstruction in the original domain block.
#'
#' @export
#' @examples
#' # Suppose 'cp' is a cross_projector, and we want only columns 1:3 of
#' # the Y block factors. Then:
#' #   inv_mat_sub <- partial_inverse_projection(cp, colind=1:3, domain="Y")
#' # The shape will be (3 x pY), so factor_scores_sub (n x 3) %*% inv_mat_sub => (n x pY).
partial_inverse_projection.cross_projector <- function(x, colind, domain=c("X","Y" ), ...) {
  domain <- match.arg(domain)
  chk::chk_numeric(colind)
  chk::chk_not_empty(colind)
  
  # Convert to integer if needed
  colind <- as.integer(colind)
  chk::chk_all(colind > 0, "Column indices must be positive")
  
  # Robust caching check
  cache_env <- x$.cache
  use_caching <- !is.null(cache_env) && is.environment(cache_env)
  key <- paste0("partial_inv_proj_", domain, "@", paste0(sort(unique(colind)), collapse = "_"))

  if (use_caching && !is.null(cache_env[[key]])) {
    return(cache_env[[key]])
  }
  
  # Compute if not cached or cache not available
  v_mat <- coef.cross_projector(x, source=domain)  # shape = (p_block x d_total)
  chk::chk_range(max(colind, na.rm=TRUE), c(1, ncol(v_mat)))
  chk::chk_range(min(colind, na.rm=TRUE), c(1, ncol(v_mat)))

  if (!requireNamespace("MASS", quietly = TRUE)) {
    stop("Package 'MASS' is required for the default pseudoinverse approach.")
  }
  
  v_sub <- v_mat[, colind, drop=FALSE]  # shape = (p_block x length(colind))
  inv_sub <- MASS::ginv(v_sub)  # minimal-norm pseudoinverse

  # Store in cache if available
  if (use_caching) {
    cache_env[[key]] <- inv_sub
  }
  
  inv_sub
}

#' shape of a cross_projector instance
#' 
#' @param source the source of the data (X or Y block)
#' @return the shape of the data
#' @export
#' @family shape
#' @inheritParams shape
shape.cross_projector <- function(x, source=c("X", "Y"), ...) {
  source <- match.arg(source)
  if (source == "X") {
    c(nrow(x$vx), ncol(x$vx))
  } else {
    c(nrow(x$vy), ncol(x$vy))
  }
}



#' @seealso \code{\link{perm_test}}, \code{\link{measure_interblock_transfer_error}}, 
#'   \code{\link{cross_projector}}
#' @export
perm_test.cross_projector <- function(x,
                                      X,
                                      Y,
                                      nperm = 100,
                                      measure_fun = NULL,
                                      shuffle_fun = NULL,
                                      fit_fun = NULL,
                                      stepwise = FALSE,
                                      parallel = FALSE,
                                      alternative = c("greater", "less", "two.sided"),
                                      ...) 
{
  alternative <- match.arg(alternative)
  
  # Validate dimensions
  if (nrow(X) != nrow(Y)) stop("X and Y must have the same number of rows.")
  
  # Capture extra arguments
  extra_args <- list(...)
  
  # ---------- Default Measure Function (x2y.mse) ----------
  if (is.null(measure_fun)) {
    measure_fun <- function(model_perm, X_orig, Y_perm, ...) {
      # Note: We evaluate the permuted model on the original X and *permuted* Y
      out <- measure_interblock_transfer_error(
        Xtrue = X_orig, Ytrue = Y_perm, 
        model = model_perm, metrics="x2y.mse"
      )
      out[["x2y.mse"]]  # single numeric
    }
    stat_name <- "x2y.mse (default)"
  } else {
    stat_name <- deparse(substitute(measure_fun))
  }
  
  # ---------- Default Shuffle Function (shuffle rows of Y) ----------
  if (is.null(shuffle_fun)) {
    shuffle_fun <- function(Xblock, Yblock, ...) {
      idx <- sample(nrow(Yblock))
      Yblock[idx, , drop=FALSE]
    }
  }
  
  # ---------- Default Fit Function (stats::cancor) ----------
  if (is.null(fit_fun)) {
    fit_fun <- function(Xtrain, Ytrain, preproc_x_orig, preproc_y_orig, ...) {
      if (!requireNamespace("stats", quietly = TRUE)) {
          stop("Package 'stats' (for cancor) needed for default fit_fun. Please install it.", call. = FALSE)
      }
      ccr_args <- c(list(x = Xtrain, y = Ytrain), list(...))
      ccr <- try(do.call(stats::cancor, ccr_args), silent = TRUE)
      if (inherits(ccr, "try-error")) {
        warning(sprintf("Default fit_fun (stats::cancor) failed: %s. Returning NULL.", as.character(ccr)))
        return(NULL)
      }
      # Need original preprocessors to build the new projector
      cross_projector(ccr$xcoef, ccr$ycoef,
                      preproc_x = preproc_x_orig, 
                      preproc_y = preproc_y_orig)
    }
  }
  
  # ---------- Observed Statistic ----------
  # Evaluate original model on original data
  measure_args_obs <- c(list(model_perm = x, X_orig = X, Y_perm = Y), extra_args)
  obs_stat <- try(do.call(measure_fun, measure_args_obs), silent = TRUE)
  
  if (inherits(obs_stat, "try-error") || !is.numeric(obs_stat) || length(obs_stat) != 1) {
      warning(sprintf("Could not calculate observed statistic using measure_fun: %s", obs_stat))
      obs_stat <- NA_real_
  }
  if (is.na(obs_stat)) {
      warning("Observed statistic is NA. Permutation test cannot proceed meaningfully.")
      # Return early with NA results? Or let it run and return NA p-value? Let it run for now.
  }

  # Track different types of failures
  n_fit_failures <- 0
  n_measure_failures <- 0
  
  # ---------- Permutation Loop Function ----------
  one_perm <- function(i, ...) {
    # Shuffle Y
    shuffle_args <- c(list(Xblock = X, Yblock = Y), extra_args)
    Yperm <- do.call(shuffle_fun, shuffle_args)
    
    # Refit model on X and permuted Y, passing original preprocessors
    fit_args <- c(list(Xtrain = X, Ytrain = Yperm, 
                       preproc_x_orig = x$preproc_x, preproc_y_orig = x$preproc_y), 
                  extra_args)
    new_mod <- try(do.call(fit_fun, fit_args), silent = TRUE)
    if (inherits(new_mod, "try-error") || is.null(new_mod)) {
        warning(sprintf("Permutation %d: fit_fun failed: %s. Returning NA.", i, 
                       ifelse(is.null(new_mod), "NULL return", as.character(new_mod))))
        n_fit_failures <<- n_fit_failures + 1
        return(NA_real_)
    }
    
    # Calculate statistic using permuted model and permuted Y
    measure_args_perm <- c(list(model_perm = new_mod, X_orig = X, Y_perm = Yperm), extra_args)
    stat_perm <- try(do.call(measure_fun, measure_args_perm), silent = TRUE)
    
    if (inherits(stat_perm, "try-error")) {
        warning(sprintf("Permutation %d: measure_fun failed: %s. Returning NA.", i, as.character(stat_perm)))
        n_measure_failures <<- n_measure_failures + 1
        return(NA_real_)
    }
     if (!is.numeric(stat_perm) || length(stat_perm) != 1) {
        warning(sprintf("Permutation %d: measure_fun did not return a single numeric value. Returning NA.", i))
        n_measure_failures <<- n_measure_failures + 1
        return(NA_real_)
    }
    stat_perm
  }
  
  # ---------- Run Permutations (Serial or Parallel) ----------
  message(sprintf("Running %d permutations for cross projector (%s)...",
                  nperm, if(parallel) "parallel" else "serial"))
  if (parallel && !requireNamespace("future.apply", quietly = TRUE))
    stop("Package 'future.apply' required for parallel execution.", call.=FALSE)
  apply_fun <- if (parallel) future.apply::future_lapply else lapply
  perm_args <- list(X = seq_len(nperm), FUN = one_perm)
  # Pass ... down to one_perm via the lapply function's ...
  if (parallel) perm_args$future.seed <- TRUE
  perm_args <- c(perm_args, extra_args) # Add ... to the apply_fun call
  
  perm_vals_list <- do.call(apply_fun, perm_args)
  perm_vals <- unlist(perm_vals_list)
  n_complete <- sum(!is.na(perm_vals))
  
  if (n_complete < nperm) {
      warning(sprintf("%d permutations failed and were excluded (%d fit failures, %d measure failures).", 
                      nperm - n_complete, n_fit_failures, n_measure_failures))
      perm_vals <- stats::na.omit(perm_vals)
  }
  if (n_complete == 0) {
      stop("All permutations failed. Cannot compute p-value.")
  }
  
  # ---------- P-value Calculation (Empirical with +1 smoothing) ----------
  pval <- NA # Initialize
  if (!is.na(obs_stat)) { # Only calculate p-value if observed stat is valid
      if (alternative == "greater") {
        b <- sum(perm_vals >= obs_stat, na.rm = TRUE)
        pval <- (b + 1) / (n_complete + 1)
      } else if (alternative == "less") {
        b <- sum(perm_vals <= obs_stat, na.rm = TRUE)
        pval <- (b + 1) / (n_complete + 1)
      } else { # two.sided
        b_greater <- sum(perm_vals >= obs_stat, na.rm = TRUE)
        b_less <- sum(perm_vals <= obs_stat, na.rm = TRUE)
        pval <- 2 * min((b_greater + 1) / (n_complete + 1), (b_less + 1) / (n_complete + 1))
        pval <- min(pval, 1.0) # Ensure p-value doesn't exceed 1
      }
  }
  
  # ---------- Output Structure ----------
  out <- structure(
    list(
      statistic = obs_stat,
      perm_values = perm_vals,
      p.value = pval,
      alternative = alternative,
      method = sprintf("Permutation test for cross_projector (measure: %s)", stat_name),
      nperm = n_complete, 
      call = match.call()
    ),
    class = "perm_test"
  )
  
  out
}

#' @export
print.cross_projector <- function(x,...) {
  cat("cross projector: ", paste0(class(x)), "\n")
  cat("input dim (X): ", shape(x, source="X")[1], "\n")
  cat("output dim (X): ", shape(x, source="X")[2], "\n")
  cat("input dim (Y): ", shape(x, source="Y")[1], "\n")
  cat("output dim (Y): ", shape(x, source="Y")[2], "\n")
}

# R/cross_projector-transfer.R (or within R/twoway_projector.R)
# --------------------------------------------------------------
# cache pseudoinverses once
# Not exported, internal helper
.cache_inv <- function(x, domain, colind = NULL, lambda = 1e-6) {
  # Robust caching check
  cache_env <- x$.cache
  use_caching <- !is.null(cache_env) && is.environment(cache_env)

  # Create a unique key based on domain, target columns, and lambda
  key_suffix <- if (!is.null(colind)) paste0(sort(colind), collapse = "_") else "all"
  key <- paste0("inv_", domain, "@", key_suffix, "_lambda", lambda)
  
  # Return cached value if available
  if (use_caching && !is.null(cache_env[[key]])) {
      return(get(key, envir = cache_env, inherits = FALSE))
  }

  # Compute if not cached or cache not available
  v <- coef.cross_projector(x, source = domain)
  
  # Apply colind if provided (subsetting components)
  if (!is.null(colind)) {
      chk::chk_vector(colind)
      chk::chk_subset(colind, 1:ncol(v))
      v <- v[, colind, drop = FALSE]
  }
  
  inv <- tryCatch(MASS::ginv(v), error = function(e) NA)
  
  # Fallback to regularized inverse if ginv fails or yields non-finite values
  if (anyNA(inv) || any(!is.finite(inv))) { 
    warning("MASS::ginv failed or returned non-finite values; using regularized inverse.")
    # Calculate t(v) %*% v + lambda*I
    vTv <- crossprod(v)
    reg_inv <- tryCatch(solve(vTv + diag(lambda, ncol(v))) %*% t(v), error = function(e) NA)
    if (any(!is.finite(reg_inv))) { 
        stop("Both ginv and regularized inverse failed for domain ", domain)
    }
    inv <- reg_inv
  }
  
  # Store in cache if available
  if (use_caching) {
    assign(key, inv, envir = cache_env)
  }
  
  inv
}
