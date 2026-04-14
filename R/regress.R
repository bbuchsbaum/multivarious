#' Multi-output linear regression
#'
#' Fit a multivariate regression model for a matrix of basis functions, `X`, and a response matrix `Y`.
#' The goal is to find a projection matrix that can be used for mapping and reconstruction.
#'
#' @param X the set of independent (basis) variables
#' @param Y the response matrix
#' @param preproc A preprocessing pipeline applied to \code{X} before fitting the model
#' @param method the regression method: `lm`, `enet`, `mridge`, or `pls`
#' @param intercept whether to include an intercept term
#' @param lambda ridge shrinkage parameter (for methods `mridge` and `enet`)
#' @param alpha the elastic net mixing parameter if method is `enet`
#' @param ncomp number of PLS components if method is `pls`
#' @param ... extra arguments sent to the underlying fitting function
#' @return a bi-projector of type `regress`. The `sdev` component of this object
#'   stores the standard deviations of the columns of the design matrix (`X` potentially
#'   including an intercept) used in the fit, not the standard deviations of latent
#'   components as might be typical in other `bi_projector` contexts (e.g., SVD).
#' @export
#' @importFrom glmnet glmnet
#' @importFrom Matrix t
#' @importFrom pls plsr
#' @importFrom stats coef lm.fit sd
#' @examples
#' # Generate synthetic data
#' set.seed(123) # for reproducibility
#' Y <- matrix(rnorm(10 * 100), 10, 100)
#' X <- matrix(rnorm(10 * 9), 10, 9)
#' 
#' # Fit regression models and reconstruct the fitted response matrix
#' r_lm <- regress(X, Y, intercept = FALSE, method = "lm")
#' recon_lm <- reconstruct(r_lm) # Reconstructs fitted Y
#' 
#' r_mridge <- regress(X, Y, intercept = TRUE, method = "mridge", lambda = 0.001)
#' recon_mridge <- reconstruct(r_mridge)
#' 
#' r_enet <- regress(X, Y, intercept = TRUE, method = "enet", lambda = 0.001, alpha = 0.5)
#' recon_enet <- reconstruct(r_enet)
#' 
#' r_pls <- regress(X, Y, intercept = TRUE, method = "pls", ncomp = 5)
#' recon_pls <- reconstruct(r_pls)
regress <- function(X, Y, preproc=pass(), method=c("lm", "enet", "mridge", "pls"), 
                    intercept=FALSE, lambda=.001, alpha=0, 
                    # Default ncomp for PLS: arbitrary, consider tuning
                    ncomp=ceiling(ncol(X)/2), ...) {
  method <- match.arg(method)
  
  # --- Preprocessing Handling ---
  # Ensure preproc is initialized and apply it to X
  if (!inherits(preproc, "pre_processor")) {
    # If it's a prepper object or similar, finalize it
    result <- fit_transform(preproc, X)
    proc <- result$preproc
    X_processed <- result$transformed
  } else {
    # Already a finalized pre_processor - use transform
    proc <- preproc
    X_processed <- transform(proc, X)
  }
  
  # Initialize and apply the transform to X
  # Note: We process X *before* adding the intercept column
  # X_processed <- init_transform(proc, X)
  
  # --- Intercept Handling ---
  # Manually add intercept column to the *processed* X if requested,
  # and tell underlying fitters NOT to add one.
  if (intercept) {
    X_fit <- cbind(Intercept = 1, X_processed)  
    intercept_flag <- FALSE            
  } else {
    X_fit <- X_processed
    intercept_flag <- FALSE # Fitters should not add intercept if not in X_fit
  }
  
  # Store the processed design matrix (including intercept if added)
  # that coefficients will correspond to
  # TODO: [INEFF] Storing the full design matrix can be memory intensive.
  #       Consider alternatives if X_fit is very large.
  scores <- X_fit 
  
  # Compute betas depending on the method
  betas <- {
    # Compute betas depending on the method (using X_fit)
    b <- if (method == "lm") {
      # Use lm.fit for potentially better performance/stability than lsfit
      lfit <- stats::lm.fit(X_fit, Y)
      # Coefficients: rows are predictors (matching X_fit cols), cols are responses
      t(stats::coef(lfit)) # Transpose to get shape (p_out x p_in)
      
    } else if (method == "mridge") {
      if (!requireNamespace("glmnet", quietly = TRUE)) {
          stop("Package 'glmnet' needed for method='mridge'. Please install it.", call. = FALSE)
      }
      gfit <- glmnet::glmnet(X_fit, Y, alpha = 0, family = "mgaussian", 
                             lambda = lambda, intercept = intercept_flag, ...) 
      cf <- stats::coef(gfit, s = lambda)
      bmat <- do.call(cbind, lapply(cf, as.matrix))
      # Transpose to get p_out x (p_in+1 if intercept fitted by glmnet)
      t(bmat)

    } else if (method == "enet") {
      if (!requireNamespace("glmnet", quietly = TRUE)) {
          stop("Package 'glmnet' needed for method='enet'. Please install it.", call. = FALSE)
      }
      gfit <- glmnet::glmnet(X_fit, Y, alpha = alpha, family = "mgaussian", 
                             lambda = lambda, intercept = intercept_flag, ...)
      cf <- stats::coef(gfit, s = lambda)
      bmat <- do.call(cbind, lapply(cf, as.matrix))
      # Transpose to get p_out x (p_in+1 if intercept fitted by glmnet)
      t(bmat)
      
    } else { # method == "pls"
      if (!requireNamespace("pls", quietly = TRUE)) {
          stop("Package 'pls' needed for method='pls'. Please install it.", call. = FALSE)
      }
      fit <- pls::plsr(Y ~ scores, ncomp = ncomp, data = data.frame(Y=Y, scores=scores), ...) # Need data frame for formula
      cf <- stats::coef(fit, ncomp = ncomp, intercept = FALSE) # Intercept already in 'scores'
      # pls::coef returns a 3D array (p_in x p_out x 1), drop the third dimension
      cf <- drop(cf)
      # Result is typically p_in x p_out, need p_out x p_in
      t(cf)
    }
    
    # FIX: Remove intercept column from glmnet betas to avoid dimension issues
    # glmnet always includes an "(Intercept)" coefficient as the first column
    # of the returned matrix, even when we supply our own intercept column.
    # Drop that column unconditionally for the mridge and enet methods.
    if (method %in% c("mridge", "enet")) {
      b <- b[, -1, drop = FALSE]
    }
    b
  }
  
  # Create a bi_projector
  # v = betas (coefficients, p_out x p_in)
  # s = scores (design matrix X_fit, N x p_in)
  # Y_approx = s %*% t(v)
  
  # Calculate sdev for the scores matrix (X_fit)
  sds <- matrixStats::colSds(as.matrix(scores))
  # Handle columns with zero standard deviation (like intercept)
  zero_sd_idx <- sds < .Machine$double.eps
  if (any(zero_sd_idx)) {
      # Optional: Warn the user
      # warning("Columns ", paste(which(zero_sd_idx), collapse=", "), " in the design matrix have zero standard deviation. Setting sdev to 1.")
      sds[zero_sd_idx] <- 1
  }
  
  p <- bi_projector(v = betas, 
                    s = scores,
                    sdev = sds, # Pass the calculated standard deviations
                    preproc=proc, # Store the *initialized* preprocessor
                    coefficients = betas, # Store betas explicitly
                    method = method,
                    classes = "regress")
  p
}


#' @export
inverse_projection.regress <- function(x,...) {
  # inverse projection should map scores back to Y (or approx Y)
  # Y_approx = scores %*% t(betas)
  # If v = betas, then inverse_projection = t(v) = t(betas)
  t(x$coefficients) # This matches if v = betas
}

#' Reconstruct fitted or subsetted outputs for a `regress` object
#'
#' For regression-based bi_projectors, reconstruction should map from the
#' design matrix side (scores) to the output space using the regression
#' coefficients, without applying any reverse preprocessing (which belongs
#' to the input/basis side).
#'
#' @param x A `regress` object produced by \code{regress()}.
#' @param comp Integer vector of component indices (columns of the design matrix / predictors) to use.
#' @param rowind Integer vector of row indices in the design matrix (observations) to reconstruct.
#' @param colind Integer vector of output indices (columns of Y) to reconstruct.
#' @param ... Ignored.
#' @export
reconstruct.regress <- function(x,
                                comp = 1:ncol(x$coefficients),
                                rowind = 1:nrow(scores(x)),
                                colind = 1:nrow(x$coefficients),
                                ...) {
  # scores(x): design matrix (Nobs x p_in)
  # coefficients: betas (p_out x p_in)
  S <- scores(x)[rowind, comp, drop = FALSE]
  Bsub_t <- t(x$coefficients[colind, comp, drop = FALSE]) # (length(comp) x length(colind))
  S %*% Bsub_t
}

#' @export
project_vars.regress <- function(x, new_data,...) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data)
  }
  # Check dimension: new_data rows = nrow(scores(x))
  chk::chk_equal(nrow(new_data), nrow(scores(x)))
  
  # project_vars for regress: t(new_data) %*% scores(x)
  # Projects new variables onto the original predictor space.
  # If new_data is NxM and scores is NxC, result is MxC
  t(new_data) %*% (scores(x))
}

#' Partial Inverse Projection for a `regress` Object
#'
#' This function computes a sub-block inversion of the regression coefficients,
#' allowing you to focus on only certain columns (e.g. partial factors).
#' If your coefficient matrix is not orthonormal or is not square, we use a
#' pseudoinverse approach (via `corpcor::pseudoinverse`) to find a minimal-norm
#' solution. 
#'
#' @param x A `regress` object (created by \code{\link{regress}}).
#' @param colind A numeric vector specifying which columns of the \emph{factor space}
#'        (i.e., the second dimension of \code{x$coefficients}) you want to invert.
#'        Typically these refer to a subset of canonical / PCA / PLS components.
#' @param ... Further arguments passed to or used by methods (not used here).
#'
#' @return A matrix of shape \code{(length(colind) x nrow(x$coefficients))}. When
#'         multiplied by partial factor scores \code{(n x length(colind))}, it yields
#'         an \code{(n x nrow(x$coefficients))} reconstruction in the original domain.
#'
#' @export
partial_inverse_projection.regress <- function(x, colind, ...) {
  # We assume x$coefficients is shape (p_out x p_in),
  # where p_out is # of outputs, and p_in is # of (X dimension).
  # For partial factors, we interpret 'colind' as columns
  # in that factor dimension. So we subset horizontally.
  
  # Subset columns of x$coefficients
  # e.g., if x$coefficients is (p_out x d_total),
  # then betas_sub is (p_out x length(colind)).
  betas_sub <- x$coefficients[, colind, drop=FALSE]
  
  # Inverse it (in a minimal-norm sense) with corpcor::pseudoinverse
  # This yields (length(colind) x p_out).
  if (!requireNamespace("corpcor", quietly = TRUE)) {
    stop("Package 'corpcor' must be installed for partial_inverse_projection in `regress`.")
  }
  inv_mat_sub <- corpcor::pseudoinverse(betas_sub)
  
  inv_mat_sub
}

#' Pretty Print Method for `regress` Objects
#'
#' Display a human-readable summary of a `regress` object using crayon formatting, 
#' including information about the method and dimensions.
#'
#' @param x A `regress` object (a bi_projector with regression info).
#' @param ... Additional arguments passed to `print()`.
#' @export
print.regress <- function(x, ...) {
  cat(crayon::bold(crayon::green("Regression bi_projector object:\n")))
  
  # Display method
  if (!is.null(x$method)) {
    cat(crayon::yellow("  Method: "), crayon::cyan(x$method), "\n", sep="")
  } else {
    cat(crayon::yellow("  Method: "), crayon::cyan("unknown"), "\n", sep="")
  }
  
  # Input/Output dims from v
  cat(crayon::yellow("  Input dimension: "), nrow(x$v), "\n", sep="")
  cat(crayon::yellow("  Output dimension: "), ncol(x$v), "\n", sep="")
  
  # Check if intercept was used: If intercept present, betas includes an extra row
  # but we have no direct flag. The code doesn't store intercept explicitly,
  # so we won't guess. Let's just print coefficients dim:
  cat(crayon::yellow("  Coefficients dimension: "),
      paste(dim(x$coefficients), collapse=" x "), "\n")
  
  invisible(x)
}
