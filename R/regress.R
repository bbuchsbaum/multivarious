#' Multi-output linear regression
#'
#' Fit a multivariate regression model for a matrix of basis functions, `X`, and a response matrix `Y`.
#' The goal is to find a projection matrix that can be used for mapping and reconstruction.
#'
#' @param X the set of independent (basis) variables
#' @param Y the response matrix
#' @param preproc the pre-processor (currently unused)
#' @param method the regression method: `lm`, `enet`, `mridge`, or `pls`
#' @param intercept whether to include an intercept term
#' @param lambda ridge shrinkage parameter (for methods `mridge` and `enet`)
#' @param alpha the elastic net mixing parameter if method is `enet`
#' @param ncomp number of PLS components if method is `pls`
#' @param ... extra arguments sent to the underlying fitting function
#' @return a bi-projector of type `regress`
#' @export
#' @importFrom glmnet glmnet
#' @importFrom Matrix t
#' @importFrom pls plsr
#' @importFrom stats coef
#' @examples
#' # Generate synthetic data
#' Y <- matrix(rnorm(100 * 10), 10, 100)
#' X <- matrix(rnorm(10 * 9), 10, 9)
#' # Fit regression models and reconstruct the response matrix
#' r_lm <- regress(X, Y, intercept = FALSE, method = "lm")
#' recon_lm <- reconstruct(r_lm)
#' r_mridge <- regress(X, Y, intercept = TRUE, method = "mridge", lambda = 0.001)
#' recon_mridge <- reconstruct(r_mridge)
#' r_enet <- regress(X, Y, intercept = TRUE, method = "enet", lambda = 0.001, alpha = 0.5)
#' recon_enet <- reconstruct(r_enet)
#' r_pls <- regress(X, Y, intercept = TRUE, method = "pls", ncomp = 5)
#' recon_pls <- reconstruct(r_pls)
regress <- function(X, Y, preproc=NULL, method=c("lm", "enet", "mridge", "pls"), 
                    intercept=FALSE, lambda=.001, alpha=0, ncomp=ceiling(ncol(X)/2), ...) {
  method <- match.arg(method)
  
  # If intercept=TRUE, prepend a column of ones
  if (intercept) {
    scores <- cbind(rep(1, nrow(X)), X)
  } else {
    scores <- X
  }
  
  # Compute betas depending on the method
  betas <- if (method == "lm") {
    # Ordinary least squares
    lfit <- stats::lsfit(X, Y, intercept=intercept)
    as.matrix(t(coef(lfit)))
    
  } else if (method == "mridge") {
    # Multivariate ridge regression via glmnet with alpha=0
    gfit <- glmnet(X, Y, alpha=0, family="mgaussian", lambda=lambda, intercept=intercept, ...)
    # coef(gfit) returns a list of coefficients (one per response)
    # do.call(cbind, ...) combines them into a matrix
    # If no intercept, drop intercept column
    if (!intercept) {
      as.matrix(Matrix::t(do.call(cbind, stats::coef(gfit))))[,-1,drop=FALSE]
    } else {
      as.matrix(Matrix::t(do.call(cbind, stats::coef(gfit))))
    }
    
  } else if (method == "enet") {
    # Elastic net for each response column separately
    out <- do.call(rbind, lapply(1:ncol(Y), function(i) {
      gfit <- glmnet(X, Y[,i], alpha=alpha, family="gaussian", lambda=lambda, intercept=intercept, ...)
      # Extract coefficients for this response
      if (!intercept) {
        coef(gfit)[-1,1]
      } else {
        stats::coef(gfit)[,1]
      }
    }))
    out
    
  } else {
    # PLS regression
    dfl <- list(x=scores, y=Y)
    # plsr expects a formula: y ~ x
    fit <- plsr(y ~ x, data=dfl, ncomp=ncomp, ...)
    as.matrix(t(stats::coef(fit)[,,1]))
  }
  
  # Remove references to X and Y (not strictly necessary, but original code does it)
  rm(X)
  rm(Y)
  
  # Create a bi_projector
  # v = t(pseudoinverse(betas))
  # s = scores
  # sdev = standard deviations of scores columns
  # store coefficients=betas and method
  p <- bi_projector(v = t(corpcor::pseudoinverse(betas)), 
                    s = scores,
                    sdev = apply(scores,2,stats::sd),
                    coefficients = betas,
                    method = method,
                    classes = "regress")
  p
}


#' @export
inverse_projection.regress <- function(x,...) {
  # inverse projection = t(coefficients)
  t(x$coefficients)
}

#' @export
project_vars.regress <- function(x, new_data,...) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data)
  }
  # Check dimension: new_data rows = nrow(scores(x))
  chk::chk_equal(nrow(new_data), nrow(scores(x)))
  
  # project_vars for regress: t(new_data) %*% scores(x)
  # If new_data is NxM and scores is NxC, result is MxC
  t(new_data) %*% (scores(x))
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

