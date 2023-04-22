
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
  
  #procres <- prep(preproc, X)
  #Xp <- procres$init(X)
  ## we have a basis set, X and data Y
  
  # Y ~ basis*betas
  # Y * b_inv = basis
  
  ## basis * betas
  ## scores(x)[rowind,comp] %*% t(components(x)[,comp,drop=FALSE])[,colind]
  
  
  if (intercept) {
    scores <- cbind(rep(1, nrow(X)), X)
  } else {
    scores <- X
  }
  
  betas <- if (method == "lm") {
    lfit = stats::lsfit(X, Y, intercept=intercept)
    as.matrix(t(coef(lfit)))
    
  } else if (method == "mridge") {
    gfit <- glmnet(X, Y, alpha=0, family="mgaussian", lambda=lambda, intercept=intercept, ...)
  
    if (!intercept) {
      as.matrix(Matrix::t(do.call(cbind, stats::coef(gfit))))[,-1,drop=FALSE]
    } else {
      as.matrix(Matrix::t(do.call(cbind, stats::coef(gfit))))
    }
  } else if (method == "enet") {
    out <- do.call(rbind, lapply(1:ncol(Y), function(i) {
      gfit <- glmnet(X, Y[,i], alpha=alpha, family="gaussian", lambda=lambda, intercept=intercept, ...)
      #gfit <- glmnet(X, Y[,i], alpha=alpha, family="gaussian", lambda=lambda, intercept=intercept)
      if (!intercept) coef(gfit)[-1,1] else stats::coef(gfit)[,1]
    }))
  
    
  } else {
    
    dfl <- list(x=scores, y=Y)
    fit <- plsr(y ~ x, data=dfl, ncomp=ncomp,...)
    as.matrix(t(stats::coef(fit)[,,1]))
  }
  
  #print(dim(betas))
  rm(X)
  rm(Y)
  
  p <- bi_projector(v=t(corpcor::pseudoinverse(betas)), 
                    s=scores,
                    sdev=apply(scores,2,stats::sd),
                    coefficients=betas,
                    method=method,
                    classes="regress")
  
}

#' @export
inverse_projection.regress <- function(x,...) {
  t(x$coefficients)
}


#' @export
project_vars.regress <- function(x, new_data,...) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data)
  }
  chk::chk_equal(nrow(new_data), nrow(scores(x)))
  t(new_data) %*% (scores(x))
}



