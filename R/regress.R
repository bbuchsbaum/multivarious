
#' Matrix-variate linear regression 
#' 
#' Fit a multivariate regression model for a matrix of basis functions, `X`, and a response matrix `Y`.
#' The goal is to find a projection matrix that can be used for mapping and reconstruction.
#' 
#' 
#' @param X the set of independent (basis) variables
#' @param Y the response matrix
#' @param preproc the pre-processor
#' @param method the regression method: `linear` or `ridge`.
#' @export
#' @importFrom glmnet glmnet
#' @return 
#' 
#' an `projector` instance
regress <- function(X, Y, preproc=NULL, method=c("linear", "ridge"), 
                    intercept=FALSE, lambda=.001) {
  method <- match.arg(method)
  
  #procres <- prep(preproc, X)
  #Xp <- procres$init(X)
  ## we have a basis set, X and data Y
  
  # Y ~ basis*betas
  # Y * b_inv = basis
  
  ## basis * betas
  ## scores(x)[rowind,comp] %*% t(components(x)[,comp,drop=FALSE])[,colind]
  
  
  if (method == "linear") {
    lfit = lsfit(X, Y, intercept=intercept)
    
    if (intercept) {
      scores <- cbind(rep(1, nrow(Y)), Y)
    } else {
      scores <- Y
    }
    
    betas <- as.matrix(t(coef(lfit)))
    
  } else {
    
    gfit <- glmnet(X, Y, alpha=0, family="mgaussian", lambda=lambda, intercept=intercept)
    betas <- as.matrix(t(do.call(cbind, coef(gfit))))
    
    if (intercept) {
      scores <- cbind(rep(1, nrow(X)), X)
    } else {
      scores <- X
    }
  }
  
  
  p <- bi_projector(v=corpcor::pseudoinverse(betas), 
                    s=scores,
                    sdev=apply(s,2,sd),
                    coefficients=betas,
                    classes="regress")
  
}

inverse_projection.regress <- function(x) {
  x$coefficients
}

project_vars.regress <- function(x, new_data) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data)
  }
  chk::chk_equal(nrow(new_data), nrow(scores(x)))
  t(new_data) %*% (scores(x))
}



