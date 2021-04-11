
#' Multi-output linear regression 
#' 
#' Fit a multivariate regression model for a matrix of basis functions, `X`, and a response matrix `Y`.
#' The goal is to find a projection matrix that can be used for mapping and reconstruction.
#' 
#' 
#' @param X the set of independent (basis) variables
#' @param Y the response matrix
#' @param preproc the pre-processor
#' @param method the regression method: `linear` or `enet` `mridge`, or `pls`
#' @param lambda ridge shrinkage parameter
#' @param alpha the elastic net mixing parameter if method is `enet`
#' @param ncomp number of pls components
#' @export
#' @importFrom glmnet glmnet
#' @importFrom pls plsr
#' @return 
#' 
#' an `bi-projector` of type `regress`
#' 
#' @examples
#' 
#' Y <- matrix(rnorm(100*10), 10, 100)
#' X <- matrix(rnorm(10*9), 10, 9)
#' r <- regress(X,Y, intercept=FALSE)
#' recon <- reconstruct(r)
#' r <- regress(X,Y, intercept=TRUE)
#' recon <- reconstruct(r)
#' r <- regress(X,Y, intercept=TRUE, method="ridge")
#' recon <- reconstruct(r)
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
    lfit = lsfit(X, Y, intercept=intercept)
    as.matrix(t(coef(lfit)))
    
  } else if (method == "mridge") {
    
    gfit <- glmnet(X, Y, alpha=0, family="mgaussian", lambda=lambda, intercept=intercept, ...)
  
    if (!intercept) {
      as.matrix(t(do.call(cbind, coef(gfit))))[,-1,drop=FALSE]
    } else {
      as.matrix(t(do.call(cbind, coef(gfit))))
    }
  } else if (method == "enet") {
    out <- do.call(rbind, lapply(1:ncol(Y), function(i) {
      gfit <- glmnet(X, Y[,i], alpha=alpha, family="gaussian", lambda=lambda, intercept=intercept, ...)
      #gfit <- glmnet(X, Y[,i], alpha=alpha, family="gaussian", lambda=lambda, intercept=intercept)
      if (!intercept) coef(gfit)[-1,1] else coef(gfit)[,1]
    }))
  
    
  } else {
    
    dfl <- list(x=scores, y=Y)
    fit <- plsr(y ~ x, data=dfl, ncomp=ncomp,...)
    as.matrix(t(coef(fit)[,,1]))
  }
  
  #print(dim(betas))
  
  p <- bi_projector(v=t(corpcor::pseudoinverse(betas)), 
                    s=scores,
                    sdev=apply(scores,2,sd),
                    coefficients=betas,
                    meethod=method,
                    classes="regress")
  
}

#' @export
inverse_projection.regress <- function(x) {
  t(x$coefficients)
}


#' @export
project_vars.regress <- function(x, new_data) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data)
  }
  chk::chk_equal(nrow(new_data), nrow(scores(x)))
  t(new_data) %*% (scores(x))
}



