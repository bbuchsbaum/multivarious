
#' PCA Bootstrap Resampling
#'
#' Perform bootstrap resampling for Principal Component Analysis (PCA) to estimate component and score variability.
#'
#' @param x A fitted PCA model object.
#' @param nboot The number of bootstrap resamples (default: 100).
#' @param k The number of components to bootstrap (default: all components in the fitted PCA model).
#' @param ... Additional arguments to be passed to the specific model implementation of `bootstrap`.
#' @return A `list` containing bootstrap z-scores for the loadings (`zboot_loadings`) and scores (`zboot_scores`).
#' @export
#' @examples
#' X <- matrix(rnorm(10*100), 10, 100)
#' x <- pca(X, ncomp=9)
#' bootstrap_results <- bootstrap(x)
#'
#' @references Fisher, Aaron, Brian Caffo, Brian Schwartz, and Vadim Zipunnikov. 2016.
#' "Fast, Exact Bootstrap Principal Component Analysis for P > 1 Million." \emph{Journal of the American Statistical Association} 111 (514): 846-60.
#' @family bootstrap
bootstrap.pca <- function(x, nboot=100, k=ncomp(x),...) {
  DUt <- t(scores(x))
  n <- dim(DUt)[2]
  
  gen <- function() {
    sidx <- sample(1:ncol(DUt), replace=TRUE)
    list(DUt=DUt[,sidx,drop=FALSE], idx=sidx)
  }
  
  
  res <- boot_svd(nboot=nboot, k=k, as.matrix(coefficients(x)), gen)
  
  zboot <- do.call(cbind, lapply(1:k, function(ki) {
    res$EVs[[ki]]/res$sdVs[[ki]]
  }))
  
  zscores <- do.call(cbind, lapply(1:k, function(ki) {
    res$EScores[[ki]]/res$sdScores[[ki]]
  }))
  
  ret <- list(zboot_loadings=zboot, zboot_scores=zscores, nboot=nboot, k=k)
  class(ret) <- c("bootstrap_result", "list")
  ret
  
}


svd_dutp <- function(DUtP, k) {
  n <- dim(DUtP)[2]
  svdDUtP <- svd(DUtP)
  sb <- svdDUtP$d
  
  sign_flip <- sign(diag(svdDUtP$u))
  sign_flip[sign_flip == 0] <- 1
  sign_flip <- sign_flip[1:k]
  
  # Add drop=FALSE to ensure these are matrices even if k=1
  Ab <- svdDUtP$u[1:min(dim(DUtP)), 1:k, drop=FALSE]
  Ub <- svdDUtP$v[1:n, 1:k, drop=FALSE]
  
  Ab <- sweep(Ab, 2, sign_flip, "*")
  Ub <- sweep(Ub, 2, sign_flip, "*")
  
  list(d=sb, Ab=Ab, Ub=Ub)
}

#' @keywords internal
#' @noRd
#' @keywords internal
#' @noRd
boot_sum <- function(res, k, v, epsilon = 1e-15) {
  
  AsByK <- lapply(1:k, function(ki) {
    do.call(rbind, lapply(res, function(a) {
      a$svdfit$Ab[, ki, drop=TRUE]
    }))
  })
  
  ScoresByK <- lapply(1:k, function(ki) {
    do.call(rbind, lapply(res, function(a) {
      u <- a$svdfit$Ub[, ki] * a$svdfit$d[ki]
      u2 <- rep(NA, length(u))
      # Restore original sample positions
      u2[a$idx] <- u
      u2
    }))
  })
  
  # Mean and SD of scores
  EScores <- lapply(ScoresByK, function(s) apply(s, 2, mean, na.rm=TRUE))
  sdScores <- lapply(ScoresByK, function(s) apply(s, 2, sd, na.rm=TRUE))
  
  # Replace zeros or near-zeros in sdScores
  sdScores <- lapply(sdScores, function(sd_vec) {
    sd_vec[sd_vec < epsilon] <- epsilon
    sd_vec
  })
  
  # Mean of A (EAs)
  EAs <- lapply(AsByK, colMeans)
  
  # EVs = loadings * EAs
  EVs <- lapply(EAs, function(EA) v %*% matrix(EA, ncol=1))
  
  varAs <- lapply(AsByK, var)
  
  varVs <- lapply(seq_along(AsByK), function(i) {
    # rowSums((v %*% varAs[[i]]) * v)
    tmp <- (v %*% varAs[[i]]) * v
    rowSums(tmp)
  })
  
  sdVs <- lapply(varVs, sqrt)
  # Replace zero or near-zero in sdVs
  sdVs <- lapply(sdVs, function(sd_vec) {
    sd_vec[sd_vec < epsilon] <- epsilon
    sd_vec
  })
  
  list(res=res, EAs=EAs, EVs=EVs, varAs=varAs, sdVs=sdVs, EScores=EScores, sdScores=sdScores)
}

#' @keywords internal
#' @noRd
boot_svd <- function(nboot, k, v, gen_DUtP) {
  
  ## Generate nboot resamples of scores
  res <- lapply(1:nboot, function(i) {
    sam <- gen_DUtP()
    DUtP <- sam$DUt
    #DUtP <- if(x$center) t(scale(t(DUt[,sidx]),center=TRUE,scale=FALSE)) else DUt[,sidx]
    list(svdfit=svd_dutp(DUtP,k), idx=sam$idx)
  })
  
  
  boot_sum(res,k, v)
  
}
