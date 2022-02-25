#' principal components analysis
#' 
#' Compute the directions of maximal variance in a matrix via the singula value decomposition
#' 
#' @param X the data matrix
#' @param ncomp the number of requested components to estimate
#' @param preproc the pre_processor
#' @param method th svd method (passed to `svd_wrapper`)
#' @param ... extra arguments to send to `svd_wrapper`
#' @export
#' 
#' @examples 
#' 
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' res <- pca(X, ncomp=4)
#' tres <- truncate(res, 3)
#' 
#' 
pca <- function(X, ncomp=min(dim(X)), preproc=center(), 
                method = c("fast", "base", "irlba", "propack", "rsvd", "svds"), ...) {
  chk::chkor(chk::chk_matrix(X), chk::chk_s4_class("Matrix"))
  
  method <- match.arg(method)
  svdres <- svd_wrapper(X, ncomp, preproc, method=method, ...)
  
  ## todo add rownames slot to `bi_projector`?
  if (!is.null(row.names(scores))) {
    row.names(scores) <- row.names(X)[seq_along(svdres$d)]
  }
  

  attr(svdres, "class") <- c("pca", attr(svdres, "class"))
  svdres
}

orth_distances.pca <- function(x, ncomp, xorig) {
  resid <- residuals(x, ncomp, xorig)
  scores <- scores(x)
  loadings <- coef(x)
  
  scoresn <- x$u
  
  Q <- matrix(0, nrow = nrow(scores), ncol = ncomp)
  
  for (i in seq_len(ncomp)) {
    res <- resid
    if (i < ncomp) {
      res <- res +
        tcrossprod(
          scores[, (i + 1):ncomp, drop = F],
          loadings[, (i + 1):ncomp, drop = F]
        )
    }
    
    Q[, i] <- rowSums(res^2)
    #T2[, i] <- rowSums(scoresn[, seq_len(i), drop = F]^2)
  }
  
  Q
}

score_distances.pca <- function(x, ncomp, xorig) {
  scores <- scores(x)
  loadings <- coef(x)
  
  scoresn <- x$u
  
  T2 <- matrix(0, nrow = nrow(scores), ncol = ncomp)
  for (i in seq_len(ncomp)) {
    T2[, i] <- rowSums(scoresn[, seq_len(i), drop = F]^2)
  }
  
  T2
  
}

#' @export
#' @importFrom chk chk_range
truncate.pca <- function(x, ncomp) {
  chk::chk_range(ncomp, c(1, ncomp(x)))
  x$v <- x$v[,1:ncomp, drop=FALSE]
  x$sdev <- x$sdev[1:ncomp]
  x$s <- x$s[,1:ncomp,drop=FALSE]
  x$u <- x$u[, 1:ncomp, drop=FALSE]
  x
}

#' @export
perm_ci.pca <- function(x, X, nperm=100, k=4,...) {
  Q <- ncomp(x)
  k <- min(Q-1,k)
  
  evals <- x$sdev^2
  Fa <- sapply(1:k, function(i) evals[i]/sum(evals[i:Q]))
  
  Xp <- x$preproc$transform(X)
  
  F1_perm <- sapply(1:nperm, function(i) {
    Xperm <- apply(Xp, 2, function(x) sample(x))
    #pp <- fresh(x$preproc$preproc)
    fit <- pca(Xperm, ncomp=Q, preproc=pass())
    evals <- fit$sdev^2
    F1_perm <- evals[1]/sum(evals)
  })
  
  ip <- inverse_projection(x)
  
  if (Q > 1) {
    Fq <- parallel::mclapply(2:k, function(a) {
      ret <- sapply(1:nperm, function(j) {
        cnums <- 1:(a-1)
        recon <- scores(x)[,cnums, drop=FALSE] %*% ip[cnums,,drop=FALSE]
        Ea <- Xp-recon
        Ea_perm <- apply(Ea, 2, function(x) sample(x))
        
        I <- diag(nrow(X))
        
        uu <- Reduce("+", lapply(1:(a-1), function(i) {
          x$u[,i,drop=FALSE] %*% t(x$u[,i,drop=FALSE]) 
        }))
        
        Ea_perm_proj <- (I - uu) %*% Ea_perm
        #pp <- fresh(x$preproc$preproc)
        fit <- pca(Ea_perm_proj, ncomp=Q, preproc=pass())
        evals <- fit$sdev^2
        Fq_perm <- evals[1]/sum(evals[1:(Q-(a-1))])
      })
    })
    
    Fq <- do.call(cbind, Fq)
    Fq <- cbind(F1_perm, Fq)
  } else {
    Fq <- as.matrix(F1_perm)
  }
  
  cfuns <- lapply(1:ncol(Fq), function(i) {
    vals <- Fq[,i]
    fit <- fitdistrplus::fitdist(vals, distr="gamma")
    f <- function(x) {
      1-stats::pgamma(x,fit$estimate[[1]],fit$estimate[[2]])
    }
    ci <- c(stats::qgamma(.025, fit$estimate[[1]],fit$estimate[[2]]),
            stats::qgamma(.975, fit$estimate[[1]],fit$estimate[[2]]))
    
    pval <- f(Fa[1])
    
    list(cdf=f, lower_ci=ci[1], upper_ci=ci[2], p=pval)
  })
  
} 

perm_cdf <- function(vals, distr="norm") {
  fit <- fitdistrplus::fitdist(vals, distr=distr)
  
  qfun <- get(paste0("q", distr))
  pfun <- get(paste0("p", distr))
  
  f <- function(x) {
    1- do.call(pfun, as.list(c(x, fit$estimate)))
  }
  
  fun <- get(paste0("q", distr))
  ci <- c(do.call(qfun, as.list(c(.025, fit$estimate))),
          do.call(qfun, as.list(c(.975, fit$estimate))))
  
  list(cdf=f, lower_ci=ci[1], upper_ci=ci[2])
  
}

# jackstraw.pca <- function(x, X, prop=.1, n=100) {
#   vars <- round(max(1, prop*ncol(X)))
#   Xp <- x$preproc$transform(X)
#   res <- do.call(cbind, lapply(1:n, function(i) {
#     vi <- sample(1:ncol(Xp), vars)
#     ##vi <- 1
#     Xperm <- Xp
#     Xperm[,vi] <- do.call(cbind, lapply(vi, function(i) sample(Xperm[,i])))
#     #pp <- fresh(x$preproc$preproc)
#     fit <- pca(Xperm, ncomp=Q, preproc=pass())
#     fit$v[,vi,drop=FALSE]
#     #cor(Xperm[,vi], fit$s)
#   }))
#   
# }


