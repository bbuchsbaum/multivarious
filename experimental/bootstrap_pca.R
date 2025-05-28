#' PCA Bootstrap Resampling
#'
#' Perform bootstrap resampling for Principal Component Analysis (PCA) to estimate component and score variability.
#'
#' @param x A fitted PCA model object.
#' @param nboot The number of bootstrap resamples (default: 100).
#' @param k The number of components to bootstrap (default: all components in the fitted PCA model).
#' @param ... Additional arguments to be passed to the specific model implementation of `bootstrap`.
#' @param parallel Logical flag indicating whether to use parallel processing.  
#' @param cores The number of cores to use for parallel processing.
#' @param seed The seed for reproducibility.
#' @param epsilon A small value to prevent division by zero in variance calculations.
#' 
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
bootstrap.pca <- function(x, nboot = 100, k = ncomp(x),
                          parallel = FALSE, cores = 2,
                          seed = NULL, epsilon = 1e-15, ...) {

  if (k > ncomp(x)) stop("k exceeds available components.")
  if (!is.null(seed)) withr::local_seed(seed)

  ## Fisher 2016 — Y = D^{1/2} U^T
  U      <- scores(x)                 # n × k_max
  d_half <- sqrt(sdev(x))[seq_len(k)]
  DhalfUt <- sweep(t(U[, seq_len(k), drop = FALSE]),
                   1, d_half, "*")    # k × n

  n <- ncol(DhalfUt)
  samp_fun <- function() sample.int(n, replace = TRUE)

  svd_one <- function(idx) {
    Ystar <- DhalfUt[, idx, drop = FALSE]
    sv    <- svd(Ystar, nu = k, nv = k)   # fast: only k vecs
    ## deterministic sign
    sgn   <- sign(sv$u[1, ])
    list(A = sweep(sv$u[, 1:k, drop = FALSE], 2, sgn, "*"),
         scores = sweep(sv$v[, 1:k, drop = FALSE], 2,
                        sv$d[1:k] * sgn, "*"),
         idx = idx)
  }

  FUN <- if (parallel) {
           requireNamespace("parallel", quietly = TRUE)
           function(i) svd_one(samp_fun()) 
          } else {
           svd_one
          }

  res <- (if (parallel) {
            parallel::mclapply(seq_len(nboot), FUN, mc.cores = cores)
          } else {
            lapply(seq_len(nboot), FUN)
          })

  ## Stack results
  A_arr  <- simplify2array(lapply(res, `[[`, "A"))      # k × k × nboot
  S_list <- lapply(res, function(r) {
    # Need to re-index scores to original sample order
    S_boot <- matrix(NA_real_, nrow = n, ncol = k)
    S_boot[r$idx, ] <- r$scores
    S_boot
  })
  S_arr  <- simplify2array(S_list) # n × k × nboot


  EA  <- apply(A_arr, 1:2, mean)
  VarA <- apply(A_arr, 1:2, var)
  sdV <- sqrt(rowSums((coefficients(x) %*% VarA) * coefficients(x)))
  sdV[sdV < epsilon] <- epsilon

  z_load <- (coefficients(x) %*% EA) / sdV

  ES <- apply(S_arr, 1:2, mean, na.rm=TRUE)
  sdS <- apply(S_arr, 1:2, sd, na.rm=TRUE)
  sdS[sdS < epsilon] <- epsilon
  z_score <- ES / sdS

  ret <- list(zboot_loadings = z_load,
              zboot_scores   = z_score,
              nboot = nboot,
              k = k,
              call = match.call())
              
  class(ret) <- c("bootstrap_result", "list")
  ret
}


#' @keywords internal
#' @noRd
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

=
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
