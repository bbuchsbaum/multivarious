#' Principal Components Analysis (PCA)
#'
#' Compute the directions of maximal variance in a data matrix using the Singular Value Decomposition (SVD).
#'
#' @param X The data matrix.
#' @param ncomp The number of requested components to estimate (default is the minimum dimension of the data matrix).
#' @param preproc The pre-processing function to apply to the data matrix (default is centering).
#' @param method The SVD method to use, passed to \code{svd_wrapper} (default is "fast").
#' @param ... Extra arguments to send to \code{svd_wrapper}.
#' @return A \code{bi_projector} object containing the PCA results.
#' @export
#' @seealso \code{\link{svd_wrapper}} for details on SVD methods.
#' @examples
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' res <- pca(X, ncomp = 4)
#' tres <- truncate(res, 3)
pca <- function(X, ncomp=min(dim(X)), preproc=center(), 
                method = c("fast", "base", "irlba", "propack", "rsvd", "svds"), ...) {
  chk::chkor_vld(chk::vld_matrix(X), chk::vld_s4_class(X, "Matrix"))
  
  method <- match.arg(method)
  svdres <- svd_wrapper(X, ncomp, preproc, method=method, ...)
  
  ## todo add rownames slot to `bi_projector`?
  if (!is.null(row.names(scores))) {
    row.names(scores) <- row.names(X)[seq_along(svdres$d)]
  }
  

  attr(svdres, "class") <- c("pca", attr(svdres, "class"))
  svdres
}


#' @keywords internal
#' @noRd
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


#' @keywords internal
#' @noRd
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



#' Permutation-Based Confidence Intervals for PCA Components
#'
#' Perform a permutation test to assess the significance of variance explained by PCA components.
#'
#' @param x A PCA object from `pca()`.
#' @param X The original data matrix used for PCA.
#' @param nperm Number of permutations.
#' @param k Number of components (beyond the first) to test. Default tests up to `min(Q-1, k)`.
#' @param distr Distribution to fit to the permutation results ("gamma", "norm", or "empirical").
#' @param parallel Logical, whether to use parallel processing for permutations.
#' @param ... Additional arguments passed to `fitdistrplus::fitdist` or parallelization.
#' 
#' @return A list containing:
#' \describe{
#'   \item{observed}{The observed F_a values for tested components.}
#'   \item{perm_values}{A matrix of permuted F-values. Each column corresponds to a component.}
#'   \item{fit}{A list of fit objects or NULL if empirical chosen.}
#'   \item{ci}{Computed confidence intervals for each component.}
#'   \item{p}{p-values for each component.}
#' }
#'
#' @details
#' The function computes a statistic `F_a` for each component `a`, representing the fraction
#' of variance explained relative to the remaining components. It then uses permutations of
#' the preprocessed data to generate a null distribution. The first component uses the full data;
#' subsequent components are tested by partialing out previously identified components and
#' permuting the residuals.
#'
#' By default, a gamma distribution is fit to the permuted values to derive CIs and p-values.
#' If `distr="empirical"`, it uses empirical quantiles instead.
#'
#' @export
perm_ci.pca <- function(x, X, nperm=100, k=4, distr="gamma", parallel=FALSE, ...) {
  Q <- ncomp(x)
  k <- min(Q-1, k)
  
  # Observed evals & Fa
  evals <- x$sdev^2
  Fa <- sapply(1:k, function(i) evals[i]/sum(evals[i:Q]))
  
  # Preprocess data
  Xp <- apply_transform(x$preproc, X) # previously Xp <- x$preproc$transform(X)
  
  # Compute permutation null dist for first component
  perm_fun <- function(i) {
    Xperm <- apply(Xp, 2, sample) # Permute each column
    fit <- pca(Xperm, ncomp=Q, preproc=pass())
    evals_p <- fit$sdev^2
    evals_p[1]/sum(evals_p)
  }
  
  if (parallel) {
    perm_values_first <- parallel::mclapply(1:nperm, perm_fun, ...)
  } else {
    perm_values_first <- lapply(1:nperm, perm_fun)
  }
  F1_perm <- unlist(perm_values_first)
  
  # For subsequent components, partial out previously identified components
  if (Q > 1 && k > 1) {
    # Create a function to compute Ea_perm_proj for each component 'a'
    partial_perm_fun <- function(a) {
      # Project out first (a-1) components
      uu <- Reduce("+", lapply(1:(a-1), function(i) {
        x$u[,i,drop=FALSE] %*% t(x$u[,i,drop=FALSE]) 
      }))
      I <- diag(nrow(X))
      function(i) {
        # Recompute recon with first (a-1) comps
        cnums <- 1:(a-1)
        recon <- scores(x)[,cnums, drop=FALSE] %*% inverse_projection(x)[cnums,,drop=FALSE]
        Ea <- Xp - recon
        Ea_perm <- apply(Ea, 2, sample)
        
        Ea_perm_proj <- (I - uu) %*% Ea_perm
        fit <- pca(Ea_perm_proj, ncomp=Q, preproc=pass())
        evals_p <- fit$sdev^2
        evals_p[1]/sum(evals_p[1:(Q-(a-1))])
      }
    }
    
    # Compute permutations for each subsequent component
    Fq_list <- list(F1_perm)
    for (a in 2:k) {
      fun_a <- partial_perm_fun(a)
      if (parallel) {
        vals_a <- parallel::mclapply(1:nperm, fun_a, ...)
      } else {
        vals_a <- lapply(1:nperm, fun_a)
      }
      Fq_list[[a]] <- unlist(vals_a)
    }
    Fq <- do.call(cbind, Fq_list)
  } else {
    Fq <- as.matrix(F1_perm)
  }
  
  # Now fit distributions or use empirical
  cfuns <- lapply(seq_len(ncol(Fq)), function(i) {
    vals <- Fq[,i]
    observed_val <- Fa[i]
    
    if (distr == "gamma" || distr == "norm") {
      # Try fitting
      fit_res <- tryCatch({
        fitdistrplus::fitdist(vals, distr=distr)
      }, error=function(e) NULL)
      
      if (is.null(fit_res)) {
        warning("Fitting distribution failed, using empirical quantiles.")
        distr <- "empirical"
      }
    }
    
    if (distr == "empirical") {
      # Use empirical quantiles
      lower_ci <- quantile(vals, 0.025)
      upper_ci <- quantile(vals, 0.975)
      pval <- mean(vals > observed_val)
      list(cdf=NULL, lower_ci=lower_ci, upper_ci=upper_ci, p=pval, fit=NULL)
    } else {
      # distr is "gamma" or "norm"
      qfun <- get(paste0("q", distr))
      pfun <- get(paste0("p", distr))
      
      est <- fit_res$estimate
      lower_ci <- do.call(qfun, c(list(0.025), as.list(est)))
      upper_ci <- do.call(qfun, c(list(0.975), as.list(est)))
      
      # p-value = 1 - CDF(observed_val)
      # cdf = P(X <= x)
      # For gamma/norm: 1 - pfun(x,...)
      pv <- 1 - do.call(pfun, c(list(observed_val), as.list(est)))
      
      list(cdf=function(x) 1-do.call(pfun, c(list(x), as.list(est))),
           lower_ci=lower_ci, upper_ci=upper_ci, p=pv, fit=fit_res)
    }
  })
  
  # Return a structured result
  list(
    observed=Fa,
    perm_values=Fq,
    results=cfuns,
    distribution=distr
  )
}



#' Rotate PCA Loadings
#'
#' Apply a specified rotation to the component loadings of a PCA model. This function leverages
#' the GPArotation package to apply orthogonal or oblique rotations. 
#'
#' @param x A PCA model object, typically created using the `pca()` function.
#' @param ncomp The number of components to rotate. Must be <= ncomp(x).
#' @param type The type of rotation to apply. Supported rotation types:
#'   \describe{
#'     \item{"varimax"}{Orthogonal Varimax rotation}
#'     \item{"quartimax"}{Orthogonal Quartimax rotation}
#'     \item{"promax"}{Oblique Promax rotation}
#'   }
#' @param loadings_type For oblique rotations, which loadings to use:
#'   \describe{
#'     \item{"pattern"}{Use pattern loadings as `v`}
#'     \item{"structure"}{Use structure loadings (`pattern_loadings %*% Phi`) as `v`}
#'   }
#'   Ignored for orthogonal rotations.
#' @param score_method How to recompute scores after rotation:
#'   \describe{
#'     \item{"auto"}{For orthogonal rotations, use `scores_new = scores_original %*% t(R)`. For oblique rotations, recompute from pseudoinverse.}
#'     \item{"recompute"}{Always recompute scores from `X_proc` and the pseudoinverse of rotated loadings.}
#'     \item{"original"}{For orth rotations same as `auto`, but may not work for oblique rotations.}
#'   }
#' @param ... Additional arguments passed to GPArotation functions.
#'
#' @return A modified PCA object with class `rotated_pca` and additional fields:
#'   \item{v}{Rotated loadings}
#'   \item{s}{Rotated scores}
#'   \item{sdev}{Updated standard deviations of rotated components}
#'   \item{explained_variance}{Proportion of explained variance for each rotated component}
#'   \item{rotation}{A list with rotation details: type, R (orth) or Phi (oblique), and loadings_type}
#'
#' @export
#' @examples
#' # Perform PCA on iris dataset
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' res <- pca(X, ncomp=4)
#'
#' # Apply varimax rotation to the first 3 components
#' rotated_res <- rotate(res, ncomp=3, type="varimax")
rotate.pca <- function(x, ncomp, type=c("varimax", "quartimax", "promax"),
                       loadings_type=c("pattern", "structure"),
                       score_method=c("auto", "recompute", "original"),
                       ...) {
  type <- match.arg(type)
  loadings_type <- match.arg(loadings_type)
  score_method <- match.arg(score_method)
  
  if (!requireNamespace("GPArotation", quietly = TRUE)) {
    stop("GPArotation package is required for rotations. Please install it.")
  }
  
  if (ncomp > ncomp(x)) {
    stop("ncomp cannot exceed the number of available components in 'x'.")
  }
  
  # Extract loadings and scores for the specified components
  loadings_to_rotate <- x$v[, 1:ncomp, drop=FALSE]
  scores_original <- x$s[, 1:ncomp, drop=FALSE]
  
  # We'll need X_proc to recompute scores in 'recompute' mode or for oblique rotations
  # If we don't have X_proc directly, we can reconstruct it:
  # X_proc ≈ scores_original %*% t(loadings_to_rotate)
  # This relies on the PCA model: X_proc = s * v', assuming preproc applied.
  
  # Just in case we need full pre-processed data:
  # For 'recompute' or oblique rotations, we must have a stable way to get X_proc.
  # We know: X_proc = scores_original %*% t(loadings_to_rotate)
  
  # Create loadings object for GPArotation
  L <- loadings_to_rotate
  class(L) <- "loadings"
  
  # Perform rotation
  if (type %in% c("varimax", "quartimax")) {
    # Orthogonal rotation
    rot_res <- GPArotation::GPForth(L, method=type, ...)
    rotated_loadings <- rot_res$loadings
    R <- rot_res$Th  # rotation matrix
    
    # Compute scores_new:
    # Depending on score_method:
    if (score_method == "auto" || score_method == "original") {
      # For orth rotations: scores_new = scores_original %*% t(R)
      scores_new <- scores_original %*% t(R)
    } else if (score_method == "recompute") {
      # recompute from X_proc:
      X_proc <- scores_original %*% t(loadings_to_rotate)
      inv_rotated <- corpcor::pseudoinverse(rotated_loadings)
      scores_new <- X_proc %*% inv_rotated
    }
    
    # Update sdev and explained variance
    variances <- apply(scores_new, 2, var)
    sdev_new <- sqrt(variances)
    explained_variance <- variances / sum(variances)
    
    # Update object
    x$v[, 1:ncomp] <- as.matrix(rotated_loadings)
    x$s[, 1:ncomp] <- scores_new
    x$sdev[1:ncomp] <- sdev_new
    x$explained_variance <- explained_variance
    
    x$rotation <- list(type=type, loadings_type="N/A (orthogonal)", R=R, Phi=NULL)
    
  } else {
    # Oblique rotation
    rot_res <- GPArotation::GPFoblq(L, method=type, ...)
    pattern_loadings <- rot_res$loadings
    Phi <- rot_res$Phi
    
    # Choose loadings based on loadings_type
    if (loadings_type == "pattern") {
      chosen_loadings <- pattern_loadings
    } else {
      # structure loadings = pattern_loadings %*% Phi
      chosen_loadings <- pattern_loadings %*% Phi
    }
    
    # Compute scores_new:
    # For oblique rotations, if score_method == "original" doesn't make sense because original was orth-based.
    # We'll handle "original" by warning or just do what "auto" does for oblique (which is recompute).
    
    if (score_method == "original") {
      warning("For oblique rotations, 'original' score_method is not valid. Using 'auto'.")
      score_method <- "auto"
    }
    
    if (score_method == "auto") {
      # auto = oblique => recompute from pseudoinverse
      X_proc <- scores_original %*% t(loadings_to_rotate)
      inv_chosen <- corpcor::pseudoinverse(chosen_loadings)
      scores_new <- X_proc %*% inv_chosen
    } else if (score_method == "recompute") {
      # Same as above
      X_proc <- scores_original %*% t(loadings_to_rotate)
      inv_chosen <- corpcor::pseudoinverse(chosen_loadings)
      scores_new <- X_proc %*% inv_chosen
    }
    
    # Update sdev and explained variance
    variances <- apply(scores_new, 2, var)
    sdev_new <- sqrt(variances)
    explained_variance <- variances / sum(variances)
    
    # Update object
    x$v[, 1:ncomp] <- chosen_loadings
    x$s[, 1:ncomp] <- scores_new
    x$sdev[1:ncomp] <- sdev_new
    x$explained_variance <- explained_variance
    
    x$rotation <- list(type=type, loadings_type=loadings_type, R=NULL, Phi=Phi)
  }
  
  # Add rotated_pca class
  if (!("rotated_pca" %in% class(x))) {
    class(x) <- c("rotated_pca", class(x))
  }
  
  x
}


# 
# rotate.pca <- function(x, X, ncomp, type=c("varimax", "promax")) {
#   type <- match.arg(type)
#   
#   L     <- x$v[,1:ncomp] %*% diag(x$sdev, ncomp, ncomp)
#   
#   ### these are still scaled
#   RL <- varimax(L)$loadings
#   ###
#   invRL     <- t(corpcor::pseudoinverse(RL))
#   
#   scores    <- reprocess(x, X) %*% invRL
#   
#   bi_projector(invRL, scores, rotated_loadings=RL, 
#                sdev=apply(scores,2, function(x) sum(x^2)),
#                classes=c("rotated", class(x)))
#      
# }

#' Rotate PCA loadings
#' 
#' Apply a specified rotation to the component loadings of a PCA model.
#' 
#' @param x A PCA model object, typically created using the `pca()` function.
#' @param ncomp The number of components to rotate.
#' @param type The type of rotation to apply. Supported rotation types are "varimax", "quartimax", and "promax".
#' @return A modified PCA object with updated components and scores after applying the specified rotation.
#' @export
#' @examples
#' # Perform PCA on the iris dataset
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' res <- pca(X, ncomp=4)
#'
#' # Apply varimax rotation to the first 3 components
#' rotated_res <- rotate(res, ncomp=3, type="varimax")
#' @export
# rotate.pca <- function(x, ncomp, type) {
#   # Check if rotation type is supported
#   supported_rotations <- c("varimax", "quartimax", "promax")
#   if (!(type %in% supported_rotations)) {
#     stop(sprintf("Unsupported rotation type. Choose from: %s", paste(supported_rotations, collapse = ", ")))
#   }
#   
#   # Load the GPArotation package if not already loaded
#   if (!requireNamespace("GPArotation", quietly = TRUE)) {
#     stop("GPArotation package is required to perform rotations. Please install it using 'install.packages(\"GPArotation\")'")
#   }
#   
#   # Extract components and scores from the pca object
#   components <- x$v
#   scores <- x$s
#   
#   # Perform the rotation
#   rotation <- GPArotation::rotate(components[, 1:ncomp], type)
#   
#   # Update components and scores with rotated values
#   x$v[, 1:ncomp] <- rotation$loadings
#   x$s[, 1:ncomp] <- scores %*% rotation$rotation
#   
#   return(x)
# }

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


