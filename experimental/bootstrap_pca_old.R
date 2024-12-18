#' Bootstrap PCA Analysis
#'
#' This function performs bootstrap resampling on a fitted PCA model to estimate the variability of loadings and scores.
#' It resamples observations (with replacement) from the original dataset's scores, then re-estimates the PCA for each bootstrap
#' sample. The resulting distributions of loadings and scores are used to compute z-scores, indicating how stable each component
#' and score is under resampling.
#'
#' @param x A fitted PCA model object. Must provide `scores(x)` and `coefficients(x)` methods.
#' @param nboot Integer. Number of bootstrap resamples (default: 100).
#' @param k Integer. Number of components to analyze (default: all components in `x`).
#' @param method Character. SVD method passed to `svd_wrapper`. Options: "base", "fast", "irlba", "propack", "rsvd", "svds".
#'        Default: "base".
#' @param sign_method Character. Method for sign alignment of component loadings/scores across bootstraps.
#'        Options:
#'        - "max_abs": Flip signs so the element with the largest absolute value in each vector is positive.
#'        - "first_element": Flip signs so the first element of each vector is positive.
#'        - "none": Do not adjust signs.
#'        Default: "max_abs".
#' @param seed Optional integer to set the random seed for reproducibility.
#' @param ... Additional arguments passed to `svd_wrapper`.
#'
#' @return A list of class `bootstrap_result`:
#' \describe{
#'   \item{zboot_loadings}{A matrix (D x k) of z-scores for loadings. D = number of features.}
#'   \item{zboot_scores}{A matrix (N x k) of z-scores for scores. N = number of samples.}
#'   \item{nboot}{Number of bootstrap samples performed.}
#'   \item{k}{Number of components analyzed.}
#' }
#'
#' @details
#' The procedure is:
#' 1. Extract scores (N x ncomp) and loadings (D x ncomp) from the fitted PCA.
#' 2. For each bootstrap iteration:
#'    - Resample N observations (with replacement).
#'    - Perform SVD (via `svd_wrapper`) on the resampled score matrix to get a new PCA estimate.
#'    - Align signs of components if `sign_method` is used.
#' 3. Aggregate all bootstrap results to compute mean and SD of loadings and scores.
#' 4. Compute z-scores = mean/SD for both loadings and scores.
#'
#' By default, sign alignment ("max_abs") ensures consistency in the direction of components.
#'
#' @examples
#' # Assuming 'x' is a PCA model fit from a function like pca(), with ncomp=5:
#' # bootstrap_results <- bootstrap(x, nboot=200, k=3, method="irlba", sign_method="max_abs", seed=123)
#'
#' @export
bootstrap.pca <- function(x, nboot=100, k=ncomp(x), 
                          method="base", 
                          sign_method=c("max_abs","first_element","none"),
                          seed=NULL,
                          ...) {
  sign_method <- match.arg(sign_method)
  
  if (!is.null(seed)) set.seed(seed)
  
  # Extract scores and loadings from the fitted PCA model
  scores_mat <- scores(x)         # N x ncomp
  loadings   <- coefficients(x)   # D x ncomp
  
  if (nboot <= 0) stop("nboot must be a positive integer.")
  
  k <- min(k, ncol(scores_mat))
  
  # Transpose scores for convenience: DUt = t(scores) = (ncomp x N)
  DUt <- t(scores_mat)
  N <- ncol(DUt)
  D <- nrow(loadings)
  
  # Check dimensions
  if (k > ncol(scores_mat)) {
    warning("Requested k is greater than the number of available components. Using all components.")
    k <- ncol(scores_mat)
  }
  
  # Generator function for bootstrap samples
  gen_sample <- function() {
    sidx <- sample(N, replace=TRUE)
    list(DUt_boot = DUt[, sidx, drop=FALSE], idx = sidx)
  }
  
  # Preallocate list to store bootstrap results
  # Each element will hold: svdfit = list(u,d,v) from partial SVD, idx = resample indices
  res_list <- vector("list", nboot)
  
  # Compute bootstrap samples
  for (i in seq_len(nboot)) {
    sam <- gen_sample()
    svdfit <- compute_svd_on_boot(sam$DUt_boot, k, method=method, sign_method=sign_method, ...)
    res_list[[i]] <- list(svdfit=svdfit, idx=sam$idx)
  }
  
  # Summarize bootstrap results to compute z-scores
  summary_res <- summarize_bootstrap_results(res_list, k, loadings)
  
  zboot_loadings <- do.call(cbind, lapply(seq_len(k), function(ci) {
    summary_res$EVs[[ci]] / summary_res$sdVs[[ci]]
  }))
  
  zboot_scores <- do.call(cbind, lapply(seq_len(k), function(ci) {
    summary_res$EScores[[ci]] / summary_res$sdScores[[ci]]
  }))
  
  ret <- list(zboot_loadings=zboot_loadings, zboot_scores=zboot_scores, nboot=nboot, k=k)
  class(ret) <- c("bootstrap_result", "list")
  ret
}


#' Compute SVD for a Bootstrap Sample
#'
#' This internal function takes a bootstrap sample (component matrix DUt_boot),
#' applies `svd_wrapper` to get a partial SVD (u, d, v), and then
#' extracts the top k components. It also applies sign flipping if requested.
#'
#' @param DUt_boot (k x N) matrix: transposed scores after resampling.
#' @param k Number of components to extract.
#' @param method SVD method for `svd_wrapper`.
#' @param sign_method How to align signs.
#' @param ... Passed to `svd_wrapper`.
#'
#' @return A list with elements:
#' \describe{
#'   \item{d}{Vector of singular values of length k}
#'   \item{U}{(k x k) left singular vectors}
#'   \item{V}{(N x k) right singular vectors}
#' }
#' @keywords internal
compute_svd_on_boot <- function(DUt_boot, k, method, sign_method, ...) {
  # Use svd_wrapper with preproc=pass() since DUt_boot is already "processed"
  # We'll just run SVD directly.
  
  # svd_wrapper expects data as samples x features. Here DUt_boot is (k x N), akin to "scores" transposed.
  # But we need an SVD that treats DUt_boot in a consistent manner:
  # The original code did svd on DUtP = (k x N), producing u (k x k), v (N x k).
  # We'll do the same: just call svd_wrapper with appropriate args.
  
  # Since svd_wrapper expects X as samples x features, and we have (k x N), 
  # let's consider rows as samples (k) and columns as features (N).
  # This is consistent since we just need top k comps:
  
  fit <- svd_wrapper(DUt_boot, ncomp=k, preproc=pass(), method=method, ...)
  
  # fit$u is (k x k), fit$v is (N x k)
  # We have: DUt_boot = U d V^T in traditional sense (like original code)
  d <- fit$sdev
  U <- fit$u   # (N rows from original vantage, but here it's actually k x k)
  # Wait, we must confirm the orientation from svd_wrapper:
  # svd_wrapper returns a bi_projector with v as loadings (features x components) and s = scores.
  # For an SVD: X = U d V^T.
  # By default, svd_wrapper sets:
  # v = right singular vectors (features x ncomp)
  # u = left singular vectors (samples x ncomp)
  # we passed DUt_boot as X, so "samples" = k, "features" = N.
  
  # Therefore:
  # u in fit is (k x k)
  # v in fit is (N x k)
  
  U_mat <- fit$u   # (k x k)
  V_mat <- fit$v   # (N x k)
  
  # Sign flipping
  U_mat <- apply_sign_method(U_mat, method=sign_method)
  V_mat <- apply_sign_method(V_mat, method=sign_method)
  
  list(d=d, U=U_mat, V=V_mat)
}


#' Apply Sign Method to SVD Components
#'
#' Ensures consistent sign orientation of singular vectors.
#'
#' @param mat A matrix of singular vectors (either U or V).
#' @param method One of "max_abs", "first_element", "none".
#'
#' @return The matrix with columns potentially flipped in sign for consistency.
#' @keywords internal
apply_sign_method <- function(mat, method="max_abs") {
  if (method == "none") return(mat)
  
  for (j in seq_len(ncol(mat))) {
    col_j <- mat[, j]
    if (method == "max_abs") {
      # Flip sign so largest abs element is positive
      idx_max <- which.max(abs(col_j))
      if (col_j[idx_max] < 0) mat[, j] <- -col_j
    } else if (method == "first_element") {
      # Flip sign if first element is negative
      if (col_j[1] < 0) mat[, j] <- -col_j
    }
  }
  
  mat
}


#' Summarize Bootstrap Results
#'
#' Aggregates results from all bootstrap samples to compute means and standard deviations of loadings and scores,
#' and then prepares for z-score computation.
#'
#' @param res_list A list of length nboot, each element with `svdfit` (d,U,V) and `idx` (resampled indices).
#' @param k Number of components.
#' @param loadings Original loadings matrix (D x ncomp).
#'
#' @return A list with elements EAs, EVs, varAs, sdVs, EScores, sdScores.
#' @keywords internal
summarize_bootstrap_results <- function(res_list, k, loadings) {
  nboot <- length(res_list)
  
  # Extract A (loadings in U space) and Scores per component
  # A ~ left singular vectors = U in our notation
  # Scores ~ right singular vectors * d: V * diag(d)
  
  # Collect A's for each component (nboot rows, k columns)
  A_by_comp <- vector("list", k)
  Scores_by_comp <- vector("list", k)
  
  for (ci in seq_len(k)) {
    A_mat <- matrix(NA, nrow=nboot, ncol=k)    # Actually we only need the ci-th column of U (just one column)
    # Wait, original code took the entire U but we only need the ci-th column of U from each bootstrap
    # Actually original code: A ~ Ab from code. It used Ab[,ki].
    # Our U = (k x k). Ab in original code = U from our code?
    # Actually, original code had A as from svd on DUt. 
    # If original code took a$svdfit$Ab[,ki], that was the i-th column of U. 
    # We can store each column individually.
    # But we need them all for var computations. We'll store entire columns anyway.
    
    # Actually, let's align with original: A_by_comp[[ci]] holds all bootstrap A-values for component ci.
    # U is (k x k). The ci-th component is just the ci-th column of U.
    # So A_mat should be nboot x k only if we needed all components at once, but we only store the ci-th column:
    A_mat <- numeric(nboot * k)
    dim(A_mat) <- c(nboot, k)
    
    # Scores: from each bootstrap: Scores = (V * diag(d)) for component ci is V[,ci] * d[ci]
    # We must restore original order of samples. idx in res_list[[i]] gives the bootstrap sample indices.
    # We'll store them in a matrix nboot x N (N = number of samples)
    # But that can be large. Original code does that. We'll do the same.
    # We'll do memory-lazy approach: 
    # Actually, original code computed mean and sd for each column. Let's do that too.
    # We'll store them all then mean/sd.
    # Potentially large but we trust user environment.
    ScoresMat <- matrix(NA, nrow=nboot, ncol=ncol(loadings)) # Wait, must match samples = from `idx`.
    # Actually, scores dimension: N samples from original, we must restore them: 
    # N = number of samples, we can get from length(res_list[[1]]$idx)
    N <- length(res_list[[1]]$idx)
    ScoresMat <- matrix(NA, nrow=nboot, ncol=N)
    
    for (b in seq_len(nboot)) {
      svdfit <- res_list[[b]]$svdfit
      idx <- res_list[[b]]$idx
      
      # U is k x k, we want the ci-th column of U:
      A_mat[b, ] <- svdfit$U[, ci] # This is a k-length vector, representing left singular vector component ci?
      # Wait, U[,ci] is length k. The original code only took a$svdfit$Ab[,ki] which was a single vector.
      # In the original code, Ab was analogous to U. They took Ab[,ki], which is a vector length k, representing loadings in reduced space.
      # We'll keep as is. It's consistent with original logic.
      
      # Scores: V (N x k), we want V[, ci]*d[ci]
      scores_b <- svdfit$V[, ci] * svdfit$d[ci]
      # Place them in correct sample order:
      # idx are the sampled indices, we must place scores back in original order?
      # The original code does:
      # u2[a$idx] <- u, meaning we restore original indexing.
      # Let's do the same:
      full_scores <- rep(NA, N)
      full_scores[idx] <- scores_b
      ScoresMat[b, ] <- full_scores
    }
    # Store results
    A_by_comp[[ci]] <- A_mat
    Scores_by_comp[[ci]] <- ScoresMat
  }
  
  # Compute means and sds
  EScores <- lapply(Scores_by_comp, function(SM) apply(SM, 2, mean, na.rm=TRUE))
  sdScores <- lapply(Scores_by_comp, function(SM) apply(SM, 2, sd, na.rm=TRUE))
  
  # Mean of A (EAs)
  EAs <- lapply(A_by_comp, colMeans)
  
  # Compute EVs = loadings * EAs
  # varAs and varVs: variance of A and resulting variance in V space
  varAs <- lapply(A_by_comp, var)
  
  varVs <- lapply(seq_len(k), function(ci) {
    # varVs = rowSums((loadings %*% varAs[[ci]]) * loadings)
    # varAs[[ci]] is k x k var matrix
    rowSums((loadings %*% varAs[[ci]]) * loadings)
  })
  
  sdVs <- lapply(varVs, sqrt)
  
  # EVs = loadings %*% EAs[[ci]] (EAs[[ci]] is length k)
  EVs <- lapply(EAs, function(EA) loadings %*% matrix(EA, ncol=1))
  
  list(EAs=EAs, EVs=EVs, varAs=varAs, sdVs=sdVs, EScores=EScores, sdScores=sdScores)
}
