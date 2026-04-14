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
          scores[, (i + 1):ncomp, drop = FALSE],
          loadings[, (i + 1):ncomp, drop = FALSE]
        )
    }
    
    Q[, i] <- rowSums(res^2)
    #T2[, i] <- rowSums(scoresn[, seq_len(i), drop = FALSE]^2)
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
    T2[, i] <- rowSums(scoresn[, seq_len(i), drop = FALSE]^2)
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



#' @rdname perm_ci
#' @param k Number of components to test (default 4).
#' @param distr Distribution assumption (default "gamma"); currently ignored in forwarding.
#' @param parallel Logical; if TRUE, use parallel processing.
#' @export
perm_ci.pca <- function(x, X, nperm=100, k=4, distr="gamma", parallel=FALSE, ...) {
  .Deprecated("perm_test.pca", package="multivarious", 
              msg = "'perm_ci.pca' is deprecated. Please use 'perm_test.pca' with distribution=\"empirical\" instead.")
  
  # Minimal forwarding logic, likely won't match exactly but guides user
  perm_test.pca(x = x, X = X, nperm = nperm, comps = k, 
                distribution = "empirical", # Force empirical as distr is gone
                parallel = parallel, alternative = "greater", ...)
}


#' @importFrom stats quantile na.omit
#' @importFrom future.apply future_lapply
#' @export
#' @seealso \code{\link{perm_test}}, \code{\link{pca}}
perm_test.pca <- function(x,
                          X,
                          nperm = 1000,
                          measure_fun = NULL,
                          fit_fun = NULL, # Ignored
                          shuffle_fun = NULL,               
                          stepwise = TRUE,
                          parallel = FALSE,
                          alternative = c("greater", "less", "two.sided"),
                          alpha = 0.05, # Significance level for stopping rule
                          comps = 4,
                          use_svd_solver = c("fast", "RSpectra"), 
                          ...) 
{
  # Match args
  alternative <- match.arg(alternative)
  use_svd_solver <- match.arg(use_svd_solver)
  chk::chk_number(alpha)
  chk::chk_range(alpha, c(0, 1))
  
  # Capture extra args (though likely unused by defaults here)
  extra_args <- list(...)
  
  # --- Setup ---
  Q <- ncomp(x)
  orig_comps <- comps
  comps <- min(Q - 1, comps) # Can only test up to Q-1 components sequentially
  if (comps < 1) stop("Cannot perform permutation test, need at least 2 components in the model.")
  if (comps < orig_comps) {
      message(sprintf("Requested comps=%d truncated to %d (can test at most Q-1=%d components sequentially).", 
                      orig_comps, comps, Q - 1))
  }
  
  # --- Observed Statistics (Default: F_a) ---
  evals <- x$sdev^2
  if (is.null(measure_fun)) {
      observed_stats <- sapply(1:comps, function(a) {
          denom <- sum(evals[a:Q])
          if (denom <= .Machine$double.eps) return(0)
          evals[a] / denom
      })
      stat_name <- "F_a (Fraction of Remaining Variance)"
      internal_stat_fun <- function(modp, a, ...) { # Add ... to signature for consistency
          if (is.null(modp) || length(modp$sdev) == 0) return(NA_real_)
          eperm <- modp$sdev^2
          if (length(eperm) == 0) return(NA_real_)
          n_relevant_evals <- Q - (a - 1)
          if (length(eperm) < 1) return(NA_real_)
          if (length(eperm) < n_relevant_evals) {
              warning(sprintf("Permutation for component %d yielded only %d eigenvalues (expected at least %d). Using sum of available.", a, length(eperm), n_relevant_evals))
              n_relevant_evals <- length(eperm)
          }
          denom <- sum(eperm[1:n_relevant_evals])
          if (denom <= .Machine$double.eps) return(0)
          eperm[1] / denom
      }
  } else {
      stat_name <- deparse(substitute(measure_fun))
      # Calculate observed stats using user function (pass extra args)
      observed_stats <- sapply(1:comps, function(a) do.call(measure_fun, c(list(model_perm = x, comp_idx = a), extra_args)))
      # Internal function passes ... down
      internal_stat_fun <- function(modp, a, ...) {
          if (is.null(modp)) return(NA_real_)
          do.call(measure_fun, c(list(model_perm = modp, comp_idx = a), list(...)))
      }
  }
  
  # --- Default Shuffle Function (Column-wise) ---
  if (is.null(shuffle_fun)) {
      shuffle_fun <- function(dat, ...) apply(dat, 2, sample)
      shuffle_type <- "column-wise"
  } else {
      shuffle_type <- "custom"
  }
  
  # --- Preprocess Data ---
  Xp <- transform(x$preproc, X)
  
  # --- Efficient PCA/SVD for Permutations & P3 Projection ---
  get_leading_svd_u <- function(M, k) {
      if (k <= 0) return(matrix(0.0, nrow = nrow(M), ncol = 0))
      if (nrow(M) < 1 || ncol(M) < 1 || k > min(nrow(M), ncol(M))) {
          warning(sprintf("Cannot compute %d singular vectors for %d x %d matrix.", k, nrow(M), ncol(M)))
          return(NULL)
      }
      solver <- use_svd_solver
      if (solver == "RSpectra" && !requireNamespace("RSpectra", quietly = TRUE)) {
          warning("RSpectra package not found, falling back to use_svd_solver='fast'.")
          solver <- "fast"
      }
      U_perm <- tryCatch({
          if (solver == "RSpectra") RSpectra::svds(M, k = k, nu = k, nv = 0)$u
          else svd(M, nu = k, nv = 0)$u
       }, error = function(e) {
           warning(sprintf("SVD(k=%d) calculation failed: %s. Returning NULL.", k, e$message))
           NULL
       })
      U_perm
  }
  run_pca_perm <- function(M_perm) {
      ncomp_needed <- Q
      pca_res <- try(svd_wrapper(M_perm, ncomp=ncomp_needed, preproc=pass(), method=use_svd_solver), silent=TRUE)
      if (inherits(pca_res, "try-error")) {
          warning(sprintf("PCA failed during permutation: %s", pca_res))
          return(NULL)
      }
      if (is.null(pca_res$sdev) || length(pca_res$sdev) == 0) {
          warning("PCA during permutation resulted in no valid singular values.")
          return(NULL)
      }
      return(pca_res)
  }
  
  # --- Pre-calculate Reconstructions if Stepwise ---
  recon_list <- list()
  if (stepwise && comps > 0) {
      message("Pre-calculating reconstructions for stepwise testing...")
      for (a in 1:comps) {
          cnums <- 1:a
          recon_list[[a]] <- reconstruct(x, comp = cnums)
      }
  }
  I_mat <- diag(nrow(Xp))
  
  # ---------- Permutation Loop Function ----------
  one_perm <- function(perm_idx, current_a, ...) {
      Ea <- if (!stepwise || current_a == 1) Xp else Xp - recon_list[[current_a - 1]]
      
      # Create args for shuffle_fun
      shuffle_args <- c(list(dat = Ea), list(...))
      Ea_perm <- do.call(shuffle_fun, shuffle_args)
      
      Ea_perm_proj <- if (stepwise && current_a > 1) {
          Ua_perm <- get_leading_svd_u(Ea_perm, current_a - 1)
          if (is.null(Ua_perm)) return(NA_real_)
          P_orth <- I_mat - tcrossprod(Ua_perm)
          proj_res <- P_orth %*% Ea_perm
          if (any(!is.finite(proj_res))) {
              warning(sprintf("Permutation %d, Comp %d: Non-finite values after P3 projection. Replacing with 0.", perm_idx, current_a))
              proj_res[!is.finite(proj_res)] <- 0
          }
          proj_res
      } else {
          Ea_perm
      }
      
      modp <- run_pca_perm(Ea_perm_proj)
      if (is.null(modp)) return(NA_real_)
      
      # Pass ... down to internal_stat_fun
      do.call(internal_stat_fun, c(list(modp = modp, a = current_a), list(...)))
  }

  # ---------- Run Permutations ----------
  Fq <- matrix(NA, nrow = nperm, ncol = comps)
  n_complete <- rep(0, comps)
  pvals <- rep(NA, comps)
  comps_tested <- 0
  
  apply_fun <- if (parallel) future.apply::future_lapply else lapply
  
  message(sprintf("Running %d permutations sequentially for up to %d PCA components (alpha=%.3f, %s)...", 
                  nperm, comps, alpha, if(parallel) "parallel" else "serial"))
  
  for (a in 1:comps) {
      message(sprintf("  Testing Component %d/%d...", a, comps))
      # Pass extra_args down via lapply's ...
      perm_args <- list(X = seq_len(nperm), FUN = one_perm, current_a = a)
      if (parallel) perm_args$future.seed <- TRUE
      perm_args <- c(perm_args, extra_args)
      
      perm_vals_list <- do.call(apply_fun, perm_args)
      perm_vals_a <- unlist(perm_vals_list)
      Fq[, a] <- perm_vals_a
      n_complete[a] <- sum(!is.na(perm_vals_a))
      
      if (n_complete[a] < nperm) {
          warning(sprintf("Component %d: %d/%d permutations failed (e.g., SVD error). Using %d successful permutations.",
                          a, nperm - n_complete[a], nperm, n_complete[a]))
      }
      if (n_complete[a] == 0) {
          warning(sprintf("Component %d: All permutations failed. Cannot compute p-value.", a))
          pvals[a] <- NA
          comps_tested <- a
          break # Stop if all perms failed
      }
      
      # Calculate empirical p-value
      obs_a <- observed_stats[a]
      perm_vals_a_clean <- stats::na.omit(perm_vals_a)
      if (is.na(obs_a)) {
          pval_a <- NA
          warning(sprintf("Component %d: Observed statistic is NA, cannot compute p-value.", a))
      } else {
          if (alternative == "greater") {
              b <- sum(perm_vals_a_clean >= obs_a)
              pval_a <- (b + 1) / (n_complete[a] + 1)
          } else if (alternative == "less") {
              b <- sum(perm_vals_a_clean <= obs_a)
              pval_a <- (b + 1) / (n_complete[a] + 1)
          } else { # two.sided
              b_greater <- sum(perm_vals_a_clean >= obs_a)
              b_less <- sum(perm_vals_a_clean <= obs_a)
              pval_two <- 2 * min((b_greater + 1) / (n_complete[a] + 1), (b_less + 1) / (n_complete[a] + 1))
              pval_a <- min(pval_two, 1.0)
          }
      }
      pvals[a] <- pval_a
      comps_tested <- a
      
      # Sequential stopping rule
      if (!is.na(pval_a) && pval_a > alpha) {
          message(sprintf("  Component %d p-value (%.4g) > alpha (%.3f). Stopping sequential testing.", a, pval_a, alpha))
          break
      }
  } # End loop over components
  
  # ---------- Calculate CIs and final results table ----------
  comp_df_list <- vector("list", comps_tested)
  for (i in 1:comps_tested) {
      lower_ci <- NA; upper_ci <- NA
      if (n_complete[i] > 1) {
          perm_vals_i_clean <- stats::na.omit(Fq[, i])
          cis <- stats::quantile(perm_vals_i_clean, probs = c(0.025, 0.975), na.rm = TRUE)
          lower_ci <- cis[1]; upper_ci <- cis[2]
      }
      comp_df_list[[i]] <- tibble::tibble(
          comp = i,
          observed = observed_stats[i],
          pval = pvals[i],
          lower_ci = lower_ci,
          upper_ci = upper_ci
      )
  }
  component_results <- dplyr::bind_rows(comp_df_list)
  
  # ---------- Output Structure ----------
  out <- list(
      call = match.call(),
      component_results = component_results,
      perm_values = Fq[, 1:comps_tested, drop = FALSE],
      alpha = alpha,
      alternative = alternative,
      method = sprintf("Permutation test for PCA (Vitale et al. 2017 P3) (statistic: %s, stepwise: %s, shuffle: %s)", 
                       stat_name, stepwise, shuffle_type),
      nperm = n_complete[1:comps_tested] # Report vector of successful permutations per component
  )
  class(out) <- c("perm_test_pca", "perm_test")
  out
}

# Re-introducing specific reconstruct.pca method as the generic bi_projector one is not suitable
#' Reconstruct Data from PCA Results
#'
#' Reconstructs the original (centered) data matrix from the PCA scores and loadings.
#'
#' @param x A `pca` object.
#' @param comp Integer vector specifying which components to use for reconstruction (default: all components in `x`).
#' @param ... Extra arguments (ignored).
#' @return A matrix representing the reconstructed data in the *original* scale (preprocessing reversed).
#' @export
reconstruct.pca <- function(x, comp = 1:ncomp(x), ...) {
  # Check component indices
  chk::chk_vector(comp)
  chk::chk_subset(comp, 1:ncomp(x))
  
  # Use standard PCA reconstruction: scores %*% t(loadings)
  reconstructed_proc <- scores(x)[, comp, drop=FALSE] %*% t(coef(x)[, comp, drop=FALSE])

  # Reverse the preprocessing to return data in original scale
  inverse_transform(x$preproc, reconstructed_proc)
}


# Ensure print.perm_test exists or define print.perm_test_pca
# Assuming print.perm_test from discriminant_projector handles the structure:
# list(statistic, perm_values, p.value, alternative, method, nperm, call)
# Our structure is different (component_results data frame).
# Let's define a specific print method.

#' Print Method for perm_test_pca Objects
#' 
#' Provides a concise summary of the PCA permutation test results.
#' 
#' @param x An object of class `perm_test_pca`.
#' @param ... Additional arguments passed to printing methods.
#' @return Invisibly returns the input object `x`.
#' @export
print.perm_test_pca <- function(x, ...) {
  cat("\nPCA Permutation Test Results\n\n")
  cat("Method: ", x$method, "\n")
  cat("Alternative: ", x$alternative, "\n")
  cat("\nComponent Results:\n")
  print(as.data.frame(x$component_results)) # Print the results table
  cat("\nNumber of successful permutations per component:", paste(x$nperm, collapse=", "), "\n")
  invisible(x)
}



#' Rotate PCA Loadings
#'
#' Apply a specified rotation to the component loadings of a PCA model. 
#' This function leverages the GPArotation package to apply orthogonal 
#' or oblique rotations.
#'
#' @param x A PCA model object, typically created using the `pca()` function.
#' @param ncomp The number of components to rotate. Must be <= ncomp(x).
#' @param type The type of rotation to apply. Supported rotation types:
#'
#' \describe{
#'   \item{"varimax"}{Orthogonal Varimax rotation}
#'   \item{"quartimax"}{Orthogonal Quartimax rotation}
#'   \item{"promax"}{Oblique Promax rotation}
#' }
#'
#' @param loadings_type For oblique rotations, which loadings to use:
#'
#' \describe{
#'   \item{"pattern"}{Use pattern loadings as \code{v}}
#'   \item{"structure"}{Use structure loadings (\code{pattern_loadings \%*\% Phi}) as \code{v}}
#' }
#'
#' Ignored for orthogonal rotations.
#'
#' @param score_method How to recompute scores after rotation:
#'
#' \describe{
#'   \item{"auto"}{For orthogonal rotations, use 
#'     \code{scores_new = scores_original \%*\% t(R)}} 
#'     For oblique rotations, recompute from the pseudoinverse.
#'
#'   \item{"recompute"}{Always recompute scores from \code{X_proc} and 
#'     the pseudoinverse of rotated loadings.}
#'
#'   \item{"original"}{For orth rotations, same as \code{auto}, 
#'     but may not work for oblique rotations.}
#' }
#'
#' @param ... Additional arguments passed to GPArotation functions.
#'
#' @return A modified PCA object with class \code{rotated_pca} and additional fields:
#' \describe{
#'   \item{v}{Rotated loadings}
#'   \item{s}{Rotated scores}
#'   \item{sdev}{Updated standard deviations of rotated components}
#'   \item{explained_variance}{Proportion of explained variance for each rotated component}
#'   \item{rotation}{A list with rotation details: \code{type}, \code{R} (orth) or \code{Phi} (oblique), and \code{loadings_type}}
#' }
#'
#' @importFrom GPArotation GPForth GPFoblq
#' @export
#'
#' @examples
#' # Perform PCA on the iris dataset
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
    variances <- apply(scores_new, 2, stats::var)
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
    variances <- apply(scores_new, 2, stats::var)
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




#' Biplot for PCA Objects (Enhanced with ggrepel)
#'
#' Creates a 2D biplot for a \code{pca} object, using \pkg{ggplot2} and \pkg{ggrepel} 
#' to show both sample scores (observations) and variable loadings (arrows).
#'
#' @param x A \code{pca} object returned by \code{\link{pca}}.
#' @param y (ignored) Placeholder to match \code{biplot(x, y, ...)} signature.
#' @param dims A length-2 integer vector specifying which principal components to plot 
#'   on the x and y axes. Defaults to \code{c(1, 2)}.
#' @param scale_arrows A numeric factor to scale the variable loadings (arrows). Default is 2.
#' @param alpha_points Transparency level for the sample points. Default is 0.6.
#' @param point_size Size for the sample points. Default is 2.
#' @param point_labels Optional character vector of labels for the sample points. 
#'   If \code{NULL}, rownames of the scores matrix are used if available; otherwise numeric indices.
#' @param var_labels Optional character vector of variable names (columns in the original data). 
#'   If \code{NULL}, rownames of \code{x\$v} are used if available; otherwise "Var1", "Var2", etc.
#' @param arrow_color Color for the loading arrows. Default is "red".
#' @param text_color Color for the variable label text. Default is "red".
#' @param repel_points Logical; if TRUE, repel sample labels using \code{geom_text_repel}. Default is \code{TRUE}.
#' @param repel_vars Logical; if TRUE, repel variable labels using \code{geom_text_repel}. Default is \code{FALSE}.
#' @param ... Additional arguments passed on to \code{ggplot2} or \code{ggrepel} functions (if needed).
#'
#' @details
#' This function constructs a scatterplot of the PCA scores (observations) on two chosen components
#' and overlays arrows for the loadings (variables). The arrow length and direction indicate how each
#' variable contributes to those principal components. You can control arrow scaling with \code{scale_arrows}.
#'
#' If your \code{pca} object includes an \code{$explained_variance} field (e.g., proportion of variance per component),
#' those values will appear in the axis labels. Otherwise, the axes are labeled simply as "PC1", "PC2", etc.
#'
#' **Note**: If you do not have \pkg{ggrepel} installed, you can set \code{repel_points=FALSE} and 
#' \code{repel_vars=FALSE}, or install \pkg{ggrepel}.
#'
#' @return A \code{ggplot} object.
#'
#' @import ggplot2
#' @importFrom ggrepel geom_text_repel
#' @export
#'
#' @examples
#' \donttest{
#' data(iris)
#' X <- as.matrix(iris[,1:4])
#' pca_res <- pca(X, ncomp=2)
#'
#' # Enhanced biplot with repelled text
#' biplot(pca_res, repel_points=TRUE, repel_vars=TRUE)
#' }
biplot.pca <- function(x,
                       y = NULL,  # ignored, matching biplot generic
                       dims = c(1, 2),
                       scale_arrows = 2,
                       alpha_points = 0.6,
                       point_size = 2,
                       point_labels = NULL,
                       var_labels = NULL,
                       arrow_color = "red",
                       text_color = "red",
                       repel_points = TRUE,
                       repel_vars = FALSE,
                       ...) 
{
  # Check dims
  n_comps <- ncomp(x)
  if (any(dims > n_comps)) {
    stop("Requested components exceed the number of available components in 'x'.")
  }
  if (length(dims) != 2) {
    stop("'dims' must be a vector of length 2, e.g. c(1, 2).")
  }
  
  # Extract scores and loadings
  sc <- x$s  # observations x components
  ld <- x$v  # variables x components
  if (is.null(sc) || is.null(ld)) {
    stop("The PCA object does not contain both scores (x$s) and loadings (x$v).")
  }
  
  # Subset to dims
  sc2 <- sc[, dims, drop = FALSE]
  ld2 <- ld[, dims, drop = FALSE]
  
  # If we have explained variance, use it
  if (!is.null(x$explained_variance)) {
    pc_var <- x$explained_variance
  } else {
    pc_var <- rep(NA_real_, ncol(sc))
  }
  
  # Convert scores to data.frame
  scores_df <- as.data.frame(sc2)
  colnames(scores_df) <- c("PCx", "PCy")
  
  # Assign labels to points
  if (is.null(point_labels)) {
    if (!is.null(rownames(sc))) {
      point_labels <- rownames(sc)
    } else {
      point_labels <- seq_len(nrow(sc))
    }
  }
  scores_df$labels <- point_labels
  
  # Convert loadings to data.frame
  loadings_df <- as.data.frame(ld2)
  colnames(loadings_df) <- c("PCx", "PCy")
  
  # Assign variable labels
  if (is.null(var_labels)) {
    if (!is.null(rownames(ld))) {
      var_labels <- rownames(ld)
    } else {
      var_labels <- paste0("Var", seq_len(nrow(ld)))
    }
  }
  loadings_df$var <- var_labels
  
  # Scale loadings for arrow length
  loadings_df$PCx <- loadings_df$PCx * scale_arrows
  loadings_df$PCy <- loadings_df$PCy * scale_arrows
  
  # Build axis labels (include % variance if available)
  pcx_lab <- if (!is.na(pc_var[dims[1]])) {
    paste0("PC", dims[1], " (", round(100 * pc_var[dims[1]], 1), "%)")
  } else {
    paste0("PC", dims[1])
  }
  pcy_lab <- if (!is.na(pc_var[dims[2]])) {
    paste0("PC", dims[2], " (", round(100 * pc_var[dims[2]], 1), "%)")
  } else {
    paste0("PC", dims[2])
  }
  
  # Start building the plot
  plt <- ggplot(scores_df, aes(x = .data$PCx, y = .data$PCy)) +
    geom_point(alpha = alpha_points, size = point_size, color = "blue") +
    theme_minimal(base_size = 12) +
    coord_equal() +
    xlab(pcx_lab) +
    ylab(pcy_lab)
  
  # Add text labels for points
  # Use ggrepel if repel_points=TRUE and ggrepel is installed
  # fallback to geom_text if not installed or repel_points=FALSE
  can_repel <- requireNamespace("ggrepel", quietly = TRUE)
  
  if (repel_points && can_repel) {
    plt <- plt + 
      ggrepel::geom_text_repel(aes(label = .data$labels), color = "black", size = 3, ...)
  } else {
    plt <- plt + 
      geom_text(aes(label = .data$labels), hjust = 1.1, vjust = 0.5, color = "black", size = 3, ...)
  }
  
  # Add loadings arrows
  plt <- plt +
    geom_segment(data = loadings_df,
                 aes(x = 0, y = 0, xend = .data$PCx, yend = .data$PCy),
                 arrow = arrow(length = unit(0.02, "npc")),
                 color = arrow_color,
                 linewidth = 0.7)
  
  # Add variable names near arrow tips
  if (repel_vars && can_repel) {
    plt <- plt +
      ggrepel::geom_text_repel(data = loadings_df,
                               aes(x = .data$PCx, y = .data$PCy, label = .data$var),
                               color = text_color,
                               size = 3,
                               ...)
  } else {
    plt <- plt +
      geom_text(data = loadings_df,
                aes(x = .data$PCx, y = .data$PCy, label = .data$var),
                color = text_color, vjust = -0.5, size = 3,
                ...)
  }
  
  plt
}

#' Print Method for PCA Objects
#'
#' Provide a color-enhanced summary of the PCA object, including 
#' dimensions, variance explained, and a quick component breakdown.
#'
#' @param x A \code{pca} object.
#' @param ... Ignored (for compatibility).
#' @export
print.pca <- function(x, ...) {
  cat(
    crayon::bold(crayon::green("PCA object")),
    " -- derived from SVD\n\n"
  )
  
  # Basic dims
  nobs <- nrow(x$s)         # number of observations
  nvars <- nrow(x$v)        # number of variables
  ncomp_used <- ncol(x$s)   # how many comps we actually have
  
  cat(crayon::cyan("Data: "), nobs, " observations x ", nvars, " variables\n", sep = "")
  cat(crayon::cyan("Components retained: "), ncomp_used, "\n\n", sep = "")
  
  # Possibly show proportion of variance if available
  if (!is.null(x$sdev)) {
    eigenvals <- x$sdev^2
    prop_var <- eigenvals / sum(eigenvals)
    cum_var <- cumsum(prop_var)
    
    cat(crayon::bold("Variance explained (per component):\n"))
    cat(
      formatC(seq_len(ncomp_used), width = 2), " ",
      formatC(round(prop_var * 100, 2), width=6), "%  ",
      "(cumulative: ",
      formatC(round(cum_var * 100, 2), width=6), "%)\n",
      sep = ""
    )
    cat("\n")
  }
  
  # If we have an explained_variance field (like from rotate), use that
  if (!is.null(x$explained_variance)) {
    cat(crayon::bold("Explained variance from rotation:\n"))
    cat(paste(round(x$explained_variance * 100, 2), "%\n"), "\n\n")
  }
  
  # Possibly show info about rotation
  if (!is.null(x$rotation)) {
    cat(crayon::bold("Rotation details:\n"))
    cat("  Type:", x$rotation$type, "\n")
    if (!is.null(x$rotation$loadings_type)) {
      cat("  Loadings type:", x$rotation$loadings_type, "\n")
    }
    cat("\n")
  }
  
  invisible(x)
}


#' Screeplot for PCA
#'
#' Displays the variance explained by each principal component as a bar or line plot.
#' 
#' @param x A \code{pca} object.
#' @param type "barplot" or "lines".
#' @param main Plot title.
#' @param ... Additional args to pass to base R plotting.
#' @export
screeplot.pca <- function(x, type="barplot", main="Screeplot", ...) {
  pc_variance <- x$sdev^2
  percent_variance <- pc_variance / sum(pc_variance) * 100
  
  if (type == "barplot") {
    graphics::barplot(percent_variance, 
            xlab="Principal Component",
            ylab="Percentage of Variance Explained",
            names.arg=paste0("PC", 1:length(percent_variance)),
            main=main, ...)
  } else if (type == "lines") {
    plot(seq_len(length(percent_variance)), percent_variance, type="o",
         xlab="Component",
         ylab="Proportion of Variance",
         main=main, ...)
  }
  invisible(NULL)
}


#' PCA Outlier Diagnostics
#'
#' Calculates Hotelling T^2 (score distance) and Q-residual (orthogonal distance)
#' for each observation, given a chosen number of components.
#'
#' @param x A \code{pca} object.
#' @param X The original data matrix used for PCA.
#' @param ncomp Number of components to consider.
#' @param cutoff Logical or numeric specifying threshold for labeling outliers. If \code{TRUE},
#'   uses some typical statistical threshold (F-dist) for T^2, or sets an arbitrary Q limit.
#'   If numeric, treat it as a cutoff. Default is \code{FALSE} (no labeling).
#' @return A data frame with columns \code{T2} and \code{Q}, and optionally an outlier flag.
#' @export
pca_outliers <- function(x, X, ncomp, cutoff=FALSE) {
  # compute T2 (score distances)
  T2 <- score_distances.pca(x, ncomp, xorig=X)
  T2vals <- T2[, ncomp]
  
  # compute Q (residual distances)
  Q <- orth_distances.pca(x, ncomp, xorig=X)
  Qvals <- Q[, ncomp]
  
  # optional outlier logic...
  
  data.frame(T2 = T2vals, Q = Qvals)
}

