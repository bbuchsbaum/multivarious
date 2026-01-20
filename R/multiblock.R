#' Create a Multiblock Projector
#'
#' Constructs a multiblock projector using the given component matrix (`v`), a preprocessing function, and a list of block indices. 
#' This allows for the projection of multiblock data, where each block represents a different set of variables or features.
#'
#' @param v A matrix of components with dimensions `nrow(v)` by `ncol(v)` (columns = number of components).
#' @param preproc A pre-processing function for the data (default: `prep(pass())`).
#' @param block_indices A list of numeric vectors specifying the indices of each data block.
#' @param classes (optional) A character vector specifying additional class attributes of the object, default is NULL.
#' @param ... Extra arguments.
#' @return A `multiblock_projector` object.
#'
#' @seealso projector
#' @export
#' @examples
#' # Generate some example data
#' X1 <- matrix(rnorm(10 * 5), 10, 5)
#' X2 <- matrix(rnorm(10 * 5), 10, 5)
#' X <- cbind(X1, X2)
#'
#' # Compute PCA on the combined data
#' pc <- pca(X, ncomp = 8)
#'
#' # Create a multiblock projector using PCA components and block indices
#' mb_proj <- multiblock_projector(pc$v, block_indices = list(1:5, 6:10))
#'
#' # Project multiblock data using the multiblock projector
#' mb_scores <- project(mb_proj, X)
multiblock_projector <- function(v, preproc=prep(pass()), ..., block_indices, classes=NULL) {
  chk::chk_list(block_indices)
  sumind <- sum(sapply(block_indices, length))
  chk::chk_equal(sumind, nrow(v))
  
  projector(v, preproc, block_indices=block_indices, ..., classes=c(classes, "multiblock_projector"))
}


#' Create a Multiblock Bi-Projector
#'
#' Constructs a multiblock bi-projector using the given component matrix (`v`), score matrix (`s`), singular values (`sdev`),
#' a preprocessing function, and a list of block indices. This allows for two-way mapping with multiblock data.
#'
#' @param v A matrix of components (nrow = number of variables, ncol = number of components).
#' @param s A matrix of scores (nrow = samples, ncol = components).
#' @param sdev A numeric vector of singular values or standard deviations.
#' @param preproc A pre-processing object (default: `prep(pass())`).
#' @param block_indices A list of numeric vectors specifying data block variable indices.
#' @param classes Additional class attributes (default NULL).
#' @param ... Extra arguments.
#' @return A `multiblock_biprojector` object.
#'
#' @seealso bi_projector, multiblock_projector
#' @export
multiblock_biprojector <- function(v, s, sdev, preproc=prep(pass()), ..., block_indices, classes=NULL) {
  sumind <- sum(sapply(block_indices, length))
  chk::chk_equal(sumind, nrow(v))
  bi_projector(v, s=s, sdev=sdev, preproc=preproc, block_indices=block_indices, ..., classes=c(classes, "multiblock_biprojector", "multiblock_projector"))
}


#' Extract the Block Indices from a Multiblock Projector
#'
#' @param x A `multiblock_projector` object.
#' @param i Ignored.
#' @param ... Ignored.
#' @return The list of block indices.
#' @export
block_indices.multiblock_projector <- function(x,i,...) {
  x$block_indices
}

#' @export
block_lengths.multiblock_projector <- function(x) {
  sapply(block_indices(x), length)
}

#' @export
nblocks.multiblock_projector <- function(x) {
  length(block_indices(x))
}

#' Project Data onto a Specific Block
#'
#' Projects the new data onto the subspace defined by a specific block of variables.
#'
#' @param x A `multiblock_projector` object.
#' @param new_data The new data to be projected.
#' @param block The block index (1-based) to project onto.
#' @param least_squares Logical. If `TRUE` (default), use least squares projection.
#' @param ... Additional arguments passed to `partial_project`.
#' @return The projected scores for the specified block.
#' @export
project_block.multiblock_projector <- function(x, new_data, block,least_squares=TRUE, ...) {
  # Check block validity
  nb <- nblocks(x)
  if (block < 1 || block > nb) {
    stop("Block index out of range.")
  }
  
  ind <- block_indices(x)[[block]]
  partial_project(x, new_data, colind=ind, least_squares,...)
}

#' Coefficients for a Multiblock Projector
#'
#' Extracts the components (loadings) for a given block or the entire projector.
#'
#' @param object A `multiblock_projector` object.
#' @param block Optional block index. If missing, returns loadings for all variables.
#' @param ... Additional arguments.
#' @return A matrix of loadings.
#' @export
coef.multiblock_projector <- function(object, block,...) {
  if (missing(block)) {
    # Instead of NextMethod(object), just use NextMethod() to call coef.projector
    NextMethod()
  } else {
    nb <- nblocks(object)
    if (block < 1 || block > nb) {
      stop("Block index out of range.")
    }
    ind <- object$block_indices[[block]]
    object$v[ind,,drop=FALSE]
  }
}

#' Pretty Print Method for `multiblock_biprojector` Objects
#'
#' Display a summary of a `multiblock_biprojector` object.
#'
#' @param x A `multiblock_biprojector` object.
#' @param ... Additional arguments passed to `print()`.
#' @return Invisible `multiblock_biprojector` object.
#' @export
print.multiblock_biprojector <- function(x, ...) {
  cat("Multiblock Bi-Projector object:\n")
  cat("  Projection matrix dimensions: ", nrow(x$v), "x", ncol(x$v), "\n")
  
  # Print block indices in a nicer format
  cat("  Block indices:\n")
  lapply(seq_along(x$block_indices), function(i) {
    cat("    Block ", i, ": ", paste(x$block_indices[[i]], collapse = ","), "\n", sep="")
  })
  
  invisible(x)
}

# -------------------------------------------------------------
# Default multiblock shuffle helper
# -------------------------------------------------------------
#' Shuffle rows of a matrix or list of matrices
#'
#' Helper used internally by permutation tests to randomize the
#' observation order of either a single matrix or each matrix in a
#' list.  Each block (or the single matrix) has its rows permuted
#' independently.
#' @param data A matrix or list of matrices with the same number of rows.
#' @keywords internal
#' @noRd
.default_multiblock_shuffle <- function(data) {
  if (is.list(data)) {
    lapply(data, function(M) M[sample(nrow(M)), , drop = FALSE])
  } else if (is.matrix(data)) {
    data[sample(nrow(data)), , drop = FALSE]
  } else {
    stop("Unsupported data type for multiblock shuffling")
  }
}





#' @importFrom stats var
#' @importFrom utils combn
#' @importFrom RSpectra svds
#' @importFrom future.apply future_lapply
#' @export
perm_test.multiblock_biprojector <- function(
    x,
    Xlist          = NULL,          # optional list of blocks
    nperm          = 500,
    comps          = 4,             # max # components to test
    alpha          = 0.05,
    shuffle_fun    = NULL,          # custom shuffler
    parallel       = FALSE,
    alternative    = c("greater","less","two.sided"),
    use_rspectra   = TRUE,
    ...
){
  alternative <- match.arg(alternative)

  ## ------------------------------------------------------------------
  ## 1.  pull block indices & component count
  ## ------------------------------------------------------------------
  blk_ind   <- block_indices(x)
  B         <- length(blk_ind)
  Kmax      <- ncol(x$v)                    # components available
  comps     <- min(comps, Kmax)
  if (comps < 1L) stop("Need at least one component to test.")

  n         <- nrow(scores(x))

  ## ------------------------------------------------------------------
  ## 2.  helper  -  get block-wise score matrix  (n x B)  for comp k
  ## ------------------------------------------------------------------
  get_Tk <- function(comp_k, data_list = NULL, S_perm = NULL){
      if (is.null(data_list)){          # use stored scores -- fast
          # If S_perm is provided (already shuffled full score matrix), use it
          # Otherwise, use the original scores (only for calculating observed stat)
          sc_matrix <- if (!is.null(S_perm)) S_perm else scores(x)
          sc <- sc_matrix[ , comp_k, drop = FALSE]   # n x 1
          # Return a matrix with B identical columns of these scores
          do.call(cbind, replicate(B, sc, simplify = FALSE))
      } else {                          # re-project if user gave data
          # Project each block in data_list onto component k using original model x
          lapply(seq_len(B), function(b){
              xb <- data_list[[b]]
              # project block b onto *single* component k
              nd_proc <- transform(x$preproc, xb, blk_ind[[b]])
              v_sub   <- x$v[ blk_ind[[b]] , comp_k , drop = FALSE]
              as.vector(nd_proc %*% v_sub)
          }) |> do.call(cbind, args = _) # Result is n x B matrix
      }
  }

  ## ------------------------------------------------------------------
  ## 3.  statistic per component  --  leading eigenvalue of  T^T T
  ## ------------------------------------------------------------------
  comp_stat <- function(Tk){
      S <- crossprod(Tk)                       # B x B
      if (B == 1) return(sum(Tk^2))            # trivial 1-block case
      if (use_rspectra && requireNamespace("RSpectra", quietly = TRUE))
           RSpectra::svds(S, k = 1, nu = 0, nv = 1)$d[1]
      else eigen(S, symmetric = TRUE, only.values = TRUE)$values[1]
  }

  ## observed statistics ------------------------------------------------
  T_list_obs <- lapply(seq_len(comps), function(k){
      comp_stat( get_Tk(k, Xlist) )
  })
  obs_vec <- unlist(T_list_obs)

  ## ------------------------------------------------------------------
  ## 4.  decide what data to shuffle and set default shuffle
  ## ------------------------------------------------------------------
  if (is.null(Xlist)) {
      # Shuffle the scores matrix once per permutation
      data_to_shuffle            <- scores(x)
      use_shuffled_scores_matrix <- TRUE
  } else {
      if (length(Xlist) != B)
         stop("Length of Xlist must equal number of blocks in the model.")
      data_to_shuffle            <- Xlist
      use_shuffled_scores_matrix <- FALSE
  }

  if (is.null(shuffle_fun))
      shuffle_fun <- .default_multiblock_shuffle

  ## ------------------------------------------------------------------
  ## 5.  permutation loop (sequential stopping like PCA)
  ## ------------------------------------------------------------------
  Fperm   <- matrix(NA_real_, nrow = nperm, ncol = comps)
  n_ok    <- integer(comps)

  apply_fun <- if (parallel) future.apply::future_lapply else lapply
  if (parallel) message("perm_test.multiblock_biprojector : permutations in parallel ...")

  # Pre-calculate observed stats outside loop
  obs_stats <- sapply(seq_len(comps), function(k) {
    Tk_obs <- get_Tk(k, Xlist) # Use original Xlist or scores
    comp_stat(Tk_obs)
  })

  for (k in seq_len(comps)){ # Loop over components for sequential testing
      perm_fun <- function(i){ # Function for one permutation
          # Shuffle either the original scores matrix OR the list of X blocks
          data_perm <- shuffle_fun(data_to_shuffle)
          
          # Calculate statistic for component k based on shuffled data
          if (use_shuffled_scores_matrix) {
              # Pass the shuffled full score matrix (data_perm) to get_Tk
              Tk_perm <- get_Tk(comp_k = k, S_perm = data_perm)
          } else {
              # Pass the shuffled list of X blocks (data_perm) to get_Tk for re-projection
              Tk_perm <- get_Tk(comp_k = k, data_list = data_perm)
          }
          comp_stat(Tk_perm)
      }
      
      # Run permutations for component k
      plist   <- apply_fun(seq_len(nperm), perm_fun)
      perm_vals_k <- unlist(plist)
      Fperm[, k] <- perm_vals_k # Store results for component k
      n_ok[k]    <- sum(is.finite(perm_vals_k))

      ## empirical p-value for component k
      obs_k <- obs_stats[k]
      g      <- sum(perm_vals_k >= obs_k, na.rm = TRUE)
      l      <- sum(perm_vals_k <= obs_k, na.rm = TRUE)
      pval   <- switch(alternative,
                       greater   = (g + 1)/(n_ok[k] + 1),
                       less      = (l + 1)/(n_ok[k] + 1),
                       two.sided = min(1, 2*min((g + 1)/(n_ok[k]+1),
                                                (l + 1)/(n_ok[k]+1))))
      if (!is.na(pval) && pval > alpha){          # Vitale-like stop rule
          comps_tested <- k
          Fperm <- Fperm[, seq_len(comps_tested), drop = FALSE]
          n_ok  <- n_ok[seq_len(comps_tested)]
          obs_stats <- obs_stats[seq_len(comps_tested)]
          break # Stop testing further components
      }
      comps_tested <- k # Record that this component was tested
  }# end loop over components
  
  # Rename obs_vec to obs_stats to avoid clash
  obs_vec <- obs_stats 

  ## ------------------------------------------------------------------
  ## 6.  tidy return object  (same shape as perm_test.pca())
  ## ------------------------------------------------------------------
  comp_df <- data.frame(comp      = seq_along(obs_vec),
                        observed  = obs_vec,
                        pval      = sapply(seq_along(obs_vec), function(j){
                                         g <- sum(Fperm[,j] >= obs_vec[j], na.rm = TRUE)
                                         (g + 1)/(n_ok[j] + 1)}),
                        lower_ci  = apply(Fperm, 2, quantile, 0.025, na.rm = TRUE),
                        upper_ci  = apply(Fperm, 2, quantile, 0.975, na.rm = TRUE))

  structure(
     list(call              = match.call(),
          component_results = comp_df,
          perm_values       = Fperm,
          alpha             = alpha,
          alternative       = alternative,
          method            = sprintf(
              "Permutation test for multiblock consensus (stat = leading-eigenvalue, %d blocks)", B),
          nperm             = n_ok),
     class = c("perm_test_multiblock","perm_test")
  )
}



#' @export
perm_test.multiblock_projector <- function(x,
                                           Xlist,
                                           nperm       = 500,
                                           comps       = 4,
                                           shuffle_fun = NULL,
                                           measure_fun = NULL,
                                           parallel    = FALSE,
                                           alternative = c("greater", "less",
                                                           "two.sided"),
                                           alpha       = 0.05,
                                           ...) {

  ## ---------- safety checks ----------
  if (is.null(Xlist)) 
    stop("Xlist is required for perm_test.multiblock_projector")
  if (!is.list(Xlist))
    stop("Xlist must be a list of block matrices.")

  B <- nblocks(x)
  if (length(Xlist) != B)
    stop("Length(Xlist) (", length(Xlist),
         ") differs from nblocks(x) (", B, ").")

  blk_idx <- block_indices(x)
  p_each  <- sapply(blk_idx, length)
  for (b in seq_len(B))
    if (ncol(Xlist[[b]]) != p_each[b])
      stop("Block ", b, ": #cols in Xlist (", ncol(Xlist[[b]]), 
           ") does not match block_indices (", p_each[b], ").")

  N <- nrow(Xlist[[1]])
  if (any(sapply(Xlist, nrow) != N))
    stop("All blocks must have the same number of rows.")

  Kmax <- min(comps, ncol(x$v))
  alternative <- match.arg(alternative)

  ## ---------- helpers ----------
  # block-specific preprocessing (reuse x$preproc) ----
  prep_all <- function(Xl) {
    # Concatenate, apply transform to full matrix, then split back
    p_all <- sum(p_each)
    Xall <- matrix(0.0, nrow=N, ncol=p_all) # Pre-allocate
    cidx <- 1
    for(b in 1:B) {
      Xall[, cidx:(cidx+p_each[b]-1)] <- Xl[[b]]
      cidx <- cidx + p_each[b]
    }
  Xproc_all <- transform(x$preproc, Xall)
    
    # Split back into list
    Xp_list <- vector("list", B)
    cidx <- 1
    for(b in 1:B) {
      Xp_list[[b]] <- Xproc_all[, cidx:(cidx+p_each[b]-1), drop=FALSE]
      cidx <- cidx + p_each[b]
    }
    Xp_list
  }

  # compute block scores for first K comps using original model 'x' ----
  get_block_scores <- function(Xp_list, K) {
    v      <- coef(x)        # Use original model's loadings (p_tot x d)
    vlist  <- lapply(seq_len(B), function(b)
                     v[ blk_idx[[b]], 1:K, drop = FALSE ])
    mapply(`%*%`, Xp_list, vlist, SIMPLIFY = FALSE)  # list of (N x K) matrices
  }

  # default statistic: mean |corr| across block pairs ----
  default_measure <- function(Scores_list, K) {
    # Scores_list is a list (length B) of (N x K) score matrices
    out <- numeric(K)
    if (B < 2) return(rep(NA_real_, K)) # Correlation requires at least 2 blocks
    for (k in seq_len(K)) {
      corrs <- combn(B, 2, FUN = function(idx) {
        # Ensure scores are vectors for cor()
        score1 <- Scores_list[[idx[1]]][, k]
        score2 <- Scores_list[[idx[2]]][, k]
        if (stats::var(score1) < .Machine$double.eps || stats::var(score2) < .Machine$double.eps) {
            # Return NA or 0 if variance is zero to avoid cor error/NA
            return(NA_real_) 
        } else {
            stats::cor(score1, score2)
        }
      })
      # Handle potential NAs from zero variance columns
      out[k] <- mean(abs(corrs), na.rm = TRUE)
      if (is.nan(out[k])) out[k] <- 0 # If all pairs had zero variance, result is NaN -> 0
    }
    out
  }

  ## ---------- defaults for shuffle / fit / measure ----------
  if (is.null(shuffle_fun))
    # Use the standardized helper name
    shuffle_fun <- .default_multiblock_shuffle
    # Original shuffle_fun <- function(Blocks, ...) lapply(Blocks, function(M) M[sample(N), , drop = FALSE])

  if (is.null(measure_fun))
    measure_fun <- default_measure

  ## ---------- observed statistic ----------
  Xp_obs <- prep_all(Xlist) # Preprocess original data
  Scores_obs <- get_block_scores(Xp_obs, Kmax) # Project using original model
  T_obs <- measure_fun(Scores_obs, Kmax, ...) # Measure consensus on observed scores

  ## ---------- permutation loop ----------
  worker <- function(iter, ...) {
    Xperm <- shuffle_fun(Xlist, ...) # Shuffle original data
    Xp    <- prep_all(Xperm)         # Preprocess shuffled data
    
    # Project shuffled preprocessed data using original model 'x'
    Scores_perm <- get_block_scores(Xp, Kmax) 
    
    # Measure consensus on permuted scores
    stat_perm <- measure_fun(Scores_perm, Kmax, ...)
    if (length(stat_perm) != Kmax) {
        warning(sprintf("Permutation %d: measure_fun returned %d values, expected %d. Filling with NA.", iter, length(stat_perm), Kmax))
        stat_perm_full <- rep(NA_real_, Kmax)
        valid_len <- min(length(stat_perm), Kmax)
        stat_perm_full[1:valid_len] <- stat_perm[1:valid_len]
        return(stat_perm_full)
    }
    stat_perm
  }

  applyFUN <- if (parallel) future.apply::future_lapply else lapply
  perm_mat <- do.call(rbind,
                      applyFUN(seq_len(nperm), worker, ...))
  ok       <- stats::complete.cases(perm_mat)
  if (!all(ok)) {
    warning(sum(!ok), " permutations failed; removed.")
    perm_mat <- perm_mat[ok, , drop = FALSE]
  }
  n_ok <- nrow(perm_mat)
  if (n_ok == 0) stop("All permutations failed.")

  ## ---------- p-values & sequential stop ----------
  keep   <- seq_len(Kmax)
  pvals  <- rep(NA_real_, Kmax)
  stopat <- Kmax
  for (k in keep) {
    if (alternative == "greater")
      pvals[k] <- (sum(perm_mat[, k] >= T_obs[k]) + 1) / (n_ok + 1)
    else if (alternative == "less")
      pvals[k] <- (sum(perm_mat[, k] <= T_obs[k]) + 1) / (n_ok + 1)
    else {          # two-sided
      p_high <- (sum(perm_mat[, k] >= T_obs[k]) + 1) / (n_ok + 1)
      p_low  <- (sum(perm_mat[, k] <= T_obs[k]) + 1) / (n_ok + 1)
      pvals[k] <- 2 * min(p_high, p_low)
    }
    if (pvals[k] > alpha) { stopat <- k; break }
  }

  keep <- seq_len(stopat)

  ci <- t(apply(perm_mat[, keep, drop = FALSE], 2,
                stats::quantile, probs = c(.025, .975)))

  comp_res <- data.frame(comp      = keep,
                         observed  = T_obs[keep],
                         p.value   = pvals[keep],
                         lower_ci  = ci[, 1],
                         upper_ci  = ci[, 2])

  structure(list(call              = match.call(),
                 component_results = comp_res,
                 perm_values       = perm_mat[, keep, drop = FALSE],
                 method            = "Permutation test for multiblock_projector (score-based consensus)",
                 nperm             = n_ok),
            class = c("perm_test_multiblock", "perm_test")
  )
}
