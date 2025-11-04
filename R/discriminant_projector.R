#' Construct a Discriminant Projector
#'
#' A `discriminant_projector` is an instance that extends `bi_projector` with a projection that maximizes class separation.
#' This can be useful for dimensionality reduction techniques that take class labels into account, such as Linear Discriminant Analysis (LDA).
#'
#' @param v The projection matrix (often `X %*% v`). Rows correspond to observations, columns to components.
#' @param s The score matrix (often `X %*% v`). Rows correspond to observations, columns to components.
#' @param sdev The standard deviations associated with the scores or components (e.g., singular values from LDA).
#' @param preproc A `prepper` or `pre_processor` object, or a pre-processing function (e.g., `center`, `pass`).
#' @param labels A factor or character vector of class labels corresponding to the rows of `X` (and `s`).
#' @param classes Additional S3 classes to prepend.
#' @param ... Extra arguments passed to `bi_projector`.
#' @return A `discriminant_projector` object.
#'
#' @seealso bi_projector
#' @importFrom stats setNames
#' @export
#' @examples
#' # Simulate data and labels
#' set.seed(123)
#' X <- matrix(rnorm(100 * 10), 100, 10)
#' labels <- factor(rep(1:2, each = 50))
#'
#' # Perform LDA and create a discriminant projector
#' lda_fit <- MASS::lda(X, labels)
#'
#' dp <- discriminant_projector(lda_fit$scaling, X %*% lda_fit$scaling, sdev = lda_fit$svd, 
#' labels = labels)
#' @export
discriminant_projector <- function(v, s, sdev, preproc=prep(pass()), labels, classes=NULL, ...) {
  
  chk::vld_matrix(v)
  chk::vld_matrix(s)
  chk::vld_numeric(sdev)
  chk::chk_equal(length(sdev), ncol(s))
  chk::chk_equal(ncol(v), length(sdev))
  chk::chk_equal(length(labels), nrow(s))
  
  # Ensure labels are factor; preserve level order if already factor
  labels <- if (is.factor(labels)) factor(labels, levels = levels(labels))
            else factor(labels)
  counts <- table(labels, dnn = NULL)
  
  out <- bi_projector(v, s=s, sdev=sdev, preproc=preproc, labels=labels, 
                      counts=counts, classes=c(classes, "discriminant_projector"), ...)
}

#' Predict method for a discriminant_projector, supporting LDA or Euclid
#'
#' This produces class predictions or posterior-like scores for new data. We first
#' project the data into the subspace defined by \code{x$v}, then either:
#' \enumerate{
#'   \item \strong{LDA approach} (\code{method="lda"}), which uses a (simplified)
#'         linear discriminant formula or distance to class means in the subspace
#'         combined with prior probabilities. 
#'   \item \strong{Euclid approach} (\code{method="euclid"}), which uses plain
#'         Euclidean distance to each class mean in the subspace.
#' }
#' We return either a \code{type="class"} label or \code{type="prob"} posterior-like
#' matrix.
#'
#' @param object A \code{discriminant_projector} object (extending \code{bi_projector}),
#'   which has \code{x$v} for column loadings, \code{x$s} for row scores, and
#'   \code{x$labels} for class labels.
#' @param new_data A numeric matrix (or vector) with the same # of columns as
#'   the original data (unless partial). Rows=observations, columns=features.
#' @param method Either \code{"lda"} (the default discriminant approach) or
#'   \code{"euclid"} (simple nearest-mean in subspace).
#' @param type \code{"class"} (default) for predicted class labels or \code{"prob"}
#'   for posterior-like probabilities. 
#' @param ... further arguments (not used or for future expansions).
#'
#' @return 
#' If \code{type="class"}, a factor vector of length n (predicted classes).
#' If \code{type="prob"}, an (n x #classes) numeric matrix of posterior-like values, with row names matching \code{new_data} if available.
#'
#' Predict method for a discriminant_projector
#'
#' This produces class predictions or posterior-like scores for new data, based on:
#' \itemize{
#'   \item \strong{LDA approach} (\code{method="lda"}), which uses a linear discriminant
#'         formula with a pooled covariance matrix if \code{x\$Sigma} is given, or
#'         the identity matrix if \code{Sigma=NULL}. If that covariance matrix is
#'         not invertible, a pseudo-inverse is used and a warning is emitted.
#'   \item \strong{Euclid approach} (\code{method="euclid"}), which uses plain
#'         Euclidean distance to each class mean in the subspace.
#' }
#'
#' We return either a \code{type="class"} label or \code{type="prob"} posterior-like
#' matrix.
#'
#' @param object A \code{discriminant_projector} object.
#' @param new_data A numeric matrix (or vector) with the same # of columns as
#'   the original data (unless partial usage). Rows=observations, columns=features.
#' @param method Either \code{"lda"} (the default) or \code{"euclid"} (nearest-mean).
#' @param type \code{"class"} (default) for predicted class labels, or \code{"prob"}
#'   for posterior-like probabilities. 
#' @param colind (optional) if partial columns are used, specify which columns 
#'   map to the subspace. If \code{NULL}, assume full columns. 
#' @param ... further arguments (not used or for future expansions).
#'
#' @return 
#' If \code{type="class"}, a factor vector of length n (predicted classes).
#' If \code{type="prob"}, an (n x #classes) numeric matrix of posterior-like values.
#' @importFrom stats median
#' @exportS3Method predict discriminant_projector
#' @export
predict.discriminant_projector <- function(object,
                                           new_data,
                                           method = c("lda", "euclid"),
                                           type   = c("class", "prob"),
                                           colind = NULL,
                                           ...) 
{
  method <- match.arg(method)
  type   <- match.arg(type)
  
  # 1) Reprocess new_data, possibly partial columns if colind is set
  #    Note that 'reprocess' can handle colind argument if partial usage is supported
  if (is.null(colind)) {
    new_data_proc <- reprocess(object, new_data) 
  } else {
    new_data_proc <- reprocess(object, new_data, colind = colind)
  }
  
  # ensure new_data_proc is at least 2D
  if (is.vector(new_data_proc)) {
    new_data_proc <- matrix(new_data_proc, nrow = 1)
  }
  
  # Handle potential colind subsetting for v
  if (!is.null(colind)) {
      if (max(colind) > nrow(object$v) || min(colind) < 1) {
           stop(sprintf("Invalid 'colind' provided. Indices must be between 1 and %d.", nrow(object$v)))
      }
      v_use <- object$v[colind, , drop = FALSE]
  } else {
      v_use <- object$v
  }
  
  # multiply by object$v (possibly subsetted) => subspace
  scores_new <- new_data_proc %*% v_use   # shape: (n x d)
  
  # 2) Extract class info from training
  labs         <- object$labels
  class_levels <- levels(as.factor(labs))
  nclass       <- length(class_levels)
  
  # 3) Compute class means in subspace from object$s
  #    object$s has shape (n_train x d)
  class_means <- matrix(NA, nrow = nclass, ncol = ncol(object$s))
  for (i in seq_len(nclass)) {
    rows <- which(labs == class_levels[i])
    class_means[i, ] <- colMeans(object$s[rows, , drop = FALSE])
  }
  
  # 4) If we want real LDA, we can use stored Sigma or identity if not present
  if (method == "lda") {
    sigma_pooled <- object$Sigma
    subspace_dim <- ncol(object$s)
    if (is.null(sigma_pooled)) {
      sigma_pooled <- diag(subspace_dim)
    } else {
      sigma_mat <- if (inherits(sigma_pooled, "Matrix")) as.matrix(sigma_pooled) else as.matrix(sigma_pooled)
      if (ncol(sigma_mat) != subspace_dim) {
        v_sigma <- if (is.null(colind)) object$v else object$v[colind, , drop = FALSE]
        if (nrow(v_sigma) == ncol(sigma_mat)) {
          sigma_mat <- crossprod(v_sigma, sigma_mat %*% v_sigma)
        } else {
          warning("Stored covariance has incompatible dimension; using identity in score space.")
          sigma_mat <- diag(subspace_dim)
        }
      }
      sigma_pooled <- 0.5 * (sigma_mat + t(sigma_mat))
    }
    inv_sigma <- tryCatch(
      solve(sigma_pooled),
      error = function(e) {
        warning("Covariance matrix not invertible; attempting pseudo-inverse via MASS::ginv")
        if (!requireNamespace("MASS", quietly = TRUE)) {
          stop("Covariance matrix not invertible and package 'MASS' is unavailable for pseudo-inverse computation.", call. = FALSE)
        }
        MASS::ginv(sigma_pooled)
      }
    )
    
    # Precompute (for linear discriminants):
    means_inv  <- class_means %*% inv_sigma
    means_quad <- rowSums(means_inv * class_means)
    logpri     <- log(object$counts / sum(object$counts))
    
    # Build discriminant matrix
    disc_mat <- matrix(0, nrow = nrow(scores_new), ncol = nclass)
    for (i in seq_len(nclass)) {
      dotvals      <- scores_new %*% means_inv[i, ] 
      disc_mat[,i] <- dotvals - 0.5 * means_quad[i] + logpri[i]
    }
    
    if (type == "class") {
      best_idx <- max.col(disc_mat, ties.method = "first")
      return(factor(class_levels[best_idx], levels = class_levels))
    } else {
      # Softmax rowwise
      if (requireNamespace("matrixStats", quietly = TRUE)) {
        row_max <- matrixStats::rowMaxs(disc_mat)
      } else {
        row_max <- apply(disc_mat, 1, max)
      }
      disc_exp <- exp(sweep(disc_mat, 1, row_max, "-"))
      prob_mat <- disc_exp / rowSums(disc_exp)
      colnames(prob_mat) <- class_levels
      rownames(prob_mat) <- rownames(new_data_proc) # Add row names
      return(prob_mat)
    }
    
  } else {
    # method = "euclid"
    # nearest-mean in subspace
    dist_mat <- matrix(NA, nrow = nrow(scores_new), ncol = nclass)
    for (i in seq_len(nclass)) {
      diff_ij      <- sweep(scores_new, 2, class_means[i, ], FUN = "-")
      dist_mat[,i] <- rowSums(diff_ij^2)
    }
    
    if (type == "class") {
      # pick argmin distance
      best_idx <- max.col(-dist_mat, ties.method = "first")
      return(factor(class_levels[best_idx], levels = class_levels))
    } else {
      # posterior-like = normalized inverse distance
      # Use relative epsilon based on median distance to avoid Inf for zero distances
      eps <- 1e-6 * stats::median(dist_mat[dist_mat > 0], na.rm = TRUE) 
      if (!is.finite(eps) || eps <= 0) {
         eps <- 1e-8 # Fallback if median is non-positive or NaN/Inf
      }
      
      inv_dist <- 1 / (dist_mat + eps)
      prob_mat <- inv_dist / rowSums(inv_dist)
      colnames(prob_mat) <- class_levels
      rownames(prob_mat) <- rownames(new_data_proc) # Add row names
      return(prob_mat)
    }
  }
}



#' @importFrom stats quantile predict na.omit
#' @importFrom future.apply future_lapply
#' @export
#'@family perm_test
#' @seealso \code{\link{perm_test}} for the generic permutation test function
perm_test.discriminant_projector <- function(
    x, X,
    nperm = 1000,
    measure_fun = NULL, 
    fit_fun = NULL,
    shuffle_fun = NULL,
    predict_method = c("lda", "euclid"), # Added explicit argument
    parallel = FALSE,
    alternative = c("greater", "less", "two.sided"),
    ...) {
  
  # Match arguments
  alternative <- match.arg(alternative)
  predict_method <- match.arg(predict_method)

  
  # Ensure labels are factor and dimensions match
  y <- factor(x$labels)
  if (nrow(X) != length(y)) {
      stop(sprintf("Number of rows in X (%d) does not match number of labels in x (%d).", nrow(X), length(y)))
  }
  
  # Capture extra arguments to pass down
  extra_args <- list(...)
  
  # ---------- Default Measure Function (Accuracy) ----------
  if (is.null(measure_fun)) {
    measure_fun <- function(model_perm, X_orig, y_perm, predict_method, ...) {
      # Predict using the *permuted* model on original X
      pred_class <- predict(model_perm, X_orig, method = predict_method, type="class")
      
      # Compare predictions to the *permuted* labels (y_perm)
      valid_idx <- !is.na(y_perm)
      if (any(!valid_idx)) {
          y_perm <- y_perm[valid_idx]
          pred_class <- pred_class[valid_idx]
          if (length(y_perm) == 0) return(NA_real_) # Return NA if no valid data
      }
      # Return only the numeric accuracy value
      sum(pred_class == y_perm) / length(y_perm)
    }
    stat_name <- "accuracy (default)"
  } else {
    stat_name <- deparse(substitute(measure_fun))
  }
  
  # ---------- Default Shuffle Function ----------
  if (is.null(shuffle_fun)) {
    shuffle_fun <- function(labels, ...) sample(labels)
  }
  
  # ---------- Default Fit Function (Re-uses original preprocessing & MASS::lda) ----------
  if (is.null(fit_fun)) {
    fit_fun <- function(Xtrain_orig, labelstrain, preproc_obj, ...) {
      # Fix: Use apply_transform instead of reprocess, as preproc_obj is the processor, not the model
      Xtrain_proc <- try(transform(preproc_obj, Xtrain_orig), silent = TRUE)
      if (inherits(Xtrain_proc, "try-error")) {
        # Improved error message
        stop(sprintf("Failed to apply original preprocessor transform in default fit_fun: %s", Xtrain_proc))
      }
      if (!requireNamespace("MASS", quietly = TRUE)) {
        stop("Package 'MASS' needed for default fit_fun. Please install it.", call. = FALSE)
      }
      # Pass ... down to lda if provided
      lda_args <- c(list(x=Xtrain_proc, grouping = labelstrain), list(...))
      lda_fit <- do.call(MASS::lda, lda_args)
      s <- Xtrain_proc %*% lda_fit$scaling
      discriminant_projector(
        v = lda_fit$scaling, s = s, sdev = lda_fit$svd,
        labels = labelstrain, preproc = preproc_obj, Sigma = lda_fit$covariance
      )
    }
  }
  
  # ---------- Observed Statistic ----------
  # Evaluate the original model on the original data with original labels
  # The measure_fun signature expects permuted labels, but for observed stat,
  # we compare model predictions on X to the true labels y.
  # We adapt the call slightly for the observed statistic calculation.
  obs_pred <- predict(x, X, method = predict_method, type = "class")
  valid_obs_idx <- !is.na(y)
  if (any(!valid_obs_idx)) {
      y_valid <- y[valid_obs_idx]
      obs_pred_valid <- obs_pred[valid_obs_idx]
      if (length(y_valid) == 0) {
          obs_stat <- NA_real_
      } else {
          obs_stat <- sum(obs_pred_valid == y_valid) / length(y_valid)
      }
  } else {
      obs_stat <- sum(obs_pred == y) / length(y)
  }
  
  if (!is.numeric(obs_stat) || length(obs_stat) != 1) {
      # This case should primarily happen if measure_fun returns NA
      # or if user provided a measure_fun that returned non-numeric/vector
      if (is.na(obs_stat)) {
          warning("Observed statistic could not be calculated (likely due to NAs). Permutation test cannot proceed meaningfully.")
      } else {
          stop("Internal error or custom measure_fun: Observed statistic is not a single numeric value.")
      }
  }
  
  # ---------- Permutation Loop Function ----------
  one_perm <- function(i, ...) {
    # Create arguments list for shuffle_fun, including extra_args
    shuffle_args <- c(list(labels = y), extra_args)
    y_perm <- do.call(shuffle_fun, shuffle_args)
    
    # Create arguments list for fit_fun, including extra_args
    # Pass original preprocessor object from 'x' explicitly
    fit_args <- c(list(Xtrain_orig = X, labelstrain = y_perm, preproc_obj = x$preproc), extra_args)
    mod <- try(do.call(fit_fun, fit_args), silent = TRUE)
    if (inherits(mod, "try-error")) {
        warning(sprintf("Permutation %d: fit_fun failed: %s. Returning NA.", i, mod))
        return(NA_real_)
    }
    
    # Create arguments list for measure_fun, including predict_method and extra_args
    measure_args <- c(list(model_perm = mod, X_orig = X, y_perm = y_perm, 
                           predict_method = predict_method), extra_args)
    stat_perm <- try(do.call(measure_fun, measure_args), silent = TRUE)
    
    if (inherits(stat_perm, "try-error")) {
        warning(sprintf("Permutation %d: measure_fun failed: %s. Returning NA.", i, stat_perm))
        return(NA_real_)
    }
    if (!is.numeric(stat_perm) || length(stat_perm) != 1) {
        warning(sprintf("Permutation %d: measure_fun did not return a single numeric value. Returning NA.", i))
        return(NA_real_)
    }
    stat_perm
  }
  
  # ---------- Run Permutations (Serial or Parallel) ----------
  message(sprintf("Running %d permutations for discriminant projector (%s)...", 
                  nperm, if(parallel) "parallel" else "serial"))
  apply_fun <- if (parallel) future.apply::future_lapply else lapply
  perm_args <- list(X = seq_len(nperm), FUN = one_perm)
  # Pass ... down to one_perm via the lapply function's ...
  if (parallel) perm_args$future.seed <- TRUE
  perm_args <- c(perm_args, extra_args) # Add ... to the apply_fun call
  
  perm_vals_list <- do.call(apply_fun, perm_args)
  perm_vals <- unlist(perm_vals_list)
  n_complete <- sum(!is.na(perm_vals))
  
  if (n_complete < nperm) {
      warning(sprintf("%d permutations failed and were excluded.", nperm - n_complete))
      perm_vals <- stats::na.omit(perm_vals)
  }
  if (n_complete == 0) {
      stop("All permutations failed. Cannot compute p-value.")
  }
  
  # ---------- P-value Calculation (Empirical with +1 smoothing) ----------
  if (alternative == "greater") {
    b <- sum(perm_vals >= obs_stat, na.rm = TRUE) # Added na.rm for safety
    pval <- (b + 1) / (n_complete + 1)
  } else if (alternative == "less") {
    b <- sum(perm_vals <= obs_stat, na.rm = TRUE)
    pval <- (b + 1) / (n_complete + 1)
  } else { # two.sided
    b_greater <- sum(perm_vals >= obs_stat, na.rm = TRUE)
    b_less <- sum(perm_vals <= obs_stat, na.rm = TRUE)
    pval <- 2 * min((b_greater + 1) / (n_complete + 1), (b_less + 1) / (n_complete + 1))
    pval <- min(pval, 1.0) # Ensure p-value doesn't exceed 1
  }
  
  # ---------- Output Structure ----------
  out <- structure(
    list(
      statistic = obs_stat,
      perm_values = perm_vals,
      p.value = pval,
      alternative = alternative,
      method = sprintf("Permutation test for discriminant_projector (measure: %s)", stat_name),
      nperm = n_complete, 
      call = match.call()
    ),
    class = "perm_test"
  )
  
  out
}

#' Print Method for perm_test Objects
#' 
#' Provides a concise summary of the permutation test results.
#' 
#' @param x An object of class `perm_test`.
#' @param ... Additional arguments passed to printing methods.
#' @return Invisibly returns the input object `x`.
#' @export
print.perm_test <- function(x, ...) {
  cat("\nPermutation Test Results\n\n")
  cat("Method: ", x$method, "\n")
  cat("Alternative: ", x$alternative, "\n")
  cat(sprintf("Observed Statistic = %.4g\n", x$statistic))
  cat(sprintf("Empirical P-value = %.4g (based on %d successful permutations)\n", 
              x$p.value, x$nperm))
  
  # Optionally show CI
  if (!is.null(x$perm_values) && length(x$perm_values) > 1) {
      ci <- stats::quantile(x$perm_values, c(0.025, 0.975), na.rm = TRUE)
      cat(sprintf("95%% CI for Permutation Stats: [%.4g, %.4g]\n", ci[1], ci[2]))
  }
  cat("\n")
  invisible(x)
}

#' @export
print.discriminant_projector <- function(x,...) {
  print.projector(x)
  # Print named counts
  cat("Label counts: \n")
  print(x$counts) 
}
