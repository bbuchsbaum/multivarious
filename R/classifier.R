#' Multiblock Bi-Projector Classifier
#'
#' Constructs a k-Nearest Neighbors (k-NN) classifier based on a fitted
#' `multiblock_biprojector` model object. The classifier uses the projected scores
#' as the feature space for k-NN.
#'
#' Users can specify whether to use the globally projected scores stored within the model
#' (`global_scores = TRUE`) or to generate reference scores by projecting provided `new_data`
#' (`global_scores = FALSE`). Partial projections based on `colind` or `block` can be used
#' when `global_scores = FALSE` or when `new_data` is provided alongside `colind`/`block`.
#' Prediction behavior is further controlled by arguments passed to `predict.classifier`.
#'
#' @param x A fitted `multiblock_biprojector` object.
#' @param colind An optional numeric vector specifying column indices from the original data space.
#'   If provided when `global_scores=FALSE`, these indices are used to perform a partial projection for the reference scores.
#'   If provided when `global_scores=TRUE`, this value is stored but does not affect the
#'   reference scores (which remain global); however, it may influence the default projection
#'   behavior during prediction unless overridden there. See `predict.classifier`.
#' @param labels A factor or vector of class labels for the training data.
#' @param new_data An optional data matrix used to generate reference scores when `global_scores=FALSE`,
#'   or when `global_scores=TRUE` but `colind` or `block` is also provided (overriding `global_scores`).
#'   Must be provided if `global_scores=FALSE`.
#' @param block An optional integer specifying a predefined block index.
#'   Used for partial projection if `global_scores=FALSE` or if `new_data` is also provided.
#'   Cannot be used simultaneously with `colind`.
#' @param knn The integer number of nearest neighbors (k) for the k-NN algorithm (default: 1).
#' @param global_scores Logical. **DEPRECATED** This argument is deprecated and its behavior has changed.
#'   Reference scores are now determined automatically:
#'   - If `new_data` is NULL: Uses the globally projected scores stored in `x` (`scores(x)`).
#'   - If `new_data` is provided: Always projects `new_data` to generate reference scores
#'     (using `partial_project`/`project_block` if `colind`/`block` are given, `project` otherwise).
#' @param ... Additional arguments (currently ignored).
#' @return An object of class `multiblock_classifier`, which also inherits from `classifier`.
#' @export 
#' @family classifier
classifier.multiblock_biprojector <- function(x, colind=NULL, labels, new_data=NULL,
                                              block=NULL, global_scores=TRUE, knn=1,...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_numeric(colind)
    chk::chk_true(all(colind == as.integer(colind)))
    chk::chk_true(all(colind > 0))
    if (!is.null(block)) {
      rlang::abort("can either supply `colind` or `block` but not both")
    }
  }
  
  # Check knn
  if (!is.numeric(knn) || knn < 1 || knn != as.integer(knn)) {
    stop("knn must be a positive integer")
  }
  
  scores_ref <- NULL # reference scores
  
  # Fix 3: Revised logic for reference scores
  if (is.null(new_data)) {
    # Case 1: No new_data provided -> Use global scores from projector
    scores_ref <- scores(x)
    # Ensure labels match the number of global scores
    if (length(labels) != nrow(scores_ref)) {
      stop("Length of labels does not match the number of rows in scores(x). Provide new_data or ensure labels match existing scores.")
    }
  } else {
    # Case 2: new_data IS provided -> Always project new_data
    # Ensure labels match the provided new_data
    chk::chk_equal(length(labels), nrow(new_data))
    
    # Project new_data to get reference scores.
    # Use partial projection if colind/block are given, otherwise global.
    if (!is.null(colind)) {
      scores_ref <- partial_project(x, new_data, colind = colind)
    } else if (!is.null(block)) {
      chk::chk_whole_number(block)
      scores_ref <- project_block(x, new_data, block = block)
    } else {
      scores_ref <- project(x, new_data)
    }
  }
  
  # Deprecate global_scores argument (no longer used in logic)
  if (!missing(global_scores) && !is.null(global_scores)) {
     warning("'global_scores' argument is deprecated and ignored. Reference scores are now determined automatically based on whether 'new_data' is provided.")
  }

  new_classifier(x, labels=labels, scores=scores_ref, colind=colind, block=block, knn=knn, # global_scores=global_scores,
                 classes="multiblock_classifier")
}


#' Create a k-NN classifier for a discriminant projector
#'
#' @param x the discriminant projector object
#' @param colind an optional vector specifying the column indices of the components
#' @param knn the number of nearest neighbors (default=1)
#' @param ... extra arguments
#' @return a classifier object
#' @export
#' @examples
#' # Assume dp is a fitted discriminant_projector object
#' # classifier(dp, knn = 5) # Basic example
classifier.discriminant_projector <- function(x, colind=NULL, knn=1,...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_numeric(colind)
    chk::chk_true(all(colind == as.integer(colind)))
    chk::chk_true(all(colind > 0))
  }
  if (!is.numeric(knn) || knn < 1 || knn != as.integer(knn)) {
    stop("knn must be a positive integer")
  }
  
  new_classifier(x, x$labels, scores(x), colind=colind, knn=knn)
}


#' Create a new k-NN classifier
#'
#' @param x the model fit
#' @param labels class labels
#' @param scores scores used for classification
#' @param colind optional component indices
#' @param knn number of nearest neighbors
#' @param classes additional S3 classes
#' @param ... extra args
#' @keywords internal
#' @noRd
new_classifier <- function(x, labels, scores, colind=NULL, knn=1, classes=NULL, ...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_numeric(colind)
    chk::chk_true(all(colind == as.integer(colind)))
    chk::chk_true(all(colind > 0))
  }

  if (!is.numeric(knn) || knn < 1 || knn != as.integer(knn)) {
    stop("knn must be a positive integer")
  }
  
  if (knn > nrow(scores)) { # Check against reference scores
    stop("knn cannot exceed the number of training cases in scores")
  }
  
  # C1: Coerce labels to factor
  if (!is.factor(labels)) {
      labels <- factor(labels)
  }

  chk::chk_equal(length(labels), nrow(scores))

  structure(
    c(
      list(...), # Put ... first to prevent overwriting named entries
      list(
        projector=x,
        labels=labels,
        scores=scores,
        colind=colind,
        knn=knn
      )
    ),
    class=c(classes, "classifier")
  )
}


#' Create a random forest classifier
#' 
#' Uses `randomForest` to train a random forest on the provided scores and labels.
#'
#' @param x a projector object
#' @param colind optional col indices
#' @param labels class labels
#' @param scores reference scores
#' @param ... passed to `randomForest`
#' @export
#' @return a `rf_classifier` object with rfres (rf model), labels, scores
#' @family classifier
#' @seealso \code{\link[randomForest]{randomForest}}
#' @examples
#' # Assume proj is a fitted projector object
#' # Assume lbls are labels and sc are scores
#' # if (requireNamespace("randomForest", quietly = TRUE)) {
#' #   rf_classifier(proj, labels = lbls, scores = sc)
#' # }
rf_classifier.projector <- function(x, colind=NULL, labels, scores, ...) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Please install package 'randomForest' for 'rf_classifier'")
  }

  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_numeric(colind)
    chk::chk_true(all(colind == as.integer(colind)))
    chk::chk_true(all(colind > 0))
  }
  
  # C1: Coerce labels to factor (if not already factor)
  if (!is.factor(labels)) {
    labels <- factor(labels)
  }
  
  chk::chk_equal(length(labels), nrow(scores))
  
  # C4: Convert scores matrix to data.frame for randomForest
  scores_df <- as.data.frame(scores, stringsAsFactors = FALSE)
  if (!is.null(colnames(scores))) {
    colnames(scores_df) <- make.names(colnames(scores)) # Ensure valid names
  } else {
    colnames(scores_df) <- paste0("Comp", 1:ncol(scores_df))
  }
  
  rfres <- randomForest::randomForest(scores_df, labels, ...) # Pass ... here
  
  # Store rf variable importance if needed
  imp <- NULL
  if ("importance" %in% names(rfres)) {
    imp <- rfres$importance
  }
  
  structure(
    list(
      projector=x,
      rfres=rfres,
      labels=labels,
      scores=scores,
      feat_names=colnames(scores_df), # Store sanitized feature names
      importance=imp,
      colind=colind
      # Do not pass ... here unless intended for storage
      ),
    class=c("rf_classifier", "classifier")
  )
}


#' create classifier from a projector
#'
#' @param x projector
#' @param colind ...
#' @param labels ...
#' @param new_data ...
#' @param knn ...
#' @param global_scores ...
#' @param ... extra args
#' @rdname classifier
#' @export
#' @family classifier
#' @examples
#' # Assume proj is a fitted projector object
#' # Assume lbls are labels and dat is new data
#' # classifier(proj, labels = lbls, new_data = dat, knn = 3)
classifier.projector <- function(x, colind=NULL, labels, new_data=NULL, knn=1, global_scores=TRUE, ...) {
  # Deprecate global_scores argument
  if (!missing(global_scores) && !is.null(global_scores)) {
     warning("'global_scores' argument is deprecated and ignored. Reference scores are now determined automatically based on whether 'new_data' is provided.")
  }

  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_numeric(colind)
    chk::chk_true(all(colind == as.integer(colind)))
    chk::chk_true(all(colind > 0))
  }
  
  if (!is.numeric(knn) || knn < 1 || knn != as.integer(knn)) {
    stop("knn must be a positive integer")
  }
  
  # Fix 3: Revised logic for reference scores
  scores_ref <- NULL
  if (is.null(new_data)) {
    # Case 1: No new_data -> Use global scores from projector if they exist
    if (!is.null(scores(x))) {
        scores_ref <- scores(x)
        # Ensure labels match global scores
        chk::chk_equal(length(labels), nrow(scores_ref))
    } else {
        stop("If new_data is NULL, the projector 'x' must contain scores (e.g., be a bi_projector). Alternatively, provide new_data.")
    }
  } else {
    # Case 2: new_data provided -> Always project new_data
    chk::chk_equal(length(labels), nrow(new_data))
    if (!is.null(colind)) {
      # Use partial projection if colind is given
      scores_ref <- partial_project(x, new_data, colind = colind)
    } else {
      # Use global projection otherwise
      scores_ref <- project(x, new_data)
    }
  }
  
  new_classifier(x, labels, scores_ref, colind, knn, ...)
}


#' Calculate Rank Score for Predictions
#'
#' Computes the rank score (normalized rank of the true class probability) for each observation.
#' Lower rank scores indicate better predictions (true class has higher probability).
#'
#' @param prob Numeric matrix of predicted probabilities (observations x classes).
#'   Column names must correspond to class labels.
#' @param observed Factor or vector of observed class labels. Must be present in `colnames(prob)`.
#' @return A `data.frame` with columns `prank` (the normalized rank score) and `observed` (the input labels).
#' @export
#' @family classifier evaluation
#' @examples
#' probs <- matrix(c(0.1, 0.9, 0.8, 0.2), 2, 2, byrow=TRUE,
#'                dimnames = list(NULL, c("A", "B")))
#' obs <- factor(c("B", "A"))
#' rank_score(probs, obs)
rank_score <- function(prob, observed) {
  pnames <- colnames(prob)
  # B3: Check if pnames is NULL (can happen if prob matrix lacks colnames)
  if (is.null(pnames)) {
      stop("Input probability matrix 'prob' must have column names corresponding to class labels.")
  }
  chk::chk_true(all(observed %in% pnames))

  # B2: Use matrixStats::rowRanks for efficiency
  # Ensure matrixStats is suggested/imported
  if (!requireNamespace("matrixStats", quietly = TRUE)) {
    stop("Please install package 'matrixStats' for rank_score")
  }
  # Note: matrixStats::rowRanks ranks *lowest* value as 1 by default.
  # We want *highest* probability to have rank 1, so use negative probabilities.
  # ties.method = "random" matches the original apply logic.
  prank_mat <- matrixStats::rowRanks(-prob, ties.method = "random")

  # Normalize ranks to be within (0, 1)
  # Original code: rp / (length(rp) + 1) -> ncol(prob) + 1
  ncols <- ncol(prob)
  prank_mat_norm <- prank_mat / (ncols + 1)

  mids <- match(observed, pnames)
  # Efficiently extract the rank for the observed class for each row
  # Use matrix indexing: rows are 1:n, columns are the matched indices
  pp <- prank_mat_norm[cbind(seq_along(observed), mids)]

  data.frame(prank = pp, observed = observed)
}


#' top-k accuracy indicator
#' 
#' Determines if the true class label is among the top `k` predicted probabilities for each observation.
#'
#' @param prob Numeric matrix of predicted probabilities (observations x classes).
#'   Column names must correspond to class labels.
#' @param observed Factor or vector of observed class labels. Must be present in `colnames(prob)`.
#' @param k Integer; the number of top probabilities to consider.
#' @return A `data.frame` with columns `topk` (logical indicator: `TRUE` if observed class is in top-k) and `observed`.
#' @export
#' @family classifier evaluation
#' @examples
#' probs <- matrix(c(0.1, 0.9, 0.8, 0.2, 0.3, 0.7), 3, 2, byrow=TRUE,
#'                 dimnames = list(NULL, c("A", "B")))
#' obs <- factor(c("B", "A", "B"))
#' topk(probs, obs, k=1)
#' topk(probs, obs, k=2)
topk <- function(prob, observed, k) {
  pnames <- colnames(prob)
  # B3: Check for NULL colnames
  if (is.null(pnames)) {
      stop("Input probability matrix 'prob' must have column names corresponding to class labels.")
  }
  chk::chk_true(all(observed %in% pnames))
  chk::chk_whole_number(k)
  chk::chk_gt(k, 0)
  chk::chk_lte(k, ncol(prob))

  # B2: Vectorized approach
  # Get indices of top k probabilities for each row
  # Use Rfast::rowOrder or base order within apply (more compatible)
  # apply is likely fine here unless this is extremely performance critical
  # Use base 'order' with decreasing=TRUE
  topk_indices <- t(apply(prob, 1, order, decreasing = TRUE))[, 1:k, drop = FALSE]

  observed_indices <- match(observed, pnames)

  # Check if observed index is in the top k for each row
  # Vectorized check using row/column indexing with any()
  # We create a matrix comparing each observed_index to the rows of topk_indices
  topk_result <- apply(topk_indices == observed_indices, 1, any)

  data.frame(topk = topk_result, observed = observed)
}


#' Normalize rows of a probability matrix to sum to 1
#' @keywords internal
#' @noRd
normalize_probs <- function(p) {
  # Normalize rows to sum to 1
  # Avoid division by zero for rows summing to 0
  rs <- rowSums(p, na.rm = TRUE)
  
  # Handle rows where all values are zero or negative
  zero_rows <- rs <= 0
  if (any(zero_rows)) {
      warning("Some rows in probability matrix had zero or negative sum; converting to uniform probability.")
      # Make truly zero rows uniform probability 1/ncol
      p[zero_rows,] <- 1
      rs[zero_rows] <- ncol(p) # Will lead to 1/ncol below
  }
  # Use pmax to avoid division by zero
  sweep(p, 1, pmax(rs, .Machine$double.eps), "/")
}

#' Calculate Average Probabilities per Class from Similarity Matrix
#'
#' This function calculates the average similarity (or inverse distance) of test samples
#' to each training class, based on a similarity/distance matrix between test and training samples.
#'
#' @param sim_mat A matrix of similarities (or inverse distances), with training samples as rows
#'                and test samples as columns (n_train x n_test).
#' @param train_labels A factor vector of labels corresponding to the rows (training samples) of `sim_mat`.
#' @return A matrix of average probabilities (n_test x n_classes), where rows sum to 1.
#' @keywords internal
#' @noRd
avg_probs <- function(sim_mat, train_labels) {
  # Ensure train_labels is a factor
  if (!is.factor(train_labels)) train_labels <- factor(train_labels)
  all_levels <- levels(train_labels)
  n_classes <- length(all_levels)
  
  # Use rowsum to sum similarities *column-wise* for each class
  # We want (n_classes x n_test) matrix where entry (c, j) is sum of sims between class c and test sample j
  summed_sims_by_class <- rowsum(sim_mat, train_labels, reorder = FALSE, na.rm = TRUE)
  
  # Count occurrences of each label in the training set
  label_counts <- table(train_labels)
  
  # Divide summed similarities by counts to get averages
  # Ensure alignment and handle classes with 0 counts (though unlikely if they are in levels)
  # Need to align counts to the row names of summed_sims_by_class (which are the class levels)
  avg_sims <- summed_sims_by_class / as.numeric(label_counts[rownames(summed_sims_by_class)])
  
  # Transpose to get (n_test x n_classes)
  avg_sims_t <- t(avg_sims)
  
  # Ensure all original classes are present as columns, even if they had 0 similarity sums
  if (ncol(avg_sims_t) != n_classes) {
      missing_cols <- setdiff(all_levels, colnames(avg_sims_t))
      if (length(missing_cols) > 0) {
          add_mat <- matrix(0, nrow = nrow(avg_sims_t), ncol = length(missing_cols),
                            dimnames = list(rownames(avg_sims_t), missing_cols))
          avg_sims_t <- cbind(avg_sims_t, add_mat)
      }
      # Reorder columns to match original levels
      avg_sims_t <- avg_sims_t[, all_levels, drop = FALSE]
  }
  
  # Normalize rows (test samples) to sum to 1 to get probabilities
  normalize_probs(avg_sims_t) # Reuse the improved normalize_probs
}

#' Find Nearest Class(es) based on Similarity/Distance
#'
#' Determines the predicted class for each test sample based on the k-nearest neighbors
#' in the training set using a precomputed similarity or distance matrix.
#'
#' @param sim_mat A matrix of similarities (higher is better) or inverse distances,
#'                with training samples as rows and test samples as columns (n_train x n_test).
#' @param train_labels A factor vector of labels corresponding to the rows (training samples) of `sim_mat`.
#' @param knn The number of nearest neighbors to consider.
#' @param higher_is_better Logical indicating if higher values in `sim_mat` mean closer/better.
#' @return A factor vector of predicted class labels for the test samples.
#' @keywords internal
#' @noRd
nearest_class <- function(sim_mat, train_labels, knn=1, higher_is_better = TRUE) {
  # Ensure labels is factor
  if(!is.factor(train_labels)) train_labels <- factor(train_labels)
  all_levels <- levels(train_labels)
  
  # Apply per column (test sample)
  predicted_labels <- apply(sim_mat, 2, function(scores_for_one_test_sample) {
    # Find the indices of the top knn training samples
    neighbor_indices <- order(scores_for_one_test_sample, decreasing = higher_is_better)[1:knn]
    
    # Get the labels of these neighbors
    neighbor_labels <- train_labels[neighbor_indices]
    
    # Find the most frequent label among neighbors (majority vote)
    # Use table and which.max, handling ties by picking the first alphabetically
    tab <- table(factor(neighbor_labels, levels=all_levels)) # Ensure all levels present
    if (length(tab) == 0) return(NA) # Should not happen if knn >= 1
    # Handle ties explicitly: which.max returns the first max
    names(which.max(tab))
  })
  
  # Return as factor with all original levels
  factor(predicted_labels, levels = all_levels)
}

#' @export
project.classifier <- function(x, new_data, ...) {
  # Retrieve potentially stored projector arguments
  proj_args <- list(...)
  if (!is.null(x$projector_args)) { # Assuming constructor stores relevant args
      proj_args <- utils::modifyList(x$projector_args, proj_args)
  }

  # Use colind stored in classifier if not overridden in ...
  use_colind <- proj_args$colind
  if (is.null(use_colind)) {
      use_colind <- x$colind
  }

  # Remove colind and new_data from proj_args to prevent duplicate matching
  proj_args$colind <- NULL
  proj_args$new_data <- NULL

  # Project using appropriate method based on colind
  scores <- if (!is.null(use_colind)) {
    rlang::exec(partial_project, x$projector, new_data = new_data, colind = use_colind, !!!proj_args)
  } else {
    rlang::exec(project, x$projector, new_data = new_data, !!!proj_args)
  }

  scores
}

#' @noRd
prepare_predict <- function(object, colind=NULL, ncomp=NULL, new_data,...) {
  # Determine the colind or block to use for projection
  use_colind <- colind
  use_block <- NULL

  # Check for stored block (only for multiblock classifiers)
  if (is.null(use_colind) && !is.null(object$block)) {
    use_block <- object$block
  }

  # Default to stored colind if neither colind nor block provided
  if (is.null(use_colind) && is.null(use_block)) {
    use_colind <- object$colind # Default to colind stored in classifier
  }

  # Ensure colind and block are mutually exclusive
  if (!is.null(use_colind) && !is.null(use_block)) {
    rlang::abort("Cannot use both `colind` and `block` simultaneously.")
  }

  # C2 & C3: Robust handling of new_data dimensions and type
  expected_cols_original <- shape(object$projector)[1] # Expect cols matching original data space
  if (!is.null(use_colind)) {
      # If colind is used, new_data should have columns matching the length of colind
      expected_cols_subset <- length(use_colind)
  } else if (!is.null(use_block)) {
      # If block is used, need to determine expected columns from block
      # This requires the projector to support block_indices
      if (!is.null(object$projector$block_indices)) {
        expected_cols_subset <- length(object$projector$block_indices[[use_block]])
      } else {
        expected_cols_subset <- expected_cols_original
      }
  } else {
      expected_cols_subset <- expected_cols_original
  }

  if (is.vector(new_data)) {
    # Check length against expected columns (subset or full)
    chk::chk_equal(length(new_data), expected_cols_subset)
    # Reshape as a single row matrix
    new_data <- matrix(new_data, nrow = 1, ncol = expected_cols_subset)
  } else if (is.matrix(new_data) || is.data.frame(new_data)) {
      # Ensure matrix/data.frame input has correct columns for the projection type
      chk::chk_equal(ncol(new_data), expected_cols_subset)
  } else {
      stop("'new_data' must be a numeric vector, matrix, or data frame.")
  }

  # Determine number of components
  max_comps <- shape(object$projector)[2]
  if (is.null(ncomp)) {
    ncomp <- max_comps
  } else {
    chk::chk_whole_number(ncomp)
    chk::chk_range(ncomp, c(1, max_comps))
  }

  # Project the data
  proj <- if (!is.null(use_colind)) {
    partial_project(object$projector, new_data, colind = use_colind, ...) # Pass ...
  } else if (!is.null(use_block)) {
    project_block(object$projector, new_data, block = use_block, ...) # Pass ...
  } else {
    project(object$projector, new_data, ...) # Pass ...
  }

  # Return list with projected data and determined parameters
  list(proj=proj, new_data=new_data, colind=use_colind, block=use_block, ncomp=ncomp)
}


#' Predict Class Labels using a Classifier Object
#'
#' Predicts class labels and probabilities for new data using a fitted `classifier` object.
#' It performs k-Nearest Neighbors (k-NN) classification in the projected component space.
#'
#' The function first projects the `new_data` into the component space defined by the
#' classifier's internal projector. If `colind` is specified, a partial projection using
#' only those features is performed. This projection is then compared to the reference scores
#' stored within the `classifier` object (`object$scores`) using the specified `metric`.
#' The k-NN algorithm identifies the `k` nearest reference samples (based on similarity or distance)
#' and predicts the class via majority vote. Probabilities are estimated based on the average
#' similarity/distance to each class among the neighbors or all reference points.
#'
#' @param object A fitted object of class `classifier`.
#' @param new_data A numeric matrix or vector of new observations to classify. Rows are observations,
#'   columns are variables matching the original data space used by the projector OR matching `colind` if provided.
#' @param ncomp Optional integer; the number of components to use from the projector for classification (default: all components used during classifier creation).
#' @param colind Optional numeric vector specifying column indices from the original data space.
#'   If provided, `new_data` is projected using only these features (`partial_project`). This overrides any
#'   `colind` stored default in the `object`. The resulting projection is compared against the
#'   reference scores (`object$scores`) stored in the classifier.
#' @param metric Character string specifying the similarity or distance metric for k-NN.
#'   Choices: "euclidean", "cosine", "ejaccard".
#' @param normalize_probs Logical; **DEPRECATED** Normalization behavior is now implicit in `prob_type="avg_similarity"`.
#' @param prob_type Character string; method for calculating probabilities:
#'   - "knn_proportion" (default): Calculates the proportion of each class among the `k` nearest neighbors.
#'   - "avg_similarity": Calculates average similarity to all training points per class (uses `avg_probs` helper).
#' @param ... Extra arguments passed down to projection methods (`project`, `partial_project`)
#'   or potentially to distance/similarity calculations (e.g., for `proxy::simil` if used with `ejaccard`).
#' @return A list containing:
#'   \item{class}{A factor vector of predicted class labels for `new_data`.} 
#'   \item{prob}{A numeric matrix (rows corresponding to `new_data`, columns to classes) of estimated class probabilities.}
#' @export
#' @family classifier predict
#' @seealso \code{\link{classifier.projector}}, \code{\link{classifier.multiblock_biprojector}}, \code{\link{partial_project}}
#' @examples
#' # Assume clf is a fitted classifier object (e.g., from classifier.projector)
#' # Assume new_dat is a matrix of new observations
#' # preds <- predict(clf, new_data = new_dat, metric = "cosine")
#' # print(preds$class)
#' # print(preds$prob)
predict.classifier <- function(object, new_data, ncomp=NULL,
                               colind=NULL,
                               metric=c("euclidean", "cosine", "ejaccard"),
                               normalize_probs=FALSE, # Deprecated
                               prob_type = c("knn_proportion", "avg_similarity"),
                               ...) {

  # Check proxy package availability early
  if (!requireNamespace("proxy", quietly = TRUE)) {
    stop("Package 'proxy' is required for predict.classifier. Please install it with: install.packages('proxy')")
  }

  metric <- match.arg(metric)
  prob_type <- match.arg(prob_type)

  # Deprecate normalize_probs
  if (!missing(normalize_probs) && normalize_probs) {
      warning("'normalize_probs' argument is deprecated and ignored. Normalization is implicit in prob_type='avg_similarity'.")
  }
  
  # Prepare data: project new_data, determine ncomp etc.
  prep <- prepare_predict(object, colind, ncomp, new_data,...) 
  proj_test <- prep$proj # Projected test data (n_test x d)
  ncomp <- prep$ncomp    # Number of components to use (d)
  
  # Reference scores (training data in projected space)
  scores_train <- as.matrix(object$scores) # (n_train x D_full)
  train <- scores_train[, 1:ncomp, drop=FALSE] # (n_train x d)
  test <- as.matrix(proj_test)[, 1:ncomp, drop=FALSE] # (n_test x d)
  
  # Training labels and levels
  train_labels <- object$labels
  if (!is.factor(train_labels)) train_labels <- factor(train_labels)
  all_levels <- levels(train_labels)
  n_classes <- length(all_levels)
  n_test <- nrow(test)
  knn <- object$knn
  
  # Compute similarities or distances (train vs test)
  # Resulting matrix `sim_dist_mat` will be (n_train x n_test)
  if (metric == "cosine") {
    sim_dist_mat <- proxy::simil(train, test, method="cosine")
    higher_is_better <- TRUE
  } else if (metric == "euclidean") {
    sim_dist_mat <- proxy::dist(train, test, method="euclidean")
    higher_is_better <- FALSE
  } else if (metric == "ejaccard") {
    sim_dist_mat <- proxy::simil(train, test, method="ejaccard")
    higher_is_better <- TRUE
  } else {
    # Should not happen due to match.arg
    stop("Unsupported metric")
  }
  sim_dist_mat <- as.matrix(sim_dist_mat)
  
  # --- Predict Classes (k-NN Majority Vote) ---
  predicted_classes <- nearest_class(sim_dist_mat, train_labels, knn, higher_is_better)

  # --- Calculate Probabilities --- 
  prob_mat <- matrix(0.0, nrow = n_test, ncol = n_classes,
                     dimnames = list(rownames(test), all_levels))

  if (prob_type == "knn_proportion") {
      # Calculate proportion of each class among k neighbors for each test sample
      for (j in 1:n_test) {
          scores_for_one_test_sample <- sim_dist_mat[, j]
          neighbor_indices <- order(scores_for_one_test_sample, decreasing = higher_is_better)[1:knn]
          neighbor_labels <- train_labels[neighbor_indices]
          # Use table to count classes among neighbors, ensuring all levels are included
          counts <- table(factor(neighbor_labels, levels = all_levels))
          prob_mat[j, ] <- counts / knn # Simple proportion
      }
  } else if (prob_type == "avg_similarity") {
      # Use the older avg_probs logic (average similarity to all training points per class)
      # Note: This doesn't directly use knn for probabilities
      if (!higher_is_better) {
          # Convert distance to similarity using 1 / (1 + D) for stability
          sim_mat_for_avg <- 1 / (1 + sim_dist_mat)
      } else {
          sim_mat_for_avg <- sim_dist_mat
          # Shift similarities to be non-negative if they were originally similarities
          # This prevents issues in avg_probs -> normalize_probs with negative sums
          min_sim <- min(sim_mat_for_avg, na.rm = TRUE)
          if (min_sim < 0) {
              sim_mat_for_avg <- sim_mat_for_avg - min_sim
          }
      }
      # Calculate average similarity per class and normalize
      # Ensure the matrix passed to avg_probs is (n_train x n_test)
      prob_mat <- avg_probs(sim_mat_for_avg, train_labels)
  } else {
      stop("Unsupported prob_type")
  }

  # Ensure prob_mat has correct dimensions and names
  if (nrow(prob_mat) != n_test || ncol(prob_mat) != n_classes) {
      stop("Internal error: Probability matrix has incorrect dimensions.")
  }
  if (!identical(colnames(prob_mat), all_levels)) {
       # Reorder columns if necessary (should be handled by avg_probs now)
       prob_mat <- prob_mat[, all_levels, drop = FALSE]
  }
  
  list(class = predicted_classes, prob = prob_mat)
}


#' Predict Class Labels using a Random Forest Classifier Object
#'
#' Predicts class labels and probabilities for new data using a fitted `rf_classifier` object.
#' This method projects the `new_data` into the component space and then uses the stored
#' `randomForest` model to predict outcomes.
#'
#' @inheritParams predict.classifier
#' @param object A fitted object of class `rf_classifier`.
#' @param ... Extra arguments passed to `predict.randomForest`.
#' @return A list containing:
#'   \item{class}{Predicted class labels (typically factor) from the random forest model.} 
#'   \item{prob}{A numeric matrix of predicted class probabilities from the random forest model.}
#' @family classifier predict
#' @seealso \code{\link{rf_classifier.projector}}, \code{\link[randomForest]{predict.randomForest}}
#' @export
predict.rf_classifier <- function(object, new_data, ncomp=NULL,
                                  colind=NULL, ...) {
  
  prep <- prepare_predict(object, colind, ncomp, new_data,...)
  proj <- prep$proj[, 1:prep$ncomp, drop=FALSE] # Ensure correct number of components

  # Use sanitized feature names stored during training
  if (!is.null(object$feat_names)) {
    if (ncol(proj) == length(object$feat_names)) {
      colnames(proj) <- object$feat_names
    } else if (ncol(proj) <= length(object$feat_names)) {
      # Use first ncomp sanitized names if dimensions match subset
      colnames(proj) <- object$feat_names[1:ncol(proj)]
    } else {
      warning("Projection has more components than original feat_names.")
      colnames(proj) <- paste0("Comp", 1:ncol(proj)) # Fallback naming
    }
  } else {
    # Fallback if feat_names not available (shouldn't happen with new code)
    colnames(proj) <- paste0("Comp", 1:ncol(proj))
  }

  # Coerce proj to data.frame as expected by randomForest predict
  proj_df <- as.data.frame(proj)
  
  # Ensure randomForest package is available
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Package 'randomForest' needed but is not available.")
  }
  
  # Predict class and probabilities
  cls <- predict(object$rfres, newdata = proj_df, type = "response", ...)
  prob <- predict(object$rfres, newdata = proj_df, type = "prob", ...)
  
  list(class=cls, prob=prob)
}


#' Evaluate Feature Importance for a Classifier
#'
#' Estimates the importance of features or blocks of features for the classification performance
#' using either a "marginal" (leave-one-block-out) or "standalone" (use-only-one-block) approach.
#'
#' Importance is measured by the change in a performance metric (`fun`) when features are
#' removed (marginal) or used exclusively (standalone).
#'
#' @inheritParams predict.classifier
#' @param x A fitted `classifier` object.
#' @param new_data The data matrix used for evaluating importance (typically validation or test data).
#' @param true_labels The true class labels corresponding to the rows of `new_data`.
#' @param blocks A list where each element is a numeric vector of feature indices (columns in the original
#'   data space) defining a block. If `NULL`, each feature is treated as its own block.
#' @param fun A function to compute the performance metric (e.g., `rank_score`, `topk`, or a custom function).
#'   The function should take a probability matrix and observed labels and return a data frame
#'   where the first column is the metric value per observation.
#' @param fun_direction Character string, either "lower_is_better" or "higher_is_better", indicating
#'   whether lower or higher values of the metric calculated by `fun` signify better performance.
#'   This is used to interpret the importance score correctly.
#' @param approach Character string: "marginal" (calculates importance as change from baseline when block is removed)
#'   or "standalone" (calculates importance as performance using only the block).
#' @param ... Additional arguments passed to `predict.classifier` during internal predictions.
#' @return A `data.frame` with columns `block` (character representation of feature indices in the block)
#'   and `importance` (numeric importance score). Higher importance values generally indicate more influential blocks,
#'   considering `fun_direction`.
#' @export
#' @family feature_importance classifier
#' @seealso \code{\link{rank_score}}, \code{\link{topk}}
#' @examples
#' # Assume clf is a fitted classifier object, dat is new data, true_lbls are correct labels for dat
#' # Assume blocks_list defines feature groups e.g., list(1:5, 6:10)
#' # feature_importance(clf, new_data = dat, true_labels = true_lbls, blocks = blocks_list)
feature_importance.classifier <- function(x, new_data, 
                                          true_labels, # Added true_labels argument
                                          ncomp = NULL,
                                          blocks = NULL, 
                                          metric = c("cosine", "euclidean", "ejaccard"), 
                                          fun = rank_score,
                                          fun_direction = c("lower_is_better", "higher_is_better"), 
                                          approach = c("marginal", "standalone"),
                                          ...) {
  metric <- match.arg(metric)
  approach <- match.arg(approach)
  fun_direction <- match.arg(fun_direction) 
  
  # Check true_labels length
  chk::chk_equal(nrow(new_data), length(true_labels))
  # Ensure true_labels are factor if not already, using levels from training data for consistency
  if (!is.factor(true_labels)) {
      true_labels <- factor(true_labels, levels = levels(x$labels))
  } else {
      # If already factor, ensure levels match training levels
      if (!identical(levels(true_labels), levels(x$labels))) {
          warning("Levels of provided 'true_labels' do not match training labels. Attempting to relevel.")
          true_labels <- factor(true_labels, levels = levels(x$labels))
      }
  }
  
  # Determine the full set of features available to the projector
  full_feature_indices <- 1:shape(x$projector)[1]
  
  if (is.null(blocks)) {
    # If blocks is NULL, treat each feature as a block
    blocks <- as.list(full_feature_indices)
  }
  
  # Check blocks validity
  all_block_indices <- unlist(blocks)
  if (any(all_block_indices <= 0) || any(all_block_indices > length(full_feature_indices))) {
      stop("Block indices are out of range for the features defined by the projector.")
  }
  if (any(duplicated(all_block_indices)) && approach == "marginal") {
      warning("Some features belong to multiple blocks; marginal importance calculation might be ambiguous.")
  }
  
  # Baseline performance with all features
  predict_args <- list(object = x, new_data = new_data, ncomp = ncomp,
                       colind = NULL, # Force use of all features for baseline
                       metric = metric, 
                       ...) 
  predict_args$normalize_probs <- NULL 
  base_pred <- do.call(predict, predict_args)
  
  # Use true_labels here
  base_accuracy <- fun(base_pred$prob, true_labels) 
  base_score <- mean(base_accuracy[,1], na.rm = TRUE)
  
  results <- lapply(seq_along(blocks), function(i) {
    block_indices_original <- blocks[[i]]
    current_score <- NA # Initialize
    importance <- NA # Initialize importance

    tryCatch({ 
        predict_args_block <- list(object = x, ncomp = ncomp,
                                   metric = metric,  
                                   ...) 
        predict_args_block$normalize_probs <- NULL
        
        if (approach == "marginal") {
          remaining_features_original <- setdiff(full_feature_indices, block_indices_original)
          
          if (length(remaining_features_original) == 0) {
            warning("Marginal approach: Removing block ", i, " (", paste(block_indices_original, collapse=","), ") leaves no features. Importance set to NA.")
            return(data.frame(block = paste(block_indices_original, collapse = ","), importance = NA))
          }
          
          subset_data <- new_data[, remaining_features_original, drop=FALSE]
          predict_args_block$new_data <- subset_data
          # Use the original indices for the remaining features when calling partial_project
          predict_args_block$colind <- remaining_features_original 
          
          preds <- do.call(predict, predict_args_block)
          # Use true_labels here
          accuracy <- fun(preds$prob, true_labels) 
          current_score <- mean(accuracy[,1], na.rm = TRUE)

          if (fun_direction == "lower_is_better") {
             importance <- current_score - base_score
          } else { 
             importance <- base_score - current_score
          }

        } else { # standalone
          if (length(block_indices_original) == 0) {
             warning("Standalone approach: Block ", i, " has no features. Importance set to NA.")
             return(data.frame(block = paste(block_indices_original, collapse = ","), importance = NA))
          }
          
          subset_data <- new_data[, block_indices_original, drop=FALSE]
          predict_args_block$new_data <- subset_data
          # Use the original indices for this block when calling partial_project
          predict_args_block$colind <- block_indices_original 
          
          preds <- do.call(predict, predict_args_block)
          # Use true_labels here
          accuracy <- fun(preds$prob, true_labels) 
          current_score <- mean(accuracy[,1], na.rm = TRUE)
          importance <- current_score 

           if (fun_direction == "lower_is_better") {
                importance <- 1 - current_score 
           }
           
        }
     }, error = function(e) {
        warning("Error calculating importance for block ", i, " (", paste(block_indices_original, collapse=","), "): ", e$message)
        importance <<- NA 
     }) 

    if (is.na(current_score) && is.na(importance)) {
         importance <- NA
    }
    
    data.frame(block = paste(block_indices_original, collapse = ","), importance = importance)
  })
  
  ret <- do.call(rbind, results)
  ret <- ret[order(ret$importance, decreasing = TRUE, na.last = TRUE), ]
  ret
}


#' Pretty Print Method for `classifier` Objects
#'
#' Display a human-readable summary of a `classifier` object.
#'
#' @param x A `classifier` object.
#' @param ... Additional arguments.
#' @return `classifier` object.
#' @export
#' @examples
#' # Assume clf is a fitted classifier object
#' # print(clf)
print.classifier <- function(x, ...) {
  # Helper function for optional coloring
  maybe_cyan <- function(text) {
    if (requireNamespace("crayon", quietly = TRUE)) {
      crayon::cyan(text)
    } else {
      text
    }
  }

  cat("k-NN Classifier object:\n")
  cat(maybe_cyan("  k-NN Neighbors (k):"), x$knn, "\n")
  cat(maybe_cyan("  Number of Training Samples:"), nrow(x$scores), "\n")
  cat(maybe_cyan("  Number of Classes:"), length(levels(x$labels)), "\n")

  if (!is.null(x$colind)) {
    cat(maybe_cyan("  Default Feature Subset (colind):"), paste(x$colind, collapse=", "), "\n")
  }
  if (inherits(x, "multiblock_classifier") && !is.null(x$block)){
      cat(maybe_cyan("  Default Block Subset:"), x$block, "\n")
  }

  cat(maybe_cyan("  Underlying Projector Details:"), "\n")
  # Indent projector print output
  proj_output <- utils::capture.output(print(x$projector))
  cat(paste("    ", proj_output), sep = "\n")

  invisible(x)
}


#' Pretty Print Method for `rf_classifier` Objects
#'
#' Display a human-readable summary of an `rf_classifier` object.
#'
#' @param x An `rf_classifier` object.
#' @param ... Additional arguments passed to `print.randomForest`.
#' @return `rf_classifier` object.
#' @export
#' @examples
#' # Assume rf_clf is a fitted rf_classifier object
#' # print(rf_clf)
print.rf_classifier <- function(x, ...) {
  # Helper function for optional coloring
  maybe_cyan <- function(text) {
    if (requireNamespace("crayon", quietly = TRUE)) {
      crayon::cyan(text)
    } else {
      text
    }
  }

  cat("Random Forest Classifier object:\n")

  # Print details about the underlying projector
  cat(maybe_cyan("  Underlying Projector Details:"), "\n")
  proj_output <- utils::capture.output(print(x$projector))
  cat(paste("    ", proj_output), sep = "\n")

  # Print details about the Random Forest model
  cat(maybe_cyan("  Random Forest Model Details (from randomForest package):"), "\n")
  # Indent RF print output
  rf_output <- utils::capture.output(print(x$rfres, ...))
  cat(paste("    ", rf_output), sep = "\n")

  if (!is.null(x$colind)) {
    cat(maybe_cyan("  Default Feature Subset (colind) for Projection:"), paste(x$colind, collapse=", "), "\n")
  }

  invisible(x)
}

