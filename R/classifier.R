#' Multiblock Bi-Projector Classifier
#'
#' Constructs a classifier for a multiblock bi-projector model object. 
#' Either global or partial scores can be used. If `colind` or `block` are provided 
#' and `global_scores=FALSE`, partial projection is performed. Otherwise, global projection is used.
#'
#' @param x A fitted multiblock bi-projector model object.
#' @param colind An optional vector of column indices used for prediction (default: NULL).
#' @param labels A factor or vector of class labels for the training data.
#' @param new_data An optional data matrix for which to generate predictions (default: NULL).
#' @param block An optional block index for prediction (default: NULL).
#' @param knn The number of nearest neighbors to consider in the classifier (default: 1).
#' @param global_scores Whether to use the global scores or the partial scores for reference space (default: TRUE).
#' @param ... Additional arguments.
#' @return A multiblock classifier object.
#' @export
#' @family classifier
classifier.multiblock_biprojector <- function(x, colind=NULL, labels, new_data=NULL, 
                                              block=NULL, global_scores=TRUE, knn=1,...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
    if (!is.null(block)) {
      rlang::abort("can either supply `colind` or `block` but not both")
    }
  }
  
  # Check knn
  if (!is.numeric(knn) || knn < 1 || knn != as.integer(knn)) {
    stop("knn must be a positive integer")
  }
  
  scores <- if (!is.null(colind) && !global_scores) {
    chk::chk_not_null(new_data)
    partial_project(x, new_data, colind=colind)
  } else if (!is.null(block) && !global_scores) {
    chk::chk_whole_number(block)
    chk::chk_not_null(new_data)
    project_block(x, new_data, block)
  } else {
    if (!is.null(new_data)) {
      project(x,new_data)
    } else {
      scores(x)
    }
  }
  
  new_classifier(x, labels=labels, scores=scores, colind=colind, block=block, knn=knn, global_scores=global_scores,
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
classifier.discriminant_projector <- function(x, colind=NULL, knn=1,...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
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
    chk::chk_true(all(colind>0))
  }
  
  if (!is.numeric(knn) || knn < 1 || knn != as.integer(knn)) {
    stop("knn must be a positive integer")
  }
  
  chk::chk_equal(length(labels), nrow(scores))
  
  structure(
    list(
      projector=x,
      labels=labels,
      scores=scores,
      colind=colind,
      knn=knn,
      ...),
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
rf_classifier.projector <- function(x, colind=NULL, labels, scores, ...) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Please install package 'randomForest' for 'rf_classifier'")
  }
  
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
  }
  
  chk::chk_equal(length(labels), nrow(scores))
  
  rfres <- randomForest::randomForest(scores, labels, ...)
  
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
      importance=imp,
      colind=colind,
      ...),
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
#' @export
classifier.projector <- function(x, colind=NULL, labels, new_data, knn=1, global_scores=TRUE, ...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
  }
  
  if (!is.numeric(knn) || knn < 1 || knn != as.integer(knn)) {
    stop("knn must be a positive integer")
  }
  
  chk::chk_equal(length(labels), nrow(new_data))
  
  scores <- if (!is.null(colind) && !global_scores) {
    partial_project(x, new_data, colind=colind)
  } else {
    project(x, new_data)
  }
  
  new_classifier(x, labels, scores, colind, knn)
}


#' Calculate Rank Score for Predictions
#'
#' @param prob matrix of predicted probabilities (observations x classes)
#' @param observed vector of observed class labels
#' @return data.frame with prank and observed
#' @export
rank_score <- function(prob, observed) {
  pnames <- colnames(prob)
  chk::chk_true(all(observed %in% pnames))
  prank_mat <- apply(prob, 1, function(p) {
    rp <- rank(p, ties.method="random")
    rp / (length(rp) + 1)
  })
  
  # prank_mat is cbinded by apply
  # actually apply with MARGIN=1 returns a vector or matrix transposed
  # ensure correct indexing:
  # 'prank_mat' is likely a matrix nclasses x nobs if apply MARGIN=1
  # we used apply(prob, 1, ...) => MARGIN=1 means by row => returns a vector or array
  # For clarity, do t():
  prank_mat <- t(prank_mat)
  
  mids <- match(observed, pnames)
  pp <- prank_mat[cbind(seq_along(observed), mids)]
  
  data.frame(prank = pp, observed = observed)
}


#' top-k accuracy indicator
#' 
#' @keywords internal
#' @noRd
topk <- function(prob, observed, k) {
  pnames <- colnames(prob)
  chk::chk_true(all(observed %in% pnames))
  
  topk_indices <- t(apply(prob, 1, function(p) order(p, decreasing = TRUE)[1:k]))
  observed_indices <- match(observed, pnames)
  
  topk_result <- sapply(seq_len(nrow(prob)), function(i) observed_indices[i] %in% topk_indices[i, ])
  
  data.frame(topk = topk_result, observed = observed)
}


#' @keywords internal
#' @noRd
normalize_probs <- function(p) {
  apply(p, 2, function(v) {
    v2 <- v - min(v)
    s <- sum(v2)
    if (s == 0) {
      # If all v are same, just uniform:
      v2[] <- 1/length(v2)
      v2
    } else {
      v2/s
    }
  })
}

#' @keywords internal
#' @noRd
avg_probs <- function(prob, labels) {
  pmeans <- t(group_means(labels, prob))
  t(apply(pmeans, 1, function(v) v/sum(v)))
}

#' @keywords internal
#' @noRd
nearest_class <- function(prob, labels, knn=1) {
  apply(prob, 2, function(v) {
    ord <- order(v, decreasing=TRUE)[1:knn]
    l <- labels[ord]
    tab <- table(l)
    names(which.max(tab))
  })
}

#' @export
project.classifier <- function(x, new_data, ...) {
  scores <- if (!is.null(x$colind)) {
    partial_project(x$projector, new_data, colind=x$colind, ...)
  } else {
    project(x$projector, new_data, ...)
  }
  
  scores
}

#' @noRd
prepare_predict <- function(object, colind=NULL, ncomp=NULL, new_data,...) {
  if (is.null(colind)) {
    colind <- object$colind
  } else {
    if (!is.null(object$colind) && !identical(object$colind, colind)) {
      warning("colind in predict differs from colind in classifier; using provided colind.")
    }
  }
  
  if (is.vector(new_data)) {
    if (is.null(colind)) {
      chk::chk_equal(length(new_data), shape(object$projector)[1])
    } else {
      chk::chk_equal(length(new_data), length(colind))
    }
    new_data <- matrix(new_data, nrow=1)
  }
  
  if (is.null(ncomp)) {
    ncomp <- shape(object$projector)[2]
  } else {
    chk::chk_range(ncomp, c(1,shape(object$projector)[2]))
  }
  
  if (!is.null(colind)) {
    if (length(colind) == 1 && is.vector(new_data)) {
      new_data <- as.matrix(new_data)
    }
    chk::chk_equal(length(colind), ncol(new_data))
    proj <- partial_project(object$projector, new_data, colind, ...)
  } else {
    proj <- project(object$projector, new_data, ...)
  }
  
  list(proj=proj, new_data=new_data,colind=colind, ncomp=ncomp)
}


#' predict with a classifier object
#' 
#' @param object classifier
#' @param new_data new data
#' @param ncomp number of components
#' @param colind column indices
#' @param metric similarity metric
#' @param normalize_probs logical
#' @param ... extra args
#' @return list with class and prob
#' @export
predict.classifier <- function(object, new_data, ncomp=NULL,
                               colind=NULL, 
                               metric=c("euclidean", "cosine", "ejaccard"), 
                               normalize_probs=FALSE, ...) {
  
  metric <- match.arg(metric)
  
  prep <- prepare_predict(object, colind, ncomp, new_data,...) 
  proj <- prep$proj
  ncomp <- prep$ncomp
  
  sc <- as.matrix(object$scores)
  train <- sc[,1:ncomp,drop=FALSE]
  test <- as.matrix(proj)[,1:ncomp,drop=FALSE]
  
  # Compute similarities or distances
  if (metric == "cosine") {
    p <- proxy::simil(train, test, method="cosine")
    p <- as.matrix(p)
  } else if (metric == "euclidean") {
    D <- proxy::dist(train, test, method="euclidean")
    D <- as.matrix(D)
    # Convert distances to similarities via exp(-D) or something:
    # The original code tries exp(-D), ensure no negative:
    p <- exp(-D) 
  } else if (metric == "ejaccard") {
    p <- proxy::simil(train, test, method="ejaccard")
    p <- as.matrix(p)
  }
  
  # Normalize probabilities if requested
  if (normalize_probs) {
    # subtract min from entire p
    p_min <- min(p)
    p <- p - p_min
    # Avoid division by zero rows:
    p <- t(apply(p, 1, function(row) {
      s <- sum(row)
      if (s == 0) {
        # uniform distribution if all zero
        rep(1/length(row), length(row))
      } else {
        row/s
      }
    }))
  }
  
  # doit function from original code
  doit <- function(p) {
    pmeans <- avg_probs(p, object$labels)
    # Convert pmeans to probabilities via softmax or just ensure positivity:
    # pmeans was normalized already in avg_probs:
    # Actually avg_probs returns rows sum to 1, so no softmax needed:
    cls <- nearest_class(p, object$labels, object$knn)
    list(class=cls, prob=pmeans)
  }
  
  doit(p)
}


#' @export
predict.rf_classifier <- function(object, new_data, ncomp=NULL,
                                  colind=NULL, ...) {
  
  prep <- prepare_predict(object, colind, ncomp, new_data,...) 
  proj <- prep$proj
  # Ensure proj is a data frame with same colnames as object$scores if possible:
  if (!is.null(colnames(object$scores))) {
    if (ncol(proj) == ncol(object$scores)) {
      colnames(proj) <- colnames(object$scores)
    } else {
      warning("Projection does not match original score dimensions for naming.")
    }
  }
  
  proj <- as.data.frame(proj)
  
  cls <- predict(object$rfres, proj)
  prob <- predict(object$rfres, proj, type="prob")
  list(class=cls, prob=prob)
}


#' Evaluate Feature Importance
#'
#' Uses "marginal" or "standalone" approaches:
#' - marginal: remove block and see change in accuracy
#' - standalone: use only that block and measure accuracy
#'
#' @param x classifier
#' @param new_data new data
#' @param ncomp ...
#' @param blocks a list of feature indices
#' @param metric ...
#' @param fun a function to compute accuracy (default rank_score)
#' @param normalize_probs logical
#' @param approach "marginal" or "standalone"
#' @param ... args to projection
#' @return a data.frame with block and importance
#' @export
feature_importance.classifier <- function(x, new_data, 
                                          ncomp = NULL,
                                          blocks = NULL, 
                                          metric = c("cosine", "euclidean", "ejaccard"), 
                                          fun = rank_score,
                                          normalize_probs = FALSE,
                                          approach = c("marginal", "standalone"),
                                          ...) {
  metric <- match.arg(metric)
  approach <- match.arg(approach)
  
  if (is.null(blocks)) {
    blocks <- lapply(seq_len(ncol(new_data)), function(i) i)
  }
  
  # Check blocks validity
  # Ensure all blocks indices are within new_data columns and colind if present
  if (!is.null(x$colind)) {
    max_feat <- length(x$colind)
  } else {
    max_feat <- ncol(new_data)
  }
  for (b in blocks) {
    if (any(b > max_feat)) {
      stop("Block indices exceed the number of available features.")
    }
  }
  
  # base accuracy with all features
  base_pred <- predict(x, new_data = new_data, ncomp = ncomp,
                       metric = metric, normalize_probs = normalize_probs, ...)
  base_accuracy <- fun(base_pred$prob, x$labels)
  base_score <- mean(base_accuracy[,1]) # mean rank or accuracy metric
  
  results <- lapply(seq_along(blocks), function(i) {
    block <- blocks[[i]]
    if (approach == "marginal") {
      # remove block
      remaining_features <- setdiff(seq_len(max_feat), block)
      if (length(remaining_features) == 0) {
        return(data.frame(block = paste(block, collapse = ","), importance = NA))
      }
      reduced_data <- new_data[, remaining_features, drop = FALSE]
      # if colind used, subset colind too
      use_colind <- if (!is.null(x$colind)) x$colind[remaining_features] else NULL
      
      preds <- predict(x, new_data = reduced_data, ncomp = ncomp,
                       colind = use_colind,
                       metric = metric, normalize_probs = normalize_probs, ...)
      accuracy <- fun(preds$prob, x$labels)
      current_score <- mean(accuracy[,1])
      importance <- base_score - current_score
      
    } else { # standalone
      # use only this block
      if (length(block) == 0) {
        return(data.frame(block = paste(block, collapse = ","), importance = NA))
      }
      
      reduced_data <- new_data[, block, drop = FALSE]
      use_colind <- if (!is.null(x$colind)) x$colind[block] else block
      
      preds <- predict(x, new_data = reduced_data, ncomp = ncomp,
                       colind = use_colind,
                       metric = metric, normalize_probs = normalize_probs, ...)
      accuracy <- fun(preds$prob, x$labels)
      current_score <- mean(accuracy[,1])
      importance <- current_score # standalone score interpreted as importance
    }
    data.frame(block = paste(block, collapse = ","), importance = importance)
  })
  
  ret <- do.call(rbind, results)
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
print.classifier <- function(x, ...) {
  cat("classifier object:\n")
  cat("  Model fit:\n")
  print(x$projector)
  cat("  Scores matrix dimensions:", nrow(x$scores), "x", ncol(x$scores), "\n")
  cat("  k-NN:", x$knn, "\n")
  if (!is.null(x$colind)) {
    cat("  Using subset of components/features indices:", paste(x$colind, collapse=", "), "\n")
  }
  invisible(x)
}

