#' Multiblock Bi-Projector Classifier
#'
#' Constructs a classifier for a multiblock bi-projector model that can generate predictions for new data points.
#'
#' @param x A fitted multiblock bi-projector model object.
#' @param colind An optional vector of column indices used for prediction (default: NULL).
#' @param labels A factor or vector of class labels for the training data.
#' @param new_data An optional data matrix for which to generate predictions (default: NULL).
#' @param block An optional block index for prediction (default: NULL).
#' @param knn The number of nearest neighbors to consider in the classifier (default: 1).
#' @param ... Additional arguments to be passed to the specific model implementation of `classifier`.
#' @return A multiblock classifier object.
#' @export
#' @family classifier
classifier.multiblock_biprojector <- function(x, colind=NULL, labels, new_data=NULL, 
                                              block=NULL, knn=1,...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
    if (!is.null(block)) {
      rlang::abort("can either supply `colind` or `block` but not both")
    }
  }
  
  scores <- if (!is.null(colind)) {
    chk::chk_not_null(new_data)
    scores <- partial_project(x, new_data, colind=colind)
  } else if (!is.null(block)) {
    chk::chk_whole_number(block)
    project_block(x, new_data, block)
  } else {
    if (!is.null(new_data)) {
      project(x,new_data)
    } else {
      scores(x)
    }
  }
  
  new_classifier(x,labels=labels,scores=scores, colind=colind, block=block, knn=knn, classes="multiblock_classifier")
}


# 
# classifier.multiblock_projector <- function(x, colind=NULL, labels, new_data, block=NULL, knn=1,...) {
#   if (!is.null(colind)) {
#     chk::chk_true(length(colind) <= shape(x)[1])
#     chk::chk_true(all(colind>0))
#     if (!is.null(block)) {
#       rlang::abort("can either supply `colind` or `block` but not both")
#     }
#   }
#   
#   scores <- if (!is.null(colind)) {
#     scores <- partial_project(x, new_data, colind=colind)
#   } else if (!is.null(block)) {
#     chk::chk_whole_number(block)
#     project_block(x, new_data, block)
#   }
#   
#   new_classifier(x,labels,scores, colind, block=block, knn=knn, classes="multiblock_classifier")
#   
# }



#' Create a k-NN classifier for a discriminant projector
#'
#' Constructs a k-NN classifier for a discriminant projector, with an option to use a subset of the components.
#' 
#' @param x the discriminant projector object
#' @param colind an optional vector specifying the column indices of the components to use for prediction (NULL by default)
#' @param knn the number of nearest neighbors to consider in the k-NN classifier (default is 1)
#' @param ... extra arguments
#' @return a classifier object
#' @export
classifier.discriminant_projector <- function(x, colind=NULL, knn=1,...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
  }
  
  new_classifier(x, x$labels, scores(x), colind=colind, knn=knn)
}


#' Create a new k-NN classifier for a model fit
#'
#' Constructs a new k-NN classifier based on a model fit and the corresponding scores matrix, with an option to use a subset of the components.
#' 
#' @param x the model fit
#' @param labels a vector of class labels corresponding to the data points
#' @param scores a matrix of scores used for classification
#' @param colind an optional vector specifying the column indices of the components to use for prediction (NULL by default)
#' @param knn the number of nearest neighbors to consider in the k-NN classifier (default is 1)
#' @param classes additional S3 classes to assign to the classifier object
#' @param ... extra arguments
#' @return a classifier object
#' @keywords internal
#' @noRd
new_classifier <- function(x, labels, scores, colind=NULL, knn=1, classes=NULL, ...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
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

#' create a random forest classifier
#' 
#' 
#' @export
#' 
#' @inheritParams rf_classifier
#' @inheritParams classifier.multiblock_biprojector
#' 
#' @param scores a matrix of references scores used for classification
#' @return a `rf_classifier` object
#' @examples
#' data(iris)
#' X <- iris[,1:4]
#' pcres <- pca(as.matrix(X),2)
#' cfier <- rf_classifier(pcres, labels=iris[,5], scores=scores(pcres))
#' p <- predict(cfier, new_data=as.matrix(iris[,1:4]))
rf_classifier.projector <- function(x, colind=NULL, labels, scores, ...) {
  if (!requireNamespace("randomForest", quietly = TRUE)) {
    stop("Please install package 'randomForest' when using 'rf_classifier'")
  }
  
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
  }
  
  
  chk::chk_equal(length(labels), nrow(scores))
  
  rfres <- randomForest::randomForest(scores, labels, ...)
  
  
  structure(
    list(
      projector=x,
      rfres=rfres,
      labels=labels,
      scores=scores,
      colind=colind,
      ...),
    class=c("rf_classifier", "classifier")
  )
  
  
}

#' create `classifier` from a `projector`
#' 
#' @inheritParams classifier
#' @param labels the labels associated with the rows of the projected data (see `new_data`)
#' @param new_data reference data associated with `labels` and to be projected into subspace (required).
#' @param knn the number of nearest neighbors to use when classifying a new point. 
#' @export
#' 
#' @family classifier
#' 
#' @return a `classifier` object
#' 
#' @examples
#' data(iris)
#' X <- iris[,1:4]
#' pcres <- pca(as.matrix(X),2)
#' cfier <- classifier(pcres, labels=iris[,5], new_data=as.matrix(iris[,1:4]))
#' p <- predict(cfier, as.matrix(iris[,1:4]))
classifier.projector <- function(x, colind=NULL, labels, new_data, knn=1,...) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
  }
  
  chk::chk_equal(length(labels), nrow(new_data))
  
  scores <- if (!is.null(colind)) {
    partial_project(x, new_data, colind=colind)
  } else {
    project(x, new_data)
  }
  
  new_classifier(x, labels, scores, colind, knn)
  
}


#' @keywords internal
#' @noRd
rank_score <- function(prob, observed) {
  pnames <- colnames(prob)
  chk::chk_true(all(observed %in% pnames))
  prank <- apply(prob, 1, function(p) {
    rp <- rank(p, ties.method="random")
    rp/length(rp)
  })
  
  mids <- match(observed, pnames)
  pp <- prank[cbind(mids, 1:length(observed))]
  
  data.frame(prank=pp, observed=observed)
}

#' @keywords internal
#' @noRd
normalize_probs <- function(p) {
  apply(p, 2, function(v) {
    v2 <- v - min(v)
    v2/sum(v2)
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
nearest_class <- function(prob, labels,knn=1) {
  
  apply(prob, 2, function(v) {
    ord <- order(v, decreasing=TRUE)[1:knn]
    l <- labels[ord]
    table(l)
    names(which.max(table(l)))
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
prepare_predict <- function(object, colind, ncomp, new_data,...) {
  if (is.null(colind)) {
    colind <- object$colind
  } else {
    ### colind overrides object$colind, should emit warning?
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
    proj <- project(object$projector, new_data,...)
  }
  
  list(proj=proj, new_data=new_data,colind=colind, ncomp=ncomp)
}


#' predict with a classifier object
#' 
#' @param object the model fit
#' @param new_data new data to predict on
#' @param ncomp the number of components to use
#' @param colind the column indices to select in the projection matrix
#' @param metric the similarity metric ("euclidean" or "cosine")
#' @param ... additional arguments to projection function
#' 
#' @importFrom stats predict
#' @return a list with the predicted class and probabilities
#' @export
predict.classifier <- function(object, new_data, ncomp=NULL,
                               colind=NULL, metric=c("cosine", "euclidean"), ...) {
  
  metric <- match.arg(metric)

  prep <- prepare_predict(object, colind, ncomp, new_data,...) 
  proj <- prep$proj
  ncomp <- prep$ncomp
  
  doit <- function(p) {
    prob <- normalize_probs(p)
    pmeans <- avg_probs(prob, object$labels)
    cls <- nearest_class(prob, object$labels, object$knn)
    list(class=cls, prob=pmeans)
  }
  
  sc <- as.matrix(object$scores)
  
  if (metric == "cosine") {
    p <- proxy::simil(sc[,1:ncomp,drop=FALSE], as.matrix(proj)[,1:ncomp,drop=FALSE], method="cosine")
    doit(p)
  } else if (metric == "euclidean") {
    D <- proxy::dist(sc[,1:ncomp,drop=FALSE], as.matrix(proj)[,1:ncomp,drop=FALSE], method="euclidean")
    D <- D/max(D)
    doit(exp(-D))
    #-D
  }
  
}


#' @export
predict.rf_classifier <- function(object, new_data, ncomp=NULL,
                               colind=NULL, ...) {
  
  
  prep <- prepare_predict(object, colind, ncomp, new_data,...) 
  proj <- prep$proj
  ncomp <- prep$ncomp
  
  cls <- predict(object$rfres, proj)
  prob <- predict(object$rfres, proj, type="prob")
  list(class=cls, prob=prob)
}



#' Pretty Print Method for `classifier` Objects
#'
#' Display a human-readable summary of a `classifier` object, including information about the k-NN classifier, the model fit, and the dimensions of the scores matrix.
#'
#' @param x A `classifier` object.
#' @param ... Additional arguments passed to `print()`.
#' @return `classifier` object.
#' @export
print.classifier <- function(x, ...) {
  cat("classifier object:\n")
  cat("  k-NN classifier with k =", x$knn, "\n")
  cat("  Model fit: \n")
  print(x$projector)
  cat("  Scores matrix dimensions: ", nrow(x$scores), "x", ncol(x$scores), "\n")
  invisible(x)
}

