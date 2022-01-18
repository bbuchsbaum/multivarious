#' @export
classifier.multiblock_biprojector <- function(x, labels, new_data=NULL, colind=NULL, block=NULL, knn=1) {
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


#' @export
classifier.multiblock_projector <- function(x, labels, new_data, colind=NULL, block=NULL, knn=1) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
    if (!is.null(block)) {
      rlang::abort("can either supply `colind` or `block` but not both")
    }
  }
  
  scores <- if (!is.null(colind)) {
    scores <- partial_project(x, new_data, colind=colind)
  } else if (!is.null(block)) {
    chk::chk_whole_number(block)
    project_block(x, new_data, block)
  }
  
  new_classifier(x,labels,scores, colind, block=block, knn=knn, classes="multiblock_classifier")
  
}



#' @export
classifier.discriminant_projector <- function(x, colind=NULL, knn=1) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
  }
  
  new_classifier(x, x$labels, scores(x), colind=colind, knn=knn)
}

#' @keywords internal
#' @export
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

#' create a classifier
#' 
#' construct a classifier from a `projector` instance
#' 
#' @param x the `projector` instance
#' @param labels the labels associated with the rows of the projected data (see `new_data`)
#' @param new_data reference data associated with `labels` and to be projected into subspace (required).
#' @param colind the subset of column indices in the fitted model to use.
#' @param knn the number of nearest neighbors to use when classifying a new point. 
#' @export
#' 
#' @examples
#' data(iris)
#' X <- iris[,1:4]
#' pcres <- pca(as.matrix(X),2)
#' cfier <- classifier(pcres, iris[,5], new_data=iris[,1:4])
classifier.projector <- function(x, labels, new_data, colind=NULL, knn=1) {
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
normalize_probs <- function(p) {
  apply(p, 2, function(v) {
    v2 <- v - min(v)
    v2/sum(v2)
  })
}

#' @keywords internal
avg_probs <- function(prob, labels) {
  pmeans <- t(group_means(labels, prob))
  t(apply(pmeans, 1, function(v) v/sum(v)))
}


#' @keywords internal
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

#' @param ncomp the number of components to use
#' @param colind the column indices to select in the projection matrix
#' @param metric the similarity metric ("euclidean" or "cosine")
#' @param additional arguments to projection function
#' @export
predict.classifier <- function(object, new_data, ncomp=NULL,
                               colind=NULL, metric=c("cosine", "euclidean"), ...) {

  if (is.vector(new_data)) {
    chk::chk_equal(length(new_data), shape(object$projector)[1])
    new_data <- matrix(new_data, nrow=1)
  }
  
  if (is.null(ncomp)) {
    ncomp <- shape(object$projector)[2]
  } else {
    chk::chk_range(ncomp, c(1,shape(object$projector)[2]))
  }
  
  metric <- match.arg(metric)
  

  if (!is.null(colind)) {
    ### colind overrides object$colind, should emit warning?
    if (length(colind) == 1 && is.vector(new_data)) {
      new_data <- as.matrix(new_data)
    }
    chk::chk_equal(length(colind), ncol(new_data))
    proj <- partial_project(object$projector, new_data, colind, ...)
    
  } else if (!is.null(object$colind)) {
    chk::chk_equal(length(object$colind), ncol(new_data))
    proj <- partial_project(object$projector, new_data, object$colind, ...)
  } else {
    proj <- project(object$projector, new_data,...)
  }
  
  
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

