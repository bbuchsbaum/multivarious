
#' @export
classifier.discriminant_projector <- function(x, colind=NULL, knn=1) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
  }
  
  structure(
    list(
      projector=x,
      labels=x$labels,
      scores=scores(x),
      colind=colind,
      knn=knn),
    class="classifier"
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
#' cfier <- classifier(pcres, iris[,5])
classifier.projector <- function(x, labels, new_data, colind=NULL, knn=1) {
  if (!is.null(colind)) {
    chk::chk_true(length(colind) <= shape(x)[1])
    chk::chk_true(all(colind>0))
  }
  
  scores <- if (!is.null(colind)) {
    scores <- partial_project(x, new_data, colind=colind)
  } else {
    project(x, new_data)
  }
  
  chk::chk_equal(length(labels), nrow(newdata))
  
  structure(
    list(
      projector=x,
      labels=labels,
      scores=scores,
      colind=colind,
      knn=knn),
    class="classifier"
  )
  
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
project.classifier <- function(x, new_data) {
  scores <- if (!is.null(x$colind)) {
    partial_project(x$projector, new_data, colind=colind)
  } else {
    project(x$projector, new_data)
  }
}

#' @export
predict.classifier <- function(object, new_data, ncomp=ncomp(object$projector),
                               metric=c("cosine", "euclidean")) {
  if (is.vector(new_data)) {
    chk::chk_equal(length(new_data), shape(object$projector)[1])
    new_data <- matrix(new_data, nrow=1)
  }
  
  metric <- match.arg(metric)
  
  proj <- project(object, new_data)
  
  doit <- function(p) {
    prob <- normalize_probs(p)
    pmeans <- avg_probs(prob, object$labels)
    cls <- nearest_class(prob, object$labels, object$knn)
    
    list(class=cls, prob=pmeans)
  }
  
  
  if (metric == "cosine") {
    p <- proxy::simil(as.matrix(object$scores)[,1:ncomp,drop=FALSE], as.matrix(proj)[,1:ncomp,drop=FALSE], method="cosine")
    doit(p)
    
  } else if (metric == "euclidean") {
    D <- proxy::dist(as.matrix(object$scores)[,1:ncomp,drop=FALSE], as.matrix(proj)[,1:ncomp,drop=FALSE], method="euclidean")
    doit(exp(-D))
  }
  
}

