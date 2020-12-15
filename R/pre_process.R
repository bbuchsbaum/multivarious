


#' construct a new pre-processing pipeline
#' 
#' @keywords internal
prepper <- function() {
  steps <- list()
  ret <- list(steps=steps)
  class(ret) <- c("prepper", "list")
  ret
}

#' @export
add_node.prepper <- function(preproc, step) {
  preproc$steps[[length(preproc$steps)+1]] <- step
  preproc
}


#' @importFrom purrr compose
#' @export
prep.prepper <- function(x) {
  
  tinit <- function(X) {
    xin <- X
    for (i in 1:length(x$steps)) {
      xin <- x$steps[[i]]$forward(xin)
    }
    
    xin
  }
  
  tform <- function(X, colind=NULL) {
    xin <- X
    for (i in 1:length(x$steps)) {
      xin <- x$steps[[i]]$apply(xin, colind)
    }
    
    xin
  }
  
  rtform <- function(X, colind=NULL) {
    xin <- X
    for (i in length(x$steps):1) {
      xin <- x$steps[[i]]$reverse(xin, colind)
    }
    
    xin
  }
  
  #Xp <- if (!missing(X)) {
  #  tinit(X)
  #} 
  
  ret <- list(
    preproc=x,
    init=tinit,
    transform=tform,
    reverse_transform=rtform)
  
  
  class(ret) <- "pre_processor"
  ret
  
}

fresh.prepper <- function(x) {
  p <- prepper()
  for (step in x$steps) {
    p <- prep_node(p, step$name, step$create)
  }
  p
}

init_transform.pre_processor <- function(x, X) {
  x$init(X)
}

apply_transform.pre_processor <- function(x, X) {
  x$transform(X)
}

reverse_transform.pre_processor <- function(x, X) {
  x$reverse_transform(X)
}

fresh.pre_processor <- function(x, preproc=prepper()) {
  p <- x$create()
}

#' prepare a new node and add to pipeline
#' 
#' @param pipeline the pre-processing pipeline
#' @param name the name of the step to add
#' @param create the creation function
#' 
#' @keywords internal
prep_node <- function(pipeline, name, create,  ...) {
  node <- create()
  ret <- list(name=name,
              create=create,
              forward=node$forward,
              reverse=node$reverse,
              apply=node$apply,
              ...)
  class(ret) <- c(name, "pre_processor")
  add_node(pipeline, ret)
}


#' a no-op pre-processing step
#' 
#' `pass` simply passes its data through the chain
#' 
#' @param preproc the pre-processing pipeline
#' @export
pass <- function(preproc=prepper()) {
  
  create <- function() {
    list(
      forward = function(X, colind=NULL) {
        X
      },
      
      reverse = function(X, colind=NULL) {
        X
      },
      
      apply = function(X, colind=NULL) {
        X
      }
    )
  }
  
  prep_node(preproc, "pass", create)
  
}

## TODO for centering sparse matrices, see:
## https://stackoverflow.com/questions/39284774/column-rescaling-for-a-very-large-sparse-matrix-in-r
## 


#' center a data matrix
#' 
#' remove mean of all columns in matrix
#' 
#' @inheritParams pass
#' @export
#' @importFrom Matrix colMeans
center <- function(preproc = prepper(), cmeans=NULL) {
  create <- function() {
    env = new.env()
    env[["cmeans"]] <- cmeans
    
    list(
      forward = function(X) {
        if (is.null(env[["cmeans"]])) {
          cmeans <- colMeans(X)
          env[["cmeans"]] <- cmeans
        } else {
          cmeans <- env[["cmeans"]]
          chk::chk_equal(ncol(X), length(cmeans))
        }
        
        #print(cmeans)
        #message("forward cmeans:", env[["cmeans"]])
        sweep(X, 2, cmeans, "-")
      },
      
      apply = function(X, colind = NULL) {
        cmeans <- env[["cmeans"]]
        #message("apply cmeans:", cmeans)
        if (is.null(colind)) {
          sweep(X, 2, cmeans, "-")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, cmeans[colind], "-")
        }
      },
      
      reverse = function(X, colind = NULL) {
        chk::chk_not_null(env[["cmeans"]])
        if (is.null(colind)) {
          #message("reverse cmeans: ", env[["cmeans"]])
          sweep(X, 2, env[["cmeans"]], "+")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, env[["cmeans"]][colind], "+")
        }
      }
    )
  }
  
  prep_node(preproc, "center", create)
}


#' scale a data matrix
#' 
#' normalize each column by a scale factor.
#' 
#' @export
colscale <- function(preproc = prepper(),
                     type = c("unit", "z", "weights"),
                     weights = NULL) {
  type <- match.arg(type)
  
  if (type != "weights" && !is.null(weights)) {
    warning("colscale: weights ignored because type != 'weights'")
  }
  if (type == "weights") {
    chk::chk_not_null(weights)
  }
  
  create <- function() {
    env = new.env()
    list(
      forward = function(X) {
        wts <- if (type == "weights") {
          chk::chk_equal(length(weights), ncol(X))
          weights
        } else {
          sds <- matrixStats::colSds(X)
          
          if (type == "unit") {
            sds <- sds * sqrt(nrow(X) - 1)
          }
          
          sds[sds == 0] <- median(sds)
          1 / sds
        }
        env[["weights"]] <- wts
        sweep(X, 2, wts, "*")
        
      },
      
      apply = function(X, colind = NULL) {
        if (is.null(colind)) {
          sweep(X, 2, env[["weights"]], "*")
        } else {
          assert_that(ncol(X) == length(colind))
          sweep(X, 2, env[["weights"]][colind], "*")
        }
      },
      
      reverse = function(X, colind = NULL) {
        if (is.null(colind)) {
          sweep(X, 2, env[["weights"]], "/")
        } else {
          assert_that(ncol(X) == length(colind))
          sweep(X, 2, env[["weights"]][colind], "/")
        }
      }
    )
  }
  
  prep_node(preproc, "colscale", create)
}


print.prepper <- function(object) {
  nn <- sapply(object$steps, function(x) x$name)
  cat("preprocessor: ", paste(nn, collapse="->"))
  
}



