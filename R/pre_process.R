


#' construct a new pre-processing pipeline
#' 
#' @keywords internal
#' @noRd
prepper <- function() {
  steps <- list()
  ret <- list(steps=steps)
  class(ret) <- c("prepper", "list")
  ret
}

#' @export
add_node.prepper <- function(x, step,...) {
  x$steps[[length(x$steps)+1]] <- step
  x
}


#' @importFrom purrr compose
#' @export
prep.prepper <- function(x,...) {
  steps <- x$steps
  tinit <- function(X) {
    xin <- X
    for (i in 1:length(steps)) {
      xin <- steps[[i]]$forward(xin)
    }
    rm(X)
    xin
  }
  
  tform <- function(X, colind=NULL) {
    xin <- X
    for (i in 1:length(steps)) {
      xin <- steps[[i]]$apply(xin, colind)
    }
    rm(X)
    xin
  }
  
  rtform <- function(X, colind=NULL) {
    xin <- X
    for (i in length(steps):1) {
      xin <- steps[[i]]$reverse(xin, colind)
    }
    rm(X)
    xin
  }
  
  #Xp <- if (!missing(X)) {
  #  tinit(X)
  #} 
  
  #environment(tinit) <- rlang::new_environment()
  #environment(tform) <- rlang::new_environment()
  #environment(rtform) <- rlang::new_environment()
  
  ret <- list(
    preproc=x,
    init=tinit,
    transform=tform,
    reverse_transform=rtform)
  
  
  class(ret) <- "pre_processor"
  ret
  
}

#' @export
fresh.prepper <- function(x,...) {
  p <- prepper()
  for (step in x$steps) {
    p <- prep_node(p, step$name, step$create)
  }
  p
}

#' @export
init_transform.pre_processor <- function(x, X,...) {
  x$init(X)
}

#' @export
apply_transform.pre_processor <- function(x, X, colind=NULL,...) {
  x$transform(X,colind)
}

#' @export
reverse_transform.pre_processor <- function(x, X, colind=NULL,...) {
  x$reverse_transform(X, colind)
}

#' @export
fresh.pre_processor <- function(x, preproc=prepper(),...) {
  p <- x$create()
}

#' prepare a new node and add to pipeline
#' 
#' @param pipeline the pre-processing pipeline
#' @param name the name of the step to add
#' @param create the creation function
#' 
#' @keywords internal
#' @noRd
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


new_pre_processor <- function(x) {
  chk::chk_not_null(x[["forward"]])
  chk::chk_not_null(x[["apply"]])
  chk::chk_not_null(x[["reverse"]])
  chk::chk_function(x[["forward"]])
  chk::chk_function(x[["apply"]])
  chk::chk_function(x[["reverse"]])
  
  funlist <- x
  structure(funlist,
            class="pre_processing_step")
}


#' a no-op pre-processing step
#' 
#' `pass` simply passes its data through the chain
#' 
#' @param preproc the pre-processing pipeline
#' @return a `prepper` list 
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
#' @param cmeans optional vector of precomputed column means
#' 
#' @inheritParams pass
#' @export
#' @importFrom Matrix colMeans
#' @return a `prepper` list 
center <- function(preproc = prepper(), cmeans=NULL) {
  create <- function() {
    #env = new.env()
    env <- rlang::new_environment()
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
#' @inheritParams pass
#' 
#' @param type the kind of scaling, `unit` norm, `z`-scoring, or precomputed `weights`
#' @param weights optional precomputed weights
#' @return a `prepper` list 
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
    #env = new.env()
    env <- rlang::new_environment()
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
          
          sds[sds == 0] <- stats::median(sds)
          1 / sds
        }
        env[["weights"]] <- wts
        sweep(X, 2, wts, "*")
        
      },
      
      apply = function(X, colind = NULL) {
        if (is.null(colind)) {
          sweep(X, 2, env[["weights"]], "*")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, env[["weights"]][colind], "*")
        }
      },
      
      reverse = function(X, colind = NULL) {
        if (is.null(colind)) {
          sweep(X, 2, env[["weights"]], "/")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, env[["weights"]][colind], "/")
        }
      }
    )
  }
  
  prep_node(preproc, "colscale", create)
}

#' center and scale each vector of a matrix
#' 
#' @param cmeans an optional vector of column means
#' @param sds an optional vector of sds
#' @inheritParams pass
#' @return a `prepper` list 
#' @export
standardize <- function(preproc = prepper(), cmeans=NULL, sds=NULL) {
  create <- function() {
    #env = new.env()
    env <- rlang::new_environment()
    list(
      forward = function(X) {
        if (is.null(sds)) {
          sds <- matrixStats::colSds(X)
        } else {
          chk::chk_equal(length(sds), ncol(X))
        }
        
        if (is.null(cmeans)) {
          cmeans <- colMeans(X)
        } else {
          chk::chk_equal(length(cmeans), ncol(X))
        }
        
        sds[sds == 0] <- mean(sds)
        
        env[["sds"]] <- sds
        env[["cmeans"]] <- cmeans
        
        x1 <- sweep(X, 2, cmeans, "-")
        sweep(x1, 2, sds, "/")
      },
      
      apply = function(X, colind = NULL) {
        if (is.null(colind)) {
          x1 <- sweep(X, 2, env[["cmeans"]], "-")
          sweep(x1, 2, env[["sds"]], "/")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          x1 <- sweep(X, 2, env[["cmeans"]][colind], "-")
          sweep(x1, 2, env[["sds"]][colind], "/")
        }
      },
      
      reverse = function(X, colind = NULL) {
        if (is.null(colind)) {
          x0 <- sweep(X, 2, env[["sds"]], "*")
          sweep(x0, 2, env[["cmeans"]], "+")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          x0 <- sweep(X, 2, env[["sds"]][colind], "*")
          sweep(x0, 2, env[["cmeans"]][colind], "+")
        }
      }
    )
  }
  prep_node(preproc, "standardize", create)
}


#' bind together blockwise pre-processors
#' 
#' 
#' concatenate a sequence of pre-processors, each previously applied to a block of data.
#' 
#' @param preprocs a list of initialized `pre-processor` objects
#' @param block_indices a list of block indices where each vector in the list 
#' contains the global indices of the variables.
#' @return a new `prepper` object
#' @examples 
#' 
#' p1 <- center() |> prep()
#' p2 <- center() |> prep()
#' 
#' x1 <- rbind(1:10, 2:11)
#' x2 <- rbind(1:10, 2:11)
#' 
#' p1a <- init_transform(p1,x1)
#' p2a <- init_transform(p2,x2)
#' 
#' clist <- concat_pre_processors(list(p1,p2), list(1:10, 11:20))
#' t1 <- apply_transform(clist, cbind(x1,x2))
#' 
#' t2 <- apply_transform(clist, cbind(x1,x2[,1:5]), colind=1:15)
#' @export
concat_pre_processors <- function(preprocs, block_indices) {
  chk::chk_equal(length(preprocs), length(block_indices))
  unraveled_ids <- unlist(block_indices)
  blk_ids <- rep(seq_along(block_indices), sapply(block_indices,length))
  idmap <- data.frame(id_global=unraveled_ids, 
                      id_block=unlist(lapply(block_indices, function(x) seq_along(x))),
                      block=blk_ids)
  

  apply_fun <- function(f, X, colind) {
    #browser()
    chk::chk_equal(ncol(X), length(colind))
    keep <- idmap$id_global %in% colind
    blks <- sort(unique(idmap$block[keep]))
    
    idmap2 <- idmap[keep,]
    do.call(cbind, lapply(blks, function(i) {
      loc <- idmap2$id_block[idmap2$block == i]
      offset <- which(idmap2$block == i)
      f(preprocs[[i]], X[,offset,drop=FALSE], colind=loc)
    }))
  }
  
  ret <- list(
    transform = function(X, colind = NULL) {
        if (!is.null(colind)) {
          apply_fun(apply_transform, X, colind)
        } else {
          chk::chk_equal(ncol(X), length(unraveled_ids))
          do.call(cbind, lapply(1:length(block_indices), function(i) {
            apply_transform(preprocs[[i]], X[,block_indices[[i]]])
          }))
        }
        
      },
      reverse_transform = function(X, colind = NULL) {
        if (!is.null(colind)) {
          apply_fun(reverse_transform, X, colind)
        } else {
          chk::chk_equal(ncol(X), length(unraveled_ids))
          do.call(cbind, lapply(1:length(block_indices), function(i) {
            reverse_transform(preprocs[[i]], X[,block_indices[[i]]])
          }))
        }
        
      }
  )
  
  class(ret) <- c("concat_pre_processor", "pre_processor")
  ret
 
}


#' @export
print.prepper <- function(x,...) {
  nn <- sapply(x$steps, function(x) x$name)
  cat("preprocessor: ", paste(nn, collapse="->"))
}



