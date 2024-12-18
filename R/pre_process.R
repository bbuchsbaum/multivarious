#' construct a new pre-processing pipeline
#' 
#' Creates a bare prepper object (a pipeline holder).
#' 
#' @keywords internal
#' @noRd
prepper <- function() {
  steps <- list()
  ret <- list(steps=steps)
  class(ret) <- c("prepper", "list")
  ret
}

#' Add a pre-processing node to a pipeline
#' 
#' @param x A `prepper` pipeline
#' @param step The pre-processing step to add
#' @param ... Additional arguments
#' @export
add_node.prepper <- function(x, step,...) {
  x$steps[[length(x$steps)+1]] <- step
  x
}


#' finalize a prepper pipeline
#' 
#' Prepares a pre-processing pipeline for application by creating `init`, `transform`, and `reverse_transform` functions.
#'
#' @export
prep.prepper <- function(x,...) {
  steps <- x$steps
  
  # init transform: applies all forward steps
  tinit <- function(X) {
    xin <- X
    for (st in steps) {
      xin <- st$forward(xin)
    }
    xin
  }
  
  # transform: apply steps in forward direction using partial 'colind'
  tform <- function(X, colind=NULL) {
    xin <- X
    for (st in steps) {
      xin <- st$apply(xin, colind)
    }
    xin
  }
  
  # reverse_transform: apply steps in reverse order
  rtform <- function(X, colind=NULL) {
    xin <- X
    for (i in seq_along(steps)) {
      st <- steps[[length(steps)-i+1]]
      xin <- st$reverse(xin, colind)
    }
    xin
  }
  
  ret <- list(
    preproc=x,
    init=tinit,
    transform=tform,
    reverse_transform=rtform
  )
  
  class(ret) <- "pre_processor"
  ret
}

#' Create a fresh pipeline from an existing prepper
#'
#' Recreates the pipeline structure without any learned parameters.
#'
#' @export
fresh.prepper <- function(x,...) {
  p <- prepper()
  for (step in x$steps) {
    # Recreate each node by calling 'prep_node' again
    p <- prep_node(p, step$name, step$create)
  }
  p
}

#' @export
init_transform.pre_processor <- function(x, X,...) {
  chk::chk_matrix(X)
  x$init(X)
}

#' @export
apply_transform.pre_processor <- function(x, X, colind=NULL,...) {
  chk::chk_matrix(X)
  if (!is.null(colind)) {
    chk::chk_range(max(colind), c(1,ncol(X)))
    chk::chk_range(min(colind), c(1,ncol(X)))
  }
  x$transform(X, colind)
}

#' @export
reverse_transform.pre_processor <- function(x, X, colind=NULL,...) {
  chk::chk_matrix(X)
  if (!is.null(colind)) {
    chk::chk_range(max(colind), c(1,ncol(X)))
    chk::chk_range(min(colind), c(1,ncol(X)))
  }
  x$reverse_transform(X, colind)
}

#' @export
fresh.pre_processor <- function(x, preproc=prepper(),...) {
  # Attempt to recreate the original pipeline from stored steps in x$preproc
  # similar to fresh.prepper
  chk::chk_s3_class(x$preproc, "prepper")
  fresh(x$preproc)
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
    env <- rlang::new_environment()
    env[["cmeans"]] <- cmeans
    
    list(
      forward = function(X) {
        if (is.null(env$cmeans)) {
          cm <- colMeans(X)
          env$cmeans <- cm
        } else {
          cm <- env$cmeans
          chk::chk_equal(ncol(X), length(cm))
        }
        sweep(X, 2, cm, "-")
      },
      
      apply = function(X, colind = NULL) {
        cm <- env$cmeans
        if (is.null(colind)) {
          sweep(X, 2, cm, "-")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, cm[colind], "-")
        }
      },
      
      reverse = function(X, colind = NULL) {
        chk::chk_not_null(env$cmeans)
        if (is.null(colind)) {
          sweep(X, 2, env$cmeans, "+")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, env$cmeans[colind], "+")
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
    env <- rlang::new_environment()
    list(
      forward = function(X) {
        wts <- if (type == "weights") {
          chk::chk_equal(length(weights), ncol(X))
          weights
        } else {
          sds <- matrixStats::colSds(X)
          if (all(sds == 0)) {
            # If all zeros, set them to 1 to avoid division by zero
            sds[] <- 1
          }
          if (type == "unit") {
            # Unit norm: scale by sqrt(nrow(X)-1)
            sds[sds == 0] <- mean(sds)
            sds <- sds * sqrt(nrow(X) - 1)
          } else {
            # z-scaling sds already fine
            sds[sds == 0] <- mean(sds)
          }
          1 / sds
        }
        env$weights <- wts
        sweep(X, 2, wts, "*")
      },
      
      apply = function(X, colind = NULL) {
        wts <- env$weights
        if (is.null(colind)) {
          sweep(X, 2, wts, "*")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, wts[colind], "*")
        }
      },
      
      reverse = function(X, colind = NULL) {
        wts <- env$weights
        if (is.null(colind)) {
          sweep(X, 2, wts, "/")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, wts[colind], "/")
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
    env <- rlang::new_environment()
    list(
      forward = function(X) {
        if (is.null(sds)) {
          sds2 <- matrixStats::colSds(X)
        } else {
          chk::chk_equal(length(sds), ncol(X))
          sds2 <- sds
        }
        
        if (is.null(cmeans)) {
          cmeans2 <- colMeans(X)
        } else {
          chk::chk_equal(length(cmeans), ncol(X))
          cmeans2 <- cmeans
        }
        
        if (all(sds2 == 0)) {
          sds2[] <- mean(sds2[sds2>0], na.rm=TRUE)
          sds2[is.na(sds2)] <- 1 # fallback if all zero
        }
        
        env$sds <- sds2
        env$cmeans <- cmeans2
        
        x1 <- sweep(X, 2, cmeans2, "-")
        sweep(x1, 2, sds2, "/")
      },
      
      apply = function(X, colind = NULL) {
        sds2 <- env$sds
        cmeans2 <- env$cmeans
        if (is.null(colind)) {
          x1 <- sweep(X, 2, cmeans2, "-")
          sweep(x1, 2, sds2, "/")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          x1 <- sweep(X, 2, cmeans2[colind], "-")
          sweep(x1, 2, sds2[colind], "/")
        }
      },
      
      reverse = function(X, colind = NULL) {
        sds2 <- env$sds
        cmeans2 <- env$cmeans
        if (is.null(colind)) {
          x0 <- sweep(X, 2, sds2, "*")
          sweep(x0, 2, cmeans2, "+")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          x0 <- sweep(X, 2, sds2[colind], "*")
          sweep(x0, 2, cmeans2[colind], "+")
        }
      }
    )
  }
  prep_node(preproc, "standardize", create)
}


#' bind together blockwise pre-processors
#' 
#' concatenate a sequence of pre-processors, each applied to a block of data.
#' 
#' @param preprocs a list of initialized `pre_processor` objects
#' @param block_indices a list of integer vectors specifying the global column indices for each block
#' @return a new `pre_processor` object that applies the correct transformations blockwise
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
    chk::chk_matrix(X)
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
      chk::chk_matrix(X)
      if (is.null(colind)) {
        chk::chk_equal(ncol(X), length(unraveled_ids))
        do.call(cbind, lapply(seq_along(block_indices), function(i) {
          apply_transform(preprocs[[i]], X[,block_indices[[i]],drop=FALSE])
        }))
      } else {
        apply_fun(apply_transform, X, colind)
      }
    },
    reverse_transform = function(X, colind = NULL) {
      chk::chk_matrix(X)
      if (is.null(colind)) {
        chk::chk_equal(ncol(X), length(unraveled_ids))
        do.call(cbind, lapply(seq_along(block_indices), function(i) {
          reverse_transform(preprocs[[i]], X[,block_indices[[i]],drop=FALSE])
        }))
      } else {
        apply_fun(reverse_transform, X, colind)
      }
    }
  )
  
  class(ret) <- c("concat_pre_processor", "pre_processor")
  ret
}


#' Print a prepper pipeline
#' 
#' Uses `crayon` to produce a colorful and readable representation of the pipeline steps.
#'
#' @param x A `prepper` object.
#' @param ... Additional arguments (ignored).
#' @export
print.prepper <- function(x,...) {
  nn <- sapply(x$steps, function(st) st$name)
  if (length(nn) == 0) {
    cat(crayon::cyan("A preprocessor with no steps.\n"))
    return(invisible(x))
  }
  
  cat(crayon::bold(crayon::green("Preprocessor pipeline:\n")))
  for (i in seq_along(nn)) {
    cat(crayon::magenta(" Step ", i, ": "), crayon::cyan(nn[i]), "\n", sep="")
  }
  invisible(x)
}


#' Print a pre_processor object
#' 
#' Display information about a `pre_processor` using crayon-based formatting.
#' 
#' @param x A `pre_processor` object.
#' @param ... Additional arguments (ignored).
#' @export
print.pre_processor <- function(x, ...) {
  # A pre_processor comes from prep(prepper)
  # It has x$preproc to show original steps.
  # Let's show the chain of steps and indicate it's finalized.
  
  cat(crayon::bold(crayon::green("A finalized pre-processing pipeline:\n")))
  if (!is.null(x$preproc) && inherits(x$preproc, "prepper")) {
    nn <- sapply(x$preproc$steps, function(st) st$name)
    if (length(nn) == 0) {
      cat(crayon::cyan("  No steps.\n"))
    } else {
      for (i in seq_along(nn)) {
        cat(crayon::magenta(" Step ", i, ": "), crayon::cyan(nn[i]), "\n", sep="")
      }
    }
  } else {
    cat(crayon::cyan("  No associated prepper information.\n"))
  }
  invisible(x)
}


#' Print a concat_pre_processor object
#' 
#' @param x A `concat_pre_processor` object.
#' @param ... Additional arguments (ignored).
#' @export
print.concat_pre_processor <- function(x, ...) {
  cat(crayon::bold(crayon::green("A concatenated (blockwise) pre-processing pipeline:\n")))
  cat(crayon::cyan("  This object applies different pre-processors to distinct column blocks.\n"))
  invisible(x)
}
