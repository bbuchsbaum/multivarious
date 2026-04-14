#' construct a new pre-processing pipeline
#' 
#' Creates a bare prepper object (a pipeline holder).
#' 
#' TODO: Consider using a single environment in the finalized pre_processor
#'       to store all step parameters, instead of individual environments
#'       per step, potentially improving serialization and GC performance
#'       for very large pipelines.
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
  invisible(x) # Return invisibly for better pipe behavior
}



#' @export
prep.prepper <- function(x,...) {
  lifecycle::deprecate_warn(
    "0.3.0", 
    "prep()", 
    "fit()",
    details = "The prep() function is deprecated. Use fit() for a more standard interface."
  )
  
  # Use local to capture a stable copy of steps
  local({
    steps <- x$steps
    orig_ncol <- NULL # Placeholder for original number of columns
    
    # init transform: applies all forward steps and learns parameters
    tinit <- function(X) {
      xin <- X
      # Store original dimension when initializing
      assign("orig_ncol", ncol(X), envir = parent.env(environment()))
      for (st in steps) {
        xin <- st$forward(xin)
      }
      xin
    }
    
    # transform: apply learned steps in forward direction using partial 'colind'
    tform <- function(X, colind=NULL) {
      xin <- X
      for (st in steps) {
        xin <- st$apply(xin, colind)
      }
      xin
    }
    
    # reverse_transform: apply learned steps in reverse order
    rtform <- function(X, colind=NULL) {
      xin <- X
      # Use rev() for clarity and robustness
      for (st in rev(steps)) {
        xin <- st$reverse(xin, colind)
      }
      xin
    }
    
    ret <- list(
      preproc = x, # Store the original prepper structure if needed
      init = tinit,
      transform = tform,
      reverse_transform = rtform,
      # Store orig_ncol after init is called
      get_orig_ncol = function() orig_ncol 
    )
    
    class(ret) <- "pre_processor"
    ret
  })
}


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
  lifecycle::deprecate_warn(
    "0.3.0", 
    "init_transform()", 
    "fit_transform()",
    details = "init_transform() is deprecated. Use fit_transform() for a more standard interface."
  )
  
  chk::chk_matrix(X)
  res <- x$init(X)
  # After init, orig_ncol should be set, store it directly in the object? 
  # Modifying the object after creation might be tricky with environments.
  # For now, rely on the closure environment within prep.prepper and get_orig_ncol.
  res
}

#' @export
apply_transform.pre_processor <- function(x, X, colind=NULL,...) {
  lifecycle::deprecate_warn(
    "0.3.0", 
    "apply_transform()", 
    "transform()",
    details = "apply_transform() is deprecated. Use transform() for a more standard interface."
  )
  
  chk::chk_matrix(X)
  
  # GENERAL CHECK: Ensure preprocessor is initialized before any application
  # REMOVED: This check prevents using preprocessors with manually supplied parameters
  # orig_ncol <- x$get_orig_ncol()
  # if (is.null(orig_ncol)) {
  #     stop("Preprocessor must be initialized with 'init_transform' before applying transformations.")
  # }
  
  # COLIND SPECIFIC CHECKS: Only if colind is provided
  if (!is.null(colind)) {
    # Check colind against original dimension (orig_ncol is known to be non-NULL here)
    # Need to get orig_ncol without stopping if NULL
    orig_ncol <- x$get_orig_ncol() # May be NULL, that's okay now
    chk::chk_vector(colind)
    if (!is.null(orig_ncol)) { # Only check subset if orig_ncol is known
        chk::chk_subset(colind, 1:orig_ncol)
    }
    # Ensure the provided X matches the dimension implied by colind
    chk::chk_equal(ncol(X), length(colind))
  }
  
  x$transform(X, colind)
}

#' @export
reverse_transform.pre_processor <- function(x, X, colind=NULL,...) {
  lifecycle::deprecate_warn(
    "0.3.0", 
    "reverse_transform()", 
    "inverse_transform()",
    details = "reverse_transform() is deprecated. Use inverse_transform() for a more standard interface."
  )
  
  chk::chk_matrix(X)
  
  # GENERAL CHECK: Ensure preprocessor is initialized before any reversal
  # REMOVED: This check prevents using preprocessors with manually supplied parameters
  # orig_ncol <- x$get_orig_ncol()
  # if (is.null(orig_ncol)) {
  #     stop("Preprocessor must be initialized with 'init_transform' before reversing transformations.")
  # }
  
  # COLIND SPECIFIC CHECKS: Only if colind is provided
  if (!is.null(colind)) {
    # Check colind against original dimension (orig_ncol is known to be non-NULL here)
    # Need to get orig_ncol without stopping if NULL
    orig_ncol <- x$get_orig_ncol() # May be NULL, that's okay now
    chk::chk_vector(colind)
     if (!is.null(orig_ncol)) { # Only check subset if orig_ncol is known
        chk::chk_subset(colind, 1:orig_ncol)
    }
    # Ensure the provided X matches the dimension implied by colind
    chk::chk_equal(ncol(X), length(colind))
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
        chk::chk_not_null(cm, "Means not initialized. Run init_transform first or supply 'cmeans'.")
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
          # Use only the means corresponding to the columns present in X
          nc <- ncol(X)
          if (nc > length(env$cmeans)) {
             stop(sprintf("Internal error in center$reverse: ncol(X) [%d] > length(stored means) [%d]", 
                           nc, length(env$cmeans)))
          } 
          means_to_use <- env$cmeans[1:nc]
          sweep(X, 2, means_to_use, "+")
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
    env$weights <- weights # Store precomputed weights if provided
    
    list(
      forward = function(X) {
        if (is.null(env$weights)) { # Compute weights only if not precomputed
          sds <- matrixStats::colSds(as.matrix(X)) # Ensure matrix for colSds
          
          # Handle zero standard deviations robustly
          zero_sd <- sds < .Machine$double.eps
          if (any(zero_sd)) {
              warning(sprintf("Columns %s have zero standard deviation. Setting scale factor to 1.", 
                              paste(which(zero_sd), collapse=", ")))
          }
          sds[zero_sd] <- 1 # Set zero SDs to 1 to avoid Inf weights
          
          if (type == "unit") {
            # Unit norm scaling: weight is 1 / (sd * sqrt(N-1))
            # Ensure N > 1 for sqrt
            N <- nrow(X)
            if (N <= 1) stop("Cannot compute unit norm scaling with N <= 1.")
            scaling_factor <- sds * sqrt(N - 1)
            # Check for zeros again after scaling (unlikely but possible)
            scaling_factor[scaling_factor < .Machine$double.eps] <- 1
            wts <- 1 / scaling_factor
          } else { # type == "z"
            # z-scaling weight is 1 / sd
            wts <- 1 / sds
          }
          env$weights <- wts
        } else {
            wts <- env$weights
            chk::chk_equal(length(wts), ncol(X))
        }
        sweep(X, 2, wts, "*")
      },
      
      apply = function(X, colind = NULL) {
        chk::chk_not_null(env$weights, "Weights not initialized. Run init_transform first.")
        wts <- env$weights
        if (is.null(colind)) {
          sweep(X, 2, wts, "*")
        } else {
          # Ensure X matches colind length, and colind is valid (checked by caller)
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, wts[colind], "*")
        }
      },
      
      reverse = function(X, colind = NULL) {
        chk::chk_not_null(env$weights, "Weights not initialized. Run init_transform first.")
        wts <- env$weights
        # Handle potential division by zero if weights were somehow zero
        wts_safe <- wts
        wts_safe[abs(wts) < .Machine$double.eps] <- 1 # Avoid division by zero
        
        if (is.null(colind)) {
          # Use only the weights corresponding to the columns present in X
          nc <- ncol(X)
           if (nc > length(wts_safe)) {
             stop(sprintf("Internal error in colscale$reverse: ncol(X) [%d] > length(stored weights) [%d]", 
                           nc, length(wts_safe)))
           } 
          weights_to_use <- wts_safe[1:nc]
          sweep(X, 2, weights_to_use, "/")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          sweep(X, 2, wts_safe[colind], "/")
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
        
        # TODO: [INEFF] This uses two sweep() calls. For large matrices, 
        #       consider combining into a single pass or using optimized 
        #       functions if performance becomes critical.
        x1 <- sweep(X, 2, cmeans2, "-")
        sweep(x1, 2, sds2, "/")
      },
      
      apply = function(X, colind = NULL) {
        sds2 <- env$sds
        cmeans2 <- env$cmeans
        chk::chk_not_null(sds2, "SDs not initialized. Run init_transform first or supply 'sds'.")
        chk::chk_not_null(cmeans2, "Means not initialized. Run init_transform first or supply 'cmeans'.")
        if (is.null(colind)) {
          # TODO: [INEFF] See forward() - two sweep() calls.
          x1 <- sweep(X, 2, cmeans2, "-")
          sweep(x1, 2, sds2, "/")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          # TODO: [INEFF] See forward() - two sweep() calls.
          x1 <- sweep(X, 2, cmeans2[colind], "-")
          sweep(x1, 2, sds2[colind], "/")
        }
      },
      
      reverse = function(X, colind = NULL) {
        sds2 <- env$sds
        cmeans2 <- env$cmeans
        chk::chk_not_null(sds2, "SDs not initialized.")
        chk::chk_not_null(cmeans2, "Means not initialized.")
        
        if (is.null(colind)) {
          # Use only parameters corresponding to columns in X
          nc <- ncol(X)
          if (nc > length(sds2) || nc > length(cmeans2)) {
               stop(sprintf("Internal error in standardize$reverse: ncol(X) [%d] inconsistent with stored params [%d, %d]", 
                           nc, length(sds2), length(cmeans2)))
          }
          sds_to_use <- sds2[1:nc]
          means_to_use <- cmeans2[1:nc]
          
          # TODO: [INEFF] Two sweep() calls.
          x0 <- sweep(X, 2, sds_to_use, "*")
          sweep(x0, 2, means_to_use, "+")
        } else {
          chk::chk_equal(ncol(X), length(colind))
          # TODO: [INEFF] Two sweep() calls.
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
  # Check for overlaps and completeness more thoroughly?
  if (any(duplicated(unraveled_ids))) stop("Duplicate indices found in block_indices.")
  # Assuming indices cover a contiguous range for simplicity now, but could add checks.
  max_idx <- max(unraveled_ids)
  
  # Precompute mapping for efficiency and correct ordering
  map_list <- lapply(seq_along(block_indices), function(i) {
      list(orig_indices = block_indices[[i]], 
           proc = preprocs[[i]])
  })
  names(map_list) <- as.character(seq_along(map_list))

  # Internal helper for applying functions blockwise respecting colind order
  # TODO: [INEFF] This helper recalculates mappings on each call.
  #       If called repeatedly with the same colind structure (unlikely?),
  #       pre-calculating and caching the exact mappings might be faster.
  apply_blockwise <- function(func, X, colind) {
    chk::chk_matrix(X)
    chk::chk_vector(colind)
    # Remove unsupported named arguments x_arg, y_arg
    chk::chk_equal(ncol(X), length(colind))
    
    results_list <- vector("list", length(colind))
    processed_cols <- logical(length(colind)) # Track which columns are processed
    
    # Find which original blocks are relevant for the given colind
    for (block_num in seq_along(map_list)) {
        block_info <- map_list[[block_num]]
        # Find which of the requested `colind` fall into this block's original indices
        matched_in_colind_idx <- which(colind %in% block_info$orig_indices)
        
        if (length(matched_in_colind_idx) > 0) {
            # Get the subset of X corresponding to these columns
            X_subset <- X[, matched_in_colind_idx, drop = FALSE]
            # Get the original GLOBAL indices within this block that correspond to X_subset
            orig_indices_in_block <- colind[matched_in_colind_idx]
            
            # Map global indices to the LOCAL indices expected by the block's preprocessor
            local_indices_for_proc <- match(orig_indices_in_block, block_info$orig_indices)
            if (any(is.na(local_indices_for_proc))) {
                stop("Internal error: Mismatch between requested global indices and block's original indices.")
            }
            
            # Apply the function using the LOCAL indices relative to the block
            res_subset <- func(block_info$proc, X_subset, colind = local_indices_for_proc)
            
            # Place results back in the correct positions in the output list
            results_list[matched_in_colind_idx] <- lapply(seq_len(ncol(res_subset)), function(k) res_subset[,k])
            processed_cols[matched_in_colind_idx] <- TRUE
        }
    }
    
    if (!all(processed_cols)) {
        stop("Some requested columns in 'colind' did not map to any provided block.")
    }
    
    # Combine results respecting original colind order
    do.call(cbind, results_list)
  }
  
  ret <- list(
    # Note: init_transform is not defined for concat_pre_processor
    # User should initialize individual preprocs before concatenating.
    transform = function(X, colind = NULL) {
      chk::chk_matrix(X)
      if (is.null(colind)) {
        # Apply to full blocks in their natural order
        chk::chk_equal(ncol(X), length(unraveled_ids))
        res_list <- lapply(seq_along(map_list), function(i) {
          block_info <- map_list[[i]]
          transform(block_info$proc, X[, block_info$orig_indices, drop = FALSE])
        })
        do.call(cbind, res_list)
      } else {
        # Apply blockwise respecting colind order
        apply_blockwise(transform, X, colind)
      }
    },
    reverse_transform = function(X, colind = NULL) {
      chk::chk_matrix(X)
      if (is.null(colind)) {
        # Reverse full blocks in their natural order
        chk::chk_equal(ncol(X), length(unraveled_ids))
        res_list <- lapply(seq_along(map_list), function(i) {
          block_info <- map_list[[i]]
          inverse_transform(block_info$proc, X[, block_info$orig_indices, drop = FALSE])
        })
        do.call(cbind, res_list)
      } else {
        # Reverse blockwise respecting colind order
        apply_blockwise(inverse_transform, X, colind)
      }
    }
    # Store map_list? Might be useful for introspection.
    # map_list = map_list 
  )
  
  class(ret) <- c("concat_pre_processor", "pre_processor")
  ret
}



#' @export
apply_transform.concat_pre_processor <- function(x, X, colind=NULL,...) {
  # Directly call the transform function stored within the concat object
  x$transform(X, colind, ...)
}

#' @export
reverse_transform.concat_pre_processor <- function(x, X, colind=NULL,...) {
  # Directly call the reverse_transform function stored within the concat object
  x$reverse_transform(X, colind, ...)
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


# =============================================================================
# New S3 Methods for Modern Preprocessing API
# =============================================================================

#' @export
fit.prepper <- function(object, X, ...) {
  # Build and initialize a pre_processor while suppressing deprecated shim warnings
  proc <- suppressWarnings(prep(object))
  fitted_proc <- mark_fitted(proc, TRUE)
  suppressWarnings(init_transform(fitted_proc, X))
  fitted_proc
}

#' @export
fit_transform.prepper <- function(object, X, ...) {
  proc <- suppressWarnings(prep(object))
  fitted_proc <- mark_fitted(proc, TRUE)
  transformed <- suppressWarnings(init_transform(fitted_proc, X))
  list(preproc = fitted_proc, transformed = transformed)
}

#' @export
transform.pre_processor <- function(object, X, colind = NULL, ...) {
  check_fitted(object, "transform")
  object$transform(X, colind)
}

#' @export
inverse_transform.pre_processor <- function(object, X, colind = NULL, ...) {
  check_fitted(object, "inverse_transform")
  object$reverse_transform(X, colind)
}

#' @export
transform.concat_pre_processor <- function(object, X, colind = NULL, ...) {
  check_fitted(object, "transform")
  object$transform(X, colind)
}

#' @export
inverse_transform.concat_pre_processor <- function(object, X, colind = NULL, ...) {
  check_fitted(object, "inverse_transform")
  object$reverse_transform(X, colind)
}
