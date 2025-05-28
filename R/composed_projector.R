#' Compose Multiple Partial Projectors
#'
#' Creates a `composed_partial_projector` object that applies partial projections sequentially.
#' If multiple projectors are composed, the column indices (colind) used at each stage must be considered.
#'
#' @param ... A sequence of projectors that implement `partial_project()`, optionally named.
#' @return A `composed_partial_projector` object.
#' @export
#'
compose_partial_projector <- function(...) {
  args <- list(...)
  arg_names <- names(args)
  if (is.null(arg_names)) {
    arg_names <- paste0("stage_", seq_along(args))
  } else {
    # Ensure unnamed stages get default names
    unnamed_idx <- which(arg_names == "")
    if (length(unnamed_idx) > 0) {
      arg_names[unnamed_idx] <- paste0("stage_", unnamed_idx)
    }
  }
  # Check for duplicate names
  if (anyDuplicated(arg_names)) {
    stop("Duplicate stage names provided: ", 
         paste(arg_names[duplicated(arg_names)], collapse = ", "))
  }
  names(args) <- arg_names # Assign final names back to the list
  
  lapply(args, function(p) chk::chk_s3_class(p, "projector"))
  
  if (length(args) == 1) {
    return(args[[1]])
  }
  
  shapelist <- lapply(args, shape)
  for (i in 2:length(args)) {
    chk::chk_equal(shapelist[[i-1]][2], shapelist[[i]][1])
  }
  
  out <- structure(
    list(projectors = args),
    class = c("composed_partial_projector", "composed_projector", "projector"),
    .cache = new.env(parent = emptyenv()),
    stage_names = arg_names,
    index_map = NULL # Placeholder for lineage tracking
  )
  out
}


#' Partial Project Through a Composed Partial Projector
#'
#' Applies `partial_project()` through each projector in the composition.
#' If `colind` is a single vector, it applies to the first projector only. Subsequent projectors apply full columns.
#' If `colind` is a list, each element specifies the `colind` for the corresponding projector in the chain.
#'
#' @param x A `composed_partial_projector` object.
#' @param new_data The input data matrix or vector.
#' @param colind A numeric vector or a list of numeric vectors/NULLs. 
#'   If a single vector, applies to the first projector only. 
#'   If a list, its length should ideally match the number of projectors. 
#'   `colind[[i]]` specifies the column indices (relative to the *input* of stage `i`) 
#'   to use for the partial projection at stage `i`. A `NULL` entry means use full projection 
#'   for that stage. If the list is shorter than the number of stages, `NULL` (full projection) 
#'   is assumed for remaining stages. If a single numeric vector is provided, it is treated 
#'   as `list(colind, NULL, NULL, ...)` for backward compatibility (partial only at first stage).
#' @param ... Additional arguments passed to `partial_project()` or `project()` methods.
#'
#' @return The partially projected data after all projectors are applied.
#' @export
partial_project.composed_partial_projector <- function(x, new_data, colind = NULL, ...) {
  projs <- x$projectors
  n_proj <- length(projs)
  
  if (n_proj == 0) return(new_data) # No projectors, return data as is
  
  # Standardize colind format to a list (user provides indices relative to ORIGINAL input)
  if (!is.list(colind)) {
    if (is.null(colind)) {
      colind_list_orig <- rep(list(NULL), n_proj)
    } else {
      chk::chk_vector(colind)
      chk::chk_numeric(colind)
      colind_list_orig <- c(list(colind), rep(list(NULL), n_proj - 1))
    }
  } else {
    if (length(colind) > n_proj) {
      warning("Length of 'colind' list is greater than the number of stages. Extra elements ignored.")
      colind_list_orig <- colind[1:n_proj]
    } else if (length(colind) < n_proj) {
      colind_list_orig <- c(colind, rep(list(NULL), n_proj - length(colind)))
    } else {
      colind_list_orig <- colind
    }
    lapply(colind_list_orig, function(ci) {
      if (!is.null(ci)) {
        chk::chk_vector(ci)
        chk::chk_numeric(ci)
      }
    })
  }
  
  # Handle input data shape
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, nrow = 1)
  }
  chk::vld_matrix(new_data)
  
  # Initialize mapping: Input columns for stage 1 are original columns 1:p
  # This assumes the user provides data corresponding ONLY to the original cols 
  # specified in colind_list_orig[[1]] if it's not NULL.
  p_initial <- shape(projs[[1]])[1]
  current_input_to_orig_map <- 1:p_initial

  # Apply projectors sequentially
  current_data <- new_data
  for (i in 1:n_proj) {
    proj <- projs[[i]]
    stage_colind_orig <- colind_list_orig[[i]] # User-specified ORIGINAL indices
    
    if (is.null(stage_colind_orig)) {
      # Full projection for this stage
      # Input data (`current_data`) columns correspond to `current_input_to_orig_map`
      current_data <- project(proj, current_data, ...)
      # Output map retains all output dimensions, but lineage isn't tracked here yet
      # For now, assume the output columns map 1:k relative to the output
      # A full lineage map would be needed to track origin precisely.
      # Placeholder: The new map just indicates the dimension
      current_input_to_orig_map <- 1:shape(proj)[2] 
    } else {
      # Partial projection for this stage
      
      # 1. Find which columns of `current_data` correspond to `stage_colind_orig`
      #    We need the positions (local indices) within `current_data` that match 
      #    the original indices provided by the user.
      local_indices_to_use <- which(current_input_to_orig_map %in% stage_colind_orig)
      
      if (length(local_indices_to_use) == 0) {
           stop("Stage ", i, ": None of the requested original column indices (", 
                paste(stage_colind_orig, collapse=","), ") were found in the input data to this stage (",
                "which corresponds to original indices: ", paste(current_input_to_orig_map, collapse=","),").")
      }
      
      # 2. Check if the provided `current_data` actually contains these columns
      #    The user must provide input `new_data` that aligns with the first non-NULL colind.
      #    For subsequent stages, `current_data` *is* the input.
      #    The number of columns in the input data for this stage MUST match the number 
      #    of *available* input columns passed from the previous stage.
      #    The `partial_project` call below needs data ONLY for the `local_indices_to_use`.
      if (i == 1) {
          # First stage: ncol(new_data) must match length(stage_colind_orig)
           chk::chk_equal(ncol(current_data), length(stage_colind_orig))
           data_for_partial_proj <- current_data # User supplied correctly subsetted data
           # The local indices *for the projector* are simply 1:length(stage_colind_orig)
           # because the data is already subsetted. But the projector's `partial_project`
           # needs the indices relative to *its* full input space (original indices).
           # --> This implies partial_project methods need to accept indices relative to original.
           # Let's assume partial_project(proj, data_subset, colind_orig) works.
           # We pass the subsetted data and the ORIGINAL indices it corresponds to.
           current_data <- partial_project(proj, data_for_partial_proj, stage_colind_orig, ...)

      } else {
          # Subsequent stages: Select the necessary columns from the output of the previous stage
          data_for_partial_proj <- current_data[, local_indices_to_use, drop = FALSE]
          # The `partial_project` method needs indices relative to *its* full input dimension.
          # The full input dimension corresponds to the output of the previous stage.
          # The `local_indices_to_use` are exactly those indices.
          current_data <- partial_project(proj, data_for_partial_proj, local_indices_to_use, ...)
      }
      
      # 3. Update the map for the next stage's input
      # The output columns map 1:k relative to the output.
      # Lineage isn't fully tracked without a proper index_map structure.
      # Placeholder: The new map just indicates the dimension
      current_input_to_orig_map <- 1:shape(proj)[2]
    }
  }
  
  current_data
}


#' @export
project.composed_projector <- function(x, new_data, ...) {
  if (is.vector(new_data)) {
    new_data <- matrix(new_data, nrow=1)
  }
  
  chk::vld_matrix(new_data)
  
  # Apply each projector in sequence
  for (proj in x$projectors) {
    new_data <- project(proj, new_data, ...)
  }
  
  new_data
}

#' @export
print.composed_projector <- function(x, ...) {
  n_proj <- length(x$projectors)
  stage_names <- attr(x, "stage_names", exact = TRUE)
  if (is.null(stage_names)) { # Fallback if attribute missing somehow
      stage_names <- paste0("stage_", seq_len(n_proj))
  }
  
  cat("Composed projector object:\n")
  cat("  Number of projectors: ", n_proj, "\n")
  
  if (n_proj > 0) {
      cat(" Pipeline:\n")
      shapes <- lapply(x$projectors, shape)
      # Determine max name length for alignment
      max_name_len <- max(nchar(stage_names))
      
      # Print initial input dimension
      # cat(sprintf("  %-*s %4d vars\n", max_name_len + 2, "Input", shapes[[1]][1]))
      
      for (i in 1:n_proj) {
          stage_name <- stage_names[i]
          in_dim <- shapes[[i]][1]
          out_dim <- shapes[[i]][2]
          # Right-align stage number
          stage_label <- sprintf("%-*s", max_name_len, stage_name)
          cat(sprintf("  %2d. %s : %4d -> %4d\n", i, stage_label, in_dim, out_dim))
      }
  }
  invisible(x)
}

#' Get Coefficients of a Composed Projector
#'
#' Calculates the effective coefficient matrix that maps from the original 
#' input space (of the first projector) to the final output space (of the 
#' last projector). This is done by multiplying the coefficient matrices 
#' of all projectors in the sequence.
#'
#' @param object A `composed_projector` object.
#' @param ... Currently unused.
#'
#' @return A matrix representing the combined coefficients.
#' @export
coef.composed_projector <- function(object, ...) {
  cache_env <- attr(object, ".cache", exact = TRUE)
  use_caching <- !is.null(cache_env) && is.environment(cache_env)
  key <- "combined_coef"
  
  if (use_caching && !is.null(cache_env[[key]])) {
    return(cache_env[[key]])
  }
  
  projs <- object$projectors
  if (length(projs) == 0) {
    # Or should it return NULL or an identity matrix of appropriate size?
    # Returning NULL seems safest if there are no projectors.
    return(NULL) 
  }
  
  # Get coefficient matrix from the first projector
  combined_coef <- coef(projs[[1]])
  
  # Sequentially multiply by subsequent coefficient matrices
  if (length(projs) > 1) {
    for (i in 2:length(projs)) {
      # Ensure coef method exists for this projector stage
      stage_coef <- coef(projs[[i]]) 
      if (is.null(combined_coef) || is.null(stage_coef)) {
          warning(paste("Cannot compute combined coefficient matrix because stage", 
                        i, "or previous stages lack coefficients."))
          return(NULL)
      }
      # Matrix multiplication: V_combined = V_prev %*% V_stage_i
      combined_coef <- combined_coef %*% stage_coef
    }
  }
  
  # Store in cache if available
  if (use_caching) {
    cache_env[[key]] <- combined_coef
  }
  
  combined_coef
}

#' Compose Projectors Sequentially (Pipe Operator)
#' 
#' This infix operator provides syntactic sugar for composing projectors sequentially.
#' It is an alias for \code{\link{compose_partial_projector}}.
#' 
#' @param lhs The left-hand side projector (or a composed projector).
#' @param rhs The right-hand side projector to add to the sequence.
#' 
#' @return A \code{composed_partial_projector} object representing the combined sequence.
#' @export
#' @rdname compose_partial_projector
`%>>%` <- function(lhs, rhs) {
   # Check if lhs is already a composed projector
   # Ensure rhs is a list containing the single projector
   if (!is.list(rhs) || length(rhs) != 1 || !inherits(rhs[[1]], "projector")) {
       # If rhs is a projector, wrap it in a list for consistent handling
       if (inherits(rhs, "projector")) {
           rhs_list <- list(rhs)
           names(rhs_list) <- paste0("stage_", length(list(...)) + 1) # Needs context or better default naming
           # Better: Use a placeholder name if none provided in original call context
           rhs_list <- list(stage_new = rhs) 
           # Need to check if name 'stage_new' collides! This logic needs care.
       } else {
           stop("`rhs` must be a projector object.")
       }
   } else {
       # Assume rhs was provided like list(new_stage_name = projector_obj)
       rhs_list <- rhs
   }
   
   lhs_projs <- if (inherits(lhs, "composed_projector")) lhs$projectors else list(lhs)
   lhs_names <- if (inherits(lhs, "composed_projector")) attr(lhs, "stage_names") else "stage_1"
   
   rhs_projs <- rhs_list
   rhs_names <- names(rhs_list)
   if (is.null(rhs_names) || rhs_names == "") rhs_names <- paste0("stage_", length(lhs_projs) + 1)
   
   # Combine projectors and check for name collisions
   all_projs <- c(lhs_projs, rhs_projs)
   all_names <- c(lhs_names, rhs_names)
   
   # Ensure unique names - simple appending suffix for now
   if (anyDuplicated(all_names)) {
       warning("Duplicate stage names detected during composition with `%>>%`. Appending suffixes.")
       all_names <- make.unique(all_names, sep = ".")
   }
   
   # Use do.call with named list
   do.call(compose_partial_projector, setNames(all_projs, all_names))
}

#' @export
#' @rdname variables_used
variables_used.composed_projector <- function(x, tol = 1e-8, ...) {
    combined_coef <- coef(x)
    if (is.null(combined_coef)) {
        warning("Cannot determine variables used: combined coefficients are NULL.")
        return(numeric(0))
    }
    
    # Find rows (original variables) with any non-zero coefficient across all components
    row_sums_sq <- rowSums(combined_coef^2)
    used_indices <- which(row_sums_sq > tol^2)
    
    sort(unique(used_indices))
}

#' @export
#' @rdname vars_for_component
vars_for_component.composed_projector <- function(x, k, tol = 1e-8, ...) {
    combined_coef <- coef(x)
    if (is.null(combined_coef)) {
        warning("Cannot determine variables used: combined coefficients are NULL.")
        return(numeric(0))
    }
    
    # Check if component k is valid
    if (k < 1 || k > ncol(combined_coef)) {
        stop("Component index 'k' (", k, ") is out of bounds [1..", ncol(combined_coef), "].")
    }
    
    # Find rows (original variables) with non-zero coefficient for component k
    component_k_coefs <- combined_coef[, k]
    used_indices <- which(abs(component_k_coefs) > tol)
    
    sort(unique(used_indices))
}

#' Compute the Inverse Projection for a Composed Projector
#'
#' Calculates the pseudo-inverse of the composed projector, mapping from the 
#' final output space back towards the original input space. This is computed
#' by multiplying the pseudo-inverses of the individual projector stages in 
#' reverse order: `V_k+ %*% ... %*% V_2+ %*% V_1+`.
#' 
#' Requires that each stage implements the `inverse_projection` method.
#'
#' @param x A `composed_projector` object.
#' @param ... Additional arguments passed to the underlying `inverse_projection` methods.
#'
#' @return A matrix representing the combined pseudo-inverse.
#' @export
inverse_projection.composed_projector <- function(x, ...) {
    cache_env <- attr(x, ".cache", exact = TRUE)
    use_caching <- !is.null(cache_env) && is.environment(cache_env)
    key <- "combined_inv_proj"
    
    if (use_caching && !is.null(cache_env[[key]])) {
        return(cache_env[[key]])
    }
    
    projs <- x$projectors
    n_proj <- length(projs)
    
    if (n_proj == 0) {
        warning("Cannot compute inverse projection: no projectors in the composition.")
        return(NULL)
    }
    
    # Get inverse projection from the *last* projector first
    combined_inv_proj <- inverse_projection(projs[[n_proj]], ...)
    
    # Sequentially pre-multiply by preceding inverse projection matrices
    if (n_proj > 1) {
        # Iterate from second-to-last down to first
        for (i in (n_proj - 1):1) {
            stage_inv_proj <- inverse_projection(projs[[i]], ...)
            
            if (is.null(combined_inv_proj) || is.null(stage_inv_proj)) {
                warning(paste("Cannot compute combined inverse projection because stage", 
                              i, "or subsequent stages lack an inverse projection."))
                return(NULL)
            }
            
            # Matrix multiplication: V_combined+ = V_{i+1...n}+ %*% V_i+
            combined_inv_proj <- combined_inv_proj %*% stage_inv_proj
        }
    }
    
    # Store in cache if available
    if (use_caching) {
        cache_env[[key]] <- combined_inv_proj
    }
    
    combined_inv_proj
}

#' Reconstruct Data from Scores using a Composed Projector
#'
#' Maps scores from the final latent space back towards the original input space
#' using the composed projector's combined inverse projection. Requires scores
#' to be provided explicitly.
#' 
#' Attempts to apply the `reverse_transform` of the *first* stage's preprocessor
#' to return data in the original units. If the first stage preprocessor is 
#' unavailable or invalid, a warning is issued, and data is returned in the 
#' (potentially) preprocessed space of the first stage.
#'
#' @param x A `composed_projector` object.
#' @param scores A numeric matrix of scores (observations x components) in the 
#'   final latent space of the composed projector.
#' @param comp Numeric vector of component indices (columns of `scores`, rows of 
#'   `inverse_projection`) to use for reconstruction. Defaults to all components.
#' @param rowind Numeric vector of row indices (observations in `scores`) to 
#'   reconstruct. Defaults to all rows.
#' @param colind Numeric vector of original variable indices (columns of the final 
#'   reconstructed matrix) to return. Defaults to all original variables.
#' @param ... Additional arguments (currently unused).
#'
#' @return A matrix representing the reconstructed data, ideally in the original 
#'   data space.
#' @export
reconstruct.composed_projector <- function(x, scores, comp = NULL, rowind = NULL, colind = NULL, ...) {
  chk::chk_s3_class(x, "composed_projector")
  chk::chk_matrix(scores)

  projs <- x$projectors
  n_proj <- length(projs)
  
  if (n_proj == 0) {
      warning("Cannot reconstruct: no projectors in the composition.")
      # If no projectors, reconstruction is just the input scores?
      # Or should return error? Let's return scores for now.
      return(scores)
  }

  # --- Argument Validation --- 
  n_comp_final <- shape(projs[[n_proj]])[2] # Num components output by LAST stage
  n_vars_orig <- shape(projs[[1]])[1]      # Num variables input to FIRST stage

  # Validate score dimensions against final stage output
  chk::chk_equal(ncol(scores), n_comp_final)

  # Default component/row/col indices
  if (is.null(comp)) comp <- 1:n_comp_final
  if (is.null(rowind)) rowind <- 1:nrow(scores)
  if (is.null(colind)) colind <- 1:n_vars_orig

  # Validate indices
  chk::chk_vector(comp)
  chk::chk_vector(rowind)
  chk::chk_vector(colind)
  chk::chk_subset(comp, 1:n_comp_final)
  chk::chk_subset(rowind, 1:nrow(scores))
  chk::chk_subset(colind, 1:n_vars_orig)
  
  # Subset initial scores based on requested components and rows
  # Note: 'comp' refers to components in the *final* space
  current_recon <- scores[rowind, comp, drop = FALSE]
  
  # --- Iterative Reconstruction --- 
  # Iterate backwards from the last stage to the first
  for (i in n_proj:1) {
      proj_i <- projs[[i]]
      preproc_i <- proj_i$preproc
      inv_proj_i <- inverse_projection(proj_i)
      
      if (is.null(inv_proj_i)) {
          stop("Cannot reconstruct: Stage ", i, " (", names(projs)[i],
               ") lacks an inverse_projection method or its result is NULL.")
      }
      
      # --- Corrected Order --- 
      # 1. Apply the linear inverse projection of stage i FIRST
      #    Input: current_recon (in the output space of stage i)
      #    Output: reconstruction in the input space of stage i (before preproc i)
      num_comps_in_current <- ncol(current_recon)
      if (num_comps_in_current > nrow(inv_proj_i)) {
          stop("Dimension mismatch during reconstruction at stage ", i,
               ": Input components (", num_comps_in_current,
               ") exceed available rows in inverse projection (", nrow(inv_proj_i), ")")
      }
      # Select rows of V_i+ corresponding to the components in current_recon
      inv_proj_i_sub <- inv_proj_i[1:num_comps_in_current, , drop = FALSE]
      # Apply the linear inverse projection
      current_recon <- current_recon %*% inv_proj_i_sub
      
      # 2. Reverse the pre-processing of stage i SECOND
      #    Input: result from step 1 (in input space of stage i, but potentially still centered/scaled)
      #    Output: reconstruction in the input space of stage i (output space of stage i-1)
      if (!is.null(preproc_i) && inherits(preproc_i, "pre_processor")) {
          # Pass the result of the inverse projection to reverse_transform
          current_recon <- reverse_transform(preproc_i, current_recon)
      } # else: If no preprocessor, current_recon is already correct for this step
      
      # ------------------------
  }
  
  # --- Final Column Selection --- 
  # After the loop, current_recon is in the original variable space
  # Select the requested original columns
  final_recon <- current_recon[, colind, drop = FALSE]
  
  return(final_recon)
}

#' Truncate a Composed Projector
#' 
#' Reduces the number of output components of the composed projector by 
#' truncating the *last* stage in the sequence. 
#' 
#' Note: This implementation currently only supports truncating the final stage.
#' Truncating intermediate stages would require re-computing subsequent stages 
#' or combined attributes and is not yet implemented.
#' 
#' @param x A `composed_projector` object.
#' @param ncomp The desired number of final output components.
#' @param ... Currently unused.
#' 
#' @return A new `composed_projector` object with the last stage truncated.
#' @export
truncate.composed_projector <- function(x, ncomp, ...) {
    chk::chk_s3_class(x, "composed_projector")
    chk::chk_number(ncomp, x_arg = "ncomp")
    chk::chk_gt(ncomp, 0)
    
    projs <- x$projectors
    n_proj <- length(projs)
    
    if (n_proj == 0) {
        warning("Cannot truncate: no projectors in the composition.")
        return(x)
    }
    
    # Truncate the last projector
    last_proj <- projs[[n_proj]]
    last_proj_truncated <- truncate(last_proj, ncomp)
    
    # Replace the last projector with the truncated version
    projs_new <- projs
    projs_new[[n_proj]] <- last_proj_truncated
    
    # Re-compose (this also updates attributes like stage_names)
    # Need !!! to splice the list elements as arguments
    # Also need to preserve original names
    new_comp <- do.call(compose_partial_projector, setNames(projs_new, attr(x, "stage_names")))
    
    # Clear the cache of the new object
    cache_env <- attr(new_comp, ".cache", exact = TRUE)
    if (!is.null(cache_env) && is.environment(cache_env)) {
        rm(list = ls(cache_env), envir = cache_env)
    }
    
    # Note: Index map is likely invalidated by truncation but not recalculated here.
    # Set index_map to NULL in the new object to indicate it's invalid?
    # attr(new_comp, "index_map") <- NULL # Optional: Explicitly invalidate
    
    return(new_comp)
}

#' Summarize a Composed Projector
#' 
#' Provides a summary of the stages within a composed projector, including 
#' stage names, input/output dimensions, and the primary class of each stage.
#' 
#' @param object A `composed_projector` object.
#' @param ... Currently unused.
#' 
#' @return A `tibble` summarizing the pipeline stages.
#' @export
#' @importFrom tibble tibble
summary.composed_projector <- function(object, ...) {
  projs <- object$projectors
  n_proj <- length(projs)
  stage_names <- attr(object, "stage_names", exact = TRUE)
  
  if (n_proj == 0) {
    return(tibble::tibble(
      stage = integer(0),
      name = character(0),
      in_dim = integer(0),
      out_dim = integer(0),
      class = character(0)
    ))
  }
  
  if (is.null(stage_names) || length(stage_names) != n_proj) {
      stage_names <- paste0("stage_", seq_len(n_proj))
  }
  
  shapes <- lapply(projs, shape)
  in_dims <- vapply(shapes, `[`, 1L, 1)
  out_dims <- vapply(shapes, `[`, 1L, 2)
  classes <- vapply(projs, function(p) class(p)[1], character(1))
  
  tibble::tibble(
     stage   = seq_len(n_proj),
     name    = stage_names,
     in_dim  = in_dims,
     out_dim = out_dims,
     class   = classes
  )
}

