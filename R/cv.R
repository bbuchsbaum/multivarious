#' Compute reconstruction-based error metrics
#'
#' Given two numeric matrices \code{Xtrue} and \code{Xrec}, compute:
#'  \itemize{
#'    \item MSE  (\code{"mse"})
#'    \item RMSE (\code{"rmse"})
#'    \item R^2  (\code{"r2"})
#'    \item MAE  (\code{"mae"})
#'  }
#'
#' @param Xtrue Original data matrix, shape (n x p).
#' @param Xrec  Reconstructed data matrix, shape (n x p).
#' @param metrics Character vector of metric names, e.g. \code{c("mse","rmse","r2","mae")}.
#' @param by_column Logical, if TRUE calculate R2 metric per column and average (default: FALSE).
#' @return A one-row \code{tibble} with columns matching \code{metrics}.
#' @export
measure_reconstruction_error <- function(Xtrue, Xrec, metrics = c("mse","rmse","r2"), by_column = FALSE) {
  stopifnot(is.matrix(Xtrue), is.matrix(Xrec))
  stopifnot(all(dim(Xtrue) == dim(Xrec)))
  
  diff <- Xtrue - Xrec
  sq   <- diff^2
  mse  <- mean(sq, na.rm = TRUE)
  rmse <- sqrt(mse)
  
  # R^2 calculation
  if (by_column) {
    # Column-wise R^2: Calculate R^2 for each column, then average
    col_means_true <- colMeans(Xtrue, na.rm = TRUE)
    sstot_col <- colSums(sweep(Xtrue, 2, col_means_true, "-")^2, na.rm = TRUE)
    ssres_col <- colSums(sq, na.rm = TRUE)
    
    # Avoid division by zero or near-zero for columns with no variance
    r2_col <- ifelse(sstot_col < 1e-15, NA_real_, 1 - ssres_col / sstot_col)
    
    r2 <- mean(r2_col, na.rm = TRUE) # Average the per-column R^2 values
  } else {
    # Original R^2: 1 - SSres / SStot (using grand mean)
    sstot <- sum((Xtrue - mean(Xtrue, na.rm = TRUE))^2, na.rm = TRUE)
    ssres <- sum(sq, na.rm = TRUE)
    r2 <- if (sstot < 1e-15) NA_real_ else (1 - ssres / sstot)
  }
  
  mae <- mean(abs(diff), na.rm = TRUE)
  
  out_vals <- list()
  for (m in metrics) {
    val <- switch(m,
                  "mse"  = mse,
                  "rmse" = rmse,
                  "r2"   = r2,
                  "mae"  = mae,
                  stop("Unknown metric: ", m)
    )
    out_vals[[m]] <- val
  }
  tibble::as_tibble(out_vals)
}

#' Compute inter-block transfer error metrics for a cross_projector
#'
#' We measure how well the model can transfer from X->Y or Y->X, e.g. "x2y.mse".
#'
#' The metric names are of the form "x2y.mse", "x2y.rmse", "y2x.r2", etc.
#'
#' @param Xtrue The X block test data
#' @param Ytrue The Y block test data
#' @param model The fitted \code{cross_projector}
#' @param metrics A character vector like \code{c("x2y.mse","y2x.r2")}
#' @return A 1-row tibble with columns for each requested metric
#' @export
measure_interblock_transfer_error <- function(Xtrue, Ytrue, model,
                                              metrics = c("x2y.mse")) {
  
  # Initialize output with NA for all requested metrics for predictable shape
  out_list <- stats::setNames(rep(NA_real_, length(metrics)), metrics)
  
  # Helper to do X->Y:
  do_x2y <- function() {
    requested_x2y <- grep("^x2y\\.", metrics, value = TRUE)
    if (length(requested_x2y) == 0) return()
    
    base_metrics <- gsub("^x2y\\.", "", requested_x2y)
    
    tryCatch({
      Ypred <- transfer(model, Xtrue, from = "X", to = "Y")
      if (!all(dim(Ypred) == dim(Ytrue))) {
        stop(sprintf("X->Y transfer returned shape %s but Ytrue is %s.",
                     paste(dim(Ypred), collapse = "x"),
                     paste(dim(Ytrue), collapse = "x")))
      }
      # Assuming measure_reconstruction_error handles missing metrics gracefully
      subres <- measure_reconstruction_error(Ytrue, Ypred, metrics = base_metrics)
      # Update out_list only for successfully computed metrics
      for (nm in names(subres)) {
        out_list[[paste0("x2y.", nm)]] <- subres[[nm]]
      }
    }, error = function(e) {
      warning("Error during X->Y transfer or measurement: ", conditionMessage(e))
      # NAs for these metrics remain in out_list
    })
  }
  
  # Helper to do Y->X:
  do_y2x <- function() {
    requested_y2x <- grep("^y2x\\.", metrics, value = TRUE)
    if (length(requested_y2x) == 0) return()
    
    base_metrics <- gsub("^y2x\\.", "", requested_y2x)
    
    tryCatch({
      Xpred <- transfer(model, Ytrue, from = "Y", to = "X")
      if (!all(dim(Xpred) == dim(Xtrue))) {
        stop(sprintf("Y->X transfer returned shape %s but Xtrue is %s.",
                     paste(dim(Xpred), collapse = "x"),
                     paste(dim(Xtrue), collapse = "x")))
      }
      subres <- measure_reconstruction_error(Xtrue, Xpred, metrics = base_metrics)
      for (nm in names(subres)) {
        out_list[[paste0("y2x.", nm)]] <- subres[[nm]]
      }
    }, error = function(e) {
      warning("Error during Y->X transfer or measurement: ", conditionMessage(e))
      # NAs remain
    })
  }
  
  # If user asked for x2y.* metrics
  if (any(startsWith(metrics, "x2y."))) {
      do_x2y()
  }
  # If user asked for y2x.* metrics
  if (any(startsWith(metrics, "y2x."))) {
      do_y2x()
  }
  
  tibble::as_tibble(out_list)
}



#' Generic cross-validation engine
#'
#' For each fold (train/test indices):
#'   \enumerate{
#'     \item Subset \code{data[train, ]}
#'     \item Fit a model with \code{.fit_fun(train_data, ...)}
#'     \item Evaluate with \code{.measure_fun(model, test_data, ...)}
#'   }
#'
#' @param data A matrix or data.frame of shape (n x p).
#' @param folds A list of folds, each a list with \code{$train} and \code{$test}.
#' @param .fit_fun Function: signature \code{function(train_data, ...)\{\}}. Returns a fitted model.
#' @param .measure_fun Function: signature \code{function(model, test_data, ...)\{\}}. Returns a tibble or named list/vector of metrics.
#' @param fit_args A list of additional named arguments passed to \code{.fit_fun}.
#' @param measure_args A list of additional named arguments passed to \code{.measure_fun}.
#' @param backend Character string: "serial" (default) or "future" for parallel execution using the `future` framework.
#' @param ... Currently ignored (arguments should be passed via `fit_args` or `measure_args`).
#' @return A tibble with columns:
#'   \item{fold}{integer fold index}
#'   \item{model}{list of fitted models}
#'   \item{metrics}{list of metric tibbles/lists}
#' @importFrom dplyr bind_rows
#' @importFrom tibble tibble
#' @export
cv_generic <- function(data, folds, .fit_fun, .measure_fun, 
                       fit_args = list(), measure_args = list(), 
                       backend = c("serial", "future"), ...) {
  
  backend <- match.arg(backend)
  
  # Function to process a single fold
  process_fold <- function(i, data, folds, .fit_fun, .measure_fun, fit_args, measure_args) {
    train_idx <- folds[[i]]$train
    test_idx  <- folds[[i]]$test
    
    train_data <- data[train_idx, , drop=FALSE]
    test_data  <- data[test_idx, , drop=FALSE]
    
    # Call fit function with arguments, wrapped in tryCatch
    model_or_error <- tryCatch({
      do.call(.fit_fun, c(list(train_data), fit_args))
    }, error = function(e) {
      warning("Fitting failed for fold ", i, ": ", conditionMessage(e))
      e # Return the error object
    })
    
    # Call measure function with arguments, wrapped in tryCatch
    metrics_or_error <- if (inherits(model_or_error, "error")) {
      # If fitting failed, create an error entry for metrics
      tibble::tibble(error = paste("Fit failed:", conditionMessage(model_or_error)))
    } else {
      # If fitting succeeded, try measuring
      tryCatch({
        do.call(.measure_fun, c(list(model_or_error, test_data), measure_args))
      }, error = function(e) {
        warning("Measuring failed for fold ", i, ": ", conditionMessage(e))
        # Return a tibble containing the error message
        tibble::tibble(error = conditionMessage(e)) 
      })
    }
    
    # Ensure metrics is a list or coercible, handle potential error tibble
    metrics_tibble <- if (tibble::is_tibble(metrics_or_error)) {
      metrics_or_error
    } else if (is.list(metrics_or_error) && !is.data.frame(metrics_or_error)) {
      # Original handling for list output from measure_fun
      tibble::as_tibble(metrics_or_error)
    } else {
      # Handle vector or other coercible types
      tibble::as_tibble(as.list(metrics_or_error))
    }
    
    # Store model or error object
    model_result <- if (inherits(model_or_error, "error")) list(NULL) else list(model_or_error)
    
    tibble::tibble(
      fold    = i,
      model   = model_result, # Store model or NULL if error
      metrics = list(metrics_tibble) # Store metrics tibble or error tibble
    )
  }
  
  # Execute based on backend
  if (backend == "future") {
      if (!requireNamespace("future.apply", quietly = TRUE)) {
          stop("Package 'future.apply' required for backend='future'. Please install it.", call. = FALSE)
      }
      # future_lapply automatically handles exports typically
      res_list <- future.apply::future_lapply(seq_along(folds), process_fold, 
                                                data=data, folds=folds, 
                                                .fit_fun=.fit_fun, .measure_fun=.measure_fun,
                                                fit_args=fit_args, measure_args=measure_args,
                                                future.seed = TRUE) # Use future's RNG seeding
  } else {
      res_list <- lapply(seq_along(folds), process_fold, 
                         data=data, folds=folds, 
                         .fit_fun=.fit_fun, .measure_fun=.measure_fun,
                         fit_args=fit_args, measure_args=measure_args)
  }
  
  dplyr::bind_rows(res_list)
}




# cv.bi_projector <- function(x,
#                                   folds,
#                                   max_comp = 5,
#                                   fit_fun  = NULL,
#                                   measure  = c("mse","rmse","r2"),
#                                   measure_fun = NULL,
#                                   return_models = FALSE,
#                                   ...) # Capture extra args for fit/measure fun
# {
#   # 1) Provide a default fit_fun if none:
#   if (is.null(fit_fun)) {
#     
#     
#     fit_fun <- function(X_train, ncomp = 2, ...) {
#       pr <- prep(center()) # Use the standard preprocessor framework
#       Xtr <- init_transform(pr, X_train)
#       svdres <- svd(Xtr, nu = ncomp, nv = ncomp)
#       bi_projector(
#         v     = svdres$v,
#         s     = svdres$u %*% diag(svdres$d[1:ncomp]),
#         sdev  = svdres$d[1:ncomp],
#         preproc = pr # Pass the properly initialized preprocessor
#       )
#     }
#   }
#   
#   # If formula, convert
#   if (rlang::is_formula(fit_fun)) {
#     formula_fun <- rlang::as_function(fit_fun)
#     .fit_fun <- function(X_train, ncomp, ...) {
#       # pass X_train as .x, plus ncomp
#       formula_fun(X_train, ncomp=ncomp, ...)
#     }
#   } else if (is.function(fit_fun)) {
#     .fit_fun <- fit_fun
#   } else {
#     stop("`fit_fun` must be a function or one-sided formula.")
#   }
#   
#   # Capture extra arguments intended for fit_fun or measure_fun
#   extra_args <- list(...)
#   
#   # If measure_fun not provided, define default with measure_reconstruction_error
#   if (!is.null(measure_fun)) {
#     .measure_fun <- measure_fun
#   } else {
#     .measure_fun <- function(model, X_test, ...) {
#       # Reconstruct requires the model's preprocessor state to be applied
#       # reconstruct_new handles projection and reverse_transform internally.
#       # Original and correct logic: compare X_test to the original-space reconstruction
#       # provided by reconstruct_new (which now consistently includes reverse_transform)
#       X_rec <- reconstruct_new(model, X_test, comp = 1:ncomp(model)) # Use all comps in model_k
#       measure_reconstruction_error(X_test, X_rec, metrics = measure)
#     }
#   }
#   
#   # We'll do our own cross-validation loop manually, 
#   # because we want to fit multiple ncomp per fold.
#   results_list <- vector("list", length(folds))
#   
#   for (i in seq_along(folds)) {
#     train_idx <- folds[[i]]$train
#     test_idx  <- folds[[i]]$test
#     
#     X_train <- x[train_idx, , drop=FALSE]
#     X_test  <- x[test_idx, , drop=FALSE]
#     
#     # Fit model once with max components
#     model_full <- tryCatch({
#         do.call(.fit_fun, c(list(X_train, ncomp=max_comp), extra_args))
#       }, error = function(e) {
#           warning("Fitting model failed for fold ", i, ": ", e$message)
#           return(NULL)
#       })
#       
#     if (is.null(model_full)) {
#         comp_list <- list(tibble::tibble(comp=1:max_comp, error=NA)) # Placeholder if fit failed
#     } else {
#         # Store metrics for each truncated component level
#         comp_list <- vector("list", max_comp)
#         for (k in seq_len(max_comp)) {
#           model_k <- tryCatch({ 
#               truncate(model_full, k)
#             }, error = function(e) {
#                 warning("Truncating model failed for fold ", i, " k=", k, ": ", e$message)
#                 return(NULL)
#             })
# 
#           if(is.null(model_k)) {
#              metrics_k <- tibble::tibble(comp_error = "Truncation failed")
#           } else {
#               metrics_k <- tryCatch({ 
#                  do.call(.measure_fun, c(list(model_k, X_test), extra_args))
#                 }, error = function(e) {
#                     warning("Measuring metrics failed for fold ", i, " k=", k, ": ", e$message)
#                     tibble::tibble(comp_error = paste("Measure failed:", conditionMessage(e)))
#                 })
#           }
# 
#           # Store a row that includes comp index and each metric
#           comp_list[[k]] <- tibble::tibble(comp = k) %>% 
#              dplyr::bind_cols(tibble::as_tibble(metrics_k)) # Ensure metrics_k is tibble
#         }
#     }
#     
#     comp_tib <- dplyr::bind_rows(comp_list)  # shape: (max_comp x (1 + #metrics))
#     
#     # We only store one "row" in the results for fold i,
#     # with a nested tibble in column "component_metrics".
#     fold_result <- tibble::tibble(
#       fold              = i,
#       component_metrics = list(comp_tib)
#     )
#     if (return_models) {
#         fold_result$model_full <- list(model_full)
#     }
#     results_list[[i]] <- fold_result
#   }
#   
#   raw_results <- dplyr::bind_rows(results_list)
#   
#   # Wrap in cv_fit
#   ret <- structure(
#     list(results=raw_results),
#     class="cv_fit"
#   )
#   ret
# }

###############################################################################
## S3 summary.cv_fit -- we'll add a new approach
###############################################################################

# @export
# summary.cv_fit <- function(object, ...) {
#   df <- object$results
#   
#   # If we have "component_metrics" nested, unnest that. 
#   # If we have a "metrics" column (older style), handle that too.
#   
#   if ("component_metrics" %in% names(df)) {
#     # new approach with per-component
#     # we unnest, then summarize across folds for each comp
#     df_unnest <- tidyr::unnest(df, cols="component_metrics")
#     # columns might be: fold, comp, mse, rmse, r2, ... or comp_error
#     # we want group_by comp, then compute mean
#     metric_cols <- setdiff(names(df_unnest), c("fold","comp"))
#     # Summarize only numeric metrics, ignoring potential error columns
#     numeric_metric_cols <- metric_cols[sapply(df_unnest[metric_cols], is.numeric)]
#     if (length(numeric_metric_cols) == 0) {
#       return(tibble::tibble(note="No numeric metrics found to summarize."))
#     }
#     out <- df_unnest %>%
#       dplyr::group_by(.data$comp) %>%
#       dplyr::summarize(
#         dplyr::across(dplyr::all_of(numeric_metric_cols), ~ mean(.x, na.rm=TRUE)),
#         .groups="drop"
#       )
#     return(out)
#     
#   } else if ("metrics" %in% names(df)) {
#     # old style - handle potential error columns here too
#     df_unnest <- tidyr::unnest(df, cols="metrics")
#     metric_cols <- setdiff(names(df_unnest), c("fold","model"))
#     # Summarize only numeric metrics
#     numeric_metric_cols <- metric_cols[sapply(df_unnest[metric_cols], is.numeric)]
#     if (length(numeric_metric_cols) == 0) {
#       return(tibble::tibble(note="No numeric metrics found to summarize."))
#     }
#     out <- df_unnest %>%
#       dplyr::summarize(
#         dplyr::across(dplyr::all_of(numeric_metric_cols), ~ mean(.x, na.rm=TRUE))
#       )
#     return(out)
#   } else {
#     # No metrics at all
#     tibble::tibble(note="No metrics found.")
#   }
# }

###############################################################################
## The rest: print.cv_fit, plot.cv_fit remain the same or updated to handle multi-comp
###############################################################################

#' @export
# print.cv_fit <- function(x, ...) {
#   df <- x$results
#   cat(crayon::bold(crayon::blue("Cross-validation fit object\n")))
#   cat(crayon::silver("  Number of folds: "), nrow(df), "\n", sep="")
#   
#   if ("component_metrics" %in% names(df)) {
#     cat(crayon::silver("  Contains per-component metrics in 'component_metrics' column.\n"))
#     cat(crayon::silver("  Use summary() to see aggregated metrics.\n"))
#   } else if ("metrics" %in% names(df)) {
#     df_unnest <- tryCatch(tidyr::unnest(df, cols="metrics"), error=function(e) NULL)
#     if (!is.null(df_unnest)) {
#       metric_cols <- setdiff(names(df_unnest), c("fold","model"))
#       cat(crayon::silver("  Metrics: "), paste(metric_cols, collapse=", "), "\n", sep="")
#     }
#     cat(crayon::silver("  Use summary() to see aggregated metrics.\n"))
#     cat(crayon::silver("  Use plot() to visualize metrics by fold.\n"))
#   } else {
#     cat(crayon::silver("  No known metrics columns found.\n"))
#   }
#   
#   invisible(x)
# }

#' @export
# plot.cv_fit <- function(x, metric=NULL, ...) {
#   if (!requireNamespace("ggplot2", quietly=TRUE)) {
#     stop("`plot.cv_fit` requires ggplot2. Please install it.")
#   }
#   
#   df <- x$results
#   
#   if ("component_metrics" %in% names(df)) {
#     # new multi-comp approach
#     df_unnest <- tidyr::unnest(df, cols="component_metrics")
#     # possible columns: fold, comp, mse, rmse, r2, ...
#     numeric_cols <- setdiff(names(df_unnest)[sapply(df_unnest, is.numeric)], c("fold","comp"))
#     if (length(numeric_cols) == 0) {
#       stop("No numeric metrics found to plot.")
#     }
#     # If user didn't specify metric, pick first
#     if (is.null(metric)) {
#       metric <- numeric_cols[1]
#       message("Metric not specified, plotting first numeric metric found: ", metric)
#     }
#     if (!metric %in% names(df_unnest)) {
#       stop("Requested metric '", metric, "' not found among numeric columns: ",
#            paste(numeric_cols, collapse=", "))
#     }
#     # We'll do a line plot: x=comp, y=metric, color=fold
#     p <- ggplot2::ggplot(df_unnest, ggplot2::aes(x=.data$comp, y=.data[[metric]])) +
#       ggplot2::stat_summary(fun = mean, geom = "line", na.rm = TRUE, color="red", linewidth=1) +
#       ggplot2::stat_summary(fun.data = mean_se, geom = "ribbon", na.rm = TRUE, alpha=0.2, fill="red") + 
#       ggplot2::labs(x="Components", y=metric, color="Fold") +
#       ggplot2::theme_minimal() +
#       ggplot2::ggtitle("Cross-validation performance by components",
#                        subtitle="Red line = Mean across folds, Ribbon = +/- 1 SE")
#     return(p)
#     
#   } else if ("metrics" %in% names(df)) {
#     # old single-model approach
#     df_unnest <- tidyr::unnest(df, cols="metrics")
#     numeric_cols <- setdiff(names(df_unnest)[sapply(df_unnest, is.numeric)], "fold")
#     if (length(numeric_cols) == 0) {
#       stop("No numeric metrics found to plot.")
#     }
#     if (is.null(metric)) {
#       metric <- numeric_cols[1]
#     }
#     if (!metric %in% names(df_unnest)) {
#       stop("Requested metric '", metric, "' not found among numeric columns: ",
#            paste(numeric_cols, collapse=", "))
#     }
#     p <- ggplot2::ggplot(df_unnest, ggplot2::aes(x=factor(.data$fold), y=.data[[metric]])) +
#       ggplot2::geom_col(fill="steelblue", alpha=0.7) +
#       ggplot2::labs(x="Fold", y=metric, title=paste("Cross-validation:", metric)) +
#       ggplot2::theme_minimal()
#     return(p)
#     
#   } else {
#     stop("No metrics found to plot.")
#   }
# }
