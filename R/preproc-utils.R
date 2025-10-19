#' Check if a preprocessing object is fitted
#'
#' Determine whether a preprocessing object has been fitted to data.
#' This is used internally to provide helpful error messages when
#' users try to transform data with an unfitted preprocessor.
#'
#' @param object A preprocessing object to check
#' @return Logical: TRUE if fitted, FALSE otherwise
#' @keywords internal
is_fitted <- function(object) {
  UseMethod("is_fitted")
}

#' @export
is_fitted.pre_processor <- function(object) {
  # Check if explicitly marked as fitted via new API
  fitted_attr <- attr(object, "fitted")
  if (!is.null(fitted_attr)) {
    return(fitted_attr)
  }
  
  # For backwards compatibility, assume fitted if created via prep()
  # and has the necessary transformation functions
  !is.null(object$transform) && !is.null(object$reverse_transform)
}

#' @export
is_fitted.prepper <- function(object) {
  # A prepper is not fitted until it's been through prep() and init_transform()
  FALSE
}

#' @export  
is_fitted.concat_pre_processor <- function(object) {
  # A concat_pre_processor is fitted if it has transform functions
  !is.null(object$transform) && !is.null(object$reverse_transform)
}

#' Check if preprocessor is fitted and error if not
#'
#' Internal helper to provide consistent error messages when
#' attempting to transform with unfitted preprocessors.
#'
#' @param object A preprocessing object
#' @param action Character string describing the attempted action
#' @keywords internal
check_fitted <- function(object, action = "transform") {
  if (!is_fitted(object)) {
    rlang::abort(
      paste0("Pre-processor not fitted. Call fit() first before using ", action, "()."),
      class = "multivarious_unfitted_error"
    )
  }
}

#' Enhanced fitted state tracking
#'
#' Adds a fitted flag to preprocessing objects to track their state.
#' This is used by the new API to ensure proper workflow.
#'
#' @param object A preprocessing object
#' @param fitted Logical indicating fitted state
#' @return The object with fitted state marked
#' @keywords internal
mark_fitted <- function(object, fitted = TRUE) {
  attr(object, "fitted") <- fitted
  object
}

#' Get fitted state from attributes
#'
#' @param object A preprocessing object
#' @return Logical indicating fitted state, or NULL if not tracked
#' @keywords internal
get_fitted_state <- function(object) {
  attr(object, "fitted")
}