#' Fit a preprocessing pipeline
#'
#' Learn preprocessing parameters from training data. This function fits the 
#' preprocessing pipeline to the provided data matrix, learning parameters 
#' such as means, standard deviations, or other transformation parameters.
#'
#' @param object A preprocessing object (e.g., `prepper` or `pre_processor`)
#' @param X A matrix or data frame to fit the preprocessing pipeline to
#' @param ... Additional arguments passed to methods
#' @return A fitted preprocessing object that can be used with `transform()` and `inverse_transform()`
#' @export
#' @seealso [fit_transform()], [transform()], [inverse_transform()]
#' @examples
#' # Fit a centering preprocessor
#' X <- matrix(rnorm(100), 10, 10)
#' preproc <- center()
#' fitted_preproc <- fit(preproc, X)
#' 
#' # Transform new data
#' X_new <- matrix(rnorm(50), 5, 10)
#' X_transformed <- transform(fitted_preproc, X_new)
fit <- function(object, X, ...) {
  UseMethod("fit")
}

#' Fit and transform data in one step
#'
#' Convenience function that fits a preprocessing pipeline to data and 
#' immediately applies the transformation. This is equivalent to calling
#' `fit()` followed by `transform()` but is more efficient and convenient.
#'
#' @param object A preprocessing object (e.g., `prepper` or `pre_processor`)
#' @param X A matrix or data frame to fit and transform
#' @param ... Additional arguments passed to methods
#' @return A list with two elements: `preproc` (the fitted preprocessor) and `transformed` (the transformed data)
#' @export
#' @seealso [fit()], [transform()], [inverse_transform()]
#' @examples
#' # Fit and transform in one step
#' X <- matrix(rnorm(100), 10, 10)
#' preproc <- center()
#' result <- fit_transform(preproc, X)
#' fitted_preproc <- result$preproc
#' X_transformed <- result$transformed
fit_transform <- function(object, X, ...) {
  UseMethod("fit_transform")
}

#' Transform data using a fitted preprocessing pipeline
#'
#' Apply a fitted preprocessing pipeline to new data. The preprocessing 
#' object must have been fitted using `fit()` or `fit_transform()` before
#' calling this function.
#'
#' @param object A fitted preprocessing object
#' @param X A matrix or data frame to transform
#' @param ... Additional arguments passed to methods
#' @return The transformed data matrix
#' @export
#' @seealso [fit()], [fit_transform()], [inverse_transform()]
#' @examples
#' # Transform new data with fitted preprocessor
#' X_train <- matrix(rnorm(100), 10, 10)
#' X_test <- matrix(rnorm(50), 5, 10)
#' 
#' preproc <- center()
#' fitted_preproc <- fit(preproc, X_train)
#' X_test_transformed <- transform(fitted_preproc, X_test)
transform <- function(object, X, ...) {
  UseMethod("transform")
}

#' Inverse transform data using a fitted preprocessing pipeline
#'
#' Reverse the preprocessing transformation, converting transformed data 
#' back to the original scale. The preprocessing object must have been 
#' fitted before calling this function.
#'
#' @param object A fitted preprocessing object
#' @param X A matrix or data frame of transformed data to reverse
#' @param ... Additional arguments passed to methods
#' @return The data matrix in original scale
#' @export
#' @seealso [fit()], [fit_transform()], [transform()]
#' @examples
#' # Inverse transform data back to original scale
#' X <- matrix(rnorm(100), 10, 10)
#' preproc <- center()
#' fitted_preproc <- fit(preproc, X)
#' X_transformed <- transform(fitted_preproc, X)
#' X_reconstructed <- inverse_transform(fitted_preproc, X_transformed)
#' 
#' # X and X_reconstructed should be approximately equal
#' all.equal(X, X_reconstructed)
inverse_transform <- function(object, X, ...) {
  UseMethod("inverse_transform")
}

#' Convenience function for preprocessing workflow
#'
#' This helper function provides a simple interface for the common preprocessing
#' workflow: fit a preprocessor to data and return both the fitted preprocessor
#' and the transformed data.
#'
#' @param preproc A preprocessing object (e.g., created with `center()`, `standardize()`, etc.)
#' @param X A matrix or data frame to preprocess
#' @param ... Additional arguments passed to methods
#' @return A list with two elements:
#'   \item{preproc}{The fitted preprocessing object}
#'   \item{transformed}{The transformed data matrix}
#' @export
#' @seealso [fit()], [fit_transform()], [transform()], [inverse_transform()]
#' @examples
#' # Simple preprocessing workflow
#' X <- matrix(rnorm(100), 10, 10)
#' result <- preprocess(center(), X)
#' fitted_preproc <- result$preproc
#' X_centered <- result$transformed
#' 
#' # Equivalent to:
#' # fitted_preproc <- fit(center(), X)
#' # X_centered <- transform(fitted_preproc, X)
preprocess <- function(preproc, X, ...) {
  result <- fit_transform(preproc, X, ...)
  list(preproc = result$preproc, transformed = result$transformed)
}