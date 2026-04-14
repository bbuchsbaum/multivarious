#' New sample projection
#'
#' Project one or more samples onto a subspace. This function takes a model fit and new observations, and projects them onto the
#' subspace defined by the model. This allows for the transformation of new data into the same lower-dimensional space as the original data.
#'
#' @param x The model fit, typically an object of class bi_projector or any other class that implements a project method
#' @param new_data A matrix or vector of new observations with the same number of columns as the original data. Rows represent observations and columns represent variables
#' @param ... Extra arguments to be passed to the specific project method for the object's class
#' @return A matrix or vector of the projected observations, where rows represent observations and columns represent the lower-dimensional space
#' @export
#' @family project
#' @seealso \code{\link{bi_projector}} for an example of a class that implements a project method
#' @examples
#' # Example with the bi_projector class
#' X <- matrix(rnorm(10*20), 10, 20)
#' svdfit <- svd(X)
#' p <- bi_projector(svdfit$v, s = svdfit$u %*% diag(svdfit$d), sdev=svdfit$d)
#'
#' # Project new_data onto the same subspace as the original data
#' new_data <- matrix(rnorm(5*20), 5, 20)
#' projected_data <- project(p, new_data)
project <- function(x, new_data, ...) UseMethod("project")



#' Partially project a new sample onto subspace
#'
#' Project a selected subset of column indices (`colind`) of `new_data` onto
#' the subspace defined by the model \code{x}. Optionally do a
#' ridge-regularized least-squares solve if columns are non-orthonormal.
#'
#' @param x The fitted model, e.g. `bi_projector`, that has a partial_project method.
#' @param new_data A numeric matrix (n x length(colind)) or vector, representing
#'   the observations to be projected.
#' @param colind A numeric vector of column indices in the original data space
#'   that correspond to \code{new_data}'s columns.
#' @param least_squares Logical; if TRUE (default), do a ridge-regularized solve.
#' @param lambda Numeric; ridge penalty (default 1e-6). Ignored if `least_squares=FALSE`.
#' @param ... Additional arguments passed to class-specific partial_project methods.
#'
#' @return A numeric matrix (n x d) of factor scores in the model's subspace, for
#'   those columns only.
#' @export
partial_project <- function(x, new_data, colind,
                            least_squares = TRUE,
                            lambda = 1e-6,
                            ...) {
  UseMethod("partial_project")
}




#' Construct a partial projector
#'
#' Create a new projector instance restricted to a subset of input columns. This function allows for the generation of
#' a new projection object that focuses only on the specified columns, enabling the projection of data using a limited
#' set of variables.
#'
#' @param x The original `projector` instance, typically an object of class `bi_projector` or any other class that implements a `partial_projector` method
#' @param colind A numeric vector of column indices to select in the projection matrix. These indices correspond to the variables used for the partial projector
#' @param ... Additional arguments passed to the underlying `partial_projector` method
#' @return A new `projector` instance, with the same class as the original object, that is restricted to the specified subset of input columns
#' @export
#' @seealso \code{\link{bi_projector}} for an example of a class that implements a `partial_projector` method
#' @examples
#' # Example with the bi_projector class
#' X <- matrix(rnorm(10*20), 10, 20)
#' svdfit <- svd(X)
#' p <- bi_projector(svdfit$v, s = svdfit$u %*% diag(svdfit$d), sdev=svdfit$d)
#'
#' # Create a partial projector using only the first 10 variables
#' colind <- 1:10
#' partial_p <- partial_projector(p, colind)
partial_projector <- function(x, colind, ...) UseMethod("partial_projector")
  


#' Project a single "block" of data onto the subspace
#'
#' When observations are concatenated into "blocks", it may be useful to project one block from the set.
#' This function facilitates the projection of a specific block of data onto a subspace. It is a
#' convenience method for multi-block fits and is equivalent to a "partial projection" where the
#' column indices are associated with a given block.
#'
#' @param x The model fit, typically an object of a class that implements a `project_block` method
#' @param new_data A matrix or vector of new observation(s) with the same number of columns as the original data
#' @param block An integer representing the block ID to select in the block projection matrix. This ID corresponds to the specific block of data to be projected
#' @param least_squares Logical. If `TRUE` use least squares projection.
#' @param ... Additional arguments passed to the underlying `project_block` method
#' @return A matrix or vector of the projected data for the specified block
#' @export
#' @family project
#' @seealso \code{\link{project}} for the generic projection function
project_block <- function(x, new_data, block, least_squares, ...) UseMethod("project_block")




#' Project one or more variables onto a subspace
#'
#' This function projects one or more variables onto a subspace. It is often called supplementary variable
#' projection and can be computed for a biorthogonal decomposition, such as Singular Value Decomposition (SVD).
#'
#' @param x The model fit, typically an object of a class that implements a `project_vars` method
#' @param new_data A matrix or vector of new observation(s) with the same number of rows as the original data
#' @param ... Additional arguments passed to the underlying `project_vars` method
#' @return A matrix or vector of the projected variables in the subspace
#' @export
#' @family project
#' @seealso \code{\link{project}} for the generic projection function for samples
project_vars <- function(x, new_data, ...) UseMethod("project_vars")


#' Transpose a model
#'
#' This function transposes a model by switching coefficients and scores. It is useful when you want to
#' reverse the roles of samples and variables in a model, especially in the context of dimensionality
#' reduction methods.
#'
#' @param x The model fit, typically an object of a class that implements a `transpose` method
#' @param ... Additional arguments passed to the underlying `transpose` method
#' @return A transposed model with coefficients and scores switched
#' @export
#' @family transpose
#' @seealso \code{\link{bi_projector}} for an example of a two-way mapping model that can be transposed
transpose <- function(x,...) UseMethod("transpose")



#' Reconstruct the data
#'
#' Reconstruct a data set from its (possibly) low-rank representation. This can be useful when analyzing
#' the impact of dimensionality reduction or when visualizing approximations of the original data.
#'
#' @param x The model fit, typically an object of a class that implements a `reconstruct` method
#' @param ... Additional arguments passed to specific methods. Common parameters include:
#'   \describe{
#'     \item{`comp`}{A vector of component indices to use in the reconstruction}
#'     \item{`rowind`}{The row indices to reconstruct (optional)}
#'     \item{`colind`}{The column indices to reconstruct (optional)}
#'     \item{`scores`}{(For `composed_projector` only) A numeric matrix of scores to reconstruct from}
#'   }
#' @return A reconstructed data set based on the selected components, rows, and columns
#' @export
#' @family reconstruct
#' @seealso \code{\link{bi_projector}} for an example of a two-way mapping model that can be reconstructed
reconstruct <- function(x, ...) UseMethod("reconstruct")



#' Reconstruct new data in a model's subspace
#'
#' This function takes a model (e.g., `projector` or `bi_projector`) and a new dataset,
#' and computes the rank-d approximation of the new data in the same subspace that
#' was defined by the model. In other words, we \strong{project} the new data into
#' the fitted subspace and then \strong{map it back} to the original dimensionality.
#'
#' Similar to \code{\link{reconstruct}} but operates on an external \code{new_data}
#' rather than the original fitted data. Often used to see how well the model's
#' subspace explains unseen data.
#'
#' @param x The fitted model object (e.g., \code{bi_projector}) that defines
#'   a subspace or factorization.
#' @param new_data A numeric matrix (or data frame) of shape
#'   \code{(n x p_full)} or possibly fewer columns if you allow partial reconstruction.
#' @param ... Additional arguments passed to the specific \code{reconstruct_new} method
#'   for the class of \code{x}.
#'
#' @return A numeric matrix (same number of rows as \code{new_data}, and typically
#'   the same number of columns if you're reconstructing fully) representing the
#'   rank-d approximation in the model's subspace.
#'
#' @export
#' @family reconstruct
#' @seealso \code{\link{reconstruct}} for reconstructing the original data in the model.
reconstruct_new <- function(x, new_data, ...) {
  UseMethod("reconstruct_new")
}



#' Transfer data from one domain/block to another via a latent space
#'
#' Convert between data representations in a multiblock or cross-decomposition
#' model by projecting the input `new_data` from the `from` domain/block
#' onto a latent space and then reconstructing it in the `to` domain/block.
#'
#' @param x The model fit, typically an object that implements a `transfer` method
#'   and ideally a `block_names` method.
#' @param new_data The data to transfer, typically matching the dimension of the `from` domain.
#' @param from Character string or index identifying the source domain/block.
#'   Must be present in `block_names(x)` if that method exists.
#' @param to Character string or index identifying the target domain/block.
#'   Must be present in `block_names(x)` if that method exists.
#' @param opts A list of optional arguments controlling the transfer process:
#'   \describe{
#'     \item{`cols`}{Optional numeric vector specifying column indices of the *target* 
#'       domain to reconstruct. If NULL (default), reconstructs all columns.}
#'     \item{`comps`}{Optional numeric vector specifying which latent components to use 
#'       for the projection/reconstruction. If NULL (default), uses all components.}
#'     \item{`ls_rr`}{Logical; if TRUE, use a ridge-regularized LS approach for the 
#'       initial projection from the `from` domain. Default FALSE.}
#'     \item{`lambda`}{Numeric ridge penalty (if `ls_rr=TRUE`). Default 1e-6.}
#'   }
#' @param ... Additional arguments passed to specific methods (discouraged, prefer `opts`).
#'
#' @return A matrix or data frame representing the transferred data in the `to` domain/block 
#'   (or a subset of columns/components if specified in `opts`).
#' @export
transfer <- function(x, new_data, from, to, opts = list(), ...) {
  UseMethod("transfer")
}

#' Default method for transfer
#' @param x Object
#' @param ... Ignored
#' @importFrom cli cli_abort
#' @export
#' @noRd
transfer.default <- function(x, ...) {
    cli::cli_abort(c(
      "!" = "{.fn transfer} is not implemented for {.cls {class(x)[1]}}.",
      "i" = "Provide a method or ensure the object inherits from a supported class."
      # Consider adding: "i" = "Supported classes include: cross_projector, ..."
    ))
}


## TODO 
## partial_residuals?
## partial_reconstruct?

#' Obtain residuals of a component model fit
#'
#' Calculate the residuals of a model after removing the effect of the first `ncomp` components.
#' This function is useful to assess the quality of the fit or to identify patterns that are not
#' captured by the model.
#'
#' @param x The model fit object.
#' @param ncomp The number of components to factor out before calculating residuals.
#' @param xorig The original data matrix (X) used to fit the model.
#' @param ... Additional arguments passed to the method.
#' @return A matrix of residuals, with the same dimensions as the original data matrix.
#' @export
#' @family residuals
residuals <- function(x, ncomp, xorig, ...) UseMethod("residuals")


#' Retrieve the component scores
#'
#' Extract the factor score matrix from a fitted model. The factor scores represent the projections of the
#' data onto the components, which can be used for further analysis or visualization.
#'
#' @param x The model fit object.
#' @param ... Additional arguments passed to the method.
#' @return A matrix of factor scores, with rows corresponding to samples and columns to components.
#' @export
#' @family scores
#' @seealso \code{\link{project}} for projecting new data onto the components.
scores <- function(x,...) UseMethod("scores")



#' Compute standardized component scores
#'
#' Calculate standardized factor scores from a fitted model. Standardized scores are useful for comparing
#' the contributions of different components on the same scale, which can help in interpreting the results.
#'
#' @param x The model fit object.
#' @param ... Additional arguments passed to the method.
#' @return A matrix of standardized factor scores, with rows corresponding to samples and columns to components.
#' @export
#' @seealso \code{\link{scores}} for retrieving the original component scores.
std_scores <- function(x, ...) UseMethod("std_scores")



#' get the components
#' 
#' Extract the component matrix of a fit.
#' 
#' @param x the model fit
#' @param ... extra args
#' @return the component matrix
#' @export
components <- function(x,...) UseMethod("components")



#' Shape of the Projector
#'
#' Get the input/output shape of the projector.
#'
#' This function retrieves the dimensions of the sample loadings matrix `v` in the form of a vector with two elements.
#' The first element is the number of rows in the `v` matrix, and the second element is the number of columns.
#'
#' @param x The model fit.
#' @param ... Extra arguments.
#' @return A vector containing the dimensions of the sample loadings matrix `v` (number of rows and columns).
#' @export
shape <- function(x,...) UseMethod("shape")


#' Inverse of the Component Matrix
#'
#' Return the inverse projection matrix, which can be used to map back to data space.
#' If the component matrix is orthogonal, then the inverse projection is the transpose of the component matrix.
#'
#' @param x The model fit.
#' @param ... Extra arguments.
#' @return The inverse projection matrix.
#' @export
#' @seealso \code{\link{project}} for projecting data onto the subspace.
inverse_projection <- function(x, ...) UseMethod("inverse_projection")


#' Partial Inverse Projection of a Columnwise Subset of Component Matrix
#'
#' Compute the inverse projection of a columnwise subset of the component matrix (e.g., a sub-block).
#' Even when the full component matrix is orthogonal, there is no guarantee that the partial component matrix is orthogonal.
#'
#' @param x A fitted model object, such as a `projector`, that has been fit to a dataset.
#' @param colind A numeric vector specifying the column indices of the component matrix to consider for the partial inverse projection.
#' @param ... Additional arguments to be passed to the specific model implementation of `partial_inverse_projection`.
#' @return A matrix representing the partial inverse projection.
#' @export
partial_inverse_projection <- function(x, colind, ...) UseMethod("partial_inverse_projection")


#' Compose Two Projectors
#'
#' Combine two projector models into a single projector by sequentially applying the first projector and then the second projector.
#'
#' @param x A fitted model object (e.g., `projector`) that has been fit to a dataset and will be applied first in the composition.
#' @param y A second fitted model object (e.g., `projector`) that has been fit to a dataset and will be applied after the first projector.
#' @param ... Additional arguments to be passed to the specific model implementation of `compose_projector`.
#' @return A new `projector` object representing the composed projector, which can be used to project data onto the combined subspace.
#' @export
compose_projector <- function(x,y,...) UseMethod("compose_projector")


#' Get a fresh pre-processing node cleared of any cached data
#' 
#' @param x the processing pipeline
#' @param ... extra args
#' @return a fresh pre-processing pipeline
#' @export
fresh <- function(x,...) UseMethod("fresh")



#' add a pre-processing stage
#' 
#' @param x the processing pipeline
#' @param step the pre-processing step to add
#' @param ... extra args
#' @export
#' @return a new pre-processing pipeline with the added step
add_node <- function(x, step, ...) UseMethod("add_node")


#' prepare a dataset by applying a pre-processing pipeline
#' 
#' @param x the pipeline
#' @param ... extra args
#' @return the pre-processed data
#' @export
prep <- function(x, ...) UseMethod("prep")


#' apply pre-processing parameters to a new data matrix
#' 
#' Given a new dataset, process it in the same way the original data was processed (e.g. centering, scaling, etc.)
#' 
#' @param x the model fit object
#' @param new_data the new data to process
#' @param colind the column indices of the new data
#' @param ... extra args
#' @return the reprocessed data
#' @export
reprocess <- function(x, new_data, colind, ...) UseMethod("reprocess")


#' refit a model
#' 
#' refit a model given new data or new parameter(s)
#'
#'
#' @param x the original model fit object
#' @param new_data the new data to process
#' @param ... extra args
#' @return a refit model object
#' @export
refit <- function(x, new_data, ...) UseMethod("refit")


#' Get the number of components
#'
#' This function returns the total number of components in the fitted model.
#'
#' @param x A fitted model object.
#' @return The number of components in the fitted model.
#' @export
#' @examples
#' # Example using the svd_wrapper function
#' data(iris)
#' X <- as.matrix(iris[, 1:4])
#' fit <- svd_wrapper(X, ncomp = 3, preproc = center(), method = "base")
#' ncomp(fit) # Should return 3
ncomp <- function(x) UseMethod("ncomp")


#' standard deviations 
#' 
#' The standard deviations of the projected data matrix
#' 
#' @param x the model fit
#' @return the standard deviations
#' @export
sdev <- function(x) UseMethod("sdev")

#' is it orthogonal
#' 
#' test whether components are orthogonal
#' 
#' @param x the object
#' @param tol tolerance for checking orthogonality
#' @return a logical value indicating whether the transformation is orthogonal
is_orthogonal <- function(x, tol=1e-6) UseMethod("is_orthogonal")


#' truncate a component fit
#' 
#' take the first n components of a decomposition
#' 
#' @param x the object to truncate
#' @param ncomp number of components to retain
#' @return a truncated object (e.g. PCA with 'ncomp' components)
#' @export
truncate <- function(x, ncomp) UseMethod("truncate")


#' get block_lengths
#' 
#' extract the lengths of each block in a multiblock object
#' 
#' @param x the object
#' @export
#' @return the block lengths
block_lengths <- function(x) UseMethod("block_lengths")


#' get block_indices 
#' 
#' extract the list of indices associated with each block in a `multiblock` object
#' 
#' @param x the object
#' @param ... extra args
#' @export
#' @return a list of block indices
block_indices <- function(x, ...) UseMethod("block_indices")


#' get the number of blocks
#' 
#' The number of data blocks in a multiblock element
#' 
#' @param x the object
#' @return the number of blocks
#' @export
nblocks <- function(x) UseMethod("nblocks")


#' initialize a transform
#' 
#' @param x the pre_processor
#' @param X the data matrix
#' @keywords internal
#' @return an initialized pre-processor
#' @export
init_transform <- function(x, X, ...) UseMethod("init_transform")



#' apply a pre-processing transform
#' 
#' @inheritParams init_transform
#' @param colind column indices
#' @param ... extra args
#' @export
#' @return the transformed data
apply_transform <- function(x, X, colind, ...) UseMethod("apply_transform")


#' reverse a pre-processing transform
#' 
#' @inheritParams init_transform
#' @param colind column indices
#' @param ... extra args
#' @return the reverse-transformed data
#' @export
reverse_transform <- function(x, X, colind, ...) UseMethod("reverse_transform")



#' Bootstrap Resampling for Multivariate Models
#'
#' Perform bootstrap resampling on a multivariate model to estimate the variability of components and scores.
#'
#' @param x A fitted model object, such as a `projector`, that has been fit to a training dataset.
#' @param nboot An integer specifying the number of bootstrap resamples to perform.
#' @param ... Additional arguments to be passed to the specific model implementation of `bootstrap`.
#' @return A list containing the bootstrap resampled components and scores for the model.
#' @export
bootstrap <- function(x, nboot, ...) UseMethod("bootstrap")


#' Construct a Classifier
#'
#' Create a classifier from a given model object (e.g., `projector`). This classifier can generate predictions for new data points.
#'
#' @param x A model object, such as a `projector`, that has been fit to a training dataset.
#' @param colind Optional vector of column indices used for prediction. If not provided, all columns will be used.
#' @param ... Additional arguments to be passed to the specific model implementation of `classifier`.
#' @return A classifier function that can be used to make predictions on new data points.
#' @export
classifier <- function(x, colind, ...) UseMethod("classifier")


#' construct a random forest wrapper classifier 
#' 
#' Given a model object (e.g. `projector` construct a random forest classifier that can generate predictions for new data points.
#' 
#' @param x the model object
#' @param colind the (optional) column indices used for prediction
#' @param ... extra arguments to `randomForest` function
#' @return a random forest classifier
#' @export
rf_classifier <- function(x, colind, ...) UseMethod("rf_classifier")


#' Permutation Confidence Intervals
#'
#' Estimate confidence intervals for model parameters using permutation testing.
#'
#' @param x A model fit object.
#' @param X The original data matrix used to fit the model.
#' @param nperm The number of permutations to perform for the confidence interval estimation.
#' @param ... Additional arguments to be passed to the specific model implementation of `perm_ci`.
#' @return A list containing the estimated lower and upper bounds of the confidence intervals for model parameters.
#' @export
perm_ci <- function(x, X, nperm, ...) UseMethod("perm_ci")



#' Rotate a Component Solution
#'
#' Perform a rotation of the component loadings to improve interpretability.
#'
#' @param x The model fit, typically a result from a dimensionality reduction method like PCA.
#' @param ncomp The number of components to rotate.
#' @param type The type of rotation to apply (e.g., "varimax", "quartimax", "promax").
#' @param ... extra args
#' @return A modified model fit with the rotated components.
#' @export
rotate <- function(x, ncomp, type, ...) UseMethod("rotate")



#' Apply rotation
#' 
#' Apply a specified rotation to the fitted model
#' 
#' @param x A model object, possibly created using the `pca()` function.
#' @param rotation_matrix \code{matrix} reprsenting the rotation.
#' @param ... extra args
#' @return A modified object with updated components and scores after applying the specified rotation.
#' @export
apply_rotation <- function(x, rotation_matrix, ...) { UseMethod("apply_rotation") }


#' Evaluate feature importance
#' 
#' Calculate the importance of features in a model
#' 
#' @param x the model fit
#' @param ... extra args
#' @return the feature importance scores
#' @export
feature_importance <- function(x, ...) UseMethod("feature_importance")



#' Generic Permutation-Based Test
#'
#' This generic function implements a permutation-based test to assess the significance
#' of components or statistics in a fitted model. The actual procedure depends on
#' the method defined for the specific model class. Typical usage:
#'
#' \enumerate{
#'   \item Shuffle or permute the data in a way that breaks the structure of interest
#'         (e.g., shuffle labels for supervised methods, shuffle columns/rows for unsupervised).
#'   \item Re-fit or re-project the model on the permuted data. Depending on the class,
#'         this can be done via a \code{fit_fun} or a class-specific approach.
#'   \item Measure the statistic of interest (e.g., variance explained, classification accuracy, canonical correlation).
#'   \item Compare the distribution of permuted statistics to the observed statistic to compute an empirical p-value.
#' }
#'
#' S3 methods define the specific defaults and required signatures for the functions
#' involved in shuffling, fitting, and measuring.
#'
#' @name perm_test
#' @aliases perm_test perm_test.pca perm_test.cross_projector perm_test.discriminant_projector perm_test.multiblock_biprojector
#'
#' @usage NULL
#'
#' @param x A fitted model object (e.g. \code{pca}, \code{cross_projector}, \code{discriminant_projector}, \code{multiblock_biprojector}).
#' @param X (Used by \code{pca}, \code{cross_projector}, \code{discriminant_projector}) The original primary data matrix used to fit \code{x}. Ignored by the \code{multiblock_biprojector} method.
#' @param Y (Used by \code{cross_projector}) The secondary data block (n x pY). Ignored by other methods.
#' @param Xlist (Used by \code{multiblock_biprojector} \[optional, default \code{NULL}\] and \code{multiblock_projector} \[required\]) List of data blocks.
#' @param nperm Integer number of permutations (Default: 1000 for PCA, 500 for multiblock methods, 100 otherwise).
#' @param measure_fun (Optional; Used by \code{pca}, \code{cross_projector}, \code{discriminant_projector}, \code{multiblock_projector}) A function for computing the statistic(s) of interest. Ignored by \code{multiblock_biprojector}. Signature/default varies by method (see Details).
#' @param shuffle_fun (Optional; Used by all methods) A function for permuting the data appropriately. Signature/default varies by method (see Details).
#' @param fit_fun (Optional; Used by \code{cross_projector}, \code{discriminant_projector}) A function for re-fitting a new model. Ignored by PCA and multiblock methods. Signature/default varies by method (see Details).
#' @param stepwise (Used by \code{pca}) Logical indicating if sequential testing (P3 projection) should be performed. Default \code{TRUE}. (The multiblock methods also perform sequential testing based on \code{alpha} and \code{comps}, but this argument is ignored). Ignored by other methods.
#' @param parallel (Used by all methods) Logical; if `TRUE`, attempt parallel execution via `future.apply::future_lapply`.
#' @param alternative (Used by all methods) Character string for the alternative hypothesis: "greater" (default), "less", or "two.sided".
#' @param alpha (Used by \code{pca}, \code{multiblock_biprojector}, \code{multiblock_projector}) Significance level for sequential stopping rule (default 0.05). Passed directly as a named argument to these methods.
#' @param comps (Used by \code{pca}, \code{multiblock_biprojector}, \code{multiblock_projector}) Maximum number of components to test sequentially (default 4). Passed directly as a named argument to these methods.
#' @param use_svd_solver (Used by \code{pca}) Optional string specifying the SVD solver (default "fast").
#' @param use_rspectra (Used by \code{multiblock_biprojector}) Logical indicating whether to use RSpectra for eigenvalue calculation (default \code{TRUE}). Passed directly as a named argument.
#' @param predict_method (Used by \code{discriminant_projector}) Prediction method (`"lda"` or `"euclid"`) used by the default measure function (default "lda").
#' @param ... Additional arguments passed down to `shuffle_fun` or `measure_fun` (if applicable).
#'   Note: For \code{multiblock} methods, \code{Xlist}, \code{comps}, \code{alpha}, and \code{use_rspectra} (for biprojector) are handled as direct named arguments, not via \code{...}.
#'
#' @details
#' This function provides a framework for permutation testing in various multivariate models.
#' The specific implementation details, default functions, and relevant arguments vary by method.
#'
#' \strong{PCA Method (`perm_test.pca`):} 
#' Relevant arguments: \code{X}, \code{nperm}, \code{measure_fun}, \code{shuffle_fun}, \code{stepwise}, \code{parallel}, \code{alternative}, \code{alpha}, \code{comps}, \code{use_svd_solver}, \code{...}. Assesses significance of variance explained by each PC (Vitale et al., 2017). Default statistic: F_a. Default shuffle: column-wise. Default uses P3 projection and sequential stopping with \code{alpha}.
#'
#' \strong{Cross Projector Method (`perm_test.cross_projector`):} 
#' Relevant arguments: \code{X}, \code{Y}, \code{nperm}, \code{measure_fun}, \code{shuffle_fun}, \code{fit_fun}, \code{parallel}, \code{alternative}, \code{...}. Tests the X-Y relationship. Default statistic: `x2y.mse`. Default shuffle: rows of Y. Default fit: `stats::cancor`.
#'
#' \strong{Discriminant Projector Method (`perm_test.discriminant_projector`):} 
#' Relevant arguments: \code{X}, \code{nperm}, \code{measure_fun}, \code{shuffle_fun}, \code{fit_fun}, \code{predict_method}, \code{parallel}, \code{alternative}, \code{...}. Tests class separation. Default statistic: prediction accuracy. Default shuffle: labels. Default fit: `MASS::lda`.
#'
#' \strong{Multiblock Bi-Projector Method (`perm_test.multiblock_biprojector`):} 
#' Relevant arguments: \code{Xlist} (optional), \code{nperm}, \code{shuffle_fun}, \code{parallel}, \code{alternative}, \code{alpha}, \code{comps}, \code{use_rspectra}, \code{...}. Tests consensus using fixed internal statistic (eigenvalue) on scores for each component. The statistic is the leading eigenvalue of the covariance matrix of block scores for a given component (T^T, where T columns are scores of block \emph{b} on component \emph{k}). By default, it shuffles rows within each block independently (either from \code{Xlist} if provided via \code{...}, or using the internally stored scores). It performs sequential testing for components specified by \code{comps} using the stopping rule defined by \code{alpha} (both passed via \code{...}).
#'
#' \strong{Multiblock Projector Method (`perm_test.multiblock_projector`):} 
#' Relevant arguments: \code{Xlist} (required), \code{nperm}, \code{measure_fun}, \code{shuffle_fun}, \code{parallel}, \code{alternative}, \code{alpha}, \code{comps}, \code{...}. Tests consensus using \code{measure_fun} (default: mean abs corr) on scores projected from \code{Xlist} using the original model \code{x}. Does not refit.
#'
#' @return
#' The structure of the return value depends on the method:
#' \describe{
#'   \item{\strong{`cross_projector`} and \strong{`discriminant_projector`}:}{
#'     Returns an object of class \code{perm_test}, a list containing: \code{statistic}, \code{perm_values}, \code{p.value}, \code{alternative}, \code{method}, \code{nperm}, \code{call}.}
#'   \item{\strong{`pca`}, \strong{`multiblock_biprojector`}, and \strong{`multiblock_projector`}:}{
#'     Returns an object inheriting from \code{perm_test} (classes \code{perm_test_pca}, \code{perm_test_multiblock}, or \code{perm_test} respectively for multiblock_projector), 
#'     a list containing: \code{component_results} (data frame with observed stat, pval, CIs per component), \code{perm_values} (matrix of permuted stats), \code{alpha} (if applicable), \code{alternative}, \code{method}, \code{nperm} (vector of successful permutations per component), \code{call}.}
#' }
#'
#' @references
#' Buja, A., & Eyuboglu, N. (1992). Remarks on parallel analysis. *Multivariate Behavioral Research*, 27(4), 509-540. (Relevant for PCA permutation concepts)
#'
#' Vitale, R., Westerhuis, J. A., Næs, T., Smilde, A. K., de Noord, O. E., & Ferrer, A. (2017).
#' Selecting the number of factors in principal component analysis by permutation testing—
#' Numerical and practical aspects. *Journal of Chemometrics*, 31(10), e2937.
#' \doi{10.1002/cem.2937} (Specific to `perm_test.pca`)
#'
#' @seealso \code{\link{pca}}, \code{\link{cross_projector}}, \code{\link{discriminant_projector}},
#'   \code{\link{multiblock_biprojector}},
#'   \code{\link{measure_interblock_transfer_error}}
#' @family perm_test
#'
#' @examples
#' # PCA Example
#' data(iris)
#' X_iris <- as.matrix(iris[,1:4])
#' mod_pca <- pca(X_iris, ncomp=4, preproc=center()) # Ensure centering
#'
#' # Test first 3 components sequentially (faster with more nperm)
#' # Ensure a future plan is set for parallel=TRUE, e.g., future::plan("multisession")
#' res_pca <- perm_test(mod_pca, X_iris, nperm=50, comps=3, parallel=FALSE)
#' print(res_pca)
#'
#' # PCA Example with row shuffling (tests different null hypothesis)
#' row_shuffle <- function(dat, ...) dat[sample(nrow(dat)), ]
#' res_pca_row <- perm_test(mod_pca, X_iris, nperm=50, comps=3,
#'                          shuffle_fun=row_shuffle, parallel=FALSE)
#' print(res_pca_row)
#'
#' @export
perm_test <- function(x, ...) {
  UseMethod("perm_test")
}



#' Screeplot for PCA
#'
#' Displays the variance explained by each principal component as a bar or line plot.
#' 
#' @param x A \code{pca} object.
#' @param ... extra args
#' @export
screeplot <- function(x, ...) UseMethod("screeplot")


#' Cross-validation Framework
#'
#' Generic function for performing cross-validation on various objects or data.
#' Specific methods should be implemented for different data types or model types.
#'
#' @param x The object to perform cross-validation on (e.g., data matrix, formula, model object).
#' @param folds A list defining the cross-validation folds, typically containing `train` and `test` indices for each fold.
#' @param ... Additional arguments passed to specific methods.
#'
#' @return The structure of the return value depends on the specific S3 method.
#'   Typically, it will be an object containing the results of the cross-validation,
#'   such as performance metrics per fold or aggregated metrics.
#'
#' @details
#' The specific implementation details, default functions, and relevant arguments vary by method.
#'
#' \strong{Bi-Projector Method (`cv.bi_projector`):}
#' Relevant arguments: \code{x}, \code{folds}, \code{max_comp}, \code{fit_fun},
#'   \code{measure}, \code{measure_fun}, \code{return_models}, \code{...}.
#'
#' This method performs cross-validation specifically for \code{bi_projector} models
#' (or models intended to be used like them, typically from unsupervised methods
#' like PCA or SVD). For each fold, it fits a single model using the training data
#' with the maximum number of components specified (\code{max_comp}). It then iterates
#' from 1 to \code{max_comp} components:
#' \enumerate{
#'   \item It truncates the full model to \code{k} components using \code{truncate()}. 
#'        (Requires a \code{truncate} method for the fitted model class).
#'   \item It reconstructs the held-out test data using the k-component truncated model
#'        via \code{reconstruct_new()}.
#'   \item It calculates reconstruction performance metrics (e.g., MSE, R2) by comparing
#'        the original test data to the reconstruction using the \code{measure} argument
#'        or a custom \code{measure_fun}.
#' }
#' The \code{fit_fun} must accept an argument \code{ncomp}. Additional arguments in \code{...}
#' are passed to \code{fit_fun} and \code{measure_fun}.
#'
#' The return value is a \code{cv_fit} object (a list with class `cv_fit`), where the
#' \code{$results} element is a tibble. Each row corresponds to a fold, containing
#' the fold index (\code{fold}) and a nested tibble (\code{component_metrics}).
#' The \code{component_metrics} tibble has rows for each component evaluated (1 to
#' \code{max_comp}) and columns for the component index (\code{comp}) plus all
#' calculated metrics (e.g., \code{mse}, \code{r2}, \code{mae}) or error messages
#' (\code{comp_error}). If \code{return_models=TRUE}, the full model fitted on the training
#' data for each fold is included in a list column \code{model_full}.
#'
#' @export
#' @family cv
#' @seealso \code{\link{cv_generic}}
cv <- function(x, folds, ...) {
  UseMethod("cv")
}


#' Identify Original Variables Used by a Projector
#'
#' Determines which columns from the *original* input space contribute
#' (have non-zero influence) to *any* of the output components of the projector.
#'
#' @param x A projector object (e.g., `projector`, `composed_projector`).
#' @param tol Numeric tolerance for determining non-zero coefficients. Default is 1e-8 for some methods. Passed via `...`.
#' @param ... Additional arguments passed to specific methods.
#'
#' @return A sorted numeric vector of unique indices corresponding to the original input variables.
#' @export
variables_used <- function(x, ...) {
    UseMethod("variables_used")
}

#' Identify Original Variables for a Specific Component
#'
#' Determines which columns from the *original* input space contribute
#' (have non-zero influence) to a *specific* output component of the projector.
#'
#' @param x A projector object (e.g., `projector`, `composed_projector`).
#' @param k The index of the output component to query.
#' @param tol Numeric tolerance for determining non-zero coefficients. Default is 1e-8 for some methods. Passed via `...`.
#' @param ... Additional arguments passed to specific methods.
#'
#' @return A sorted numeric vector of unique indices corresponding to the original input variables.
#' @export
vars_for_component <- function(x, k, ...) {
    UseMethod("vars_for_component")
}


