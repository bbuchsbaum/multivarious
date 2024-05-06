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
#' p <- bi_projector(svdfit$v, s = svdfit$u %% diag(svdfit$d), sdev=svdfit$d)
#'
#' # Project new_data onto the same subspace as the original data
#' new_data <- matrix(rnorm(5*20), 5, 20)
#' projected_data <- project(p, new_data)
project <- function(x, new_data, ...) UseMethod("project")



#' Partially project a new sample onto subspace
#'
#' Project a selected subset of column indices onto the subspace. This function allows for the projection of new data
#' onto a lower-dimensional space using only a subset of the variables, as specified by the column indices.
#'
#' @param x The model fit, typically an object of class `bi_projector` or any other class that implements a `partial_project` method
#' @param new_data A matrix or vector of new observations with a subset of columns equal to length of `colind`. Rows represent observations and columns represent variables
#' @param colind A numeric vector of column indices to select in the projection matrix. These indices correspond to the variables used for the partial projection
#' @return A matrix or vector of the partially projected observations, where rows represent observations and columns represent the lower-dimensional space
#' @export
#' @family partial_project
#' @seealso \code{\link{bi_projector}} for an example of a class that implements a `partial_project` method
#' @examples
#' # Example with the bi_projector class
#' X <- matrix(rnorm(10*20), 10, 20)
#' svdfit <- svd(X)
#' p <- bi_projector(svdfit$v, s = svdfit$u %*% diag(svdfit$d), sdev=svdfit$d)
#'
#' # Partially project new_data onto the same subspace as the original data 
#' # using only the first 10 variables
#' new_data <- matrix(rnorm(5*20), 5, 20)
#' colind <- 1:10
#' partially_projected_data <- partial_project(p, new_data[,colind], colind)
partial_project <- function(x, new_data, colind) UseMethod("partial_project")




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
#' @param ... Additional arguments passed to the underlying `project_block` method
#' @return A matrix or vector of the projected data for the specified block
#' @export
#' @family project
#' @seealso \code{\link{project}} for the generic projection function
project_block <- function(x, new_data, block,...) UseMethod("project_block")




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
#' @param comp A vector of component indices to use in the reconstruction
#' @param rowind The row indices to reconstruct (optional). If not provided, all rows are used.
#' @param colind The column indices to reconstruct (optional). If not provided, all columns are used.
#' @param ... Additional arguments passed to the underlying `reconstruct` method
#' @return A reconstructed data set based on the selected components, rows, and columns
#' @export
#' @family reconstruct
#' @seealso \code{\link{bi_projector}} for an example of a two-way mapping model that can be reconstructed
reconstruct <- function(x, comp, rowind, colind, ...) UseMethod("reconstruct")



##' Transfer data from one input domain to another via common latent space
#'
#' Convert between data representations in a multiblock decomposition/alignment by projecting
#' the input data onto a common latent space and then reconstructing it in the target domain.
#'
#' @param x The model fit, typically an object of a class that implements a `transfer` method
#' @param new_data The data to transfer, with the same number of rows as the source data block
#' @param i The index of the source data block
#' @param j The index of the destination data block
#' @param comp A vector of component indices to use in the reconstruction
#' @param rowind Optional set of row indices to transfer (default: all rows)
#' @param colind Optional set of column indices to transfer (default: all columns)
#' @param ... Additional arguments passed to the underlying `convert_domain` method
#' @return A matrix or data frame representing the transferred data in the target domain
#' @export
#' @seealso \code{\link{project_block}} for projecting a single block of data onto the subspace
convert_domain <- function(x, new_data, i, j, comp, rowind, colind, ...) UseMethod("transfer")



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
#' X <- iris[, 1:4]
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
#' @return a logical value indicating whether the transformation is orthogonal
is_orthogonal <- function(x) UseMethod("is_orthogonal")


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
#' @return A modified model fit with the rotated components.
#' @export
rotate <- function(x, ncomp, type) UseMethod("rotate")




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



