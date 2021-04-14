#' new sample projection
#' 
#' Project one or more *samples* of onto a subspace. 
#' 
#' @param x the model fit
#' @param new_data a matrix or vector of new observations(s) with the same number of columns as the original data.
#' @param ... extra args
#' @export
project <- function(x, new_data, ...) UseMethod("project")


#' partially project a new sample onto subspace
#' 
#' project a selected subset of column indices onto the subspace
#' 
#' @inheritParams project
#' @param colind the column indices to select in the projection matrix
#' @export
partial_project <- function(x, new_data, colind) UseMethod("partial_project")



#' project a single "block" of data onto the subspace
#' 
#' When observations are concatenated into "blocks", it may be useful to project one block from the set.
#' This is equivalent to a "partial projection" where the column indices are associated with a given block.
#' This is therefore a convenience method of multiblock fits.
#' 
#' @inheritParams project
#' @param block the block id to select in the block projection matrix
#' @export
project_block <- function(x, new_data, block,...) UseMethod("project_block")




#' new variable projection
#' 
#' project one or more *variables* onto a subspace. This is sometimes called supplementary variable projection and can be computed
#' for a biorthogonal decomposition such as SVD.
#' 
#' 
#' @param x the model fit
#' @param new_data a matrix or vector of new observations(s) with the same number of rows as the original data.
#' @param ... extra args
#' @export
project_vars <- function(x, new_data, ...) UseMethod("project_vars")


#' transpose a model
#' 
#' transpose a model by switching coefficients and scores
#' 
#' @param x the model fit
#' @export
transpose <- function(x,...) UseMethod("transpose")



#' reconstruct the data
#' 
#' Reconstruct a data set from its (possibly) low-rank representation.
#' 
#' 
#' @param x the model fit
#' @param comp a vector of component indices to use in the reconstruction.
#' @param ... extra args
#' @export
reconstruct <- function(x, comp, rowind, colind, ...) UseMethod("reconstruct")



#' transfer data from one input domain to another via c ommon latent space
#' 
#' Convert between data representations in a multiblock decomposition/alignment
#' 
#' 
#' @param x the model fit
#' @param new_data the data to transfer
#' @param i the index of the source data block
#' @param j the index of the destination data block
#' @param comp a vector of component indices to use in the reconstruction.
#' @param ... extra args
#' @export
transfer <- function(x, comp, rowind, colind, ...) UseMethod("transfer")



## TODO 
## partial_residuals?
## partial_reconstruct?

#' get residuals of component model fit
#'
#' Extract the residuals of a model, after removing the first \code{ncomp} components
#' 
#' @param x the model fit
#' @param ncomp the number of components to factor out
#' @param xorig the original X data matrix
#' @param ... extra arguments
#' @export
residuals <- function(x, ncomp, xorig, ...) UseMethod("residuals")


#' get the component scores
#' 
#' Extract the factor score matrix from a fit. These are the projections of the data onto the components.
#' 
#' @param x the model fit
#' @param ... extra args
#' @export
scores <- function(x,...) UseMethod("scores")



#' compute standardized scores
#' 
#' 
#' @param the model fit
#' @param ... extra args
#' @export 
std_scores <- function(x, ...) UseMethod("std_scores")



#' get the components
#' 
#' Extract the component matrix of a fit.
#' 
#' @param x the model fit
#' @param ... extra args
#' @export
components <- function(x,...) UseMethod("components")



#' shape of the projector
#' 
#' Get the input/output shape of the projector
#' 
#' @param x the model fit
#' @param ... extra args
#' @export
shape <- function(x,...) UseMethod("shape")


#' inverse of the component matrix
#' 
#' return the inverse projection matrix. Can be used to map back to data space.
#' If the component matrix is orthogonal, then the inverse projection is the transpose of the component matrix.
#' 
#' 
#' @param x the model fit
#' @param ... extra args
#' @export
inverse_projection <- function(x, ...) UseMethod("inverse_projection")


#' inverse projection of a columnwise subset of component matrix (e.g. a sub-block)
#' 
#' If the component matrix is orthogonal, then the inverse projection is the transpose of the component matrix.
#' However, even when the full component matrix is orthogonal, there is no guarantee that the *partial* component matrix is
#' orthogonal.
#' 
#' 
#' @param x the model fit
#' @param ... extra args
#' @export
partial_inverse_projection <- function(x, colind, ...) UseMethod("partial_inverse_projection")


#' compose two projectors
#' 
#' @param x the first projector
#' @param y the second projector
compose_projector <- function(x,y,...) UseMethod("compose_projector")


#' Get a fresh pre-processing node cleared of any cached data
#' 
#' @param x the processing pipeline
#' @param ... extra args
#' 
#' @export
fresh <- function(x,...) UseMethod("fresh")



#' add a pre-processing stage
#' 
#' @param x the processing pipeline
#' @param step the pre-processing step to add
#' @param ... extra args
#' @export
add_node <- function(x, step, ...) UseMethod("add_node")


#' prepare a dataset by applying a pre-processing pipeline
#' 
#' @param x the pipeline
#' @param ... extra args
#' @export
prep <- function(x, ...) UseMethod("prep")


#' apply pre-processing parameters to a new data matrix
#' 
#' Given a new dataset, process it in the same way the original data was processed (e.g. centering, scaling, etc.)
#' 
#' @param x the model fit object
#' @param new_data the new data to process
#' @param colind the column indices of the new data
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
#' @export
refit <- function(x, new_data, ...) UseMethod("refit")


#' get the number of components
#' 
#' The total number of components in the fitted model
#' 
#' @param x the model fit
#' @export
ncomp <- function(x) UseMethod("ncomp")


#' standard deviations 
#' 
#' The standard deviations of the projected data matrix
#' 
#' @param x the model fit
#' @export
sdev <- function(x) UseMethod("sdev")

#' is it orthogonal
#' 
#' test whether components are orthogonal
#' 
#' @param the object
is_orthogonal <- function(x) UseMethod("is_orthogonal")


#' truncate a component fit
#' 
#' take the first n components of a decomposition
#' 
#' @param x the object to truncate
#' @param ncomp number of components to retain
#' @export
truncate <- function(x, ncomp) UseMethod("truncate")


#' get block_lengths
#' 
#' extract the lengths of each block in a multiblock object
#' 
#' @param x the object
#' @export
block_lengths <- function(x) UseMethod("block_lengths")


#' get block_indices 
#' 
#' extract the list of indices associated with each block in a `multiblock` object
#' 
#' @param x the object
#' @param ... extra args
#' @export
block_indices <- function(x, ...) UseMethod("block_indices")


#' get the number of blocks
#' 
#' The number of data blocks in a multiblock element
#' 
#' @param x the object
#' @export
nblocks <- function(x) UseMethod("nblocks")


#' initialize a transform
#' 
#' @param x the pre_processor
#' @param X the data matrix
#' @keywords internal
#' @export
init_transform <- function(x, X, ...) UseMethod("init_transform")




#' @inheritParams init_transform
#' @export
apply_transform <- function(x, X, colind, ...) UseMethod("apply_transform")



#' @inheritParams init_transform
#' @export
reverse_transform <- function(x, X, colind, ...) UseMethod("reverse_transform")

#' bootstrap resampling
#' 
#' bootstrap a multivariate model to estimate component and score variability
#' 
#' @param x the model fit
#' @param nboot the number of bootstrap resamples
#' @param ... extra args
#' @export
bootstrap <- function(x, nboot, ...) UseMethod("bootstrap")


#' construct a classifier 
#' 
#' Given a model object (e.g. `projector` construct a classifier that can generate predictions for new data points.
#' 
#' @param x the model object
#' @param colind the (optional) column indices used for prediction
#' @export
classifier <- function(x, colind, ...) UseMethod("classifier")




