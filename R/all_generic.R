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
project_block <- function(x, new_data, block_index,...) UseMethod("project_block")




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



#' reconstruct the data
#' 
#' Reconstruct a data set from its (possibly) low-rank represenation.
#' 
#' 
#' @param x the model fit
#' @param comp a vector of component indices to use in the reconstruction.
#' @param ... extra args
#' @export
reconstruct <- function(x, comp, rowind, colind, ...) UseMethod("reconstruct")



#' Extract the residuals of a model, after removing the first \code{ncomp} components
#' 
#' @param x the model fit
#' @param ncomp the number of components to factor out
#' @param ... extra arguments
residuals <- function(x, ncomp, ...) UseMethod("residuals")


#' get the component scores
#' 
#' Extract the factor score matrix from a fit. These are the projections of the data onto the components.
#' 
#' @param x the model fit
#' @param ... extra args
#' @export
scores <- function(x,...) UseMethod("scores")


#' get the components
#' 
#' Extract the component matrix of a fit.
#' 
#' @param x the model fit
#' @param ... extra args
#' @export
components <- function(x,...) UseMethod("components")



#' apply pre-processing parameters to a new data matrix
#' 
#' Given a new dataset, process it in the same way the original data was processed (e.g. centering, scaling, etc.)
#' 
#' @param x the model fit object
#' @export
reprocess <- function(x, ...) UseMethod("reprocess")


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
truncate <- function(x) UseMethod("trucate")









