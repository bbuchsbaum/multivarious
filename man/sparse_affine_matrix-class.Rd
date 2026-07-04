\name{sparse_affine_matrix-class}
\alias{sparse_affine_matrix-class}
\alias{sparse_affine_transpose-class}
\alias{\%*\%,ANY,sparse_affine_matrix-method}
\alias{\%*\%,ANY,sparse_affine_transpose-method}
\alias{\%*\%,sparse_affine_matrix,ANY-method}
\alias{\%*\%,sparse_affine_transpose,ANY-method}
\alias{[,sparse_affine_matrix,ANY,ANY,ANY-method}
\alias{[,sparse_affine_matrix,ANY,missing,ANY-method}
\alias{[,sparse_affine_matrix,missing,ANY,ANY-method}
\alias{[,sparse_affine_matrix,missing,missing,ANY-method}
\alias{as.matrix,sparse_affine_matrix-method}
\alias{as.matrix,sparse_affine_transpose-method}
\alias{crossprod,ANY,sparse_affine_matrix-method}
\alias{crossprod,sparse_affine_matrix,ANY-method}
\alias{crossprod,sparse_affine_matrix,missing-method}
\alias{crossprod,sparse_affine_transpose,ANY-method}
\alias{crossprod,sparse_affine_transpose,missing-method}
\alias{dim,sparse_affine_matrix-method}
\alias{dim,sparse_affine_transpose-method}
\alias{dimnames,sparse_affine_matrix-method}
\alias{dimnames,sparse_affine_transpose-method}
\alias{t,sparse_affine_matrix-method}
\alias{t,sparse_affine_transpose-method}
\alias{tcrossprod,ANY,sparse_affine_matrix-method}
\alias{tcrossprod,sparse_affine_matrix,ANY-method}
\alias{tcrossprod,sparse_affine_matrix,missing-method}
\alias{tcrossprod,sparse_affine_transpose,ANY-method}
\alias{tcrossprod,sparse_affine_transpose,missing-method}
\title{Lazy sparse affine matrices}
\description{
Objects of class \code{sparse_affine_matrix} represent sparse matrix
preprocessing of the form \code{sweep(x, 2, scale, "*") + shift} without
materializing dense centered columns. The companion
\code{sparse_affine_transpose} class represents its transpose.
}
\usage{
\S4method{dim}{sparse_affine_matrix}(x)
\S4method{dim}{sparse_affine_transpose}(x)

\S4method{dimnames}{sparse_affine_matrix}(x)
\S4method{dimnames}{sparse_affine_transpose}(x)

\S4method{as.matrix}{sparse_affine_matrix}(x, ...)
\S4method{as.matrix}{sparse_affine_transpose}(x, ...)

\S4method{[}{sparse_affine_matrix,ANY,ANY,ANY}(x, i, j, ..., drop = TRUE)
\S4method{[}{sparse_affine_matrix,ANY,missing,ANY}(x, i, j, ..., drop = TRUE)
\S4method{[}{sparse_affine_matrix,missing,ANY,ANY}(x, i, j, ..., drop = TRUE)
\S4method{[}{sparse_affine_matrix,missing,missing,ANY}(x, i, j, ..., drop = TRUE)

\S4method{t}{sparse_affine_matrix}(x)
\S4method{t}{sparse_affine_transpose}(x)

\S4method{\%*\%}{sparse_affine_matrix,ANY}(x, y)
\S4method{\%*\%}{ANY,sparse_affine_matrix}(x, y)
\S4method{\%*\%}{sparse_affine_transpose,ANY}(x, y)
\S4method{\%*\%}{ANY,sparse_affine_transpose}(x, y)

\S4method{crossprod}{sparse_affine_matrix,missing}(x, y = NULL, ...)
\S4method{crossprod}{sparse_affine_matrix,ANY}(x, y = NULL, ...)
\S4method{crossprod}{ANY,sparse_affine_matrix}(x, y = NULL, ...)
\S4method{crossprod}{sparse_affine_transpose,missing}(x, y = NULL, ...)
\S4method{crossprod}{sparse_affine_transpose,ANY}(x, y = NULL, ...)

\S4method{tcrossprod}{sparse_affine_matrix,missing}(x, y = NULL, ...)
\S4method{tcrossprod}{sparse_affine_matrix,ANY}(x, y = NULL, ...)
\S4method{tcrossprod}{ANY,sparse_affine_matrix}(x, y = NULL, ...)
\S4method{tcrossprod}{sparse_affine_transpose,missing}(x, y = NULL, ...)
\S4method{tcrossprod}{sparse_affine_transpose,ANY}(x, y = NULL, ...)
}
\arguments{
\item{x}{A sparse affine matrix, sparse affine transpose, or compatible
left-hand matrix argument.}
\item{y}{A compatible right-hand matrix argument.}
\item{i, j}{Row and column indices for subsetting.}
\item{drop}{Whether to drop dimensions when subsetting.}
\item{...}{Additional arguments passed to the generic.}
}
\details{
These classes are implementation details returned by sparse-aware
preprocessing steps such as \code{\link[=center]{center()}} and
\code{\link[=standardize]{standardize()}} when sparse inputs would otherwise
be densified by centering. Users normally do not construct these objects
directly.

The documented S4 methods provide the matrix operations used by downstream
estimators: dimensions and names, subsetting, explicit materialization through
\code{as.matrix()}, transpose views, matrix multiplication, and cross-products.
The algebraic methods apply the affine offset during the multiplication so that
common projections and decompositions can operate without first expanding the
preprocessed sparse input to a dense matrix.
}
\value{
Dimension and name methods return the corresponding base values. Subsetting
returns either another sparse affine view or a materialized vector or matrix
when \code{drop = TRUE}. Multiplication and cross-product methods return base
matrix results matching the corresponding dense affine calculation.
}
\seealso{
\code{\link[=prep]{prep()}}, \code{\link[=fit_transform]{fit_transform()}},
\code{\link[=transform]{transform()}}
}
\keyword{classes}
\keyword{methods}
