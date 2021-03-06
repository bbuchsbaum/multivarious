% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/all_generic.R
\name{partial_inverse_projection}
\alias{partial_inverse_projection}
\title{inverse projection of a columnwise subset of component matrix (e.g. a sub-block)}
\usage{
partial_inverse_projection(x, colind, ...)
}
\arguments{
\item{x}{the model fit}

\item{...}{extra args}
}
\description{
If the component matrix is orthogonal, then the inverse projection is the transpose of the component matrix.
However, even when the full component matrix is orthogonal, there is no guarantee that the \emph{partial} component matrix is
orthogonal.
}
