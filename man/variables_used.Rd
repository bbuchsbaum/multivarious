% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/all_generic.R, R/composed_projector.R
\name{variables_used}
\alias{variables_used}
\alias{variables_used.composed_projector}
\title{Identify Original Variables Used by a Projector}
\usage{
variables_used(x, ...)

\method{variables_used}{composed_projector}(x, tol = 1e-08, ...)
}
\arguments{
\item{x}{A projector object (e.g., \code{projector}, \code{composed_projector}).}

\item{...}{Additional arguments passed to specific methods.}
}
\value{
A sorted numeric vector of unique indices corresponding to the original input variables.
}
\description{
Determines which columns from the \emph{original} input space contribute
(have non-zero influence) to \emph{any} of the output components of the projector.
}
