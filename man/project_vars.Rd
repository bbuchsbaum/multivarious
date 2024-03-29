% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/all_generic.R
\name{project_vars}
\alias{project_vars}
\title{Project one or more variables onto a subspace}
\usage{
project_vars(x, new_data, ...)
}
\arguments{
\item{x}{The model fit, typically an object of a class that implements a \code{project_vars} method}

\item{new_data}{A matrix or vector of new observation(s) with the same number of rows as the original data}

\item{...}{Additional arguments passed to the underlying \code{project_vars} method}
}
\value{
A matrix or vector of the projected variables in the subspace
}
\description{
This function projects one or more variables onto a subspace. It is often called supplementary variable
projection and can be computed for a biorthogonal decomposition, such as Singular Value Decomposition (SVD).
}
\seealso{
\code{\link{project}} for the generic projection function for samples

Other project: 
\code{\link{project}()},
\code{\link{project.cross_projector}()},
\code{\link{project_block}()}
}
\concept{project}
