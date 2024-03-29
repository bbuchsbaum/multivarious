% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/all_generic.R
\name{project}
\alias{project}
\title{New sample projection}
\usage{
project(x, new_data, ...)
}
\arguments{
\item{x}{The model fit, typically an object of class bi_projector or any other class that implements a project method}

\item{new_data}{A matrix or vector of new observations with the same number of columns as the original data. Rows represent observations and columns represent variables}

\item{...}{Extra arguments to be passed to the specific project method for the object's class}
}
\value{
A matrix or vector of the projected observations, where rows represent observations and columns represent the lower-dimensional space
}
\description{
Project one or more samples onto a subspace. This function takes a model fit and new observations, and projects them onto the
subspace defined by the model. This allows for the transformation of new data into the same lower-dimensional space as the original data.
}
\examples{
# Example with the bi_projector class
X <- matrix(rnorm(10*20), 10, 20)
svdfit <- svd(X)
p <- bi_projector(svdfit$v, s = svdfit$u \%\% diag(svdfit$d), sdev=svdfit$d)

# Project new_data onto the same subspace as the original data
new_data <- matrix(rnorm(5*20), 5, 20)
projected_data <- project(p, new_data)
}
\seealso{
\code{\link{bi_projector}} for an example of a class that implements a project method

Other project: 
\code{\link{project.cross_projector}()},
\code{\link{project_block}()},
\code{\link{project_vars}()}
}
\concept{project}
