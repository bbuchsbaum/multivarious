% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/svd.R
\name{svd_wrapper}
\alias{svd_wrapper}
\title{Singular Value Decomposition (SVD) Wrapper}
\usage{
svd_wrapper(
  X,
  ncomp = min(dim(X)),
  preproc = pass(),
  method = c("fast", "base", "irlba", "propack", "rsvd", "svds"),
  q = 2,
  p = 10,
  tol = .Machine$double.eps,
  ...
)
}
\arguments{
\item{X}{the input matrix}

\item{ncomp}{the number of components to estimate (default: min(dim(X)))}

\item{preproc}{the pre-processor to apply on the input matrix (e.g., \code{center()}, \code{standardize()}, \code{pass()})
Can be a \code{prepper} object or a pre-processing function.}

\item{method}{the SVD method to use: 'base', 'fast', 'irlba', 'propack', 'rsvd', or 'svds'}

\item{q}{parameter passed to method \code{rsvd} (default: 2)}

\item{p}{parameter passed to method \code{rsvd} (default: 10)}

\item{tol}{minimum relative tolerance for dropping singular values (compared to the largest). Default: \code{.Machine$double.eps}.}

\item{...}{extra arguments passed to the selected SVD function}
}
\value{
an SVD object that extends \code{bi_projector}
}
\description{
Computes the singular value decomposition of a matrix using one of the specified methods.
It is designed to be an easy-to-use wrapper for various SVD methods available in R.
}
\examples{
# Load iris dataset and select the first four columns
data(iris)
X <- iris[, 1:4]

# Compute SVD using the base method and 3 components
fit <- svd_wrapper(X, ncomp = 3, preproc = center(), method = "base")
}
