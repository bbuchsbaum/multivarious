% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/geneig.R
\name{geneig}
\alias{geneig}
\title{Generalized Eigenvalue Decomposition}
\usage{
geneig(A, B, ncomp, method = c("robust", "sdiag", "geigen", "primme"), ...)
}
\arguments{
\item{A}{The left-hand side square matrix.}

\item{B}{The right-hand side square matrix, same dimension as A.}

\item{ncomp}{Number of eigenpairs to return.}

\item{method}{Method to compute the eigenvalues and eigenvectors:
\itemize{
\item "robust": Uses a stable decomposition via a whitening transform (requires B to be symmetric positive-definite).
\item "sdiag": Uses a spectral decomposition of B and transforms the problem, works when B is symmetric positive-definite.
\item "geigen": Uses the \code{geigen} package for a general solution.
\item "primme": Uses the \code{PRIMME} package for large sparse matrices.
}}

\item{...}{Additional arguments passed to the underlying methods.}
}
\value{
An object of class \code{projector} with eigenvalues stored in \code{values} and standard deviations in \code{sdev = sqrt(values)}.
}
\description{
Computes the generalized eigenvalues and eigenvectors for the problem: A x = λ B x.
Various methods are available and differ in their assumptions about A and B.
}
\examples{
if (requireNamespace("geigen", quietly = TRUE)) {
  A <- matrix(c(14, 10, 12, 10, 12, 13, 12, 13, 14), nrow=3, byrow=TRUE)
  B <- matrix(c(48, 17, 26, 17, 33, 32, 26, 32, 34), nrow=3, byrow=TRUE)
  res <- geneig(A, B, ncomp=3, method="geigen")
  # res$values and coefficients(res)
}
}
