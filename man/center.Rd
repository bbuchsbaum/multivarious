% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/pre_process.R
\name{center}
\alias{center}
\title{center a data matrix}
\usage{
center(preproc = prepper(), cmeans = NULL)
}
\arguments{
\item{preproc}{the pre-processing pipeline}

\item{cmeans}{optional vector of precomputed column means}
}
\value{
a \code{prepper} list
}
\description{
remove mean of all columns in matrix
}
