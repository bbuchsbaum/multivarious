% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/utils.R
\name{principal_angles}
\alias{principal_angles}
\title{Principal angles (two sub‑spaces)}
\usage{
principal_angles(fit1, fit2, k = NULL)
}
\arguments{
\item{fit1, fit2}{bi_projector objects (or any object with $v loadings)}

\item{k}{number of dimensions to compare (default: min(ncomp))}
}
\value{
numeric vector of principal angles (radians, length = k)
}
\description{
Principal angles (two sub‑spaces)
}
