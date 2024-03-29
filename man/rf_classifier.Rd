% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/all_generic.R
\name{rf_classifier}
\alias{rf_classifier}
\title{construct a random forest wrapper classifier}
\usage{
rf_classifier(x, colind, ...)
}
\arguments{
\item{x}{the model object}

\item{colind}{the (optional) column indices used for prediction}

\item{...}{extra arguments to \code{randomForest} function}
}
\value{
a random forest classifier
}
\description{
Given a model object (e.g. \code{projector} construct a random forest classifier that can generate predictions for new data points.
}
