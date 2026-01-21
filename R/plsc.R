#' Partial Least Squares Correlation (PLSC)
#'
#' Reference implementation of symmetric brain-behavior PLS (a.k.a. Behavior PLSC).
#' It finds paired weight vectors for X and Y that maximize their cross-block
#' covariance, obtained from the SVD of the cross-covariance (or correlation)
#' matrix \eqn{C_{XY} = X^\top Y / (n-1)}.
#'
#' @param X Numeric matrix of predictors (n x p_x).
#' @param Y Numeric matrix of outcomes/behaviors (n x p_y). Must have the same
#'   number of rows as \code{X}.
#' @param ncomp Number of latent variables to return. Defaults to
#'   \code{min(nrow(X), ncol(X), ncol(Y))}.
#' @param preproc_x Preprocessor for the X block (default: \code{standardize()}).
#'   Use \code{center()} if you want covariance-based PLSC instead of correlation.
#' @param preproc_y Preprocessor for the Y block (default: \code{standardize()}).
#' @param ... Extra arguments stored on the returned object.
#'
#' @return A \code{cross_projector} with class \code{"plsc"} containing
#'   \itemize{
#'     \item \code{vx}, \code{vy}: X and Y loading/weight matrices.
#'     \item \code{sx}, \code{sy}: subject scores for X and Y blocks.
#'     \item \code{singvals}: singular values of \eqn{C_{XY}} (strength of each LV).
#'     \item \code{explained_cov}: proportion of cross-block covariance per LV.
#'     \item \code{preproc_x}, \code{preproc_y}: fitted preprocessors for reuse.
#'   }
#' @export
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(80), 20, 4)
#' Y <- matrix(rnorm(60), 20, 3)
#' fit <- plsc(X, Y, ncomp = 3)
#' fit$singvals
plsc <- function(X,
                 Y,
                 ncomp = NULL,
                 preproc_x = standardize(),
                 preproc_y = standardize(),
                 ...) {
  chk::chk_matrix(X)
  chk::chk_matrix(Y)
  chk::chk_equal(nrow(X), nrow(Y))

  # Fit/transform each block with the modern preprocessing API
  fit_block <- function(p, M) {
    if (inherits(p, "pre_processor")) {
      list(preproc = p, transformed = transform(p, M))
    } else {
      res <- fit_transform(p, M)
      list(preproc = res$preproc, transformed = res$transformed)
    }
  }
  fx_res <- fit_block(preproc_x, X)
  fy_res <- fit_block(preproc_y, Y)
  fx <- fx_res$preproc
  fy <- fy_res$preproc
  Xp <- fx_res$transformed
  Yp <- fy_res$transformed

  max_comp <- min(nrow(Xp), ncol(Xp), ncol(Yp))
  if (is.null(ncomp)) {
    ncomp <- max_comp
  } else {
    chk::chk_range(ncomp, c(1, max_comp))
  }

  # Cross-covariance (correlation if both blocks are standardized)
  Cxy <- crossprod(Xp, Yp) / (nrow(Xp) - 1)

  sv <- svd(Cxy, nu = ncomp, nv = ncomp)
  vx <- sv$u[, seq_len(ncomp), drop = FALSE]
  vy <- sv$v[, seq_len(ncomp), drop = FALSE]
  singvals <- sv$d[seq_len(ncomp)]

  # Subject scores (sometimes called brain/behavior scores)
  sx <- Xp %*% vx
  sy <- Yp %*% vy

  explained <- singvals^2 / sum(singvals^2)

  cross_projector(
    vx, vy,
    preproc_x = fx,
    preproc_y = fy,
    sx = sx,
    sy = sy,
    singvals = singvals,
    explained_cov = explained,
    Cxy = Cxy,
    classes = "plsc",
    ...
  )
}

#' @export
print.plsc <- function(x, ...) {
  cat(crayon::bold(crayon::green("PLSC object (cross-projector)\n\n")))
  cat(crayon::cyan("Samples: "), nrow(x$sx), "\n", sep = "")
  cat(crayon::cyan("X vars: "), nrow(x$vx), " | Y vars: ", nrow(x$vy), "\n", sep = "")
  cat(crayon::cyan("Components: "), ncomp(x), "\n", sep = "")
  if (!is.null(x$singvals)) {
    cat("Singular values: ", paste(round(x$singvals, 4), collapse = ", "), "\n")
  }
  invisible(x)
}

#' Extract scores from a PLSC fit
#' @param x A \code{plsc} object.
#' @param block Which block to return scores for: "X" (default) or "Y".
#' @param ... Ignored.
#' @return Numeric matrix of scores for the chosen block.
#' @export
scores.plsc <- function(x, block = c("X", "Y"), ...) {
  block <- match.arg(block)
  if (block == "X") {
    x$sx
  } else {
    x$sy
  }
}
