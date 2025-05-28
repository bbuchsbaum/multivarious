#' @noRd
split_matrix <- function(X, fac) {
  idx <- split(1:nrow(X), fac)
  lapply(idx, function(i) X[i,])
}


#' Compute column-wise mean in X for each factor level of Y
#'
#' This function computes group means for each factor level of Y in the provided data matrix X.
#'
#' @param Y a vector of labels to compute means over disjoint sets
#' @param X a data matrix from which to compute means
#' @return a matrix with row names corresponding to factor levels of Y and column-wise means for each factor level
#' @export
#' @examples
#' # Example data
#' X <- matrix(rnorm(50), 10, 5)
#' Y <- factor(rep(1:2, each = 5))
#'
#' # Compute group means
#' gm <- group_means(Y, X)
group_means <- function (Y, X) {
  chk::chk_equal(nrow(X), length(Y))
  
  if (all(table(Y) == 1)) {
    warnings("`Y` does not contain more than one replicate of any level")
    row.names(X) <- names(table(Y))
    X
  }
  else {
    if (any(is.na(X))) {
      xspl <- split_matrix(X, Y)
      ret <- do.call(rbind, lapply(xspl, function(x) matrixStats::colMeans2(x, na.rm = TRUE)))
      row.names(ret) <- names(xspl)
      ret
    }
    else {
      Rs <- rowsum(X, Y, na.rm = TRUE)
      yt <- table(Y)
      ret <- sweep(Rs, 1, yt, "/")
      row.names(ret) <- names(yt)
      ret
    }
  }
}



#' @noRd
.avg_pair_princ_ang <- function(bases, ...) {
  warning("Subspace similarity method 'avg_pair' is not implemented.")
  return(NA_real_)
}

#' @noRd
.grassmann_dispersion <- function(bases, ...) {
  warning("Subspace similarity method 'grassmann' is not implemented.")
  return(NA_real_)
}

#' @noRd
.worst_case_angle <- function(bases, ...) {
  warning("Subspace similarity method 'worst_case' is not implemented.")
  return(NA_real_)
}

#' Compute subspace similarity
#' 
#' @param fits a list of bi_projector objects
#' @param method the method to use for computing subspace similarity
#' @param ... additional arguments to pass to the method
#' @return a numeric value representing the subspace similarity
#' @export
subspace_similarity <- function(fits,
                                method = c("avg_pair",     # mean of all pair–wise princ. angles
                                           "grassmann",    # Grassmann‑mean dispersion
                                           "worst_case"),  # max{ min angle of each pair }
                                ...) {
  method <- match.arg(method)
  bases  <- lapply(fits, function(f) qr.Q(qr(scores(f))))   # orthonormalise

  switch(method,
    avg_pair   = .avg_pair_princ_ang(bases, ...),
    grassmann  = .grassmann_dispersion(bases, ...),
    worst_case = .worst_case_angle(bases, ...)
  )
}



#' Principal angles (two sub‑spaces)
#'
#' @param fit1,fit2  bi_projector objects (or any object with $v loadings)
#' @param k          number of dimensions to compare (default: min(ncomp))
#' @return numeric vector of principal angles (radians, length = k)
#' @export
principal_angles <- function(fit1, fit2, k = NULL) {
  stopifnot(inherits(fit1, "bi_projector"),
            inherits(fit2, "bi_projector"))

  V1 <- fit1$v
  V2 <- fit2$v
  k  <- if (is.null(k)) min(ncol(V1), ncol(V2)) else k
  V1 <- qr.Q(qr(V1[, 1:k, drop = FALSE]))      # orthonormal bases
  V2 <- qr.Q(qr(V2[, 1:k, drop = FALSE]))

  s  <- svd(crossprod(V1, V2), nu = 0, nv = 0)$d  # singular values
  s  <- pmin(pmax(s, -1), 1)                      # numerical safety

  acos(s)                                         # angles in radians
}

#' Compute a regression model for each column in a matrix and return residual matrix
#' 
#' @param form the formula defining the model to fit for residuals
#' @param X the response matrix
#' @param design the \code{data.frame} containing the design variables specified in \code{form} argument.
#' @param intercept add an intercept term (default is FALSE)
#' 
#' @return a \code{matrix} of residuals
#' @examples 
#' 
#' X <- matrix(rnorm(20*10), 20, 10)
#' des <- data.frame(a=rep(letters[1:4], 5), b=factor(rep(1:5, each=4)))
#' xresid <- residualize(~ a+b, X, design=des)
#' 
#' ## design is saturated, residuals should be zero
#' xresid2 <- residualize(~ a*b, X, design=des)
#' sum(xresid2) == 0
#' @export
#' @importFrom stats model.matrix lsfit resid
residualize <- function(form, X, design, intercept=FALSE) {
  #options(contrasts = c("contr.sum", "contr.poly"))
  modmat <- model.matrix(form, data=design)
  stats::resid(lsfit(modmat, X, intercept=intercept))
}

#' Calculate Principal Angles Between Subspaces
#'
#' Computes the principal angles between two subspaces defined by the
#' columns of two orthonormal matrices Q1 and Q2.
#'
#' @param Q1 An n x p matrix whose columns form an orthonormal basis for the first subspace.
#' @param Q2 An n x q matrix whose columns form an orthonormal basis for the second subspace.
#' @return A numeric vector containing the principal angles in radians, sorted in ascending order.
#'         The number of angles is `min(p, q)`.
#' @export
#' @examples
#' # Example: Angle between xy-plane and a plane rotated 45 degrees around x-axis
#' Q1 <- cbind(c(1,0,0), c(0,1,0)) # xy-plane basis
#' theta <- pi/4
#' R <- matrix(c(1, 0, 0, 0, cos(theta), sin(theta), 0, -sin(theta), cos(theta)), 3, 3)
#' Q2 <- R %*% Q1 # Rotated basis
#' angles_rad <- prinang(Q1, Q2)
#' angles_deg <- angles_rad * 180 / pi
#' print(angles_deg) # Should be approximately 0 and 45 degrees
#'
#' # Example with PCA loadings (after ensuring orthonormality if needed)
#' # Assuming pca1$v and pca2$v are loading matrices (variables x components)
#' # Orthonormalize them first if they are not already (e.g., from standard SVD)
#' # Q1 <- qr.Q(qr(pca1$v[, 1:3]))
#' # Q2 <- qr.Q(qr(pca2$v[, 1:3]))
#' # prinang(Q1, Q2)
prinang <- function(Q1, Q2) {
  # Basic dimension checks
  if (nrow(Q1) != nrow(Q2)) {
    stop("Q1 and Q2 must have the same number of rows.")
  }
  p <- ncol(Q1)
  q <- ncol(Q2)
  if (p == 0 || q == 0) {
    warning("One or both matrices have zero columns. Returning empty vector.")
    return(numeric(0))
  }
  
  # Optional: Add checks for orthonormality (can be computationally expensive)
  # tol <- 1e-8
  # if (max(abs(crossprod(Q1) - diag(p))) > tol || max(abs(crossprod(Q2) - diag(q))) > tol) {
  #   warning("Input matrices may not be orthonormal. Results might be inaccurate.")
  # }
  
  # Compute the SVD of the cross-product
  svd_res <- svd(crossprod(Q1, Q2))
  
  # Singular values are cosines of the principal angles
  # Clamp values to [-1, 1] to avoid domain errors in acos due to potential numerical inaccuracies
  cos_thetas <- pmax(-1, pmin(1, svd_res$d))
  
  # Angles are acos of singular values
  angles <- acos(cos_thetas)
  
  # Return angles sorted in ascending order
  sort(angles)
}

