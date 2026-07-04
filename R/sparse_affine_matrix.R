#' Sparse affine matrix view
#'
#' Represents a preprocessed matrix as `sweep(X, 2, scale, "*") + shift`
#' without materializing dense centered columns.
#'
#' @keywords internal
#' @noRd
methods::setClass(
  "sparse_affine_matrix",
  slots = list(x = "ANY", scale = "numeric", shift = "numeric")
)

#' @keywords internal
#' @noRd
methods::setClass(
  "sparse_affine_transpose",
  slots = list(x = "sparse_affine_matrix")
)

#' @keywords internal
#' @noRd
new_sparse_affine_matrix <- function(x, scale = NULL, shift = NULL) {
  .check_matrix_like(x)

  if (.is_sparse_affine_matrix(x)) {
    base_scale <- x@scale
    base_shift <- x@shift
    base <- x@x
  } else {
    base_scale <- rep(1, ncol(x))
    base_shift <- rep(0, ncol(x))
    base <- x
  }

  p <- ncol(base)
  if (is.null(scale)) scale <- rep(1, p)
  if (is.null(shift)) shift <- rep(0, p)
  chk::chk_equal(length(scale), p)
  chk::chk_equal(length(shift), p)

  methods::new(
    "sparse_affine_matrix",
    x = base,
    scale = base_scale * scale,
    shift = base_shift * scale + shift
  )
}

#' @keywords internal
#' @noRd
.is_sparse_affine_matrix <- function(x) {
  methods::is(x, "sparse_affine_matrix")
}

#' @keywords internal
#' @noRd
.is_sparse_affine_transpose <- function(x) {
  methods::is(x, "sparse_affine_transpose")
}

#' @keywords internal
#' @noRd
.is_sparse_matrix <- function(x) {
  inherits(x, "sparseMatrix")
}

#' @keywords internal
#' @noRd
.is_matrix_like <- function(x) {
  is.matrix(x) || inherits(x, "Matrix") || .is_sparse_affine_matrix(x) ||
    .is_sparse_affine_transpose(x)
}

#' @keywords internal
#' @noRd
.check_matrix_like <- function(x, arg = "X") {
  if (!.is_matrix_like(x)) {
    rlang::abort(sprintf("`%s` must be a matrix or Matrix::Matrix.", arg))
  }
  invisible(x)
}

#' @keywords internal
#' @noRd
.sparse_policy <- function(policy = NULL) {
  if (is.null(policy)) {
    policy <- getOption("multivarious.sparse_policy", "operator")
  }
  policy <- match.arg(policy, c("operator", "auto", "preserve", "allow_dense"))
  if (identical(policy, "auto")) "operator" else policy
}

#' @keywords internal
#' @noRd
.with_sparse_policy <- function(policy, expr) {
  old <- options(multivarious.sparse_policy = .sparse_policy(policy))
  on.exit(options(old), add = TRUE)
  force(expr)
}

#' @keywords internal
#' @noRd
.object_sparse_policy <- function(object, sparse = NULL) {
  if (!is.null(sparse)) {
    return(.sparse_policy(sparse))
  }
  .sparse_policy(attr(object, "sparse_policy"))
}

#' @keywords internal
#' @noRd
.has_nonzero_shift <- function(shift) {
  any(is.finite(shift) & abs(shift) > .Machine$double.eps)
}

#' @keywords internal
#' @noRd
.as_base_matrix <- function(x) {
  if (inherits(x, "Matrix")) as.matrix(x) else x
}

#' @keywords internal
#' @noRd
.affine_transform_matrix <- function(X, scale = NULL, shift = NULL) {
  .check_matrix_like(X)
  p <- ncol(X)
  if (is.null(scale)) scale <- rep(1, p)
  if (is.null(shift)) shift <- rep(0, p)
  chk::chk_equal(length(scale), p)
  chk::chk_equal(length(shift), p)

  if (.is_sparse_affine_matrix(X)) {
    out <- new_sparse_affine_matrix(X, scale = scale, shift = shift)
    if (identical(.sparse_policy(), "preserve") && .has_nonzero_shift(out@shift)) {
      rlang::abort(
        "This preprocessing step would densify a sparse matrix. Use sparse = \"operator\" or sparse = \"allow_dense\".",
        class = "multivarious_sparse_densify_error"
      )
    }
    return(out)
  }

  if (.is_sparse_matrix(X)) {
    policy <- .sparse_policy()
    if (.has_nonzero_shift(shift)) {
      if (identical(policy, "preserve")) {
        rlang::abort(
          "This preprocessing step would densify a sparse matrix. Use sparse = \"operator\" or sparse = \"allow_dense\".",
          class = "multivarious_sparse_densify_error"
        )
      }
      if (identical(policy, "allow_dense")) {
        dense <- as.matrix(X)
        dense <- sweep(dense, 2, scale, "*")
        return(sweep(dense, 2, shift, "+"))
      }
      return(new_sparse_affine_matrix(X, scale = scale, shift = shift))
    }

    return(sweep(X, 2, scale, "*"))
  }

  out <- sweep(X, 2, scale, "*")
  sweep(out, 2, shift, "+")
}

#' @keywords internal
#' @noRd
.matrix_col_means <- function(X, na.rm = FALSE) {
  if (.is_sparse_affine_matrix(X)) {
    return(.matrix_col_means(X@x, na.rm = na.rm) * X@scale + X@shift)
  }
  colMeans(X, na.rm = na.rm)
}

#' @keywords internal
#' @noRd
.matrix_col_sums <- function(X, na.rm = FALSE) {
  if (.is_sparse_affine_matrix(X)) {
    return(.matrix_col_sums(X@x, na.rm = na.rm) * X@scale + nrow(X) * X@shift)
  }
  if (inherits(X, "Matrix")) {
    return(Matrix::colSums(X, na.rm = na.rm))
  }
  colSums(X, na.rm = na.rm)
}

#' @keywords internal
#' @noRd
.matrix_col_sumsq <- function(X, na.rm = FALSE) {
  if (.is_sparse_affine_matrix(X)) {
    sx <- .matrix_col_sums(X@x, na.rm = na.rm)
    sx2 <- .matrix_col_sumsq(X@x, na.rm = na.rm)
    return((X@scale^2) * sx2 + 2 * X@scale * X@shift * sx + nrow(X) * (X@shift^2))
  }
  if (.is_sparse_matrix(X)) {
    return(Matrix::colSums(X^2, na.rm = na.rm))
  }
  colSums(X^2, na.rm = na.rm)
}

#' @keywords internal
#' @noRd
.matrix_col_sds <- function(X, na.rm = FALSE) {
  if (!.is_sparse_affine_matrix(X) && !inherits(X, "Matrix")) {
    return(matrixStats::colSds(X, na.rm = na.rm))
  }
  if (na.rm && anyNA(X)) {
    return(matrixStats::colSds(as.matrix(X), na.rm = TRUE))
  }

  n <- nrow(X)
  if (n <= 1) {
    return(rep(NA_real_, ncol(X)))
  }
  means <- .matrix_col_means(X, na.rm = FALSE)
  ss <- .matrix_col_sumsq(X, na.rm = FALSE) - n * means^2
  sqrt(pmax(ss / (n - 1), 0))
}

#' @keywords internal
#' @noRd
.affine_matmul <- function(X, Y) {
  if (!.is_sparse_affine_matrix(X)) {
    out <- X %*% Y
    return(.as_base_matrix(out))
  }
  y_is_vector <- is.null(dim(Y))
  if (y_is_vector) {
    Y <- matrix(Y, ncol = 1)
  }
  chk::chk_equal(nrow(Y), ncol(X))

  base_prod <- as.matrix(X@x %*% (Y * X@scale))
  shift_prod <- drop(crossprod(X@shift, Y))
  out <- sweep(base_prod, 2, shift_prod, "+")
  if (y_is_vector) drop(out) else out
}

#' @keywords internal
#' @noRd
.matmul_affine <- function(X, Y) {
  x_is_vector <- is.null(dim(X))
  if (x_is_vector) {
    X <- matrix(X, nrow = 1)
  }
  chk::chk_equal(ncol(X), nrow(Y))

  base_prod <- as.matrix(X %*% Y@x)
  scaled <- sweep(base_prod, 2, Y@scale, "*")
  out <- scaled + rowSums(X) %o% Y@shift
  if (x_is_vector) drop(out) else out
}

#' @keywords internal
#' @noRd
.affine_crossprod <- function(X, Y = NULL) {
  if (!.is_sparse_affine_matrix(X)) {
    return(.as_base_matrix(if (is.null(Y)) crossprod(X) else crossprod(X, Y)))
  }

  if (.is_sparse_affine_matrix(Y)) {
    chk::chk_equal(nrow(Y), nrow(X))
    xty <- as.matrix(crossprod(X@x, Y@x))
    scaled <- sweep(sweep(xty, 1, X@scale, "*"), 2, Y@scale, "*")
    xsum <- .matrix_col_sums(X@x) * X@scale
    ysum <- .matrix_col_sums(Y@x) * Y@scale
    return(
      scaled +
        xsum %o% Y@shift +
        X@shift %o% ysum +
        nrow(X) * (X@shift %o% Y@shift)
    )
  }

  if (!is.null(Y)) {
    chk::chk_equal(nrow(Y), nrow(X))
    base <- as.matrix(crossprod(X@x, Y)) * X@scale
    shift <- X@shift %o% colSums(Y)
    return(base + shift)
  }

  xtx <- as.matrix(crossprod(X@x))
  sx <- .matrix_col_sums(X@x)
  scaled <- sweep(sweep(xtx, 1, X@scale, "*"), 2, X@scale, "*")
  scaled +
    (X@scale * sx) %o% X@shift +
    X@shift %o% (X@scale * sx) +
    nrow(X) * (X@shift %o% X@shift)
}

#' @keywords internal
#' @noRd
.affine_tcrossprod <- function(X, Y = NULL) {
  if (is.null(Y)) {
    Y <- X
  }

  if (.is_sparse_affine_matrix(Y)) {
    chk::chk_equal(ncol(Y), ncol(X))
    ty <- Matrix::t(Y@x)
    weighted_ty <- sweep(ty, 1, X@scale * Y@scale, "*")
    base <- as.matrix(X@x %*% weighted_ty)
    x_part <- drop(X@x %*% (X@scale * Y@shift))
    y_part <- drop(Y@x %*% (Y@scale * X@shift))
    base +
      x_part %o% rep(1, nrow(Y)) +
      rep(1, nrow(X)) %o% y_part +
      sum(X@shift * Y@shift)
  } else {
    .affine_matmul(X, t(Y))
  }
}

#' @keywords internal
#' @noRd
.normal_tcrossprod_affine <- function(X, Y) {
  x_is_vector <- is.null(dim(X))
  if (x_is_vector) {
    X <- matrix(X, nrow = 1)
  }
  chk::chk_equal(ncol(X), ncol(Y))

  base <- as.matrix(sweep(X, 2, Y@scale, "*") %*% Matrix::t(Y@x))
  out <- base + drop(X %*% Y@shift) %o% rep(1, nrow(Y))
  if (x_is_vector) drop(out) else out
}

#' @keywords internal
#' @noRd
.affine_as_center_scale <- function(X) {
  if (!.is_sparse_affine_matrix(X)) {
    return(NULL)
  }
  if (any(abs(X@scale) < .Machine$double.eps)) {
    return(NULL)
  }
  list(center = -X@shift / X@scale, scale = 1 / X@scale)
}

#' @keywords internal
#' @noRd
.svd_sparse_implicit <- function(X, k, method, ...) {
  backend <- if (identical(method, "svds")) "svds" else "irlba"
  dots <- list(...)

  if (.is_sparse_affine_matrix(X)) {
    center_scale <- .affine_as_center_scale(X)
    if (is.null(center_scale)) {
      return(NULL)
    }
    A <- X@x
  } else {
    center_scale <- NULL
    A <- X
  }

  if (identical(backend, "irlba")) {
    if (!requireNamespace("irlba", quietly = TRUE)) {
      rlang::abort(
        "Package 'irlba' is needed for sparse implicit SVD. Install it or use sparse = \"allow_dense\".",
        class = "multivarious_missing_sparse_svd_error"
      )
    }
    args <- c(list(A = A, nv = k, nu = k), dots)
    if (!is.null(center_scale)) {
      args$center <- center_scale$center
      args$scale <- center_scale$scale
    }
    return(do.call(irlba::irlba, args))
  }

  if (!requireNamespace("RSpectra", quietly = TRUE)) {
    rlang::abort(
      "Package 'RSpectra' is needed for method = \"svds\".",
      class = "multivarious_missing_sparse_svd_error"
    )
  }
  opts <- dots$opts
  dots$opts <- NULL
  if (is.null(opts)) opts <- list()
  if (!is.null(center_scale)) {
    opts <- utils::modifyList(
      list(center = center_scale$center, scale = center_scale$scale),
      opts
    )
  }
  do.call(RSpectra::svds, c(list(A = A, k = k, nu = k, nv = k, opts = opts), dots))
}

.affine_subset <- function(x, i, j, i_missing = FALSE, j_missing = FALSE, drop = TRUE) {
  if (i_missing) i <- seq_len(nrow(x))
  if (j_missing) {
    j_idx <- seq_len(ncol(x))
    out <- x@x[i, , drop = FALSE]
  } else {
    j_idx <- seq_len(ncol(x))[j]
    out <- x@x[i, j, drop = FALSE]
  }

  res <- new_sparse_affine_matrix(
    out,
    scale = x@scale[j_idx],
    shift = x@shift[j_idx]
  )
  if (drop) {
    drop(as.matrix(res))
  } else {
    res
  }
}

methods::setMethod("dim", "sparse_affine_matrix", function(x) dim(x@x))
methods::setMethod("dim", "sparse_affine_transpose", function(x) rev(dim(x@x)))

methods::setMethod("dimnames", "sparse_affine_matrix", function(x) dimnames(x@x))
methods::setMethod("dimnames", "sparse_affine_transpose", function(x) rev(dimnames(x@x)))

methods::setMethod("as.matrix", "sparse_affine_matrix", function(x, ...) {
  dense <- as.matrix(x@x)
  dense <- sweep(dense, 2, x@scale, "*")
  sweep(dense, 2, x@shift, "+")
})
methods::setMethod("as.matrix", "sparse_affine_transpose", function(x, ...) {
  t(as.matrix(x@x))
})

methods::setMethod(
  "[",
  signature(x = "sparse_affine_matrix", i = "ANY", j = "ANY", drop = "ANY"),
  function(x, i, j, ..., drop = TRUE) .affine_subset(x, i, j, drop = drop)
)
methods::setMethod(
  "[",
  signature(x = "sparse_affine_matrix", i = "ANY", j = "missing", drop = "ANY"),
  function(x, i, j, ..., drop = TRUE) .affine_subset(x, i, j, j_missing = TRUE, drop = drop)
)
methods::setMethod(
  "[",
  signature(x = "sparse_affine_matrix", i = "missing", j = "ANY", drop = "ANY"),
  function(x, i, j, ..., drop = TRUE) .affine_subset(x, i, j, i_missing = TRUE, drop = drop)
)
methods::setMethod(
  "[",
  signature(x = "sparse_affine_matrix", i = "missing", j = "missing", drop = "ANY"),
  function(x, i, j, ..., drop = TRUE) .affine_subset(x, i, j, i_missing = TRUE, j_missing = TRUE, drop = drop)
)

methods::setMethod("t", "sparse_affine_matrix", function(x) {
  methods::new("sparse_affine_transpose", x = x)
})
methods::setMethod("t", "sparse_affine_transpose", function(x) x@x)

methods::setMethod("%*%", signature(x = "sparse_affine_matrix", y = "ANY"), function(x, y) {
  if (.is_sparse_affine_transpose(y)) {
    .affine_tcrossprod(x, y@x)
  } else {
    .affine_matmul(x, y)
  }
})
methods::setMethod("%*%", signature(x = "ANY", y = "sparse_affine_matrix"), function(x, y) {
  .matmul_affine(x, y)
})
methods::setMethod("%*%", signature(x = "sparse_affine_transpose", y = "ANY"), function(x, y) {
  .affine_crossprod(x@x, y)
})
methods::setMethod("%*%", signature(x = "ANY", y = "sparse_affine_transpose"), function(x, y) {
  if (.is_sparse_affine_matrix(x)) {
    .affine_tcrossprod(x, y@x)
  } else {
    .normal_tcrossprod_affine(x, y@x)
  }
})

methods::setMethod("crossprod", signature(x = "sparse_affine_matrix", y = "missing"), function(x, y = NULL, ...) {
  .affine_crossprod(x)
})
methods::setMethod("crossprod", signature(x = "sparse_affine_matrix", y = "ANY"), function(x, y = NULL, ...) {
  .affine_crossprod(x, y)
})
methods::setMethod("crossprod", signature(x = "ANY", y = "sparse_affine_matrix"), function(x, y = NULL, ...) {
  t(.affine_crossprod(y, x))
})
methods::setMethod("crossprod", signature(x = "sparse_affine_transpose", y = "missing"), function(x, y = NULL, ...) {
  .affine_tcrossprod(x@x)
})
methods::setMethod("crossprod", signature(x = "sparse_affine_transpose", y = "ANY"), function(x, y = NULL, ...) {
  .affine_matmul(x@x, y)
})

methods::setMethod("tcrossprod", signature(x = "sparse_affine_matrix", y = "missing"), function(x, y = NULL, ...) {
  .affine_tcrossprod(x)
})
methods::setMethod("tcrossprod", signature(x = "sparse_affine_matrix", y = "ANY"), function(x, y = NULL, ...) {
  .affine_tcrossprod(x, y)
})
methods::setMethod("tcrossprod", signature(x = "ANY", y = "sparse_affine_matrix"), function(x, y = NULL, ...) {
  .normal_tcrossprod_affine(x, y)
})
methods::setMethod("tcrossprod", signature(x = "sparse_affine_transpose", y = "missing"), function(x, y = NULL, ...) {
  .affine_crossprod(x@x)
})
methods::setMethod("tcrossprod", signature(x = "sparse_affine_transpose", y = "ANY"), function(x, y = NULL, ...) {
  .affine_crossprod(x@x, t(y))
})
