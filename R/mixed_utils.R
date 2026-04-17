#' Identity basis specification
#'
#' @return A basis specification for the unregularized feature space.
#' @export
identity_basis <- function() {
  structure(
    list(type = "identity"),
    class = "mixed_basis_spec"
  )
}

#' Shared PCA basis specification
#'
#' @param ncomp Number of shared basis components.
#' @param fit_on Basis fitting target.
#' @return A basis specification for a shared PCA basis.
#' @export
shared_pca <- function(ncomp, fit_on = c("nuisance_residual", "full", "whitened_residual")) {
  fit_on <- match.arg(fit_on)
  chk::chk_count(ncomp)
  structure(
    list(type = "shared_pca", ncomp = ncomp, fit_on = fit_on),
    class = "mixed_basis_spec"
  )
}

#' Supplied basis specification
#'
#' @param basis A projector-like object with `components()`.
#' @return A basis specification wrapping the supplied basis object.
#' @export
supplied_basis <- function(basis) {
  chk::chk_s3_class(basis, "projector")
  structure(
    list(type = "supplied", basis = basis),
    class = "mixed_basis_spec"
  )
}

#' @keywords internal
#' @noRd
normalize_mixed_term <- function(term) {
  if (inherits(term, "formula")) {
    rhs <- attr(stats::terms(term), "term.labels")
    if (length(rhs) != 1) {
      stop("Formula term specification must contain exactly one term.")
    }
    rhs[[1]]
  } else if (is.character(term) && length(term) == 1) {
    term
  } else {
    stop("`term` must be a single character string or one-sided formula.")
  }
}

#' @keywords internal
#' @noRd
normalize_mixed_response <- function(Y, design = NULL) {
  if (is.array(Y) && length(dim(Y)) == 3L) {
    dims <- dim(Y)
    n_subject <- dims[[1]]
    n_within <- dims[[2]]
    p <- dims[[3]]

    Y_mat <- do.call(
      rbind,
      lapply(seq_len(n_subject), function(i) {
        Yi <- Y[i, , , drop = FALSE]
        matrix(Yi, nrow = n_within, ncol = p)
      })
    )

    if (is.null(design)) {
      design <- data.frame(
        subject = factor(rep(seq_len(n_subject), each = n_within)),
        within = factor(rep(seq_len(n_within), times = n_subject))
      )
    } else {
      chk::chk_equal(nrow(design), n_subject * n_within)
    }

    return(list(
      Y = Y_mat,
      design = design,
      input_type = "array",
      original_dims = dims
    ))
  }

  Y <- as.matrix(Y)
  if (is.null(design)) {
    stop("`design` must be supplied when `Y` is a matrix.")
  }
  chk::chk_equal(nrow(Y), nrow(design))

  list(
    Y = Y,
    design = design,
    input_type = "matrix",
    original_dims = dim(Y)
  )
}

#' @keywords internal
#' @noRd
resolve_mixed_basis <- function(basis, Y_proc, row_engine = NULL) {
  if (is.null(basis)) {
    basis <- identity_basis()
  }

  if (inherits(basis, "projector")) {
    B <- components(basis)
    return(list(
      spec = supplied_basis(basis),
      basis_fit = basis,
      B = B
    ))
  }

  if (!inherits(basis, "mixed_basis_spec")) {
    stop("`basis` must be NULL, a mixed basis specification, or a projector-like object.")
  }

  if (basis$type == "identity") {
    p <- ncol(Y_proc)
    B <- diag(1, nrow = p, ncol = p)
    basis_fit <- projector(B, preproc = fit(pass(), Y_proc))
    return(list(spec = basis, basis_fit = basis_fit, B = B))
  }

  if (basis$type == "shared_pca") {
    k <- min(basis$ncomp, nrow(Y_proc), ncol(Y_proc))
    source <- if (basis$fit_on == "full") {
      Y_proc
    } else {
      if (is.null(row_engine)) {
        stop("A fitted row engine is required for residual-based shared PCA bases.")
      }
      Yw <- row_engine$whiten(Y_proc)
      if (basis$fit_on == "whitened_residual") {
        Yw - row_engine$P_model %*% Yw
      } else if (basis$fit_on == "nuisance_residual") {
        if (length(row_engine$intercept_cols)) {
          P0 <- orthogonal_projector(row_engine$X_w[, row_engine$intercept_cols, drop = FALSE])
          Yw - P0 %*% Yw
        } else {
          Yw
        }
      } else {
        stop("Unsupported shared PCA fitting target: ", basis$fit_on)
      }
    }
    pfit <- pca(source, ncomp = k, preproc = pass())
    B <- components(pfit)
    return(list(spec = basis, basis_fit = pfit, B = B))
  }

  if (basis$type == "supplied") {
    B <- components(basis$basis)
    return(list(spec = basis, basis_fit = basis$basis, B = B))
  }

  stop("Unsupported basis specification.")
}

#' @keywords internal
#' @noRd
fit_or_apply_preproc <- function(preproc, Y) {
  if (inherits(preproc, "pre_processor")) {
    list(preproc = preproc, transformed = transform(preproc, Y))
  } else {
    res <- fit_transform(preproc, Y)
    list(preproc = res$preproc, transformed = res$transformed)
  }
}

#' @keywords internal
#' @noRd
whitened_effect_matrix <- function(fit, term) {
  meta <- fit$effects_meta[[term]]
  if (is.null(meta)) {
    stop("Unknown term: ", term)
  }
  B <- fit$basis_matrix
  Yw_basis <- fit$row_engine$whiten(fit$Y_proc %*% B)
  list(
    M = meta$P_term %*% Yw_basis,
    Yw_basis = Yw_basis,
    B = B,
    meta = meta
  )
}

#' @keywords internal
#' @noRd
svd_effect <- function(M, B, rank_max) {
  p <- nrow(B)
  if (rank_max < 1L) {
    return(list(
      v = matrix(0, nrow = p, ncol = 0),
      s = matrix(0, nrow = nrow(M), ncol = 0),
      sdev = numeric(0),
      rank = 0L
    ))
  }
  sv <- svd(M, nu = rank_max, nv = rank_max)
  idx <- seq_len(rank_max)
  v <- B %*% sv$v[, idx, drop = FALSE]
  s <- sv$u[, idx, drop = FALSE] %*%
    diag(sv$d[idx], nrow = rank_max, ncol = rank_max)
  list(v = v, s = s, sdev = sv$d[idx], rank = rank_max)
}

#' @keywords internal
#' @noRd
inverse_transform_contribution <- function(preproc, X, colind = NULL) {
  zero <- matrix(0, nrow = nrow(X), ncol = ncol(X))
  inverse_transform(preproc, X, colind = colind) -
    inverse_transform(preproc, zero, colind = colind)
}

#' @keywords internal
#' @noRd
parse_random_spec <- function(random, design) {
  if (is.null(random)) {
    return(list(
      formula = NULL,
      grouping_var = NULL,
      random_terms = character(),
      subject_blocks = NULL,
      mode = "none"
    ))
  }

  findbars_fun <- NULL
  if (requireNamespace("reformulas", quietly = TRUE)) {
    findbars_fun <- reformulas::findbars
  } else if (requireNamespace("lme4", quietly = TRUE)) {
    findbars_fun <- lme4::findbars
  } else {
    stop("Random-effects formulas require either the 'reformulas' or 'lme4' package.")
  }

  bars <- findbars_fun(random)
  if (length(bars) < 1L) {
    stop("No random-effects terms were found in `random`.")
  }

  grouping_vars <- unique(vapply(bars, function(bar) as.character(bar[[3]]), character(1)))
  if (length(grouping_vars) != 1L) {
    stop("Milestone 2 currently supports multiple random-effects terms only when they share one grouping variable.")
  }
  grouping_var <- grouping_vars[[1]]
  if (!grouping_var %in% names(design)) {
    stop("Grouping variable '", grouping_var, "' was not found in `design`.")
  }

  lhs_has_intercept <- function(lhs) {
    base::attr(stats::terms(stats::as.formula(paste("~", paste(deparse(lhs), collapse = "")))), "intercept") == 1L
  }

  has_intercept <- any(vapply(bars, function(bar) lhs_has_intercept(bar[[2]]), logical(1)))
  random_terms <- unique(unlist(lapply(bars, function(bar) setdiff(all.vars(bar[[2]]), "1")), use.names = FALSE))
  lhs_tokens <- c(if (has_intercept) "1", random_terms)
  lhs_formula <- if (length(lhs_tokens)) {
    stats::as.formula(paste("~", paste(lhs_tokens, collapse = " + ")))
  } else {
    stats::as.formula("~ 1")
  }
  split_idx <- split(seq_len(nrow(design)), design[[grouping_var]], drop = TRUE)
  subject_blocks <- unname(Filter(function(idx) length(idx) > 0, split_idx))

  list(
    formula = random,
    lhs = bars,
    lhs_formula = lhs_formula,
    grouping_var = grouping_var,
    random_terms = random_terms,
    subject_blocks = subject_blocks,
    mode = "grouped_identity",
    n_bars = length(bars)
  )
}

#' @keywords internal
#' @noRd
is_constant_within_groups <- function(x, groups) {
  split_x <- split(x, groups)
  all(vapply(split_x, function(v) {
    vals <- unique(v[!is.na(v)])
    length(vals) <= 1L
  }, logical(1)))
}

#' @keywords internal
#' @noRd
classify_term_scope <- function(term_label, design, grouping_var) {
  if (is.null(grouping_var)) {
    return("ungrouped")
  }

  vars <- strsplit(term_label, ":", fixed = TRUE)[[1]]
  vars <- trimws(vars)
  if (!length(vars)) {
    return("ungrouped")
  }

  const_flags <- vapply(vars, function(v) {
    if (!v %in% names(design)) {
      FALSE
    } else {
      is_constant_within_groups(design[[v]], design[[grouping_var]])
    }
  }, logical(1))

  if (all(const_flags)) {
    "between"
  } else if (all(!const_flags)) {
    "within"
  } else {
    "mixed"
  }
}

#' @keywords internal
#' @noRd
permute_blocks_same_size <- function(blocks) {
  sizes <- vapply(blocks, length, integer(1))
  idx_by_size <- split(seq_along(blocks), sizes)
  source_order <- seq_along(blocks)

  for (grp in idx_by_size) {
    source_order[grp] <- sample(grp, length(grp), replace = FALSE)
  }

  unlist(blocks[source_order], use.names = FALSE)
}

#' @keywords internal
#' @noRd
resample_blocks <- function(blocks, replace = TRUE) {
  picked <- sample.int(length(blocks), size = length(blocks), replace = replace)
  unlist(blocks[picked], use.names = FALSE)
}
