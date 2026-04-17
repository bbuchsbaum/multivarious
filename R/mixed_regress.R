#' Mixed-effect multivariate regression
#'
#' Fit the row-side design geometry for operator-valued ANOVA. Supports
#' fixed-effect designs (`random = NULL`) and grouped random-effects designs
#' with a single grouping variable (random intercept and optional random
#' slopes, shared-`Omega` across response features).
#'
#' @param Y Response matrix (`n_obs x p`) or 3D array (`n_subject x n_within x p`).
#' @param design Optional design data frame. Required for matrix input.
#' @param fixed Fixed-effect formula.
#' @param random Random-effect specification. Either `NULL` (fixed-effects only)
#'   or a one-sided formula of the form `~ ... | group` understood by
#'   `reformulas::findbars()` / `lme4::findbars()`. All random-effects bars
#'   must share a single grouping variable.
#' @param basis Feature basis specification or projector-like object.
#' @param preproc Response preprocessor.
#' @param ... Reserved for future extensions.
#' @return A `mixed_fit` object.
#' @export
mixed_regress <- function(Y,
                          design = NULL,
                          fixed,
                          random = NULL,
                          basis = identity_basis(),
                          preproc = center(),
                          ...) {
  normalized <- normalize_mixed_response(Y, design = design)
  Y_mat <- normalized$Y
  design_df <- normalized$design
  random_spec <- parse_random_spec(random, design_df)

  prep_res <- fit_or_apply_preproc(preproc, Y_mat)
  Y_preproc <- prep_res$preproc
  Y_proc <- prep_res$transformed

  row_engine <- build_row_engine(fixed, design_df, random_spec, Y_proc)
  basis_res <- resolve_mixed_basis(basis, Y_proc, row_engine = row_engine)

  out <- structure(
    list(
      call = match.call(),
      Y = Y_mat,
      Y_proc = Y_proc,
      design = design_df,
      fixed = fixed,
      random = random,
      random_spec = random_spec,
      grouping_var = random_spec$grouping_var,
      subject_blocks = random_spec$subject_blocks,
      row_engine = row_engine,
      basis = basis_res$spec,
      basis_fit = basis_res$basis_fit,
      basis_matrix = basis_res$B,
      preproc = Y_preproc,
      row_metric = row_engine$metric,
      effects_meta = row_engine$effects_meta,
      input_type = normalized$input_type,
      original_dims = normalized$original_dims
    ),
    class = "mixed_fit"
  )

  out
}

#' @export
print.mixed_fit <- function(x, ...) {
  cat("mixed_fit object\n\n")
  cat("Observations: ", nrow(x$Y), "\n", sep = "")
  cat("Features: ", ncol(x$Y), "\n", sep = "")
  cat("Terms: ", paste(names(x$effects_meta), collapse = ", "), "\n", sep = "")
  cat("Basis rank: ", ncol(x$basis_matrix), "\n", sep = "")
  cat("Row metric: ", x$row_metric$mode, "\n", sep = "")
  if (!is.null(x$grouping_var)) {
    cat("Grouping variable: ", x$grouping_var, "\n", sep = "")
  }
  if (x$input_type == "array") {
    cat("Input: 3D array normalized to stacked observations\n")
  }
  invisible(x)
}

#' @export
summary.mixed_fit <- function(object, ...) {
  term_df <- vapply(object$effects_meta, `[[`, numeric(1), "df_term")
  term_scope <- vapply(object$effects_meta, `[[`, character(1), "term_scope")
  out <- data.frame(
    term = names(object$effects_meta),
    df_term = term_df,
    scope = term_scope,
    stringsAsFactors = FALSE
  )
  out
}
