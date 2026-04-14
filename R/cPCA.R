#' Contrastive PCA++ (cPCA++)
#
#' Performs Contrastive PCA++ (cPCA++) to find directions that capture variation
#' enriched in a "foreground" dataset relative to a "background" dataset.
#' This implementation follows the cPCA++ approach which directly solves the
#' generalized eigenvalue problem Rf v = lambda Rb v, where Rf and Rb are
#' the covariance matrices of the foreground and background data, centered
#' using the *background mean*.
#'
#' @references
#' Abid, A., Zhang, M. J., Bagaria, V. K., & Zou, J. (2018). Exploring patterns enriched in a dataset with contrastive principal component analysis. Nature Communications, 9(1), 2134.
#'
#' Salloum, R., & Kuo, C. C. J. (2022). cPCA++: An efficient method for contrastive feature learning. Pattern Recognition, 124, 108378.
#'
#' Wu, M., Sun, Q., & Yang, Y. (2025). PCA++: How Uniformity Induces Robustness to Background Noise in Contrastive Learning. arXiv preprint arXiv:2511.12278.
#'
#' Woller, J. P., Menrath, D., & Gharabaghi, A. (2025). Generalized contrastive PCA is equivalent to generalized eigendecomposition. PLOS Computational Biology, 21(10), e1013555.
#'
#' @param X_f A numeric matrix representing the foreground dataset (samples x features).
#' @param X_b A numeric matrix representing the background dataset (samples x features).
#'        `X_f` and `X_b` must have the same number of features (columns).
#' @param ncomp Integer. The number of contrastive components to compute. Defaults to
#'        `min(ncol(X_f), nrow(X_f), nrow(X_b))`, and may be further capped by the
#'        effective background rank (especially under the sample-space strategy).
#' @param center_background Logical. If TRUE (default), both `X_f` and `X_b` are centered using the
#'        column means of `X_b`. If FALSE, it assumes data is already appropriately centered.
#' @param lambda Shrinkage intensity for covariance estimation (0 <= lambda <= 1).
#'        Defaults to 0 (no shrinkage). Uses `corpcor::cov.shrink`. Can help stabilize
#'        results if `Rb` is ill-conditioned or singular.
#' @param method A character string specifying the primary computation method. Options include:
#'    - `"geigen"` (Default): Use `geneig` from the `geigen` package.
#'    - `"primme"`: Use `geneig` with the PRIMME library backend (requires special `geigen` build).
#'    - `"sdiag"`: Use `geneig` with a spectral decomposition method.
#'    - `"corpcor"`: Use a corpcor-based whitening approach followed by standard PCA.
#' @param strategy Controls the GEVD approach when `method` is not `"corpcor"`. Options include:
#'    - `"auto"` (Default): Chooses based on dimensions (feature vs. sample space).
#'    - `"feature"`: Forces direct computation via `p x p` covariance matrices.
#'    - `"sample"`: Forces sample-space computation via SVD and a smaller GEVD (efficient for large `p`).
#' @param verbose Logical; if TRUE (default), prints brief status messages about strategy
#'    selection and defaults. Set to FALSE to silence these messages.
#' @param sample_rank Optional integer controlling the background subspace rank used in the
#'    sample-space strategy. If `NULL` (default), uses the full background rank `min(n_b-1, p)`.
#'    If provided, the solver will target approximately `sample_rank + sample_oversample` and
#'    will be bounded above by the full background rank.
#' @param sample_oversample Integer oversampling margin (default 10) applied when `sample_rank`
#'    is given. Ignored when `sample_rank` is `NULL`.
#' @param ... Additional arguments passed to the underlying computation functions 
#'    (`geigen::geneig` or `irlba::irlba` based on `method` and `strategy`).
#'
#' @details
#' **Preprocessing:** Following the cPCA++ paper, if `center_background = TRUE`, both `X_f` and `X_b`
#' are centered by subtracting the column means calculated *only* from the background data `X_b`.
#' This is crucial for isolating variance specific to `X_f`.
#'  
#' **Core Algorithm (methods "geigen", "primme", "sdiag", strategy="feature"):**
#' 1. Center `X_f` and `X_b` using the mean of `X_b`.
#' 2. Compute potentially shrunk \eqn{p \times p} covariance matrices `Rf` (from centered `X_f`) and `Rb` (from centered `X_b`) using `corpcor::cov.shrink`.
#' 3. Solve the generalized eigenvalue problem `Rf v = lambda Rb v` for the top `ncomp` eigenvectors `v` using `geigen::geneig`. These eigenvectors are the contrastive principal components (loadings).
#' 4. Compute scores by projecting the centered foreground data onto the eigenvectors: `S = X_f_centered %*% v`.
#'
#' **Core Algorithm (Large-D / Sample Space Strategy, strategy="sample"):**
#' When \eqn{p \gg n}, forming \eqn{p \times p} matrices `Rf` and `Rb` is infeasible. The "sample" strategy follows cPCA++ §3.2:
#' 1. Center `X_f` and `X_b` using the mean of `X_b`.
#' 2. Compute the SVD of centered \eqn{X_b = Ub Sb Vb^T} (using `irlba` for efficiency).
#' 3. Project centered `X_f` into the background's principal subspace: `Zf = X_f_centered %*% Vb`.
#' 4. Form small \eqn{r \times r} matrices: `Rf_small = cov(Zf)` and `Rb_small = (1/(n_b-1)) * Sb^2`.
#' 5. Solve the small \eqn{r \times r} GEVD: `Rf_small w = lambda Rb_small w` using `geigen::geneig`.
#' 6. Lift eigenvectors back to feature space: `v = Vb %*% w`.
#' 7. Compute scores: `S = X_f_centered %*% v`.
#'
#' **Alternative Algorithm (method "corpcor"):**
#' 1. Center `X_f` and `X_b` using the mean of `X_b`.
#' 2. Compute `Rb` and its inverse square root `Rb_inv_sqrt`.
#' 3. Whiten the foreground data: `X_f_whitened = X_f_centered %*% Rb_inv_sqrt`.
#' 4. Perform standard PCA (`stats::prcomp`) on `X_f_whitened`.
#' 5. The returned `v` and `s` are the loadings and scores *in the whitened space*. The loadings are *not* the generalized eigenvectors `v`. A specific class `corpcor_pca` is added to signal this.
#'
#' @return A `bi_projector`-like object with classes `c("cPCAplus", "<method_class>", "bi_projector")` containing:
#' \describe{
#'   \item{v}{Loadings matrix (features x ncomp). Interpretation depends on `method` (see Details).}
#'   \item{s}{Scores matrix (samples_f x ncomp).}
#'   \item{sdev}{Vector (length ncomp). Standard deviations (sqrt of generalized eigenvalues for `geigen` methods, PCA std devs for `corpcor`).}
#'   \item{values}{Vector (length ncomp). Generalized eigenvalues (for `geigen` methods) or PCA eigenvalues (for `corpcor`).}
#'   \item{strategy}{The strategy used ("feature" or "sample") if method was not "corpcor".}
#'   \item{preproc}{The initialized `preprocessor` object used.}
#'   \item{method}{The computation method used.}
#'   \item{ncomp}{The number of components computed.}
#'   \item{nfeatures}{The number of features.}
#' }
#'
#' @examples
#' # Simulate data where foreground has extra variance in first few dimensions
#' set.seed(123)
#' n_f <- 100
#' n_b <- 150
#' n_features <- 50
#'
#' # Background: standard normal noise
#' X_b <- matrix(rnorm(n_b * n_features), nrow=n_b, ncol=n_features)
#' colnames(X_b) <- paste0("Feat_", 1:n_features)
#'
#' # Foreground: background noise + extra variance in first 5 features
#' X_f_signal <- matrix(rnorm(n_f * 5, mean=0, sd=2), nrow=n_f, ncol=5)
#' X_f_noise <- matrix(rnorm(n_f * (n_features-5)), nrow=n_f, ncol=n_features-5)
#' X_f <- cbind(X_f_signal, X_f_noise) + matrix(rnorm(n_f * n_features), nrow=n_f, ncol=n_features)
#' colnames(X_f) <- paste0("Feat_", 1:n_features)
#' rownames(X_f) <- paste0("SampleF_", 1:n_f)
#'
#' # Apply cPCA++ (requires geigen and corpcor packages)
#' # install.packages(c("geigen", "corpcor"))
#' if (requireNamespace("geigen", quietly = TRUE) && requireNamespace("corpcor", quietly = TRUE)) {
#'   # Assuming helper constructors like bi_projector are available
#'   # library(multivarious) 
#'
#'   res_cpca_plus <- cPCAplus(X_f, X_b, ncomp = 5, method = "geigen")
#'
#'   # Scores for the foreground data (samples x components)
#'   print(head(res_cpca_plus$s))
#'
#'   # Loadings (contrastive directions) (features x components)
#'   print(head(res_cpca_plus$v))
#' }
#'
#' \donttest{
#' # Plot example (slow graphics)
#' if (requireNamespace("geigen", quietly = TRUE) && requireNamespace("corpcor", quietly = TRUE)) {
#'   set.seed(123)
#'   X_b <- matrix(rnorm(150 * 50), nrow=150, ncol=50)
#'   X_f <- cbind(matrix(rnorm(100*5, sd=2), 100, 5), matrix(rnorm(100*45), 100, 45))
#'   res <- cPCAplus(X_f, X_b, ncomp = 5, method = "geigen")
#'   plot(res$s[, 1], res$s[, 2],
#'        xlab = "Contrastive Component 1", ylab = "Contrastive Component 2",
#'        main = "cPCA++ Scores")
#' }
#' }
#'
#' @importFrom stats prcomp
#' @importFrom Matrix crossprod
#' @importFrom stats coef prcomp
#' @importFrom corpcor pseudoinverse
#' @importFrom geigen geigen
#' @export
cPCAplus <- function(X_f, X_b, ncomp = NULL,
                     center_background = TRUE,
                     lambda = 0,
                     method = c("geigen", "primme", "sdiag", "corpcor"),
                     strategy = c("auto", "feature", "sample"),
                     verbose = getOption("multivarious.verbose", TRUE),
                     sample_rank = NULL,
                     sample_oversample = 10L,
                     ...) {

  # --- Input Validation --- 
  method <- match.arg(method)
  strategy <- match.arg(strategy)
  # Use chk or assertions for validation
  stopifnot(is.matrix(X_f), is.numeric(X_f))
  stopifnot(is.matrix(X_b), is.numeric(X_b))
  if (ncol(X_f) != ncol(X_b)) {
    stop("Foreground matrix X_f (", ncol(X_f), " features) and background matrix X_b (",
         ncol(X_b), " features) must have the same number of columns.")
  }
  if (is.null(ncomp)) {
    ncomp <- min(ncol(X_f), nrow(X_f), nrow(X_b))
    if (isTRUE(verbose)) {
      message("ncomp not provided; using min(p, n_f, n_b) = ", ncomp)
    }
  }
  stopifnot(is.numeric(ncomp), length(ncomp) == 1, ncomp > 0, ncomp == floor(ncomp))
  stopifnot(ncomp <= ncol(X_f))
  stopifnot(is.logical(center_background), length(center_background) == 1)
  stopifnot(is.numeric(lambda), length(lambda) == 1, lambda >= 0, lambda <= 1)
  stopifnot(is.logical(verbose), length(verbose) == 1)
  stopifnot(is.null(sample_rank) || (is.numeric(sample_rank) && length(sample_rank) == 1 && sample_rank > 0))
  stopifnot(is.numeric(sample_oversample), length(sample_oversample) == 1)

  # Preserve original row/col names
  rn_f <- rownames(X_f)
  cn <- colnames(X_f)
  if (is.null(cn) && !is.null(colnames(X_b))) cn <- colnames(X_b)

  # --- Preprocessing --- 
  # Fix 2: Create proper pre_processor objects
  if (center_background) {
    mean_b <- colMeans(X_b, na.rm = TRUE)
    if(any(is.na(mean_b))) stop("NA values encountered in background mean calculation.")
    X_f_centered <- sweep(X_f, 2, mean_b, "-")
    X_b_centered <- sweep(X_b, 2, mean_b, "-")
    # Create a finalized preprocessor using the calculated means
    proc <- prep(center(cmeans = mean_b))
  } else {
    X_f_centered <- X_f
    X_b_centered <- X_b
    # Create a finalized identity preprocessor
    proc <- prep(pass())
  }

  # --- Core Computation --- 
  if (method == "corpcor") {
    if (!requireNamespace("corpcor", quietly = TRUE)) {
      stop("Package 'corpcor' needed for method='corpcor'. Please install it.", call. = FALSE)
    }

    # Compute Rb using cov.shrink on already centered data
    Rb <- corpcor::cov.shrink(X_b_centered, lambda = lambda, verbose = FALSE)

    # Compute Rb^(-1/2) using eigenvalue decomposition
    eig_Rb <- eigen(Rb, symmetric = TRUE)
    eig_vals_Rb_inv_sqrt <- ifelse(eig_Rb$values > sqrt(.Machine$double.eps), 1 / sqrt(eig_Rb$values), 0)
    # Check for potential issues
    if (sum(eig_vals_Rb_inv_sqrt > 0) < ncomp) {
        warning("Rank of background covariance (after thresholding) is less than ncomp. Results may be unreliable.")
    }
    Rb_inv_sqrt <- eig_Rb$vectors %*% diag(eig_vals_Rb_inv_sqrt, nrow=length(eig_vals_Rb_inv_sqrt)) %*% t(eig_Rb$vectors)

    # Whiten foreground data
    X_f_whitened <- X_f_centered %*% Rb_inv_sqrt

    # Perform standard PCA on the whitened data
    # Using prcomp for robustness and standard output
    pca_res <- tryCatch({
        stats::prcomp(X_f_whitened, center = FALSE, scale. = FALSE, rank. = ncomp)
        }, error = function(e){ 
            stop("Error during stats::prcomp on whitened data: ", e$message)
        })

    # Adjust ncomp if prcomp returned fewer components
    actual_ncomp <- min(ncomp, ncol(pca_res$rotation))
    if (actual_ncomp < ncomp) {
        warning("prcomp returned fewer components (", actual_ncomp, ") than requested (", ncomp, ").")
        ncomp <- actual_ncomp
    }
    if (ncomp == 0) stop("PCA on whitened data yielded zero components.")

    # Results are in the whitened space
    eigenvectors_whitened <- pca_res$rotation[, 1:ncomp, drop = FALSE]
    eigenvalues_pca <- (pca_res$sdev[1:ncomp])^2
    scores_pca <- pca_res$x[, 1:ncomp, drop = FALSE]

    # Back-transform loadings to original space
    v_orig <- Rb_inv_sqrt %*% eigenvectors_whitened

    # Recalculate scores by projecting original centered data onto back-transformed loadings
    scores <- X_f_centered %*% v_orig

    # Create structure using bi_projector constructor
    # Note: values/sdev are from PCA on whitened data, not generalized eigenvalues
     projector <- bi_projector(
            v = v_orig,                  # Loadings in original space
            s = scores,                  # Scores = projection onto original space loadings
            sdev = pca_res$sdev[1:ncomp], # Sdev from PCA on whitened data
            values = eigenvalues_pca,    # Eigenvalues from PCA on whitened data
            preproc = proc,
            classes = c("cPCAplus", "corpcor_pca", "bi_projector"),
            method_used = list(type = "cPCAplus", method = method, lambda=lambda, ncomp=ncomp) # Store extra info
        )

  } else {
    # --- Methods using geigen --- 
    if (!requireNamespace("geigen", quietly = TRUE)) {
      stop("Package 'geigen' needed for methods 'geigen', 'primme', 'sdiag'. Please install it.", call. = FALSE)
    }

    # Determine strategy: feature space (direct covariance) or sample space (large D)
    p <- ncol(X_f_centered)
    n_f <- nrow(X_f_centered)
    n_b <- nrow(X_b_centered)
    use_sample_space <- FALSE
    effective_strategy <- strategy

    if (strategy == "auto") {
      # Heuristic: p > 5 * max(n_f, n_b)
      if (p > 5 * max(n_f, n_b)) {
        if (isTRUE(verbose)) {
          message("Large dimension detected (p > 5*max(n_f, n_b)); switching to sample-space GEVD strategy.")
        }
        use_sample_space <- TRUE
        effective_strategy <- "sample"
      } else {
        effective_strategy <- "feature"
      }
    } else if (strategy == "sample") {
      use_sample_space <- TRUE
    } # else strategy == "feature" -> use_sample_space remains FALSE

    if (lambda != 0 && use_sample_space) {
      warning("lambda != 0 but sample-space strategy is selected; shrinkage is ignored in sample-space path.")
    }

    # Helper to compute covariance with optional shrinkage
    cov_estimate <- function(mat) {
      if (lambda == 0) {
        stats::cov(mat)
      } else {
        if (!requireNamespace("corpcor", quietly = TRUE)) {
          stop("lambda != 0 requires package 'corpcor'. Please install it.", call. = FALSE)
        }
        corpcor::cov.shrink(mat, lambda = lambda, verbose = FALSE)
      }
    }

    geneig_method <- switch(method,
                            geigen = "geigen",
                            primme = "primme",
                            sdiag  = "sdiag",
                            stop("Unsupported method: ", method))

    # --- Compute Eigen solution based on strategy --- 
    if (use_sample_space) {
        # --- Sample Space Strategy (Large D) --- 
        if (isTRUE(verbose)) message("Using sample-space strategy...")
        if (!requireNamespace("irlba", quietly = TRUE))
          stop("Package 'irlba' needed for large-D sample-space strategy. Install it.", call.=FALSE)

        # 1. Background SVD (thin)
        r_full <- min(n_b - 1L, p)
        if (r_full <= 0) stop("Background rank is non-positive.")

        if (!is.null(sample_rank)) {
          r_target <- min(r_full,
                          max(1L, as.integer(sample_rank) + as.integer(sample_oversample)))
        } else {
          r_target <- r_full
        }
        if (r_target <= 0) stop("Target rank for background SVD is non-positive.")

        svdb <- tryCatch({
          irlba::irlba(X_b_centered, nv = r_target, nu = 0)
        }, error = function(e){ 
          stop("Error during irlba SVD of background matrix X_b: ", e$message)
        })
        
        V_b  <- svdb$v            # p  x r
        # Use adjusted degrees of freedom for covariance estimate
        Sigma2 <- (svdb$d^2) / max(1, n_b - 1)   # length r 
        actual_rank_b <- length(Sigma2)

        # 2. Small matrices
        Rb_small <- diag(Sigma2, nrow = actual_rank_b)
        
        # Foreground projected into background subspace
        Zf  <- X_f_centered %*% V_b            # n_f x r
        # Use adjusted degrees of freedom for covariance estimate
        Rf_small <- crossprod(Zf) / max(1, n_f - 1)  # r x r

        # 3. Solve GEVD in r x r space
        ncomp_geigen <- min(ncomp, actual_rank_b)
        if (ncomp_geigen < ncomp) {
          warning("Reduced ncomp from ", ncomp, " to ", ncomp_geigen, " due to limited background rank.")
          ncomp <- ncomp_geigen
        }
        if (ncomp == 0) stop("Cannot compute components; ncomp is zero after rank adjustment.")

        geig_small <- tryCatch({
          geneig(A = Rf_small, B = Rb_small, ncomp = ncomp, method = geneig_method, ...)
        }, error = function(e) {
          stop("Error during small GEVD via geneig(): ", conditionMessage(e))
        })
        
        # Ensure enough valid eigenvalues/vectors returned
        k_avail_small <- length(geig_small$values)
        actual_ncomp_small <- min(ncomp, k_avail_small)
        if (actual_ncomp_small < ncomp) {
          warning("Small GEVD returned fewer eigenvalues (", k_avail_small, ") than requested (", ncomp, "). Reducing ncomp to ", actual_ncomp_small, ".")
          ncomp <- actual_ncomp_small
        }
        if (ncomp == 0) stop("Small GEVD returned zero components.")

        eigenvalues_raw <- geig_small$values[1:ncomp]
        w <- geig_small$vectors[, 1:ncomp, drop = FALSE] # r x k
        
        # 4. Lift eigenvectors back to feature space
        v_raw <- V_b %*% w                     # p x k
        
        # Ensure real
        if (is.complex(v_raw)) {
            warning("Complex eigenvectors encountered after lifting, taking the real part.")
            v_raw <- Re(v_raw)
        }
        eigenvectors_ordered <- v_raw
        eigenvalues_ordered <- eigenvalues_raw

    } else {
        # --- Feature Space Strategy (Standard/Small D) --- 
        if (isTRUE(verbose)) message("Using feature-space strategy...")
        # Compute covariance matrices using chosen estimator
        Rf <- cov_estimate(X_f_centered)
        Rb <- cov_estimate(X_b_centered)
        Rf <- as.matrix(Rf); class(Rf) <- "matrix"
        Rb <- as.matrix(Rb); class(Rb) <- "matrix"

        # Solve the generalized eigenvalue problem: Rf v = lambda Rb v
        geigen_res <- tryCatch({
          geneig(A = Rf, B = Rb, ncomp = ncomp, method = geneig_method, ...)
        }, error = function(e) {
          stop("Error during geneig(): ", conditionMessage(e))
        })

        # Extract results
        k_avail <- length(geigen_res$values)
        actual_ncomp <- min(ncomp, k_avail)
        if (actual_ncomp < ncomp) {
          warning("geneig() returned fewer eigenvalues (", k_avail, ") than requested (", ncomp, "). Reducing ncomp to ", actual_ncomp, ".")
          ncomp <- actual_ncomp
        }
        if (ncomp == 0) stop("Generalized eigenvalue decomposition returned zero components.")

        eigenvalues_raw <- geigen_res$values[1:ncomp]
        eigenvectors_raw <- geigen_res$vectors[, 1:ncomp, drop = FALSE]

        # Ensure eigenvectors are real
        if (is.complex(eigenvectors_raw)) {
          warning("Complex eigenvectors encountered, taking the real part.")
          eigenvectors_raw <- Re(eigenvectors_raw)
        }
        eigenvectors_ordered <- eigenvectors_raw
        eigenvalues_ordered <- eigenvalues_raw
    }

    # --- Post-process results common to both GEVD strategies --- 
    
    # Order by decreasing eigenvalue magnitude
    ord <- order(eigenvalues_ordered, decreasing = TRUE)
    eigenvalues <- eigenvalues_ordered[ord]
    eigenvectors <- eigenvectors_ordered[, ord, drop = FALSE]

    # Apply sign flip for consistency
    if (is.numeric(eigenvectors) && nrow(eigenvectors) > 0) { # Check if eigenvectors were successfully extracted
        signs <- apply(eigenvectors, 2, function(v) {
            if(all(is.na(v) | v == 0)) return(1) # Handle all zero/NA vectors
            # Flip based on element with largest absolute value
            sign_val <- base::sign(v[which.max(abs(v))])
            # Ensure sign is not 0 if max abs value is 0 (should not happen with check above)
            if (sign_val == 0) 1 else sign_val
            })
        signs[signs == 0] <- 1 # Handle cases where max element is 0
        eigenvectors <- sweep(eigenvectors, 2, signs, FUN = "*")
        # Normalize columns to unit 2-norm to stabilize orientation comparisons
        coln <- sqrt(colSums(eigenvectors^2))
        coln[coln < .Machine$double.eps] <- 1
        eigenvectors <- sweep(eigenvectors, 2, coln, "/")
    } else {
        warning("Could not apply sign flip: eigenvectors are not numeric or empty.")
    }

    # Compute scores: projection of centered foreground data onto eigenvectors
    scores <- X_f_centered %*% eigenvectors

    # Create a bi_projector instance
    projector <- bi_projector(
            v = eigenvectors,         # Generalized eigenvectors (Loadings)
            s = scores,               # Scores = Projection onto loadings
            sdev = sqrt(pmax(eigenvalues, 0)), # Ensure non-negative before sqrt
            values = eigenvalues,     # Generalized eigenvalues
            preproc = proc,
            classes = c("cPCAplus", paste0(effective_strategy, "_pca"), "bi_projector"),
            method_used = list(type = "cPCAplus", method = method, strategy = effective_strategy, lambda=lambda, ncomp=ncomp) 
        )
  }

  # --- Final Touches --- 
  # Assign names
  colnames(projector$v) <- paste0("cPC", 1:ncomp)
  colnames(projector$s) <- paste0("cPC", 1:ncomp)
  if (!is.null(cn)) {
      rownames(projector$v) <- cn
  }
  if (!is.null(rn_f)) {
      rownames(projector$s) <- rn_f
  }

  return(projector)
}
