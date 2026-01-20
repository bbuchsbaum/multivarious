
# ---------- helpers: which parsing / selection / inversion ----------

.normalize_which <- function(which) {
  w <- toupper(which)
  if (w %in% c("TOP", "LARGEST"))   w <- "LA"
  if (w %in% c("BOTTOM", "SMALLEST")) w <- "SA"
  valid <- c("LA", "SA", "LM", "SM")
  if (!w %in% valid) {
    stop(sprintf("Invalid 'which'='%s'. Allowed: %s", which, paste(valid, collapse = ", ")))
  }
  w
}

.invert_which_recip <- function(which_lambda) {
  switch(which_lambda,
    LA = "SA",
    SA = "LA",
    LM = "SM",
    SM = "LM"
  )
}

.select_eigs <- function(vals, vecs, which, ncomp) {
  w <- .normalize_which(which)
  ord <-
    if (w %in% c("LA", "SA")) {
      order(vals, decreasing = (w == "LA"))
    } else {
      order(abs(vals), decreasing = (w == "LM"))
    }
  keep <- ord[seq_len(min(ncomp, length(vals)))]
  list(values = vals[keep], vectors = vecs[, keep, drop = FALSE])
}

# -------------------------------------------------------------------

#' Generalized Eigenvalue Decomposition
#'
#' Computes the generalized eigenvalues and eigenvectors for the problem: A x = lambda B x.
#' Supports multiple dense and iterative solvers with a unified eigenpair selection interface.
#'
#' @param A The left-hand side square matrix.
#' @param B The right-hand side square matrix, same dimension as A.
#' @param ncomp Number of eigenpairs to return.
#' @param preproc A preprocessing function to apply to the matrices before solving the generalized eigenvalue problem.
#' @param method One of:
#'   - "robust": Uses a stable decomposition via a whitening transform (B must be symmetric PD).
#'   - "sdiag":  Uses a spectral decomposition of B (must be symmetric PD). Requires A to be symmetric for meaningful results.
#'   - "geigen": Uses the \pkg{geigen} package for a general solution (A and B can be non-symmetric).
#'   - "primme": Uses the \pkg{PRIMME} package for large/sparse symmetric problems (A and B must be symmetric).
#'   - "rspectra": Uses \pkg{RSpectra}; if B is SPD it calls \code{eigs_sym(A, B, ...)} directly, otherwise it applies a reciprocal transform to support all targets.
#'   - "subspace": Block subspace iteration for symmetric pairs with SPD B (iterative, no external package required).
#' @param which Which eigenpairs to return. One of
#'   `"LA"` (largest algebraic), `"SA"` (smallest algebraic), `"LM"` (largest magnitude),
#'   or `"SM"` (smallest magnitude). Aliases: `"top"`/`"largest"` -> `"LA"`, `"bottom"`/`"smallest"` -> `"SA"`.
#'   Dense backends select eigenpairs post hoc; `"primme"` supports `"LA"`, `"SA"`, `"SM"` (not `"LM"`);
#'   `"rspectra"` honors all four options. Default is `"LA"`.
#' @param ... Additional arguments to pass to the underlying solver.
#' @return A `projector` object with generalized eigenvectors and eigenvalues.
#' @seealso \code{\link{projector}} for the base class structure.
#'
#' @references
#' Golub, G. H. & Van Loan, C. F. (2013) *Matrix Computations*,
#'   4th ed., Section 8.7 -- textbook derivation for the "robust" (Cholesky)
#'   and "sdiag" (spectral) transforms.
#'
#' Moler, C. & Stewart, G. (1973) "An Algorithm for Generalized Matrix
#'   Eigenvalue Problems". *SIAM J. Numer. Anal.*, 10 (2): 241-256 --
#'   the QZ algorithm behind the \code{geigen} backend.
#'
#' Stathopoulos, A. & McCombs, J. R. (2010) "PRIMME: PReconditioned
#'   Iterative Multi-Method Eigensolver". *ACM TOMS* 37 (2): 21:1-21:30 --
#'   the algorithmic core of the \code{primme} backend.
#'
#' See also the \pkg{geigen} (CRAN) and \pkg{PRIMME} documentation.
#'
#' @importFrom geigen geigen
#' @importFrom RSpectra eigs
#' @importFrom Matrix Cholesky solve forceSymmetric Diagonal t isSymmetric
#' @export
#' @examples
#' \donttest{
#' # Simulate two matrices
#' set.seed(123)
#' A <- matrix(rnorm(50 * 50), 50, 50)
#' B <- matrix(rnorm(50 * 50), 50, 50)
#' A <- A %*% t(A) # Make A symmetric
#' B <- B %*% t(B) + diag(50) * 0.1 # Make B symmetric positive definite
#'
#' # Solve generalized eigenvalue problem
#' result <- geneig(A = A, B = B, ncomp = 3)
#' }
#'
 geneig <- function(A = NULL,
                    B = NULL,
                    ncomp = 2,
                    preproc = prep(pass()),
                    method = c("robust", "sdiag", "geigen", "primme", "rspectra", "subspace"),
                    which = "LA", ...) {
   method <- match.arg(method)
   which <- .normalize_which(which)

   # Validate inputs
   if (is.null(A) || is.null(B)) stop("A and B must be supplied.")
   if (!is.matrix(A) && !inherits(A, "Matrix")) stop("A must be a matrix or a Matrix::Matrix.")
   if (!is.matrix(B) && !inherits(B, "Matrix")) stop("B must be a matrix or a Matrix::Matrix.")
   if (nrow(A) != ncol(A) || nrow(B) != ncol(B) || nrow(A) != nrow(B)) {
     stop("A and B must be square and of the same dimension.")
   }
   if (!is.numeric(ncomp) || ncomp <= 0 || !chk::vld_whole_number(ncomp)) {
     stop("'ncomp' must be a positive integer.")
   }

   # Truncate ncomp if it exceeds matrix dimensions
   if (ncomp > nrow(A)) {
     warning(sprintf("'ncomp' (%d) exceeds matrix dimensions (%d), truncating.", ncomp, nrow(A)))
     ncomp <- nrow(A)
   }

   # Optional preprocessing hook if a function is provided
   if (is.function(preproc)) {
     outpb <- preproc(list(A = A, B = B))
     if (!is.null(outpb$A)) A <- outpb$A
     if (!is.null(outpb$B)) B <- outpb$B
   }

   if (method %in% c("rspectra", "primme") && ncomp >= nrow(A)) {
     warning("Iterative backends require k < n; switching to a dense backend for full decomposition.")
     method <- if ((inherits(B, "Matrix") && Matrix::isSymmetric(B)) || isSymmetric(B)) "robust" else "geigen"
   }

   sym_A <- if (inherits(A, "Matrix")) Matrix::isSymmetric(A) else isSymmetric(A)
   sym_B <- if (inherits(B, "Matrix")) Matrix::isSymmetric(B) else isSymmetric(B)

   res_raw <- switch(
     method,
     robust = {
       if (!isTRUE(sym_B)) {
         stop("For method='robust', B must be symmetric.")
       }
       B_chol_try <- try(chol(B), silent = TRUE)
       if (inherits(B_chol_try, "try-error")) {
         stop("For method='robust', B must be positive definite (Cholesky failed).")
       }
       B_chol <- B_chol_try

       Rinverse <- backsolve(B_chol, diag(nrow(B)))
       W <- forwardsolve(t(B_chol), A %*% Rinverse, upper.tri = FALSE)
       W <- (W + t(W)) / 2

       decomp <- eigen(W, symmetric = TRUE)
       pick <- .select_eigs(decomp$values, decomp$vectors, which, ncomp)

       vectors_raw <- Rinverse %*% pick$vectors
       values <- pick$values

       vectors <- b_orthonormalize(vectors_raw, B)
       list(vectors = vectors, values = values)
     },
     sdiag = {
       if (!isTRUE(sym_B)) {
         stop("For method='sdiag', B must be symmetric.")
       }
       if (!isTRUE(sym_A)) {
         warning("For method='sdiag', A is not symmetric. Results may be inaccurate or complex.")
       }
       min_eigenvalue <- sqrt(.Machine$double.eps)
       B_eig <- eigen(B, symmetric = TRUE)
       valsB <- B_eig$values
       if (any(valsB < min_eigenvalue)) {
         warning(sprintf("B has %d eigenvalues close to zero or negative (min eigenvalue=%.2e). Clamping for inversion.",
                         sum(valsB < min_eigenvalue), min(valsB)))
         valsB[valsB < min_eigenvalue] <- min_eigenvalue
       }

       B_sqrt_inv <- B_eig$vectors %*% diag(1 / sqrt(valsB), nrow = length(valsB), ncol = length(valsB)) %*% t(B_eig$vectors)
       A_transformed <- B_sqrt_inv %*% A %*% B_sqrt_inv
       A_transformed <- (A_transformed + t(A_transformed)) / 2
       A_eig <- eigen(A_transformed, symmetric = TRUE)

       pick <- .select_eigs(A_eig$values, A_eig$vectors, which, ncomp)
       vectors_raw <- B_sqrt_inv %*% pick$vectors
       values <- pick$values

       vectors <- b_orthonormalize(vectors_raw, B)
       list(vectors = vectors, values = values)
     },
     geigen = {
       if (!requireNamespace("geigen", quietly = TRUE)) {
         stop("Package 'geigen' not installed. Please install it for method='geigen'.")
       }
       geigen_symmetric <- isTRUE(sym_A) && isTRUE(sym_B)
       if (geigen_symmetric) {
         chol_try <- try(chol(B), silent = TRUE)
         if (inherits(chol_try, "try-error")) {
           geigen_symmetric <- FALSE
         }
       }
       res <- geigen::geigen(A, B, symmetric = geigen_symmetric)
       ord <-
         if (which %in% c("LA", "SA")) {
           order(Re(res$values), decreasing = (which == "LA"))
         } else {
           order(Mod(res$values), decreasing = (which == "LM"))
         }
       keep <- ord[seq_len(min(ncomp, length(ord)))]
       values <- res$values[keep]
       vectors <- res$vectors[, keep, drop = FALSE]
       list(vectors = vectors, values = values)
     },
     primme = {
       if (!requireNamespace("PRIMME", quietly = TRUE)) {
         stop("Package 'PRIMME' not installed. Please install it for method='primme'.")
       }
       if (!isTRUE(sym_A) || !isTRUE(sym_B)) {
         stop("For method='primme' using eigs_sym, both A and B must be symmetric.")
       }
       which_p <- switch(which,
         LA = "LA",
         SA = "SA",
         SM = "SM",
         LM = stop("PRIMME does not support which='LM' for generalized problems. Use 'LA', 'SA', or 'SM'.")
       )
       user_args <- list(...)
       if (!is.null(user_args$.primme_method)) {
         user_args$method <- user_args$.primme_method
         user_args$.primme_method <- NULL
       }
       protected_args <- c("A", "B", "NEig", "which")
       if (length(user_args)) {
         drop_names <- intersect(names(user_args), protected_args)
         if (length(drop_names)) {
           warning(sprintf(
             "Ignoring PRIMME arguments conflicting with geneig(): %s",
             paste(drop_names, collapse = ", "
           )))
           user_args <- user_args[setdiff(names(user_args), protected_args)]
         }
       }
       primme_defaults <- list(
         method = "PRIMME_DEFAULT_MIN_TIME",
         eps = 1e-6,
         maxBlockSize = max(1L, min(4L, as.integer(ncomp)))
       )
       primme_args <- utils::modifyList(primme_defaults, user_args)
       res <- do.call(
         PRIMME::eigs_sym,
         c(list(A = A, B = B, NEig = ncomp, which = which_p), primme_args)
       )
       list(vectors = res$vectors, values = res$values)
     },
     subspace = {
       if (!isTRUE(sym_A) || !isTRUE(sym_B)) {
         stop("For method='subspace', A and B must be symmetric.")
       }
       which_sub <- switch(which,
         LA = "largest",
         SA = "smallest",
         stop("method='subspace' currently supports which='LA' or 'SA' only.")
       )
       dot_args <- list(...)
       if (length(dot_args)) {
         allowed <- c("max_iter", "tol", "V0", "reg_S", "reg_T", "seed")
         named <- names(dot_args)
         if (is.null(named) || any(named == "")) {
           stop("All additional arguments for method='subspace' must be named.")
         }
         unknown <- setdiff(named, allowed)
         if (length(unknown) > 0) {
           warning(sprintf("Ignoring unsupported arguments for method='subspace': %s", paste(unknown, collapse = ", ")))
         }
         dot_args <- dot_args[named %in% allowed]
       }
       args <- c(list(S1 = A, S2 = B, q = ncomp, which = which_sub), dot_args)
      res <- do.call(solve_gep_subspace, args)
      if (!isTRUE(res$converged)) {
        tol_val <- if (!is.null(args$tol)) args$tol else 1e-5
        warning(sprintf(
          "geneig(method='subspace'): iteration stopped after %d steps with residual %.2e (tol=%.1e).",
          res$iterations, res$residual, tol_val
        ))
      }
      list(vectors = res$vectors, values = res$values)
    },
     rspectra = {
       if (!requireNamespace("RSpectra", quietly = TRUE)) {
         stop("Package 'RSpectra' not installed. Please install it for method='rspectra'.")
       }
       if (!isTRUE(sym_A) || !isTRUE(sym_B)) {
         stop("For method='rspectra', A and B must be symmetric.")
       }
       As <- Matrix::forceSymmetric(if (inherits(A, "Matrix")) A else Matrix::Matrix(A, sparse = FALSE))
       Bs <- Matrix::forceSymmetric(if (inherits(B, "Matrix")) B else Matrix::Matrix(B, sparse = FALSE))
       d <- nrow(As)

      Bchol_try <- try(Matrix::Cholesky(Bs, LDL = FALSE, super = TRUE), silent = TRUE)
      if (!inherits(Bchol_try, "try-error") && which %in% c("LA", "LM")) {
        direct_try <- try(RSpectra::eigs_sym(A = As, B = Bs, k = ncomp, which = which, ...), silent = TRUE)
        rs <- if (inherits(direct_try, "try-error")) {
          A_rs <- if (inherits(As, "sparseMatrix")) {
            methods::as(As, "dgCMatrix")
          } else {
            as.matrix(As)
          }
          B_rs <- if (inherits(Bs, "sparseMatrix")) {
            methods::as(Bs, "dgCMatrix")
          } else {
            as.matrix(Bs)
          }
          RSpectra::eigs_sym(A = A_rs, B = B_rs, k = ncomp, which = which, ...)
        } else {
          direct_try
        }
        lam <- as.numeric(rs$values)
        X <- b_orthonormalize(as.matrix(rs$vectors), Bs)
        list(vectors = X, values = lam)
      } else {
        Afact <- factor_mat(As, reg = 1e-8, max_tries = 6)
        ch <- Afact$ch

        Top <- function(x, args) {
          v <- Matrix::solve(ch, x, system = "L")
          w <- Bs %*% v
          y <- Matrix::solve(ch, w, system = "Lt")
          as.numeric(y)
        }

        which_T <- .invert_which_recip(which)
        rs <- RSpectra::eigs_sym(Top, k = ncomp, which = which_T, n = d, ...)
        mu <- as.numeric(rs$values)
        Y <- rs$vectors

        tiny <- sqrt(.Machine$double.eps)
        if (any(abs(mu) < tiny)) {
          warning("Some mu near 0 in the rspectra reciprocal fallback; corresponding lambda may overflow. Consider method='primme' or 'geigen' with a shift.")
        }

        lam <- 1 / mu
        Xraw <- Matrix::solve(ch, Y, system = "Lt")
        X <- b_orthonormalize(as.matrix(Xraw), Bs)
        list(vectors = X, values = lam)
      }
     }
   )

   ev <- res_raw$values
   vec <- res_raw$vectors

   if (is.complex(ev)) {
     if (any(abs(Im(ev)) > sqrt(.Machine$double.eps))) {
       warning("Complex eigenvalues found. Taking the real part.")
     }
     ev <- Re(ev)
     if (is.complex(vec)) {
       warning("Complex eigenvectors found. Taking the real part.")
       vec <- Re(vec)
     }
   }

   if (any(ev < 0)) {
     warning("Some real eigenvalues are negative. 'sdev' computed using absolute values.")
   }

   sdev <- sqrt(abs(ev))

   out <- list(
     values  = ev,
     vectors = vec,
     sdev    = sdev,
     ncomp   = ncomp,
     method  = method
   )

   class(out) <- c("geneig", "list")
   out
 }

#' Factor a matrix with regularization
#'
#' Attempts a Cholesky factorization with a diagonal `reg` until it succeeds.
#'
#' @param M A symmetric matrix to factor.
#' @param reg Initial regularization term.
#' @param max_tries Number of times to multiply reg by 10 if factorization fails.
#' @return A list with `ch` (the Cholesky factor) and `reg` (the final reg used).
#' @keywords internal
#' @importFrom Matrix Diagonal Cholesky
#' @noRd
factor_mat <- function(M, reg = 1e-3, max_tries = 5) {
  if (!inherits(M, "Matrix")) M <- Matrix::Matrix(M, sparse = FALSE)
  M <- Matrix::forceSymmetric(M)
  reg_now <- 0
  for (i in 0:max_tries) {
    reg_now <- if (i == 0) 0 else (reg * (10^(i - 1)))
    ch <- try(Matrix::Cholesky(M, LDL = FALSE, super = TRUE, Imult = reg_now), silent = TRUE)
    if (!inherits(ch, "try-error")) {
      return(list(ch = ch, reg_added = reg_now))
    }
  }
  stop(sprintf("Unable to factor matrix even after adding regularization up to %.2e.", reg_now))
}

#' Solve using a precomputed Cholesky factor
#'
#' @param ch A Cholesky factor object from `Matrix::Cholesky()`.
#' @param RHS A right-hand-side matrix/vector compatible with `Matrix::solve`.
#' @keywords internal
#' @importFrom Matrix solve
#' @noRd
solve_chol <- function(ch, RHS) {
  Matrix::solve(ch, RHS) # Use Matrix::solve method
}

#' B-orthonormalize a set of vectors
#'
#' @param X Matrix whose columns are candidate eigenvectors.
#' @param B Inner-product matrix.
#' @return Matrix with B-orthonormal columns.
#' @keywords internal
#' @noRd
b_orthonormalize <- function(X, B) {
  Bmat <- if (inherits(B, "Matrix")) B else Matrix::Matrix(B, sparse = FALSE)
  Xmat <- if (inherits(X, "Matrix")) X else Matrix::Matrix(X, sparse = FALSE)
  M <- Matrix::t(Xmat) %*% Bmat %*% Xmat
  M <- (M + Matrix::t(M)) / 2
  M_dense <- as.matrix(M)
  R <- try(chol(M_dense), silent = TRUE)
  if (!inherits(R, "try-error")) {
    Xout <- Xmat %*% solve(R)
  } else {
    eig <- eigen(M_dense, symmetric = TRUE)
    vals <- pmax(eig$values, .Machine$double.eps)
    Xout <- Xmat %*% eig$vectors %*% diag(1 / sqrt(vals), nrow = length(vals))
  }
  as.matrix(Xout)
}

#' Orthonormalize columns via QR
#'
#' @param X A numeric matrix whose columns we want to orthonormalize.
#' @return A matrix of the same dimension with orthonormal columns. Handles potential rank deficiency by returning fewer columns.
#' @keywords internal
#' @importFrom methods as is
#' @noRd
orthonormalize <- function(X) {
  if (ncol(X) == 0) return(X) # Handle empty matrix
  
  # Use Matrix::qr for potential sparse input
  # Convert to dense first if it's not already suitable for base qr
  if (!methods::is(X, "matrix")) {
      X_dense <- try(methods::as(X, "matrix"), silent = TRUE)
      if (inherits(X_dense, "try-error")){
          stop("orthonormalize: Input matrix cannot be coerced to dense matrix for QR.")
      } 
      X <- X_dense
  }
  
  QR <- qr(X)
  rank <- QR$rank
  if (rank == 0) {
      warning("orthonormalize: Input matrix has rank 0.")
      return(matrix(0.0, nrow = nrow(X), ncol = 0)) # Return empty matrix
  }
  
  Q <- qr.Q(QR)
  
  # Return only the first 'rank' columns corresponding to the independent basis
  if (rank < ncol(X)) {
    warning(sprintf("orthonormalize: Input matrix rank (%d) is less than number of columns (%d). Returning orthonormal basis for the column space.", rank, ncol(X)))
  } 
  Q[, 1:rank, drop = FALSE]
}

#' Subspace Iteration for Generalized Eigenproblem
#'
#' Iteratively solves S1 x = lambda S2 x using a subspace approach.
#' Assumes S1, S2 are symmetric. For `which = "largest"`, S2 must be SPD;
#' for `which = "smallest"`, S1 must be SPD.
#'
#' The iteration always applies the operator B^{-1} C while keeping the
#' subspace B-orthonormalized, where (C, B) equals (S1, S2) for `which = "largest"`
#' and (S2, S1) for `which = "smallest"`. The reduced problem solves
#' `T = V^T B V` (SPD) and `S = V^T C V`, with eigenvalues `mu` of
#' `R^{-T} S R^{-1}` mapping directly to `lambda` in the "largest" case and
#' `lambda = 1 / mu` in the "smallest" case.
#'
#' @param S1 A square symmetric matrix (e.g., n x n).
#' @param S2 A square symmetric positive definite matrix of the same dimension.
#' @param q Number of eigenpairs to approximate.
#' @param which "largest" or "smallest" eigenvalues to seek.
#' @param max_iter Maximum iteration count.
#' @param tol Convergence tolerance on relative change in eigenvalues or residuals.
#' @param V0 Optional initial guess matrix (n x q). If NULL, uses random.
#' @param reg_S Regularization added to S1 or S2 during factorization attempts. Default 1e-6.
#' @param reg_T Regularization for the small T matrix. Default 1e-9.
#' @param seed Optional seed for random V0 initialization. If NULL, uses current RNG state.
#' @return A list with `values` = the approximate eigenvalues, `vectors` = the approximate eigenvectors (n x q).
#' @keywords internal
#' @importFrom Matrix Diagonal solve t
#' @noRd
solve_gep_subspace <- function(S1, S2, q = 2,
                               which = c("largest", "smallest"),
                               max_iter = 100, tol = 1e-6,
                               V0 = NULL, reg_S = 1e-6, reg_T = 1e-9,
                               seed = NULL) {
  which <- match.arg(which)
  d <- nrow(S1)

  if (ncol(S1) != d || nrow(S2) != d || ncol(S2) != d) {
    stop("S1 and S2 must be square and of the same dimension.")
  }
  if (q <= 0) stop("q must be positive.")

  if (which == "largest") {
    C <- S1
    B <- S2
  } else {
    C <- S2
    B <- S1
  }

  s_fact <- factor_mat(B, reg = reg_S)
  ch <- s_fact$ch

  solve_step <- function(V) {
    RHS <- C %*% V
    solve_chol(ch, RHS)
  }

  if (is.null(V0)) {
    if (!is.null(seed)) set.seed(seed)
    V <- matrix(rnorm(d * q), nrow = d, ncol = q)
  } else {
    if (ncol(V0) != q || nrow(V0) != d) {
      stop(sprintf("V0 dimensions (%d x %d) do not match expected (%d x %d).",
                   nrow(V0), ncol(V0), d, q))
    }
    V <- V0
  }

  symm <- function(M) (M + t(M)) / 2

  residual_stats <- function(lambda_vec, V_mat) {
    if (!length(lambda_vec)) {
      return(list(max_resid = Inf))
    }
    diag_lambda <- Matrix::Diagonal(x = lambda_vec)
    residual_mat <- S1 %*% V_mat - S2 %*% (V_mat %*% diag_lambda)
    residual_dense <- as.matrix(residual_mat)
    res_norm <- sqrt(colSums(residual_dense^2))
    list(max_resid = max(res_norm / pmax(1, abs(lambda_vec))))
  }

  rayleigh_ritz <- function(V_basis, C_mat, B_mat, which_mode, invert = FALSE) {
    if (ncol(V_basis) == 0) {
      stop("Rayleigh-Ritz requires a non-empty basis.")
    }

    T_mat <- symm(crossprod(V_basis, B_mat %*% V_basis))
    S_mat <- symm(crossprod(V_basis, C_mat %*% V_basis))

    T_dense <- as.matrix(T_mat)
    R <- NULL
    for (attempt in 0:3) {
      bump <- reg_T * (10^attempt)
      T_try <- T_dense
      diag(T_try) <- diag(T_try) + bump
      R_try <- try(chol(T_try), silent = TRUE)
      if (!inherits(R_try, "try-error")) {
        R <- R_try
        break
      }
    }
    if (is.null(R)) {
      stop("Cholesky factorization of projected metric in Rayleigh-Ritz failed.")
    }

    S_small <- as.matrix(S_mat)
    M <- backsolve(R, forwardsolve(t(R), S_small))
    M <- symm(M)

    ee <- eigen(M, symmetric = TRUE)
    mu <- ee$values
    W <- backsolve(R, ee$vectors)

    if (invert) {
      eps <- 1e-300
      lambda <- 1 / pmax(abs(mu), eps) * sign(mu)
      ord <- order(lambda, decreasing = FALSE)
    } else {
      lambda <- mu
      ord <- if (which_mode == "largest") {
        order(lambda, decreasing = TRUE)
      } else {
        order(lambda, decreasing = FALSE)
      }
    }
    lambda <- lambda[ord]
    W <- W[, ord, drop = FALSE]

    V_new <- b_orthonormalize(V_basis %*% W, B_mat)
    if (ncol(V_new) < length(lambda)) {
      lambda <- lambda[seq_len(ncol(V_new))]
    }

    list(values = lambda, vectors = V_new)
  }

  V <- b_orthonormalize(V, B)
  if (ncol(V) < q) {
    warning(sprintf(
      "Initial subspace V has rank %d, less than requested q=%d. Proceeding with reduced rank.",
      ncol(V), q
    ))
    q <- ncol(V)
    if (q == 0) stop("Initial subspace V has rank 0.")
  }

  lambda_prev <- NULL
  lambda_curr <- NULL
  rel_change <- Inf
  max_resid <- Inf
  converged <- FALSE

  iter <- 0L
  for (iter in seq_len(max_iter)) {
    V_hat <- solve_step(V)
    V_new <- b_orthonormalize(V_hat, B)

    q_new <- ncol(V_new)
    if (q_new < q) {
      warning(sprintf("Subspace rank reduced to %d during iteration %d.", q_new, iter))
      q <- q_new
      if (q == 0) stop("Subspace iteration collapsed to rank 0.")
      V_new <- V_new[, seq_len(q), drop = FALSE]
      if (!is.null(lambda_prev)) lambda_prev <- lambda_prev[seq_len(q)]
      if (!is.null(lambda_curr)) lambda_curr <- lambda_curr[seq_len(q)]
    }

    T_mat <- symm(crossprod(V_new, B %*% V_new))
    S_mat <- symm(crossprod(V_new, C %*% V_new))

    T_dense <- as.matrix(T_mat)
    R <- NULL
    for (attempt in 0:3) {
      bump <- reg_T * (10^attempt)
      T_try <- T_dense
      diag(T_try) <- diag(T_try) + bump
      R_try <- try(chol(T_try), silent = TRUE)
      if (!inherits(R_try, "try-error")) {
        R <- R_try
        break
      }
    }
    if (is.null(R)) {
      stop(sprintf("Cholesky factorization of projected metric failed at iter %d.", iter))
    }

    S_small <- as.matrix(S_mat)
    M <- backsolve(R, forwardsolve(t(R), S_small))
    M <- symm(M)

    ee <- eigen(M, symmetric = TRUE)
    mu <- ee$values
    W <- backsolve(R, ee$vectors)

    if (which == "largest") {
      lambda <- mu
      ord <- order(lambda, decreasing = TRUE)
    } else {
      eps <- 1e-300
      lambda <- 1 / pmax(abs(mu), eps) * sign(mu)
      ord <- order(lambda, decreasing = FALSE)
    }
    lambda <- lambda[ord]
    W <- W[, ord, drop = FALSE]

    V <- b_orthonormalize(V_new %*% W, B)
    if (ncol(V) < length(lambda)) {
      lambda <- lambda[seq_len(ncol(V))]
    }
    q <- ncol(V)

    if (!is.null(lambda_prev)) {
      common <- min(length(lambda_prev), length(lambda))
      if (common == 0) {
        rel_change <- Inf
      } else {
        prev_vals <- lambda_prev[seq_len(common)]
        curr_vals <- lambda[seq_len(common)]
        valid <- abs(prev_vals) > 1e-12
        if (any(valid)) {
          rel_change <- max(abs(curr_vals[valid] - prev_vals[valid]) /
                              pmax(abs(prev_vals[valid]), 1e-12))
        } else {
          rel_change <- Inf
        }
      }
    } else {
      rel_change <- Inf
    }

    lambda_prev <- lambda
    lambda_curr <- lambda

    max_resid <- residual_stats(lambda_curr, V)$max_resid

    if ((iter > 1 && rel_change < tol) || max_resid < tol) {
      converged <- TRUE
      break
    }
  }

  lambda_final <- lambda_curr
  V_final <- V

  rr_try <- try(rayleigh_ritz(V, S1, S2, which), silent = TRUE)
  if (!inherits(rr_try, "try-error")) {
    lambda_final <- rr_try$values
    V_final <- rr_try$vectors
  }

  stats_final <- residual_stats(lambda_final, V_final)
  max_resid <- stats_final$max_resid
  converged <- converged || max_resid < tol
  if (!length(lambda_final)) converged <- FALSE

  if (!converged) {
    warning(sprintf(
      "Subspace iteration did not converge within %d iterations (tol=%.1e, last rel_change=%.2e, max_resid=%.2e).",
      max_iter, tol, rel_change, max_resid
    ))
  }

  list(
    values = lambda_final,
    vectors = V_final[, seq_len(ncol(V_final)), drop = FALSE],
    iterations = iter,
    converged = converged,
    residual = max_resid
  )
}
