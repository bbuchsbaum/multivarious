# Generalized Eigenvalue Decomposition

Computes the generalized eigenvalues and eigenvectors for the problem: A
x = lambda B x. Supports multiple dense and iterative solvers with a
unified eigenpair selection interface.

## Usage

``` r
geneig(
  A = NULL,
  B = NULL,
  ncomp = 2,
  preproc = NULL,
  method = c("robust", "sdiag", "geigen", "primme", "rspectra", "subspace"),
  which = "LA",
  ...
)
```

## Arguments

- A:

  The left-hand side square matrix.

- B:

  The right-hand side square matrix, same dimension as A.

- ncomp:

  Number of eigenpairs to return.

- preproc:

  A preprocessing function to apply to the matrices before solving the
  generalized eigenvalue problem.

- method:

  One of:

  - "robust": Uses a stable decomposition via a whitening transform (B
    must be symmetric PD).

  - "sdiag": Uses a spectral decomposition of B (must be symmetric PD).
    Requires A to be symmetric for meaningful results.

  - "geigen": Uses the geigen package for a general solution (A and B
    can be non-symmetric).

  - "primme": Uses the PRIMME package for large/sparse symmetric
    problems (A and B must be symmetric).

  - "rspectra": Uses RSpectra; if B is SPD it calls
    `eigs_sym(A, B, ...)` directly, otherwise it applies a reciprocal
    transform to support all targets.

  - "subspace": Block subspace iteration for symmetric pairs with SPD B
    (iterative, no external package required).

- which:

  Which eigenpairs to return. One of `"LA"` (largest algebraic), `"SA"`
  (smallest algebraic), `"LM"` (largest magnitude), or `"SM"` (smallest
  magnitude). Aliases: `"top"`/`"largest"` -\> `"LA"`,
  `"bottom"`/`"smallest"` -\> `"SA"`. Dense backends select eigenpairs
  post hoc; `"primme"` supports `"LA"`, `"SA"`, `"SM"` (not `"LM"`);
  `"rspectra"` honors all four options. Default is `"LA"`.

- ...:

  Additional arguments to pass to the underlying solver.

## Value

A `projector` object with generalized eigenvectors and eigenvalues.

## References

Golub, G. H. & Van Loan, C. F. (2013) *Matrix Computations*, 4th ed.,
Section 8.7 – textbook derivation for the "robust" (Cholesky) and
"sdiag" (spectral) transforms.

Moler, C. & Stewart, G. (1973) "An Algorithm for Generalized Matrix
Eigenvalue Problems". *SIAM J. Numer. Anal.*, 10 (2): 241-256 – the QZ
algorithm behind the `geigen` backend.

Stathopoulos, A. & McCombs, J. R. (2010) "PRIMME: PReconditioned
Iterative Multi-Method Eigensolver". *ACM TOMS* 37 (2): 21:1-21:30 – the
algorithmic core of the `primme` backend.

See also the geigen (CRAN) and PRIMME documentation.

## See also

[`projector`](https://bbuchsbaum.github.io/multivarious/reference/projector.md)
for the base class structure.

## Examples

``` r
# \donttest{
# Simulate two matrices
set.seed(123)
A <- matrix(rnorm(50 * 50), 50, 50)
B <- matrix(rnorm(50 * 50), 50, 50)
A <- A %*% t(A) # Make A symmetric
B <- B %*% t(B) + diag(50) * 0.1 # Make B symmetric positive definite

# Solve generalized eigenvalue problem
result <- geneig(A = A, B = B, ncomp = 3)
# }
```
