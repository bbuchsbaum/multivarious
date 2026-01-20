# Phase 1: Code Fixes - Research

**Researched:** 2026-01-20
**Domain:** R Package Development / CRAN Compliance
**Confidence:** HIGH

## Summary

Research identified all code issues preventing R CMD check from passing with zero errors and warnings. The package has multiple categories of issues:

1. **T/F Shorthand**: 3 active uses of `F` instead of `FALSE` in `R/pca.R` (lines 52, 53, 75)
2. **\dontrun{} Misuse**: 3 locations using `\dontrun{}` for examples that should run or use `\donttest{}`
3. **Example Errors**: Examples fail because data frames are passed instead of matrices
4. **S3 Method Issues**: Inconsistent generic/method signatures and unregistered methods
5. **Missing Imports**: `coefficients` and `combn` functions not imported
6. **Non-ASCII Characters**: Unicode characters in code files that must be escaped
7. **Documentation Gaps**: Undocumented arguments in Rd files

**Primary recommendation:** Fix issues in order of severity - example errors first (they block all other checks), then T/F shorthand, imports, S3 consistency, and documentation.

## Standard Stack

### Core Tools
| Tool | Command | Purpose | When to Use |
|------|---------|---------|-------------|
| devtools | `devtools::check()` | Full package check | Development iteration |
| R CMD check | `R CMD check --as-cran pkg.tar.gz` | CRAN-style check | Final verification |
| devtools::document() | `devtools::document()` | Generate documentation | After roxygen changes |
| testthat | `devtools::test()` | Run unit tests | After code changes |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `_R_CHECK_FORCE_SUGGESTS_=false` | Skip missing suggested packages | When randomForest unavailable |
| `tools::showNonASCIIfile()` | Find non-ASCII characters | Debugging encoding issues |

## Architecture Patterns

### CRAN Compliance Rules

**T/F vs TRUE/FALSE:**
- CRAN requires explicit `TRUE`/`FALSE`, never `T`/`F` abbreviations
- Applies to all R code, including internal functions
- Comments can use T/F but active code cannot

**Example Documentation:**
```r
# Use \donttest{} for examples that:
# - Take a long time to run
# - Require optional packages
# - Require specific system state

# Use \dontrun{} ONLY for examples that:
# - Would produce errors intentionally
# - Would write to user's file system
# - Would modify global state inappropriately

# Regular examples should run without wrappers when possible
```

**S3 Method Registration:**
```r
# All S3 methods must be registered in NAMESPACE
# Methods should match generic signature exactly

# Generic:
bootstrap <- function(x, nboot, ...) UseMethod("bootstrap")

# Method MUST match (x, nboot, ...) signature:
bootstrap.plsc <- function(x, nboot, ...) {
  # Additional args extracted from ...
}
```

### Project Structure Reference
```
R/
├── all_generic.R      # S3 generics - check signatures
├── pca.R              # T/F issues (lines 52, 53, 75)
├── cPCA.R             # \dontrun{} issue (line 118)
├── geneig.R           # Non-ASCII characters, \donttest{} example
├── multiblock.R       # Non-ASCII characters
├── bootstrap.R        # Generic definition
├── plsc_inference.R   # Method with wrong signature
├── svd.R              # Example with data frame issue
├── classifier.R       # Unregistered S3 methods
└── projector.R        # Unregistered S3 methods
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Matrix coercion | Manual checks | `as.matrix()` in examples | Standard R idiom |
| S3 registration | Manual NAMESPACE edits | roxygen2 `@export` | Automatic, consistent |
| Non-ASCII replacement | Manual find/replace | `\uXXXX` escapes | Portable encoding |

## Common Pitfalls

### Pitfall 1: Data Frame vs Matrix in Examples
**What goes wrong:** Examples use `iris[,1:4]` which returns a data.frame, but functions expect matrices
**Why it happens:** Convenient shorthand that works in interactive use but fails in strict checks
**How to avoid:** Always use `as.matrix(iris[,1:4])` in examples
**Warning signs:** Error "must be a matrix" in R CMD check example runs

### Pitfall 2: S3 Method Signature Mismatch
**What goes wrong:** Method has different signature than generic
**Why it happens:** Generic evolved but methods not updated
**How to avoid:** Method signature must match generic exactly, use `...` for extra args
**Warning signs:** "S3 generic/method consistency" WARNING in R CMD check

### Pitfall 3: Unregistered S3 Methods
**What goes wrong:** Methods exist but not in NAMESPACE
**Why it happens:** Missing `@export` or wrong function naming
**How to avoid:** All `foo.class` functions need `@export` if `foo` is exported generic
**Warning signs:** "Apparent methods for exported generics not registered"

### Pitfall 4: Non-ASCII in R Code
**What goes wrong:** Unicode characters in roxygen comments or code
**Why it happens:** Copy/paste from documents with special characters (em-dash, lambda, etc.)
**How to avoid:** Use ASCII or `\uXXXX` escapes
**Warning signs:** "Non-ASCII characters" WARNING, file listed

### Pitfall 5: Missing Imports
**What goes wrong:** Functions used without importing from package
**Why it happens:** Functions work because of implicit loading
**How to avoid:** Add `@importFrom pkg func` or `pkg::func` calls
**Warning signs:** "Undefined global functions or variables" NOTE

## Exact Issues Found

### T/F Shorthand (3 locations, all in pca.R)

| File | Line | Current | Fix |
|------|------|---------|-----|
| R/pca.R | 52 | `drop = F` | `drop = FALSE` |
| R/pca.R | 53 | `drop = F` | `drop = FALSE` |
| R/pca.R | 75 | `drop = F` | `drop = FALSE` |

Line 58 is commented out - no fix needed.

### \dontrun{} Misuse (3 locations)

| File | Lines | Context | Recommended Fix |
|------|-------|---------|-----------------|
| R/pca.R | 639-646 | biplot example with ggrepel | Convert to `\donttest{}` - runs but slow |
| R/cPCA.R | 118-129 | Plot example | Convert to `\donttest{}` - graphics example |
| R/all_generic.R | 696-734 | perm_test examples using MASS | Convert to `\donttest{}` or make conditional |

### Example Errors (2 locations, same pattern)

| File | Line | Issue | Fix |
|------|------|-------|-----|
| R/svd.R | 25-28 | `X <- iris[, 1:4]` returns data.frame | `X <- as.matrix(iris[, 1:4])` |
| R/all_generic.R | 407-409 | `X <- iris[, 1:4]` returns data.frame | `X <- as.matrix(iris[, 1:4])` |

### S3 Method Issues

**Signature Mismatch:**
| Generic | Method | Issue |
|---------|--------|-------|
| `bootstrap(x, nboot, ...)` | `bootstrap.plsc(x, X, Y, nboot, ...)` | Method adds X, Y before nboot |

**Unregistered Methods:**
| Method | File | Line |
|--------|------|------|
| `classifier.projector` | R/classifier.R | 251 |
| `inverse_projection.projector` | R/projector.R | 92 |
| `perm_ci.pca` | R/pca.R | 98 |

### Missing Imports

| Function | From | Add to |
|----------|------|--------|
| `coefficients` | stats | NAMESPACE via roxygen |
| `combn` | utils | NAMESPACE via roxygen |

### Non-ASCII Characters

| File | Lines | Characters |
|------|-------|------------|
| R/geneig.R | 40, 56, 65, 69, 73, 336, 477, 484-486 | Greek letters (lambda, mu), em-dashes, arrows |
| R/multiblock.R | 207, 211, 214, 227, 230-231, 267, 297, 306, 337, 384, 408, 411, 483, 492 | Em-dashes, multiplication signs |

### Documentation Issues

| File | Issue |
|------|-------|
| bootstrap_plsc.Rd | Undocumented `...` argument |
| perm_test.plsc.Rd | Undocumented: x, nperm, stepwise, parallel, alternative, ... |

### Other NOTEs

| Issue | Location | Fix |
|-------|----------|-----|
| Unused import | tidyr in DESCRIPTION | Remove from Imports or use |
| Vignette engine issues | vignettes/*.Rmd | Check VignetteBuilder field |

## Code Examples

### Fix T/F Shorthand
```r
# Before (R/pca.R line 52-53):
scores[, (i + 1):ncomp, drop = F],
loadings[, (i + 1):ncomp, drop = F]

# After:
scores[, (i + 1):ncomp, drop = FALSE],
loadings[, (i + 1):ncomp, drop = FALSE]
```

### Fix Example Data Frame Issue
```r
# Before (R/svd.R and R/all_generic.R):
#' X <- iris[, 1:4]

# After:
#' X <- as.matrix(iris[, 1:4])
```

### Fix S3 Method Signature
```r
# Before (R/plsc_inference.R line 288):
bootstrap.plsc <- function(x, X, Y, nboot = 500, ...) {

# After - match generic signature:
bootstrap.plsc <- function(x, nboot = 500, ...) {
  # Extract X, Y from ... or require them via check
  args <- list(...)
  X <- args$X
  Y <- args$Y
  # ... rest of function
}
```

### Add Missing Imports (roxygen)
```r
#' @importFrom stats coefficients
#' @importFrom utils combn
```

### Fix Non-ASCII Characters
```r
# Before:
#' Computes... A x = λ B x.

# After:
#' Computes... A x = \\u03BB B x.

# Or use ASCII description:
#' Computes... A x = lambda * B x.
```

### Register S3 Methods
```r
# Add @export to method definitions
#' @export
classifier.projector <- function(...) { }
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `T`/`F` shortcuts | `TRUE`/`FALSE` explicit | Always required for CRAN | Mandatory |
| `\dontrun{}` for slow examples | `\donttest{}` | R 3.0+ | Less strict checking |
| Manual NAMESPACE | roxygen2 `@export` | roxygen2 adoption | Automatic registration |

## Open Questions

1. **bootstrap.plsc signature change** - Changing the method signature may break existing user code. Consider whether to:
   - Change signature (breaking change)
   - Document the inconsistency as intentional
   - Use a different generic name

2. **Non-ASCII in comments** - The Greek letters in geneig.R are in roxygen documentation. Options:
   - Use `\u03BB` escapes
   - Use ASCII descriptions ("lambda")
   - Keep in comments only (may still trigger warning)

3. **Vignette engine issues** - Some vignettes have encoding or engine detection problems. May need separate investigation.

## Task Execution Order

Execute fixes in this order to avoid cascading failures:

1. **Example data frame fixes** (blocks all example testing)
2. **T/F shorthand fixes** (simple, no dependencies)
3. **Missing imports** (enables clean check of other code)
4. **S3 method registration** (adds exports)
5. **\dontrun to \donttest conversion** (documentation)
6. **S3 signature consistency** (may need design decision)
7. **Non-ASCII character fixes** (tedious but straightforward)
8. **Documentation gaps** (roxygen updates)

## Sources

### Primary (HIGH confidence)
- R CMD check output on local machine (direct verification)
- Writing R Extensions manual (official R documentation)
- CRAN repository policy (https://cran.r-project.org/web/packages/policies.html)

### Secondary (MEDIUM confidence)
- tools::showNonASCIIfile() output (direct verification)
- devtools::document() output (direct verification)

## Metadata

**Confidence breakdown:**
- T/F locations: HIGH - direct grep verification
- \dontrun{} locations: HIGH - direct grep verification
- Example errors: HIGH - R CMD check output
- S3 issues: HIGH - R CMD check output
- Non-ASCII: HIGH - tools::showNonASCIIfile() output

**Research date:** 2026-01-20
**Valid until:** 2026-02-20 (30 days - stable R package development practices)
