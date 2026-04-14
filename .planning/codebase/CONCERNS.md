# Codebase Concerns

**Analysis Date:** 2026-01-20

## Tech Debt

**Deprecated API in preprocessing:**
- Issue: Legacy `prep()`, `init_transform()`, `apply_transform()`, `reverse_transform()` functions still exist alongside new `fit()`, `fit_transform()`, `transform()`, `inverse_transform()` API
- Files: `R/pre_process.R` (lines 32-165)
- Impact: Confusing dual-API, maintenance burden, warnings in user code
- Fix approach: Remove deprecated functions after deprecation period; update all internal usages to new API

**Old preprocessing API usage in svd_wrapper:**
- Issue: Comment indicates old API removal planned for v1.0 but dual-path code remains
- Files: `R/svd.R` (line 39)
- Impact: Technical debt marker, potential inconsistency
- Fix approach: Complete migration to new preprocessing API when bumping to v1.0

**Deprecated classifier arguments:**
- Issue: `global_scores` parameter in `classifier.multiblock_biprojector` and `classifier.bi_projector` is deprecated but still present
- Files: `R/classifier.R` (lines 27, 82, 254)
- Impact: User confusion, warnings in output
- Fix approach: Remove parameter in next major version

**Deprecated normalize_probs argument:**
- Issue: `normalize_probs` parameter in predict.classifier marked as deprecated
- Files: `R/classifier.R` (lines 619, 640, 654)
- Impact: API clutter
- Fix approach: Remove in next major version

**Deprecated perm_ci.pca:**
- Issue: Function still exists, immediately calls `.Deprecated()` and forwards to `perm_test.pca`
- Files: `R/pca.R` (lines 98-106)
- Impact: Dead code, maintenance burden
- Fix approach: Remove function entirely in next major version

## Known Bugs

**Bug in pca() row.names assignment:**
- Symptoms: Reference to undefined variable `scores` instead of `svdres$s`
- Files: `R/pca.R` (lines 26-28)
- Trigger: Running `pca()` - the if-block references `row.names(scores)` where `scores` is not defined
- Workaround: Code likely never executes due to conditional checking undefined var

**Potential issue with TODO generic stub:**
- Symptoms: Incomplete generic function definition
- Files: `R/all_generic.R` (line 230)
- Trigger: Bare `## TODO` comment with no implementation for `partial_residuals`, `partial_reconstruct`
- Workaround: None needed currently - just incomplete feature placeholder

## Security Considerations

**No secrets/credentials detected:**
- Risk: None identified
- Files: N/A
- Current mitigation: Package is a computational library with no network/auth operations
- Recommendations: Maintain this approach

## Performance Bottlenecks

**Double sweep() calls in standardize preprocessing:**
- Problem: Two separate `sweep()` calls for centering and scaling operations
- Files: `R/pre_process.R` (lines 452-497)
- Cause: Operations split for clarity, but each sweep creates intermediate matrix
- Improvement path: Combine into single vectorized operation for large matrices; TODOs already annotated

**Blockwise preprocessing recalculates mappings:**
- Problem: `concat_pre_processors` helper recalculates column mappings on each call
- Files: `R/pre_process.R` (line 548)
- Cause: No caching of computed mappings
- Improvement path: Pre-compute and cache mapping structure if `colind` patterns repeat

**Full design matrix storage in regress:**
- Problem: Stores full design matrix `X_fit` which can be memory intensive
- Files: `R/regress.R` (line 78)
- Cause: Design decision to keep reference data
- Improvement path: Consider lazy loading or optional storage for very large X

**Large file complexity:**
- Problem: Several files exceed 500 lines, increasing maintenance difficulty
- Files:
  - `R/classifier.R` (1042 lines)
  - `R/pca.R` (894 lines)
  - `R/all_generic.R` (837 lines)
  - `R/pre_process.R` (756 lines)
  - `R/geneig.R` (743 lines)
- Cause: Feature accumulation over time
- Improvement path: Consider splitting by functionality (e.g., separate classifier types into own files)

## Fragile Areas

**Caching mechanism:**
- Files: `R/projector.R`, `R/bi_projector.R`, `R/twoway_projector.R`, `R/composed_projector.R`
- Why fragile: Cache invalidation not automatic when projector state changes; relies on `.cache` environment attribute
- Safe modification: Always call cache-aware methods; clear cache explicitly when modifying projector internals
- Test coverage: Limited explicit cache testing

**Preprocessing pipeline environments:**
- Files: `R/pre_process.R` (lines 269, 341, 427)
- Why fragile: Each step creates its own `rlang::new_environment()` to store state; complex environment hierarchy
- Safe modification: Use provided accessors; avoid direct environment manipulation
- Test coverage: Good coverage in `tests/testthat/test_preprocess.R` (19 tests)

**Generalized eigenvalue decomposition fallbacks:**
- Files: `R/geneig.R` (lines 142-450)
- Why fragile: Multiple try-catch blocks for different decomposition strategies; complex fallback logic between methods
- Safe modification: Test with edge cases (singular matrices, near-singular B)
- Test coverage: Tests exist but may not cover all fallback paths

**SVD method dispatch:**
- Files: `R/svd.R` (lines 60-88)
- Why fragile: Six different SVD backends with different requirements; wrapped in tryCatch
- Safe modification: Ensure all methods tested with edge cases (small k, near-singular matrices)
- Test coverage: Only 2 tests in `test_svd.R`; minimal coverage

## Scaling Limits

**Kernel centering in Nystrom approximation:**
- Current capacity: Works with uncentered kernels only
- Limit: True kernel centering noted as "not fully implemented" (line 23)
- Scaling path: Would require O(N^2) computation for proper kernel centering

**Dense matrix operations:**
- Current capacity: Works well for moderate-sized matrices
- Limit: Several core operations form full covariance matrices (p x p)
- Scaling path: Sample-space strategies implemented for cPCA++ provide template for other methods

## Dependencies at Risk

**Heavy dependency list (43 imports):**
- Risk: Many required packages increase installation complexity and failure points
- Impact: Users may struggle with installation; CRAN checks may fail if any dependency is archived
- Migration plan: Consider moving less critical packages (ggplot2, ggrepel, dplyr, tidyr) to Suggests

**PRIMME package:**
- Risk: Less commonly installed; may have platform-specific build issues
- Impact: `geneig()` method="primme" unavailable if not installed
- Migration plan: Already handled gracefully with fallbacks; consider moving to Suggests

**geigen package:**
- Risk: Core dependency for GEVD operations
- Impact: Required for cPCA++, geneig, and related methods
- Migration plan: None needed - actively maintained

## Missing Critical Features

**Kernel centering for Nystrom:**
- Problem: Documented as not implemented for non-linear kernels
- Blocks: Proper Kernel PCA equivalence with Nystrom approximation
- Files: `R/nystrom_embedding.R` (lines 22-25)

**Partial residuals/reconstruct:**
- Problem: Generic stubs exist but no implementations
- Blocks: Partial variable analysis workflows
- Files: `R/all_generic.R` (lines 230-232)

## Test Coverage Gaps

**SVD wrapper minimally tested:**
- What's not tested: Only 2 tests for 6 different SVD methods
- Files: `R/svd.R`, `tests/testthat/test_svd.R`
- Risk: Method-specific edge cases may fail silently
- Priority: Medium - core functionality

**Caching behavior not explicitly tested:**
- What's not tested: Cache invalidation, cache hit/miss scenarios
- Files: `R/projector.R`, `R/bi_projector.R` (cache logic)
- Risk: Stale cache could return incorrect results
- Priority: Medium - affects correctness

**Cross-validation edge cases:**
- What's not tested: Small fold sizes, single observation folds
- Files: `R/cv.R` (514 lines), `tests/testthat/test_cv.R` (3 tests)
- Risk: Failures with unusual CV configurations
- Priority: Low - users typically use standard folds

**Composed projector partial projection:**
- What's not tested: Multi-stage partial projections beyond first stage
- Files: `R/composed_projector.R` (warning on line 81)
- Risk: Warning issued but behavior may be unexpected
- Priority: Low - advanced use case

**Export-to-test ratio:**
- What's not tested: ~199 exported functions vs ~97 test_that blocks
- Files: All R/ files
- Risk: Approximately half of exported API has minimal or no direct testing
- Priority: Medium - overall coverage improvement needed

## DESCRIPTION File Issue

**Missing Author/Maintainer fields:**
- Issue: R CMD check fails with "Required fields missing or empty: 'Author' 'Maintainer'"
- Files: `DESCRIPTION`
- Impact: Cannot pass R CMD check; CRAN submission blocked
- Fix approach: Fields are specified via Authors@R which should auto-generate; may need roxygen2 rebuild or explicit Author/Maintainer fields

---

*Concerns audit: 2026-01-20*
