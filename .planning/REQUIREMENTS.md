# Requirements

## REQ-001: Fix T/F shorthand usage
**Priority:** P0 - Blocking
**Source:** CRAN Policy / Research PITFALLS.md
**Description:** Replace all `T` and `F` with `TRUE` and `FALSE` in R code. CRAN rejects packages using shorthand.
**Files:** R/pca.R (lines 52, 53, 75)
**Acceptance:** `grep -r "= T\|= F\|,T\|,F" R/` returns no matches

## REQ-002: Fix \dontrun{} misuse in examples
**Priority:** P0 - Blocking
**Source:** CRAN Policy / Research PITFALLS.md
**Description:** Convert `\dontrun{}` to `\donttest{}` for executable but slow examples. CRAN requires \dontrun{} only for truly non-executable code.
**Files:** R/cPCA.R, R/pca.R, R/all_generic.R
**Acceptance:** All \dontrun{} blocks reviewed; slow-but-executable examples use \donttest{}

## REQ-003: R CMD check passes with zero errors
**Priority:** P0 - Blocking
**Source:** CRAN Policy
**Description:** Package must pass `R CMD check` with zero ERRORs.
**Acceptance:** `devtools::check()` reports 0 errors

## REQ-004: R CMD check passes with zero warnings
**Priority:** P0 - Blocking
**Source:** CRAN Policy
**Description:** Package must pass `R CMD check` with zero WARNINGs.
**Acceptance:** `devtools::check()` reports 0 warnings

## REQ-005: Create NEWS.md
**Priority:** P1 - Required
**Source:** CRAN best practice / Research FEATURES.md
**Description:** Create NEWS.md documenting changes since last CRAN version (0.2.0). Include new features, bug fixes, deprecated functions.
**Acceptance:** NEWS.md exists and documents v0.3.0 changes

## REQ-006: Update cran-comments.md
**Priority:** P1 - Required
**Source:** CRAN submission process
**Description:** Update cran-comments.md with current R CMD check results, explain the 29 Imports NOTE, and mark as resubmission (not new release).
**Acceptance:** cran-comments.md reflects actual check results and explains NOTEs

## REQ-007: All tests pass
**Priority:** P0 - Blocking
**Source:** CRAN Policy
**Description:** All testthat tests must pass.
**Acceptance:** `devtools::test()` reports 0 failures

## REQ-008: Documentation complete
**Priority:** P1 - Required
**Source:** CRAN Policy
**Description:** All exported functions have @param, @return, and @examples tags. No missing \value sections in .Rd files.
**Acceptance:** R CMD check reports no documentation issues

## REQ-009: Cross-platform verification
**Priority:** P1 - Required
**Source:** CRAN Policy / Research ARCHITECTURE.md
**Description:** Verify package builds on Windows, macOS, and Linux. Use win-builder and mac-builder services.
**Acceptance:** Check results from win-builder and mac-builder show no new issues

## REQ-010: R-devel compatibility
**Priority:** P1 - Required
**Source:** CRAN Policy
**Description:** Package must work with R-devel (development version of R).
**Acceptance:** devtools::check_win_devel() passes

## REQ-011: Bump version appropriately
**Priority:** P2 - Should Have
**Source:** R package conventions
**Description:** Increment version number for resubmission (0.2.0 -> 0.3.0 or similar).
**Acceptance:** DESCRIPTION Version field updated; NEWS.md matches

---

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| REQ-001 | Phase 1 | Pending |
| REQ-002 | Phase 1 | Pending |
| REQ-003 | Phase 1 | Pending |
| REQ-004 | Phase 1 | Pending |
| REQ-005 | Phase 2 | Pending |
| REQ-006 | Phase 2 | Pending |
| REQ-007 | Phase 1 | Pending |
| REQ-008 | Phase 2 | Pending |
| REQ-009 | Phase 3 | Pending |
| REQ-010 | Phase 3 | Pending |
| REQ-011 | Phase 2 | Pending |

---
*Requirements defined: 2026-01-20*
*Traceability updated: 2026-01-20*
