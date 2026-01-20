# Roadmap: multivarious CRAN Resubmission

## Overview

Prepare the multivarious R package (v0.2.0 -> v0.3.0) for CRAN resubmission. The package previously passed CRAN checks but requires targeted fixes for blocking issues (T/F shorthand, \dontrun{} misuse) and documentation updates (NEWS.md, cran-comments.md) before submission.

**Phases:** 4
**Total Requirements:** 11
**Depth:** Standard

---

## Phase 1: Code Fixes

**Goal:** Package passes R CMD check with zero errors and warnings on local machine.

**Dependencies:** None (starting phase)

**Plans:** 5 plans

Plans:
- [ ] 01-01-PLAN.md - Fix example data frame issues and non-ASCII characters
- [ ] 01-02-PLAN.md - Fix S3 method consistency and registration
- [ ] 01-03-PLAN.md - Fix missing imports and remove unused tidyr
- [ ] 01-04-PLAN.md - Convert \dontrun{} to \donttest{} and document arguments
- [ ] 01-05-PLAN.md - Final verification with R CMD check and tests

**Requirements:**
- REQ-001: Fix T/F shorthand usage
- REQ-002: Fix \dontrun{} misuse in examples
- REQ-003: R CMD check passes with zero errors
- REQ-004: R CMD check passes with zero warnings
- REQ-007: All tests pass

**Success Criteria:**
1. `grep -r "= T\|= F\|,T\|,F" R/` returns no matches for T/F shorthand
2. All \dontrun{} blocks in examples are either truly non-executable OR converted to \donttest{}
3. `devtools::check()` reports 0 errors
4. `devtools::check()` reports 0 warnings
5. `devtools::test()` reports 0 test failures

**Deliverables:**
- Fixed R/pca.R (lines 52, 53, 75)
- Reviewed/fixed R/cPCA.R, R/all_generic.R examples
- Clean R CMD check output

---

## Phase 2: Documentation

**Goal:** All required CRAN documentation exists and version is bumped for submission.

**Dependencies:** Phase 1 (need clean R CMD check results for cran-comments.md)

**Requirements:**
- REQ-005: Create NEWS.md
- REQ-006: Update cran-comments.md
- REQ-008: Documentation complete
- REQ-011: Bump version appropriately

**Success Criteria:**
1. NEWS.md exists in package root and documents v0.3.0 changes including deprecated functions
2. cran-comments.md reflects actual R CMD check results and explains the 29 Imports NOTE
3. R CMD check reports no documentation warnings (missing \value, undocumented arguments)
4. DESCRIPTION Version field shows 0.3.0

**Deliverables:**
- NEWS.md
- Updated cran-comments.md
- Updated DESCRIPTION (version bump)
- Any documentation fixes identified by R CMD check

---

## Phase 3: Cross-Platform Verification

**Goal:** Package verified to work on Windows, macOS, Linux, and R-devel.

**Dependencies:** Phase 2 (submit final version for external checks)

**Requirements:**
- REQ-009: Cross-platform verification
- REQ-010: R-devel compatibility

**Success Criteria:**
1. win-builder (release and devel) returns results with no new errors or warnings
2. mac-builder returns results with no new errors or warnings
3. R-devel check passes (via win-builder devel or rhub)
4. Any platform-specific issues identified are resolved

**Deliverables:**
- win-builder check confirmation
- mac-builder check confirmation
- Any platform-specific fixes applied
- Updated cran-comments.md with cross-platform results

---

## Phase 4: Submission

**Goal:** Package submitted to CRAN and accepted.

**Dependencies:** Phase 3 (all verification complete)

**Requirements:**
- All previous requirements verified complete

**Success Criteria:**
1. `devtools::release()` completes without error
2. CRAN incoming queue shows package received
3. CRAN acceptance email received (or issues identified for next iteration)

**Deliverables:**
- Submitted package to CRAN
- Confirmation of submission receipt

---

## Progress

| Phase | Status | Requirements | Completed |
|-------|--------|--------------|-----------|
| 1 - Code Fixes | Planned | 5 | 0/5 |
| 2 - Documentation | Not Started | 4 | 0/4 |
| 3 - Cross-Platform | Not Started | 2 | 0/2 |
| 4 - Submission | Not Started | 0 | - |

**Overall:** 0/11 requirements complete

---

## Requirement Traceability

| Requirement | Description | Phase | Status |
|-------------|-------------|-------|--------|
| REQ-001 | Fix T/F shorthand usage | 1 | Pending |
| REQ-002 | Fix \dontrun{} misuse | 1 | Pending |
| REQ-003 | Zero errors in R CMD check | 1 | Pending |
| REQ-004 | Zero warnings in R CMD check | 1 | Pending |
| REQ-005 | Create NEWS.md | 2 | Pending |
| REQ-006 | Update cran-comments.md | 2 | Pending |
| REQ-007 | All tests pass | 1 | Pending |
| REQ-008 | Documentation complete | 2 | Pending |
| REQ-009 | Cross-platform verification | 3 | Pending |
| REQ-010 | R-devel compatibility | 3 | Pending |
| REQ-011 | Bump version | 2 | Pending |

---
*Roadmap created: 2026-01-20*
*Last updated: 2026-01-20*
