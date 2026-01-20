# multivarious CRAN Resubmission

## What This Is

Preparing the multivarious R package (v0.2.0) for CRAN resubmission. The package provides extensible data structures for multivariate analysis — projectors, preprocessing pipelines, dimensionality reduction. Previously accepted on CRAN, now needs to pass checks after many updates.

## Core Value

Pass R CMD check with no errors and minimal warnings/notes, ready for CRAN submission.

## Requirements

### Validated

- ✓ S3 projector class hierarchy (projector, bi_projector, cross_projector, composed_projector) — existing
- ✓ Preprocessing pipeline (center, colscale, standardize, pass) — existing
- ✓ PCA, SVD, PLSC implementations — existing
- ✓ Multiblock projector support — existing
- ✓ Classification wrappers — existing
- ✓ Cross-validation and bootstrap utilities — existing
- ✓ 14 vignettes documenting usage — existing
- ✓ testthat test suite — existing

### Active

- [ ] R CMD check passes with no errors
- [ ] R CMD check passes with no warnings
- [ ] R CMD check --as-cran passes
- [ ] All examples run without error
- [ ] All tests pass
- [ ] Documentation complete (no missing \value, \arguments, etc.)
- [ ] NAMESPACE exports correct
- [ ] Version bumped appropriately

### Out of Scope

- New features — this is a maintenance release for CRAN compliance
- Removing deprecated functions — keep backward compatibility for now
- Major refactoring — fix what's broken, don't restructure
- Increasing test coverage beyond what's needed for CRAN

## Context

**Known concerns from codebase mapping:**
- Deprecated APIs still present (preprocessing, classifier arguments) — acceptable if properly marked
- Some test coverage gaps — acceptable if existing tests pass
- Heavy dependency list (29 Imports) — may trigger CRAN note but not blocking
- PRIMME package in Imports — less common, may need conditional handling

**Previous CRAN status:** Accepted (version unknown, likely 0.1.x)

**Current version:** 0.2.0

## Constraints

- **Compatibility**: Must maintain backward compatibility with existing user code
- **Dependencies**: All Imports must be available on CRAN
- **Platform**: Must pass on Linux, macOS, Windows (CRAN check matrix)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Fix only what blocks CRAN | Minimize scope, ship faster | — Pending |
| Keep deprecated functions | Backward compatibility | — Pending |

---
*Last updated: 2026-01-20 after initialization*
