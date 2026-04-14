# Phase 4: Submission - Research

**Researched:** 2026-01-21
**Domain:** CRAN package submission process
**Confidence:** HIGH

## Summary

This research covers the CRAN submission process for the multivarious R package (v0.3.0). The package has already passed all cross-platform verification (0 errors, 0 warnings, 0 notes on macOS local, win-builder release/devel, and mac-builder).

The standard submission workflow uses `devtools::release()` or `devtools::submit_cran()`. Since this is a resubmission (not first-time), the package will enter the normal CRAN queue rather than the "newbies" queue. The process involves uploading the package tarball via CRAN's web form, confirming via email, and monitoring the incoming queue until acceptance.

**Primary recommendation:** Use `devtools::submit_cran()` directly (not `release()`) since all verification was already completed in Phase 3. Run a final fresh R CMD check immediately before submission to confirm the package still passes.

## Standard Stack

The established tools for CRAN submission:

### Core
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| devtools | 2.4+ | `submit_cran()` for uploading to CRAN | Official r-lib submission tool |
| pkgbuild | 1.4+ | Builds source tarball with `manual = TRUE` | Called by submit_cran() |

### Supporting
| Tool | Purpose | When to Use |
|------|---------|-------------|
| foghorn | `cran_incoming()` to monitor queue status | After submission to track progress |
| usethis | `use_github_release()` for post-acceptance | After CRAN acceptance |

### Alternatives Considered
| Standard | Alternative | Tradeoff |
|----------|-------------|----------|
| `submit_cran()` | `release()` | `release()` runs extra interactive checks; redundant since Phase 3 verified everything |
| `submit_cran()` | Manual web form | Manual upload via https://cran.r-project.org/submit.html; less convenient |

**No installation needed:** devtools is already installed in standard R development environment.

## Architecture Patterns

### Submission Workflow Sequence

```
Pre-submission checks
    |
    v
devtools::submit_cran()
    |
    v
Email confirmation (MUST click link)
    |
    v
CRAN automated checks (hours)
    |
    v
Queue monitoring (cran_incoming)
    |
    v
Acceptance/Rejection email
```

### Pattern 1: Direct Submission (Recommended)

**What:** Use `submit_cran()` when comprehensive verification is already done.
**When to use:** Package has been verified on multiple platforms with 0E/0W/0N.
**Example:**
```r
# Source: https://devtools.r-lib.org/reference/submit_cran.html
# After final fresh check passes
devtools::submit_cran()
```

### Pattern 2: Cautious Submission

**What:** Use `release()` for interactive checklist with extra verification.
**When to use:** When uncertain about package readiness or less verification done.
**Example:**
```r
# Source: https://r-pkgs.org/release.html
devtools::release()
# Answers prompts interactively, then calls submit_cran()
```

### Anti-Patterns to Avoid
- **Skipping email confirmation:** Package is NOT submitted until you click the link in CRAN's confirmation email
- **Submitting with uncommitted changes:** Git working directory should be clean
- **Trusting cached check results:** Always run fresh R CMD check immediately before submission
- **Responding defensively to rejections:** If rejected, fix issues calmly and resubmit

## Don't Hand-Roll

Problems that have existing solutions:

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Package upload | Manual HTTP POST | `devtools::submit_cran()` | Handles form encoding, metadata extraction |
| Queue monitoring | Web scraping | `foghorn::cran_incoming()` | Parses CRAN incoming directory properly |
| GitHub release | Manual tag/release | `usethis::use_github_release()` | Uses CRAN-SUBMISSION file automatically |

**Key insight:** The CRAN submission infrastructure has specific requirements. Tools handle the complexity of form submission, email confirmation tracking, and queue parsing.

## Common Pitfalls

### Pitfall 1: Forgetting Email Confirmation
**What goes wrong:** Package uploaded but not actually submitted because confirmation link not clicked.
**Why it happens:** Maintainers assume upload completes the process.
**How to avoid:** Immediately check email after `submit_cran()` completes; click confirmation link.
**Warning signs:** Package never appears in CRAN incoming queue after several hours.

### Pitfall 2: Dirty Git State at Submission
**What goes wrong:** Package submitted with uncommitted changes that weren't in the verification build.
**Why it happens:** Last-minute changes made after R CMD check but before submission.
**How to avoid:** Run `git status` before submission; commit or stash changes.
**Warning signs:** Modified files shown in `git status`.

### Pitfall 3: Stale Check Results
**What goes wrong:** Package fails CRAN's automated checks despite passing earlier.
**Why it happens:** R-devel or CRAN infrastructure changed since last local check.
**How to avoid:** Run fresh `R CMD check --as-cran` immediately before submission.
**Warning signs:** More than 24 hours since last check; R-devel version changed.

### Pitfall 4: Incomplete cran-comments.md
**What goes wrong:** Missing required sections like test environments or NOTE explanations.
**Why it happens:** Template not fully populated or outdated.
**How to avoid:** Review cran-comments.md matches current check output exactly.
**Warning signs:** cran-comments.md says 1 NOTE but check shows 0 NOTEs.

### Pitfall 5: Mismatched DESCRIPTION Metadata
**What goes wrong:** CRAN's automated system extracts wrong maintainer email or version.
**Why it happens:** DESCRIPTION not updated after changes.
**How to avoid:** Verify DESCRIPTION Version, Maintainer email, URLs are current.
**Warning signs:** Confirmation email goes to wrong address.

## Code Examples

Verified patterns from official sources:

### Pre-Submission Final Check
```r
# Source: https://cran.r-project.org/web/packages/submission_checklist.html
# Run immediately before submission
devtools::check(args = c("--as-cran"))
# Or from command line:
# R CMD check --as-cran multivarious_0.3.0.tar.gz
```

### Direct Submission
```r
# Source: https://devtools.r-lib.org/reference/submit_cran.html
devtools::submit_cran()
# This will:
# 1. Build package tarball with pkgbuild::build(manual = TRUE)
# 2. Upload to CRAN web form
# 3. Populate form from DESCRIPTION and cran-comments.md
# 4. Write CRAN-SUBMISSION file for later GitHub release
```

### Monitor Submission Queue
```r
# Source: https://fmichonneau.github.io/foghorn/reference/cran_incoming.html
foghorn::cran_incoming("multivarious")
# Returns: data frame with folder, package, version, time columns
# Or check web dashboard: https://cran.r-project.org/incoming/
```

### Post-Acceptance GitHub Release
```r
# Source: https://r-pkgs.org/release.html
# After receiving CRAN acceptance email:
usethis::use_github_release()
# Creates GitHub release using CRAN-SUBMISSION file info
# Pulls release notes from NEWS.md
```

## CRAN Queue Workflow

After submission, packages move through these folders:

| Folder | Meaning | Typical Duration |
|--------|---------|------------------|
| incoming | Just uploaded | Minutes |
| inspect | Awaiting manual inspection | Hours to days (first-time only) |
| pretest | Automated checks running | Hours |
| recheck | Reverse dependency checks | Hours (if has reverse deps) |
| pending | Needs closer CRAN team review | Days |
| waiting | CRAN waiting for your response | Until you respond |
| human (BA/KH/KL/UL/VW) | Assigned to specific reviewer | Days |
| publish | Approved, about to appear on CRAN | Hours |
| archive | Rejected | N/A |

**For this package:** No reverse dependencies exist, so recheck folder is N/A. This is a resubmission (not first-time), so less scrutiny expected.

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| `rhub::check_for_cran()` | Deprecated | 2024 | Use win-builder/mac-builder instead |
| Manual CRAN form | `devtools::submit_cran()` | 2018+ | Automated form population |
| CRAN-RELEASE file | CRAN-SUBMISSION file | devtools 2.x | Better machine-readable format |

**Current best practices (2025-2026):**
- Use `submit_cran()` for upload (not manual form)
- Use win-builder and mac-builder for cross-platform checks (not R-hub)
- Monitor queue via https://cran.r-project.org/incoming/ or foghorn package

## Specific Considerations for This Package

### Package State Assessment

| Item | Status | Notes |
|------|--------|-------|
| Version | 0.3.0 | Correct for resubmission |
| R CMD check | 0E/0W/0N | Verified on 4 platforms |
| cran-comments.md | Complete | Lists all test environments |
| NEWS.md | Complete | Documents all v0.3.0 changes |
| .Rbuildignore | Complete | Excludes planning/dev files |
| Reverse deps | None | No downstream breakage risk |
| Prior CRAN version | 0.2.0 | Was accepted before |

### Pre-Submission Checklist (from CONTEXT.md)

User-specified checks before submission:
1. Run fresh R CMD check immediately before submission
2. Verify cran-comments.md matches actual check output
3. Review NEWS.md for completeness
4. Verify DESCRIPTION fields (maintainer email, URLs)
5. Verify clean git state
6. Review .Rbuildignore for completeness

### Uncommitted Changes Observation

Current git status shows 49 files with uncommitted changes:
- `R/` source files (5 files)
- `docs/` pkgdown site (many files)
- `tests/` test files (5 files)
- `vignettes/` (7 files)

**Recommendation:** These should be evaluated:
- If they are intended changes: commit before submission
- If they are unintended/generated: investigate or stash

## Open Questions

1. **Git uncommitted changes**
   - What we know: 49 files show as modified
   - What's unclear: Whether these are intentional (need commit) or artifacts (need stash/revert)
   - Recommendation: Planner should verify git state before submission

## Sources

### Primary (HIGH confidence)
- [CRAN Submission Checklist](https://cran.r-project.org/web/packages/submission_checklist.html) - Official requirements
- [devtools submit_cran() documentation](https://devtools.r-lib.org/reference/submit_cran.html) - Function reference
- [R Packages (2e) - Releasing to CRAN](https://r-pkgs.org/release.html) - Comprehensive guide

### Secondary (MEDIUM confidence)
- [CRAN incoming dashboard](https://r-hub.github.io/cransays/articles/dashboard.html) - Queue monitoring
- [foghorn package](https://fmichonneau.github.io/foghorn/reference/cran_incoming.html) - Programmatic queue access

### Tertiary (LOW confidence)
- WebSearch results for workflow best practices - general community patterns

## Metadata

**Confidence breakdown:**
- Submission process: HIGH - Official devtools documentation
- Queue monitoring: HIGH - Official r-hub/foghorn documentation
- Post-acceptance: MEDIUM - R Packages book (may evolve)
- Pitfalls: MEDIUM - Community patterns, not officially documented

**Research date:** 2026-01-21
**Valid until:** 60 days (CRAN process is stable)
