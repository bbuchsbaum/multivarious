# Architecture Research: CRAN Submission Process

**Project:** multivarious (resubmission)
**Researched:** 2026-01-20
**Confidence:** HIGH (based on official CRAN documentation)

## Executive Summary

The CRAN submission process is a multi-stage workflow involving automated checks, human review, and iterative feedback cycles. For a resubmission like multivarious (version 0.2.0), the process is somewhat streamlined compared to first-time submissions since the package has already passed initial scrutiny. However, all standard checks still apply, and the existing `cran-comments.md` should be updated to reflect this as an update rather than "new release."

---

## Pre-Submission Checklist

Everything to verify before submitting to CRAN.

### Required: Zero Tolerance Items

| Item | Command/Action | Notes |
|------|----------------|-------|
| Pass `R CMD check --as-cran` | `devtools::check()` | **Must have 0 ERRORs and 0 WARNINGs** |
| Check with R-devel | `devtools::check_win_devel()` | Required by CRAN policy |
| Cross-platform checks | R-hub builder or GitHub Actions | Windows, macOS, Linux |
| URL validation | `urlchecker::url_check()` | Fix broken links, update redirects |

### Required: Documentation Items

| Item | Location | Requirement |
|------|----------|-------------|
| DESCRIPTION Title | Line 2 | Not starting with "A package" or package name |
| DESCRIPTION Description | Line 10+ | Informative, single quotes around non-English terms |
| All exported functions | R/*.R | Must have `@returns` and `@examples` tags |
| NEWS.md | Root | Document changes from previous version |
| cran-comments.md | Root | Updated with current check results |

### Required: Code Conduct Items

| Item | Requirement |
|------|-------------|
| Examples runtime | Each example < few seconds |
| Thread usage | Max 2 threads/cores in tests/examples |
| No global state changes | Don't modify global environment |
| No external writes | Only write to tempdir() without user consent |
| No process termination | Never call `quit()` or `q()` |

### Recommended: NOTEs Mitigation

NOTEs do not block submission but trigger human review. Address these:

| NOTE Type | How to Address |
|-----------|----------------|
| "New submission" | Expected for new packages; document in cran-comments |
| "No visible binding" | Use `.data$var` or declare variables with `utils::globalVariables()` |
| "Package size" | Consider data compression or moving large data to separate package |
| "Non-standard files" | Update `.Rbuildignore` |

### Reverse Dependencies Check (for updates)

If multivarious has packages depending on it:

```r
# Install revdepcheck
# install.packages("revdepcheck")

# Run reverse dependency check
revdepcheck::revdep_check(num_workers = 4)

# View results
revdepcheck::revdep_summary()
```

**Policy:** If updates break dependent packages, notify maintainers at least 2 weeks before submission.

---

## Submission Steps

The actual process from "ready to submit" to "submitted."

### Step 1: Increment Version

```r
# For update/patch release
usethis::use_version("patch")  # 0.2.0 -> 0.2.1

# Or for minor release
usethis::use_version("minor")  # 0.2.0 -> 0.3.0
```

### Step 2: Update cran-comments.md

For resubmission (not after CRAN feedback):

```markdown
## R CMD check results

0 errors | 0 warnings | 0 notes

## Submission notes

This is an update to multivarious (previous version: 0.2.0).

Changes in this version:
* [List key changes]

## Test environments

* Local: macOS [version], R [version]
* win-builder: R-devel
* R-hub: [platforms tested]
* GitHub Actions: windows-latest, macOS-latest, ubuntu-latest
```

### Step 3: Final Local Checks

```r
# Full check with CRAN settings
devtools::check(remote = TRUE, manual = TRUE)

# Build the tarball
devtools::build()

# Verify tarball passes check
R CMD check --as-cran multivarious_0.2.1.tar.gz
```

### Step 4: Submit to CRAN

**Option A: Using devtools (recommended)**
```r
# Full release workflow with interactive prompts
devtools::release()

# Or direct submission (skips local prompts)
devtools::submit_cran()
```

**Option B: Web form**
- Go to https://cran.r-project.org/submit.html
- Upload the source tarball (`.tar.gz`)
- Fill in maintainer email
- Submit

### Step 5: Confirm Submission

**Critical:** Within minutes of upload, you will receive a confirmation email. You MUST click the confirmation link and re-confirm agreement with CRAN policies. **If you skip this, your package is NOT submitted.**

---

## Review Process

What happens after submission.

### Automated Phase (Hours)

1. **Upload received** - CRAN acknowledges receipt
2. **Automated checks** - Package checked on Windows and Linux against:
   - R-release (current stable)
   - R-devel (development version)
3. **Results email** - Links to check results sent to maintainer

### Queue Stages

Your package moves through folders visible on the [CRAN incoming dashboard](https://r-hub.github.io/cransays/articles/dashboard.html):

| Stage | Meaning | Action Needed |
|-------|---------|---------------|
| **inspect** | Awaiting manual inspection (possible false positive) | Wait |
| **newbies** | First-time submission queue | Wait (not applicable for updates) |
| **pending** | CRAN member needs more time | Wait |
| **human** | Assigned to specific CRAN member | Wait; may receive questions |
| **recheck** | Reverse dependency checks running | Wait |
| **pretest** | Additional automated tests | Wait |
| **publish** | Accepted! Will appear on CRAN soon | Celebrate |

### Update vs New Package

For **updates to existing packages** (like multivarious):
- Process is potentially fully automated if checks pass
- Less human scrutiny than first-time submissions
- Still subject to reverse dependency checks
- Can still be flagged for human review if issues detected

### Monitoring Tools

```r
# Check your package status in R
foghorn::cran_incoming("multivarious")

# Or view dashboard in browser
# https://r-hub.github.io/cransays/articles/dashboard.html
# https://cransubs.r-universe.dev/
```

---

## Handling Feedback

How to respond when CRAN sends feedback or rejects a submission.

### Common Feedback Types

| Feedback | Typical Cause | Response |
|----------|---------------|----------|
| "Please fix and resubmit" | Check errors/warnings | Fix issues, resubmit |
| "Please add @returns" | Missing return documentation | Add roxygen tags |
| "Examples take too long" | Examples exceed time limit | Add `\donttest{}` wrapper |
| "Please explain NOTE" | Unexplained NOTE in submission | Add explanation to cran-comments |
| "URL is broken" | Link in docs doesn't work | Fix or remove URL |

### Resubmission Protocol

1. **Do NOT respond defensively** - Read feedback carefully
2. **Fix all identified issues** - Address every point raised
3. **Re-run checks** - Verify fixes don't introduce new problems
4. **Increment patch version** - e.g., 0.2.1 -> 0.2.2
5. **Update cran-comments.md** with resubmission section:

```markdown
## Resubmission

This is a resubmission. In this version I have:

* Added @returns tags to all exported functions (as requested)
* Wrapped long-running examples in \donttest{} (as requested)
* Fixed broken URL in vignette

## R CMD check results

0 errors | 0 warnings | 0 notes
```

6. **Submit via `devtools::submit_cran()`**

### Communication Guidelines

- **Be concise** - CRAN team is small (can be counted on one hand)
- **Be specific** - Reference exact changes made
- **Be patient** - Don't resubmit while previous submission pending
- **Don't email unless necessary** - Use cran-comments.md for explanations

### Appealing Decisions

If you believe feedback is incorrect or a false positive:
- Email `cran-submissions@r-project.org` with clear, technical explanation
- Provide evidence (check logs, context)
- Be respectful and concise

---

## Timeline

Expected duration from submission to acceptance.

### Typical Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Upload to confirmation email | Minutes | Automated |
| Automated checks complete | 2-24 hours | Depends on queue |
| Human review (if needed) | 1-7 days | Highly variable |
| Reverse dependency checks | Hours to days | If package has dependents |
| Publication after acceptance | Hours | Usually same day |

### Realistic Expectations

**Best case (clean update, no issues):**
- Submit Monday morning
- Accepted Monday afternoon/evening
- On CRAN by Tuesday

**Typical case (minor issues):**
- Submit Monday
- Feedback Tuesday/Wednesday
- Resubmit Wednesday
- Accepted Thursday/Friday

**Worst case (complex issues):**
- Multiple rounds of feedback over 1-2 weeks
- Policy questions requiring discussion
- Holidays/busy periods extending timeline

### Timing Considerations

- **Avoid submitting Friday afternoon** - May sit in queue over weekend
- **Avoid major R release periods** - CRAN team is busy
- **Avoid end of year** - Holiday period, reduced capacity
- **30-day rule** - Don't submit updates more frequently unless responding to CRAN feedback

---

## Resubmission Notes

Specific considerations for updating an existing package like multivarious.

### Current Package State

Based on examination of the repository:

| Item | Current State | Action Needed |
|------|---------------|---------------|
| `cran-comments.md` | Says "new release" | Update for resubmission |
| `.Rbuildignore` | Includes `cran-comments.md` | Good, no change needed |
| `DESCRIPTION` | Version 0.2.0 | Increment before submission |
| `NEWS.md` | Not found | **Create before submission** |
| ORCID in Authors@R | Present | Good |

### Required Updates for multivarious

1. **Create NEWS.md** documenting changes since last CRAN version:
```markdown
# multivarious 0.2.1

## Changes
* [Document all user-facing changes]

## Bug fixes
* [Document any fixes]
```

2. **Update cran-comments.md**:
```markdown
## R CMD check results

0 errors | 0 warnings | [N] notes

## Submission notes

This is an update to multivarious.

### Test environments
* [List environments tested]

### Changes in this version
* [Key changes]
```

3. **Verify all functions have `@returns` tags** - Previous review mentioned this

### API Stability

If any changes break backward compatibility:
- Check for reverse dependencies: `tools::package_dependencies("multivarious", reverse = TRUE)`
- If dependent packages exist, notify maintainers 2 weeks before submission
- Document breaking changes prominently in NEWS.md

---

## Quick Reference Commands

```r
# Pre-submission
devtools::check()                    # Local check
devtools::check_win_devel()          # Windows R-devel
urlchecker::url_check()              # Validate URLs
devtools::spell_check()              # Check spelling

# Reverse dependencies (if applicable)
revdepcheck::revdep_check(num_workers = 4)

# Submission
usethis::use_version("patch")        # Increment version
devtools::submit_cran()              # Submit to CRAN

# Monitoring
foghorn::cran_incoming("multivarious")

# Post-acceptance
usethis::use_github_release()        # Create GitHub release
usethis::use_dev_version(push = TRUE) # Bump to dev version
```

---

## Sources

### Official CRAN Documentation
- [CRAN Submission Page](https://cran.r-project.org/submit.html)
- [CRAN Submission Checklist](https://cran.r-project.org/web/packages/submission_checklist.html)
- [CRAN Repository Policy](https://cran.r-project.org/web/packages/policies.html)

### R Packages Book (Wickham & Bryan)
- [Chapter 22: Releasing to CRAN](https://r-pkgs.org/release.html)

### Devtools Documentation
- [submit_cran() Reference](https://devtools.r-lib.org/reference/submit_cran.html)

### Monitoring Tools
- [CRAN Incoming Dashboard](https://r-hub.github.io/cransays/articles/dashboard.html)
- [R-hub Builder](https://builder.r-hub.io)

### Community Resources
- [ThinkR: Prepare for CRAN](https://github.com/ThinkR-open/prepare-for-cran)
- [Marine Data Science: CRAN Checklist](https://www.marinedatascience.co/blog/2020/01/09/checklist-for-r-package-re-submissions-on-cran/)
