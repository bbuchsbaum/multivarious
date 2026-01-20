# Stack Research: CRAN Submission

**Package:** multivarious
**Researched:** 2026-01-20
**Context:** Resubmission after many updates (previously version 0.2.0 on CRAN)

## Executive Summary

The CRAN submission ecosystem in 2025-2026 is well-established with a clear toolchain: **devtools/rcmdcheck** for local checking, **rhub v2** for multi-platform testing via GitHub Actions, **win-builder/mac-builder** for final verification, and **devtools::submit_cran()** or **devtools::release()** for submission.

For multivarious specifically, the 29 Imports and 14 vignettes require attention to check times and dependency management.

---

## Local Check Tools

### Primary: devtools + rcmdcheck

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| devtools | >= 2.4.5 | Wrapper for all development tasks | Standard, integrates rcmdcheck, handles submission |
| rcmdcheck | >= 1.4.0 | R CMD check with better output | Better error reporting than raw R CMD check |

**Core commands:**
```r
# Standard local check (most common)
devtools::check()

# CRAN-equivalent check
rcmdcheck::rcmdcheck(args = c("--as-cran"))

# Check with manual building (required for CRAN)
devtools::check(manual = TRUE)
```

**Why devtools over raw R CMD check:**
- Better formatted output
- Automatic environment setup (`NOT_CRAN` env var)
- Integrated with document(), test(), and submission workflow
- Since 2025: `check(cran = TRUE)` surfaces "Namespace in Imports field not imported from" NOTEs

**Confidence:** HIGH (Official devtools documentation)

### Supporting: goodpractice

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| goodpractice | >= 1.0.4 | Extended best practices checks | Catches issues R CMD check misses |

**Usage:**
```r
goodpractice::gp()
```

**What it catches beyond R CMD check:**
- Missing tests
- Code complexity (cyclomatic complexity)
- Linting issues via lintr
- Missing Description fields (URL, BugReports)
- Style violations

**Note:** goodpractice has had stability issues historically. Run it, but don't block on all findings. Focus on:
- `no_import_package_as_a_whole` - relevant for packages with many imports
- `truefalse_not_tf` - common mistake
- `rcmdcheck_package_dependencies_present`

**Confidence:** MEDIUM (Package has had CRAN removal/restoration cycles)

### Supporting: lintr

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| lintr | >= 3.1.2 | Static code analysis | Style consistency, potential bugs |

**Usage:**
```r
lintr::lint_package()
```

**When to use:** Optional but helpful for maintaining code quality. Not required for CRAN.

**Confidence:** HIGH (CRAN published November 2025)

---

## Documentation Tools

### Primary: roxygen2

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| roxygen2 | >= 7.3.0 | Generate .Rd files from comments | Already in use (RoxygenNote: 7.3.3 in DESCRIPTION) |

**Current status:** Already configured in multivarious.

**Key commands:**
```r
devtools::document()  # Generate documentation
```

**2025 features to leverage:**
- `@aliases` ordering now controllable (better pkgdown integration)
- Markdown code blocks support alternative knitr engines
- Better PDF/SVG figure handling for different output formats

**Confidence:** HIGH (Official roxygen2 documentation)

### Secondary: pkgdown

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| pkgdown | >= 2.1.0 | Package website | Already configured (URL in DESCRIPTION) |

**Current status:** Website exists at https://bbuchsbaum.github.io/multivarious/

**Key commands:**
```r
pkgdown::build_site()
```

**Confidence:** HIGH

---

## URL and Spelling Checks

### urlchecker

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| urlchecker | >= 1.0.1 | Check URLs in package | CRAN checks all URLs; broken URLs delay submission |

**Usage:**
```r
urlchecker::url_check()        # Check all URLs
urlchecker::url_update()       # Auto-fix 301 redirects
```

**Why this matters for multivarious:** With 14 vignettes, there are likely many URLs. Broken or redirected URLs cause NOTEs.

**Confidence:** HIGH (CRAN published July 2025)

### spelling

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| spelling | >= 2.3.0 | Spell check documentation | CRAN rejects packages with typos in DESCRIPTION |

**Usage:**
```r
devtools::spell_check()
spelling::spell_check_package()
```

**Confidence:** HIGH

---

## Pre-Submission Testing

### Win-Builder

| Service | URL | Purpose | Why |
|---------|-----|---------|-----|
| win-builder | https://win-builder.r-project.org/ | Test on Windows + R-devel | CRAN tests on Windows; R-devel catches future issues |

**Usage:**
```r
devtools::check_win_devel()    # R-devel on Windows
devtools::check_win_release()  # R-release on Windows
```

**Timeline:** Results emailed within ~30 minutes.

**Confidence:** HIGH (Official R Project service)

### Mac Builder

| Service | URL | Purpose | Why |
|---------|-----|---------|-----|
| mac-builder | https://mac.r-project.org/macbuilder/submit.html | Test on macOS M1 | CRAN tests on macOS; catches ARM-specific issues |

**Usage:** Manual upload or:
```r
devtools::check_mac_release()
```

**Confidence:** HIGH (Official R Project service)

### R-hub v2 (Recommended)

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| rhub | >= 2.0.1 | Multi-platform testing via GitHub Actions | Tests many platforms/configurations automatically |

**Setup (one-time):**
```r
rhub::rhub_setup()  # Adds .github/workflows/rhub.yaml
# Push to GitHub, then:
rhub::rhub_doctor()  # Verify setup
```

**Usage:**
```r
rhub::rhub_check()  # Run checks on GitHub
```

**Available platforms (2025):**
- `linux`, `macos`, `macos-arm64`, `windows` (VM-based)
- `ubuntu-clang`, `ubuntu-gcc12`, `ubuntu-release` (container)
- `valgrind`, `rchk` (memory/pointer checking)
- Various clang/gcc versions, ASAN/UBSAN sanitizers

**Important:** rhub v2 pulls from GitHub, not local files. Push first!

**Old API deprecated:** `check_for_cran()` is defunct. Use rhub v2.

**Confidence:** HIGH (Official rhub documentation, July 2025)

---

## CI/CD Integration

### GitHub Actions (r-lib/actions)

**Current status:** multivarious has workflows but they may need updating:
- `.github/workflows/check-standard.yaml` (exists, dated March 2024)
- `.github/workflows/pkgdown.yaml` (exists)
- `.github/workflows/test-coverage.yaml` (exists)

**Recommended setup:**
```r
# Update to current r-lib/actions v2
usethis::use_github_action("check-standard")
usethis::use_github_action("test-coverage")
usethis::use_github_action("pkgdown")
```

**Key workflows:**
| Workflow | Purpose | When to Use |
|----------|---------|-------------|
| check-standard | R CMD check on Linux/Mac/Windows | Every PR, every push |
| test-coverage | Report to codecov.io | Every PR |
| pkgdown | Build website | On release/main push |

**2025 updates:**
- Use `v2` tag (sliding tag with non-breaking updates)
- Workflows now run on all PRs, not just main branch
- Automatic Quarto installation if qmd files present

**Confidence:** HIGH (r-lib/actions repository)

---

## Submission Tools

### devtools::release() vs devtools::submit_cran()

| Function | What it does | When to use |
|----------|--------------|-------------|
| `release()` | Full pre-flight checks + submission | Conservative approach, first submission |
| `submit_cran()` | Direct submission | Already validated via CI/other means |

**Recommendation for multivarious:** Use `release()` for the resubmission since there have been many updates.

**Workflow:**
```r
# 1. Final local check
devtools::check(manual = TRUE)

# 2. Win-builder check
devtools::check_win_devel()

# 3. Create release checklist
usethis::use_release_issue()  # Creates GitHub issue with checklist

# 4. Submit
devtools::release()  # Runs checks, then submits
```

**Confidence:** HIGH

### usethis::use_release_issue()

Creates a GitHub issue with a customized release checklist based on:
- Release type (patch/minor/major)
- Package characteristics

**Usage:**
```r
usethis::use_release_issue()
```

**Why use it:** Tracks progress, surfaces forgotten steps, provides accountability.

**Confidence:** HIGH (usethis 3.1.0, August 2025)

### cran-comments.md

**Current status:** Exists in multivarious, but outdated (references "new release").

**What to include for resubmission:**
```markdown
## R CMD check results

0 errors | 0 warnings | X notes

* This is a resubmission. Previous version was 0.2.0.

## Changes since last CRAN release

* [List major changes]

## Test environments

* local macOS (M1/M2), R 4.x.x
* win-builder (R-devel, R-release)
* mac-builder
* GitHub Actions (ubuntu, macos, windows)

## R CMD check results

[Paste check summary]

## Reverse dependencies

[If any exist, note they were checked]
```

**Confidence:** HIGH

---

## Reverse Dependency Checking

### revdepcheck

| Tool | Version | Purpose | Why |
|------|---------|---------|-----|
| revdepcheck | >= 1.0.0 | Check reverse dependencies | CRAN expects you to not break dependent packages |

**Usage:**
```r
revdepcheck::revdep_check()
```

**For multivarious:** Check if any packages depend on multivarious:
```r
tools::dependsOnPkgs("multivarious")
```

If none, this step is quick. If some exist, run full revdep check.

**Confidence:** HIGH

---

## Specific Recommendations for multivarious

### 1. Dependency Review (29 Imports)

**Concern:** 29 Imports is on the higher end. CRAN doesn't have a hard limit, but:
- More dependencies = longer install time
- More dependencies = more potential breakage

**Action items:**
- Review if all 29 are truly needed in `Imports`
- Consider moving rarely-used packages to `Suggests` with conditional loading
- Candidates to evaluate: `crayon` (cli provides similar), `assertthat` (chk already used)

**Confidence:** MEDIUM (Judgment call based on CRAN policy guidance)

### 2. Vignette Build Time (14 vignettes)

**Concern:** CRAN has limited check farm resources. 14 vignettes may take significant time.

**Action items:**
- Add `eval=FALSE` to expensive code chunks
- Use pre-computed results where possible
- Consider consolidating vignettes if content overlaps
- Test total build time: `devtools::build_vignettes()` and time it

**CRAN guidance:** "Long-running tests and vignette code can be made optional for checking."

**Confidence:** HIGH (CRAN policy)

### 3. Update GitHub Actions Workflows

**Current:** Workflows dated March 2024 (almost 2 years old).

**Action:**
```r
usethis::use_github_action("check-standard")
usethis::use_github_action("test-coverage")
usethis::use_github_action("pkgdown")
```

This will update to r-lib/actions v2.

**Confidence:** HIGH

### 4. Add rhub v2 Workflow

```r
rhub::rhub_setup()
# Then push and use rhub::rhub_check()
```

**Confidence:** HIGH

### 5. Update cran-comments.md

The current file references "new release" which is stale. Update before submission.

**Confidence:** HIGH

---

## Tools NOT to Use

| Tool | Why Not |
|------|---------|
| `rhub::check_for_cran()` | Deprecated/defunct in rhub v2. Use `rhub::rhub_check()` |
| Raw `R CMD check` | Use devtools::check() for better environment setup |
| Manual NAMESPACE editing | Always use roxygen2 |
| `goodpractice` as a blocker | Run for info, but don't block submission on all issues |

---

## Complete Pre-Submission Checklist

```r
# 1. Update and document
devtools::document()
devtools::check(manual = TRUE)

# 2. Check URLs and spelling
urlchecker::url_check()
devtools::spell_check()

# 3. Run goodpractice (informational)
goodpractice::gp()

# 4. Update NEWS.md and cran-comments.md

# 5. Cross-platform testing
devtools::check_win_devel()
devtools::check_win_release()
devtools::check_mac_release()  # or manual mac-builder upload

# 6. rhub multi-platform (if configured)
rhub::rhub_check()

# 7. Reverse dependency check
revdepcheck::revdep_check()

# 8. Create release issue
usethis::use_release_issue()

# 9. Submit
devtools::release()
```

---

## Sources

### Official Documentation
- [CRAN Submission Checklist](https://cran.r-project.org/web/packages/submission_checklist.html)
- [CRAN Repository Policy](https://cran.r-project.org/web/packages/policies.html)
- [R Packages (2e) - Releasing to CRAN](https://r-pkgs.org/release.html)

### Tool Documentation
- [devtools - check()](https://devtools.r-lib.org/reference/check.html)
- [rhub - Getting Started](https://r-hub.github.io/rhub/articles/rhub.html)
- [usethis - GitHub Actions](https://usethis.r-lib.org/reference/github_actions.html)
- [r-lib/actions GitHub](https://github.com/r-lib/actions)
- [usethis - use_release_issue()](https://usethis.r-lib.org/reference/use_release_issue.html)
- [urlchecker](https://github.com/r-lib/urlchecker)

### Build Services
- [Win-Builder](https://win-builder.r-project.org/)
- [Mac Builder](https://mac.r-project.org/macbuilder/submit.html)

### Community Resources
- [ThinkR - Preparing for CRAN](https://github.com/ThinkR-open/prepare-for-cran)
- [R-hub Blog - URL Checks](https://blog.r-hub.io/2020/12/01/url-checks/)
