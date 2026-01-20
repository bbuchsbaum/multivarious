# Features Research: CRAN Check Requirements

**Domain:** R Package CRAN Submission
**Researched:** 2026-01-20
**Overall Confidence:** HIGH (based on official CRAN documentation and R-pkgs book)

## Executive Summary

R CMD check performs approximately 50+ distinct checks across package structure, code, documentation, tests, and vignettes. For CRAN submission, the strict rule is: **zero ERRORs, zero WARNINGs, minimize NOTEs**. The `--as-cran` flag enables additional checks that CRAN runs during incoming inspection.

For multivarious specifically, key concerns include:
- 29 Imports (exceeds soft limit of 20, will generate NOTE)
- Deprecated functions using lifecycle (acceptable if properly documented)
- 14 vignettes (check time concerns)

---

## Errors (Must Fix - Automatic Rejection)

ERRORs cause immediate, automatic rejection. No exceptions.

### Package Structure Errors

| Error Type | Cause | Fix |
|------------|-------|-----|
| Missing DESCRIPTION | No DESCRIPTION file | Create valid DESCRIPTION |
| Invalid DESCRIPTION | Malformed fields, missing required fields | Fix syntax, add Title/Description/Author/Maintainer/License |
| Missing NAMESPACE | No NAMESPACE file | Generate via roxygen2 or create manually |
| Invalid NAMESPACE | Syntax errors in NAMESPACE | Regenerate with `devtools::document()` |
| Installation failure | Package cannot be installed | Fix compilation errors, missing dependencies |
| R syntax errors | Invalid R code in R/ directory | Fix syntax errors |
| Package cannot load | Runtime errors during load | Fix `.onLoad()`, `.onAttach()`, circular dependencies |

### Code Errors

| Error Type | Cause | Fix |
|------------|-------|-----|
| Undefined globals | Using undefined variables | Import from packages or define locally |
| Missing dependencies | Using package not in Imports/Depends | Add to DESCRIPTION |
| Examples fail with error | `stop()` called in examples | Fix example code or use `\dontrun{}` |
| Tests fail | `testthat` tests throw errors | Fix failing tests |
| Vignette build fails | Error during vignette compilation | Fix vignette code |

### Documentation Errors

| Error Type | Cause | Fix |
|------------|-------|-----|
| Missing documentation | Exported function has no .Rd file | Add roxygen2 documentation |
| Rd parse error | Invalid .Rd syntax | Fix roxygen2 comments |
| Missing `\value` | Exported function lacks return documentation | Add `@return` tag |

---

## Warnings (Should Fix - Likely Rejection)

WARNINGs cause rejection in most cases. CRAN expects zero warnings.

### Documentation Warnings

| Warning Type | Cause | CRAN Tolerance | Fix |
|--------------|-------|----------------|-----|
| Undocumented arguments | `@param` missing for function argument | None - must fix | Add `@param` for all arguments |
| Undocumented return value | Missing `@return` | None - must fix | Add `@return` tag |
| Cross-reference not found | `\link{foo}` to non-existent topic | None - must fix | Fix link or remove |
| Examples too slow | Examples take >10 seconds each | Very low | Use `\donttest{}` for slow examples |
| Rd line width | Lines >90 characters in usage/examples | Low | Break long lines |

### Code Warnings

| Warning Type | Cause | CRAN Tolerance | Fix |
|--------------|-------|----------------|-----|
| S3 method inconsistency | Generic/method signature mismatch | None | Match signatures (add `...` if generic has it) |
| Replacement function issue | Invalid replacement function signature | None | Use `x<-` naming with correct args |
| Foreign function call issue | `.Call()` registration problems | None | Use `@useDynLib` properly |
| `T`/`F` instead of `TRUE`/`FALSE` | Using abbreviations | Low | Replace with full names |
| `:::` usage | Accessing internal functions | Very low | Import properly or avoid |

### Build Warnings

| Warning Type | Cause | CRAN Tolerance | Fix |
|--------------|-------|----------------|-----|
| Large installed size | Package >5MB installed | Medium (explain) | Compress data, reduce size |
| Large tarball | Source >10MB | Medium (request exception) | Reduce size or request allowance |
| Binary files | Executables in package | None | Remove or justify in BinaryFiles |

---

## Notes (Usually Acceptable with Explanation)

NOTEs require human review. Eliminate when possible; explain when not.

### Always Acceptable NOTEs

| NOTE | When It Appears | Action Required |
|------|-----------------|-----------------|
| "New submission" | First CRAN submission | None - expected |
| "New maintainer" | Maintainer email changed | Confirm from old email if possible |
| Non-ASCII in data | UTF-8 characters in datasets | None if encoding declared correctly |
| Rd cross-reference to suggested package | Links to packages in Suggests | None - informational |

### Usually Acceptable NOTEs (Explain in cran-comments.md)

| NOTE | Threshold | Typical CRAN Response |
|------|-----------|----------------------|
| "Imports includes X packages" | >20 Imports | Accepted with explanation if justified |
| Check time | >10 minutes total | May request reduction |
| "Found the following (possibly) invalid URLs" | Transient URL failures | Accepted if URLs work |
| "GNU make required" | Using GNU-specific Makefile | Accepted for compiled code |
| Package size | 1-5MB | Usually fine |

### NOTEs to Fix (CRAN Will Ask)

| NOTE | Cause | Fix |
|------|-------|-----|
| "no visible binding for global variable" | NSE/tidyverse patterns | Use `.data$var` or add `utils::globalVariables()` |
| "Undefined global functions or variables" | Missing imports | Add to Imports and import in NAMESPACE |
| "Consider adding importFrom" | Using `::` extensively | Add `@importFrom` directives |
| Spelling errors | Typos in DESCRIPTION/documentation | Fix spelling or add to inst/WORDLIST |

---

## Documentation Requirements

### Mandatory Documentation

| Requirement | Where | Checked By |
|-------------|-------|------------|
| DESCRIPTION complete | DESCRIPTION file | R CMD check |
| All exports documented | man/*.Rd | R CMD check |
| All arguments documented | `@param` in roxygen | R CMD check |
| Return values documented | `@return` in roxygen | R CMD check |
| Examples for exports | `@examples` in roxygen | R CMD check (run by default) |

### DESCRIPTION Field Requirements

```
Title: <65 chars, title case, no period, no "A package for...">
Description: <One paragraph, proper sentences, package names in 'quotes'>
Authors@R: <Must include cph (copyright holder) role>
License: <Must be in R's license database>
URL: <Optional but recommended>
BugReports: <Optional but recommended>
```

### What CRAN Reviewers Check Manually

- Title/Description not redundant ("R package" is implicit)
- No grammar/spelling errors
- Package names in single quotes: `'ggplot2'`
- Function references with parentheses: `foo()`
- DOIs formatted correctly: `<doi:10.xxxx/yyyy>`
- No promotional language

---

## Test Requirements

### What Must Pass

| Requirement | Standard |
|-------------|----------|
| All tests pass | Zero failures |
| No test errors | Tests must not throw unhandled errors |
| Reasonable time | Tests should complete in <60 seconds ideally |
| Work without internet | Or skip gracefully with `skip_if_offline()` |
| Work on all platforms | Or skip appropriately |

### Test Best Practices for CRAN

```r
# Skip tests that need optional dependencies
skip_if_not_installed("optional_package")

# Skip tests that need internet
skip_if_offline()

# Skip on CRAN for slow/flaky tests
skip_on_cran()

# Set reasonable timeouts
withr::local_options(timeout = 60)
```

### Common Test Failures

| Failure Type | Cause | Prevention |
|--------------|-------|------------|
| Timing-dependent | Race conditions | Use deterministic tests |
| Internet-dependent | External API calls | Mock or skip_if_offline() |
| Platform-specific | OS differences | Skip or handle appropriately |
| Missing Suggests | Package in Suggests not installed | skip_if_not_installed() |

---

## Dependency Requirements

### Imports vs Suggests vs Depends

| Field | Meaning | CRAN Rule |
|-------|---------|-----------|
| **Depends** | Required AND attached (user sees) | Minimize; use for R version requirement |
| **Imports** | Required, used via `::` or import | Must be on CRAN/Bioconductor |
| **Suggests** | Optional, for examples/tests/vignettes | Use conditionally |
| **LinkingTo** | C/C++ headers needed | Must be on CRAN |
| **Enhances** | Your package enhances these | Rarely used |

### The 20 Imports Soft Limit

CRAN issues a NOTE when Imports > 20 packages:

> "Importing from so many packages makes the package vulnerable to any of them becoming unavailable. Move as many as possible to Suggests and use conditionally."

**Strategies to reduce Imports (for multivarious with 29):**

1. Move visualization packages to Suggests: `ggplot2`, `ggrepel`
2. Move optional method packages to Suggests: `glmnet`, `randomForest`
3. Move rarely-used packages to Suggests and load conditionally
4. Re-implement simple functions instead of importing packages

### Dependency Validity Rules

| Rule | Consequence |
|------|-------------|
| All Imports must be on CRAN/Bioconductor | ERROR if unavailable |
| Cannot depend on orphaned packages | NOTE, may block submission |
| Circular dependencies forbidden | ERROR |
| Version constraints must be satisfiable | ERROR |

---

## Vignette Requirements

### Build Requirements

| Requirement | Standard |
|-------------|----------|
| Build successfully | Zero errors |
| Reasonable build time | <60 seconds per vignette ideally |
| All packages available | List in Suggests |
| Output in inst/doc | Pre-built vignettes required |

### Vignette Best Practices

```yaml
# In vignette YAML header:
vignette: >
  %\VignetteIndexEntry{Title}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
```

```r
# Conditional evaluation for optional dependencies
eval_optional <- requireNamespace("optional_pkg", quietly = TRUE)
```

### With 14 Vignettes (multivarious concern)

- Total check time may exceed 10 minutes
- Consider using `knitr::opts_chunk$set(eval = FALSE)` for some
- Pre-compute expensive results and load from cache
- Ensure all packages in vignettes are in Suggests

---

## The `--as-cran` Flag

`R CMD check --as-cran` enables additional checks that CRAN runs.

### Additional Checks Enabled

| Check | Environment Variable | What It Does |
|-------|---------------------|--------------|
| Rd line widths | `_R_CHECK_RD_LINE_WIDTHS_` | Enforce 90 char limit |
| URL validation | `_R_CHECK_RD_VALIDATE_RD2HTML_` | Check all URLs work |
| Core limit | `_R_CHECK_LIMIT_CORES_` | Max 2 cores in examples |
| Top-level files | `_R_CHECK_TOPLEVEL_FILES_` | Report non-standard files |
| PDF size | (GhostScript) | Check PDF sizes |
| Rd citations | `_R_CHECK_RD_CITATIONS_` | Validate citation format |

### Running Comprehensive Checks

```r
# Basic check
devtools::check()

# CRAN-style check
devtools::check(cran = TRUE)

# Or from command line
R CMD build .
R CMD check --as-cran packagename_version.tar.gz
```

---

## Known Exceptions (What CRAN Tolerates)

### Commonly Accepted Exceptions

| Situation | How to Handle |
|-----------|---------------|
| >20 Imports for complex package | Explain in cran-comments.md |
| Slow examples | Wrap in `\donttest{}`, explain |
| Platform-specific features | Document, provide fallbacks |
| External software dependency | Document clearly in DESCRIPTION |
| Large data files | Request exception, explain need |

### Lifecycle/Deprecated Functions

Using the lifecycle package for deprecation is **fully acceptable**:

- `lifecycle::deprecate_soft()` - Warns only on direct calls
- `lifecycle::deprecate_warn()` - Warns unconditionally
- `lifecycle::deprecate_stop()` - Errors (for removed functions)

CRAN does not penalize packages for having deprecated functions **as long as**:
- The package itself passes R CMD check
- Deprecated functions still work (just warn)
- Documentation is clear about deprecation status

---

## Pre-Submission Checklist

### Automated (R CMD check)

- [ ] Zero ERRORs
- [ ] Zero WARNINGs
- [ ] NOTEs explained in cran-comments.md
- [ ] Passes on R-devel (required by policy)
- [ ] Passes on Windows (use win-builder)
- [ ] Passes on macOS (use R-hub or mac-builder)

### Manual Review

- [ ] DESCRIPTION Title/Description polished
- [ ] NEWS.md updated (for updates)
- [ ] Version number incremented
- [ ] No secrets/API keys in code
- [ ] License file correct
- [ ] URLs all working
- [ ] Author/maintainer email valid and monitored

### cran-comments.md Template

```markdown
## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission / update from version X.Y.Z

## Notes

* NOTE: Imports includes 29 packages
  - This package provides comprehensive multivariate analysis tools
    requiring matrix computation (Matrix, MASS), optimization (glmnet),
    and visualization (ggplot2) capabilities.

## Test environments

* local macOS, R 4.4.0
* win-builder (devel and release)
* R-hub (Ubuntu, Fedora, Windows)
```

---

## Sources

### Official Documentation
- [CRAN Repository Policy](https://cran.r-project.org/web/packages/policies.html)
- [CRAN Submission Checklist](https://cran.r-project.org/web/packages/submission_checklist.html)
- [Writing R Extensions](https://cran.r-project.org/doc/manuals/r-release/R-exts.html)

### R Packages Book (2e)
- [Appendix A: R CMD check](https://r-pkgs.org/R-CMD-check.html)
- [Chapter 22: Releasing to CRAN](https://r-pkgs.org/release.html)
- [Chapter 10: Dependencies Mindset](https://r-pkgs.org/dependencies-mindset-background.html)

### Tools and Resources
- [rcmdcheck package](https://rcmdcheck.r-lib.org/)
- [devtools::check()](https://devtools.r-lib.org/reference/check.html)
- [lifecycle package](https://lifecycle.r-lib.org/)
- [pkgndep - Analyze dependency heaviness](https://github.com/jokergoo/pkgndep)

### Community Resources
- [ThinkR CRAN Preparation Guide](https://github.com/ThinkR-open/prepare-for-cran)
- [Karl Broman's Package Primer](https://kbroman.org/pkg_primer/pages/cran.html)
