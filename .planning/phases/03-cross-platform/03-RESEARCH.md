# Phase 3: Cross-Platform Verification - Research

**Researched:** 2026-01-20
**Domain:** R Package Cross-Platform Testing (win-builder, mac-builder, R-devel)
**Confidence:** HIGH

## Summary

This phase verifies that the multivarious package builds and passes R CMD check on Windows, macOS, and with R-devel before CRAN submission. The package is already clean locally (0 errors, 0 warnings, 2 NOTEs) and now needs external verification using CRAN's recommended build services.

The standard approach uses two primary services:
1. **win-builder** - Windows testing on R-release, R-devel, and R-oldrelease
2. **mac-builder** - macOS Apple Silicon (M1) testing with CRAN M1 build machine setup

The devtools package provides functions to submit to these services programmatically. Results arrive via email (win-builder) or web link (mac-builder) within approximately 30 minutes. Cross-platform verification should be done in parallel - submit to both services simultaneously, then analyze results.

**Primary recommendation:** Use `devtools::check_win_release()`, `devtools::check_win_devel()`, and `devtools::check_mac_release()` to submit to all services. Wait for all results before updating cran-comments.md with test environments section.

## Standard Stack

### Core Tools
| Tool | Purpose | Why Standard |
|------|---------|--------------|
| devtools | Programmatic submission to build services | Standard R package development workflow |
| win-builder.r-project.org | Windows verification (same infrastructure as CRAN) | CRAN-recommended, uses identical check setup |
| mac.r-project.org/macbuilder | macOS M1 verification | CRAN-recommended, mirrors CRAN M1 build machine |

### Submission Functions
| Function | Target | R Version |
|----------|--------|-----------|
| `devtools::check_win_release()` | Windows | R-release (currently R 4.5.2) |
| `devtools::check_win_devel()` | Windows | R-devel (will be R 4.6.0) |
| `devtools::check_win_oldrelease()` | Windows | R-oldrelease (currently R 4.4.3) |
| `devtools::check_mac_release()` | macOS M1 | R-release |

### Function Signatures

```r
# Windows checks
devtools::check_win_release(
  pkg = ".",
  args = NULL,
  manual = TRUE,    # Build manual - required for CRAN
  email = NULL,     # Uses DESCRIPTION maintainer email
  quiet = FALSE
)

devtools::check_win_devel(pkg = ".", args = NULL, manual = TRUE, email = NULL, quiet = FALSE)
devtools::check_win_oldrelease(pkg = ".", args = NULL, manual = TRUE, email = NULL, quiet = FALSE)

# macOS check
devtools::check_mac_release(
  pkg = ".",
  dep_pkgs = character(),  # Custom dependencies
  args = NULL,
  manual = TRUE,
  quiet = FALSE
)
# Returns URL to results (invisibly)
```

## Architecture Patterns

### Submission Workflow

```
1. PARALLEL SUBMISSION (simultaneous)
   ├── devtools::check_win_release()  → email with results link
   ├── devtools::check_win_devel()    → email with results link
   └── devtools::check_mac_release()  → returns results URL

2. WAIT FOR RESULTS (~30 minutes each)
   ├── Check maintainer email for win-builder notifications
   └── Visit returned URL for mac-builder results

3. ANALYZE RESULTS
   ├── Any ERRORS? → Must fix before submission
   ├── Any WARNINGS? → Must fix before submission
   └── Any new NOTEs? → Investigate and document

4. UPDATE cran-comments.md
   └── Add test environments section with results
```

### cran-comments.md Test Environments Format

The current cran-comments.md needs a "Test environments" section updated with cross-platform results:

```markdown
## Test environments

* Local: macOS Sonoma 14.3, R 4.5.1 (aarch64-apple-darwin20)
* win-builder (R-release): R 4.5.2
* win-builder (R-devel): R 4.6.0
* mac-builder: macOS, R-release, Apple Silicon (M1)

## R CMD check results

0 errors | 0 warnings | 2 notes

[existing NOTE explanations]
```

### Results Interpretation Guide

| Result | Action |
|--------|--------|
| 0 errors, 0 warnings, same NOTEs | Ready to proceed to CRAN submission |
| 0 errors, 0 warnings, new NOTEs | Investigate; document if benign or fix |
| Any WARNING | MUST fix before CRAN submission |
| Any ERROR | MUST fix before CRAN submission |

### Timeline Expectations

| Service | Typical Response Time | Results Delivery |
|---------|----------------------|------------------|
| win-builder | ~30 minutes | Email to DESCRIPTION maintainer |
| mac-builder | ~30 minutes | Web page (URL returned by function) |

Note: Results links are deleted after approximately 72 hours.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Submitting to win-builder | Manual FTP upload | `devtools::check_win_*()` | Handles package bundling, upload, proper formatting |
| Submitting to mac-builder | Manual web form | `devtools::check_mac_release()` | Returns URL, handles upload automatically |
| Building source tarball | Manual R CMD build | devtools handles this internally | Ensures consistent build process |
| Monitoring queue | Manually checking FTP | `foghorn::winbuilder_queue()` | Shows queue status if concerned about delays |

**Key insight:** The devtools functions encapsulate all the complexity of bundling, uploading, and proper formatting. Manual submission is error-prone and unnecessary.

## Common Pitfalls

### Pitfall 1: Not Testing R-devel

**What goes wrong:** Package accepted with R-release but fails CRAN incoming checks with R-devel.
**Why it happens:** R-devel has stricter checks and API changes. CRAN tests against R-devel.
**How to avoid:** Always run `check_win_devel()` before submission.
**Warning signs:** "Non-API calls to R" notes appear only on R-devel.

### Pitfall 2: Platform-Specific Code Paths

**What goes wrong:** Code works on one platform but fails on another due to path separators, file permissions, or OS-specific behavior.
**Why it happens:** Development on single platform without testing elsewhere.
**How to avoid:** Use `file.path()` for paths, avoid platform-specific assumptions.
**Warning signs:** Tests fail on win-builder but pass locally.

### Pitfall 3: Missing Email Results

**What goes wrong:** Win-builder results never arrive.
**Why it happens:** Email filtering, server issues, or incorrect maintainer email in DESCRIPTION.
**How to avoid:**
  - Verify DESCRIPTION maintainer email is correct
  - Check spam folder
  - Use `foghorn::winbuilder_queue()` to monitor
  - Wait 30-60 minutes before re-submitting
**Warning signs:** No email within 1 hour.

### Pitfall 4: Dependencies Not Available

**What goes wrong:** Package depends on something not available on all platforms.
**Why it happens:** Platform-specific dependencies, Bioconductor packages, or rare CRAN packages.
**How to avoid:**
  - Check CRAN availability of all Imports
  - Use `requireNamespace()` with graceful fallback for optional features
**Warning signs:** "package not available" errors in check logs.

**Note on multivarious:** The PRIMME package is in Imports but used conditionally via `requireNamespace()` with proper guards. Tests skip if PRIMME is not installed. PRIMME is now available on Windows (despite SystemRequirements stating POSIX), so this should not cause issues.

### Pitfall 5: Submitting Too Quickly After Failures

**What goes wrong:** Repeated submissions flood the service, wasting maintainer resources.
**Why it happens:** Impatience, unclear error messages.
**How to avoid:**
  - Wait for full results before resubmitting
  - Debug locally using R-devel if possible
  - Fix all issues before next submission
**Warning signs:** Multiple emails from win-builder in short succession.

### Pitfall 6: Ignoring New Platform-Specific NOTEs

**What goes wrong:** CRAN rejects because of unexplained platform-specific NOTE.
**Why it happens:** Assuming local check results represent all platforms.
**How to avoid:** Compare results across all platforms, document any platform-specific NOTEs.
**Warning signs:** NOTE appears on win-builder but not local macOS check.

## Code Examples

### Complete Submission Workflow

```r
# From R console, in package directory

# 1. Submit to all platforms in parallel
# Note: These return immediately; results come via email/URL
devtools::check_win_release()
devtools::check_win_devel()
mac_url <- devtools::check_mac_release()

# 2. mac-builder returns URL immediately
message("Mac builder results will be at: ", mac_url)

# 3. Wait ~30 minutes for emails
# Check spam folder if not received

# 4. Optional: Monitor win-builder queue
foghorn::winbuilder_queue()  # Requires foghorn package
```

### Manual Submission Alternative (Web)

If devtools functions fail:

1. **Build source package:**
   ```bash
   R CMD build /Users/bbuchsbaum/code/multivarious
   ```

2. **Win-builder:**
   - Upload `multivarious_0.3.0.tar.gz` at https://win-builder.r-project.org/upload.aspx
   - Select R version (release, devel, or oldrelease)

3. **Mac-builder:**
   - Upload at https://mac.r-project.org/macbuilder/submit.html
   - Package must list you as maintainer

### Checking Results

```r
# If no email after 1 hour, check queue
foghorn::winbuilder_queue()

# Can also check FTP directory manually in browser:
# ftp://win-builder.r-project.org/R-release/
# ftp://win-builder.r-project.org/R-devel/
```

## Expected Results for multivarious

Based on local check results and package characteristics:

### Expected NOTEs

1. **Escaped LaTeX specials in plsc.Rd** - Documented, intentional
2. **Unstated dependency on albersdown in vignettes** - Documented, conditional use

### Potential Platform-Specific Issues

| Issue | Risk | Mitigation |
|-------|------|------------|
| PRIMME dependency | LOW | Uses `requireNamespace()` guard; available on Windows now |
| 27 Imports | LOW | Already documented in cran-comments.md; CRAN accepts with explanation |
| Vignette build time (~24s) | LOW | Within acceptable limits |
| RSpectra/irlba compilation | LOW | Both well-maintained CRAN packages with Windows binaries |

### Likely Result

Expect: `0 errors | 0 warnings | 2 notes` across all platforms, matching local results.

## Decision Framework: Handling Results

```
IF errors OR warnings THEN
  - Must fix before CRAN submission
  - Run local fix, re-run local check
  - Re-submit to failing platform
  - Document fix in cran-comments.md "Resubmission" section

IF new NOTE not seen locally THEN
  - Investigate cause (platform-specific?)
  - If benign: add explanation to cran-comments.md
  - If problematic: fix and resubmit

IF same NOTEs as local THEN
  - Update Test environments in cran-comments.md
  - Ready for Phase 4 (CRAN submission)
```

## Timing Considerations

| Step | Duration |
|------|----------|
| Submit to all services | < 5 minutes |
| Wait for results | 30-60 minutes |
| Review results | 10 minutes |
| Update cran-comments.md | 10 minutes |
| **Total (no issues)** | **~1 hour** |
| Fix issues + resubmit | Add ~1 hour per iteration |

## Sources

### Primary (HIGH confidence)
- [win-builder.r-project.org](https://win-builder.r-project.org/) - Official Windows build service documentation
- [mac.r-project.org/macbuilder](https://mac.r-project.org/macbuilder/submit.html) - Official macOS build service
- [devtools::check_win](https://devtools.r-lib.org/reference/check_win.html) - devtools function documentation
- [R Packages 2e - Chapter 22](https://r-pkgs.org/release.html) - Releasing to CRAN
- [CRAN Submission Checklist](https://cran.r-project.org/web/packages/submission_checklist.html) - Official CRAN requirements

### Secondary (MEDIUM confidence)
- [R-hub WinBuilder Guide](https://blog.r-hub.io/2020/04/01/win-builder/) - Best practices for using win-builder
- [ThinkR prepare-for-cran](https://github.com/ThinkR-open/prepare-for-cran) - Community checklist

### Tertiary (LOW confidence)
- WebSearch results for common platform issues - verified against official sources

## Metadata

**Confidence breakdown:**
- Submission workflow: HIGH - Official documentation from devtools and R-project
- Expected results: MEDIUM - Based on local checks and package analysis
- Timing estimates: MEDIUM - Based on multiple sources, may vary
- Platform-specific issues: HIGH - PRIMME verified available on Windows via CRAN

**Research date:** 2026-01-20
**Valid until:** 2026-02-20 (30 days - build services stable but versions update)

## Open Questions

1. **Mac-builder R version:** The service documentation doesn't clearly state which R version is used (assumed R-release based on devtools function naming). Actual results will confirm.

2. **Win-builder email reliability:** Reports of occasional email delivery issues in late 2025. Mitigation: use foghorn to monitor queue if emails don't arrive.

3. **PRIMME Windows behavior:** While CRAN shows Windows binaries available, the SystemRequirements still states "POSIX system". The `requireNamespace()` guard should handle any issues gracefully.
