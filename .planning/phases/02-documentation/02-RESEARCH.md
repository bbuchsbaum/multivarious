# Phase 2: Documentation - Research

**Researched:** 2026-01-20
**Domain:** R Package Documentation for CRAN Submission
**Confidence:** HIGH

## Summary

This phase prepares the multivarious package (currently v0.2.0) for CRAN resubmission by creating/updating required documentation files. The package previously existed on CRAN and is being resubmitted with targeted fixes from Phase 1.

The standard approach for CRAN documentation involves:
1. Creating NEWS.md with version-specific changelog entries
2. Updating cran-comments.md to explain R CMD check results and NOTEs
3. Ensuring all documentation is complete (already verified in Phase 1)
4. Bumping version in DESCRIPTION to 0.3.0
5. Updating .Rbuildignore to exclude planning files from the package bundle

**Primary recommendation:** Create NEWS.md documenting changes since 0.2.0, update cran-comments.md to explain the remaining NOTEs as acceptable, bump version to 0.3.0 in DESCRIPTION.

## Standard Stack

### Core Tools
| Tool | Version | Purpose | Why Standard |
|------|---------|---------|--------------|
| roxygen2 | 7.3.3 | Documentation generation | Already in use; generates .Rd files |
| usethis | any | Helper functions | `use_news_md()`, `use_version()` |
| devtools | any | Package development | `check()`, `document()` |

### Files to Create/Modify
| File | Action | Purpose |
|------|--------|---------|
| NEWS.md | Create | Document changes for v0.3.0 |
| cran-comments.md | Update | Explain R CMD check results |
| DESCRIPTION | Modify | Bump Version: 0.2.0 -> 0.3.0 |
| .Rbuildignore | Modify | Exclude .planning directory |

## Architecture Patterns

### NEWS.md Structure (CRAN-compliant)

The heading format must follow one of these patterns (R parses this for `utils::news()`):
- `# packagename X.Y.Z`
- `# packagename vX.Y.Z`
- `# Version X.Y.Z`
- `# Changes in X.Y.Z`

**Recommended format for multivarious:**
```markdown
# multivarious 0.3.0

## Bug Fixes

* Fixed T/F shorthand usage to TRUE/FALSE in `pca()` for CRAN compliance.
* Converted `\dontrun{}` to `\donttest{}` in examples for executable but slow code.
* Fixed `bootstrap.plsc()` to avoid duplicate argument errors.
* Fixed `regress()` PLS method dimension handling.

## Internal Changes

* Registered previously unregistered S3 methods: `classifier.projector`,
  `inverse_projection.projector`, `perm_ci.pca`.
* Added missing `importFrom` directives for `coefficients` and `combn`.
* Replaced non-ASCII characters with ASCII equivalents in documentation.

## Deprecated

* `prep()` is deprecated in favor of `fit()` for preprocessing pipelines.
* `perm_ci.pca()` is deprecated.
* `perm_test.plsc()` is deprecated.

# multivarious 0.2.0

* Initial CRAN release.
```

### cran-comments.md Structure

Based on authoritative sources ([R Packages 2e](https://r-pkgs.org/release.html), [usethis](https://usethis.r-lib.org/reference/use_cran_comments.html)):

```markdown
## Resubmission

This is a resubmission. In this version I have:

* [specific changes made]

## R CMD check results

0 errors | 0 warnings | X notes

* NOTE 1: [full message]
  - Explanation: [why acceptable]

* NOTE 2: [full message]
  - Explanation: [why acceptable]

## Reverse dependencies

There are currently no downstream dependencies for this package.
```

### DESCRIPTION Version Format

```
Version: 0.3.0
```

Standard semantic versioning. Patch increment (0.2.0 -> 0.2.1) is typical for bug fix resubmissions, but minor increment (0.2.0 -> 0.3.0) is acceptable for resubmissions with broader changes.

### .Rbuildignore Patterns

Current file already excludes cran-comments.md. Need to add:
```
^\.planning$
^\.claude$
^CLAUDE\.md$
^figure$
^check\.log$
^README\.html$
```

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Version bumping | Manual DESCRIPTION edit | `usethis::use_version("minor")` | Updates NEWS.md header too |
| NEWS.md creation | Empty file | `usethis::use_news_md()` | Creates proper template |
| .Rbuildignore patterns | Manual regex | `usethis::use_build_ignore()` | Proper escaping |

**Key insight:** The usethis package handles escaping and formatting automatically; manual edits risk malformed regex patterns in .Rbuildignore.

## Common Pitfalls

### Pitfall 1: NEWS.md Heading Format

**What goes wrong:** Heading like `## Version 0.3.0` or `# v0.3.0` without package name causes `utils::news()` to fail parsing.
**Why it happens:** Different packages use different styles, not all are compatible.
**How to avoid:** Use `# multivarious 0.3.0` format consistently.
**Warning signs:** `utils::news(package="multivarious")` returns NULL or errors.

### Pitfall 2: Not Explaining NOTEs

**What goes wrong:** CRAN reviewer manually checks unexplained NOTEs, causing delays or rejection.
**Why it happens:** Assumption that NOTEs are "fine" and don't need explanation.
**How to avoid:** Every NOTE should have a one-line explanation in cran-comments.md.
**Warning signs:** CRAN response asks about specific NOTEs.

### Pitfall 3: Forgetting .Rbuildignore Updates

**What goes wrong:** Planning files (.planning/, CLAUDE.md) included in package tarball, triggering NOTEs.
**Why it happens:** Files added during development not excluded from build.
**How to avoid:** Add all non-package files to .Rbuildignore before building.
**Warning signs:** "Non-standard files/directories found at top level" NOTE.

### Pitfall 4: Version Mismatch

**What goes wrong:** DESCRIPTION says 0.3.0 but NEWS.md says 0.2.1 (or vice versa).
**Why it happens:** Manual editing of multiple files.
**How to avoid:** Use `usethis::use_version()` which updates both files.
**Warning signs:** Version appears different in different places.

### Pitfall 5: Missing \value in .Rd Files

**What goes wrong:** R CMD check warns "Missing \value sections".
**Why it happens:** roxygen2 `@return` tag missing or incomplete.
**How to avoid:** Every exported function needs `@return` tag.
**Warning signs:** R CMD check warning about documentation.

**Current status:** Phase 1 verification showed 0 documentation warnings, but 15 .Rd files lack \value sections. Most are print/plot methods (acceptable to omit) or internal functions (@keywords internal). R CMD check did not flag these as warnings.

## Code Examples

### Creating NEWS.md

```r
# Source: https://usethis.r-lib.org/reference/use_news_md.html
usethis::use_news_md()
# Creates NEWS.md with template, opens for editing
```

### Bumping Version

```r
# Source: https://r-pkgs.org/release.html
usethis::use_version("minor")  # 0.2.0 -> 0.3.0
# Updates DESCRIPTION and adds NEWS.md heading
```

### Adding to .Rbuildignore

```r
# Source: https://usethis.r-lib.org/reference/use_build_ignore.html
usethis::use_build_ignore(".planning")
usethis::use_build_ignore(".claude")
usethis::use_build_ignore("CLAUDE.md")
usethis::use_build_ignore("figure")
usethis::use_build_ignore("check.log")
usethis::use_build_ignore("README.html")
```

### Manual DESCRIPTION Edit

```
# Before
Version: 0.2.0

# After
Version: 0.3.0
```

### Manual .Rbuildignore Edit

```
# Add these lines (use ^ for start anchor, escape dots)
^\.planning$
^\.claude$
^CLAUDE\.md$
^figure$
^check\.log$
^README\.html$
```

## Current R CMD Check NOTEs

From Phase 1 verification and fresh check (2026-01-20):

### NOTE 1: Hidden files and directories
```
Found the following hidden files and directories:
  .claude
  .planning
```
**Resolution:** Add to .Rbuildignore
**Explanation for cran-comments.md:** "Development artifacts excluded via .Rbuildignore"

### NOTE 2: Non-standard top-level files
```
Non-standard files/directories found at top level:
  'CLAUDE.md' 'README.html' 'check.log' 'figure'
```
**Resolution:** Add to .Rbuildignore
**Explanation for cran-comments.md:** "Development artifacts excluded via .Rbuildignore"

### NOTE 3: Escaped LaTeX specials
```
checkRd: (-1) plsc.Rd:17: Escaped LaTeX specials: \_
checkRd: (-1) plsc.Rd:19: Escaped LaTeX specials: \_
```
**Resolution:** Minor formatting; not a warning
**Explanation for cran-comments.md:** "Escaped underscores in documentation are intentional for rendering."

### NOTE 4: Vignette engine declaration
```
Files named as vignettes but with no recognized vignette engine
```
**Resolution:** This appears to be a parsing issue with the vignette YAML headers - they have proper engine declarations but the format may have whitespace issues. Requires investigation.
**Risk level:** MEDIUM - may need vignette header reformatting

### NOTE 5: Unstated dependency on albersdown
```
'::' or ':::' import not declared from: 'albersdown'
```
**Resolution:** The albersdown package is used conditionally in vignettes for theming. It's not a required dependency.
**Explanation for cran-comments.md:** "albersdown is used conditionally for vignette styling via `requireNamespace()` guard; not a runtime dependency."

### Post-.Rbuildignore Expected NOTEs

After fixing .Rbuildignore, expect only 2-3 NOTEs:
1. Escaped LaTeX specials (informational)
2. Vignette-related (may need fixing)
3. 29 Imports (needs explanation)

The 29 Imports NOTE is expected:
```
Imports includes 29 non-default packages.
```
**Explanation for cran-comments.md:** "The package provides a comprehensive framework for multivariate analysis requiring these dependencies. Each import is actively used for specific functionality (e.g., Matrix for sparse matrices, RSpectra for efficient eigendecomposition, pls for partial least squares)."

## Order of Operations

**Recommended task sequence:**

1. **Update .Rbuildignore first** - Reduces NOTEs before final check
2. **Bump version in DESCRIPTION** - Sets the version all other files reference
3. **Create NEWS.md** - Documents changes for this version
4. **Update cran-comments.md** - Explains remaining NOTEs
5. **Run final R CMD check** - Verify all documentation complete
6. **Run devtools::document()** - Regenerate .Rd files if any @return tags added

**Rationale:** .Rbuildignore changes reduce noise in subsequent checks. Version bump happens before NEWS.md creation so the version is consistent. Final check confirms everything works.

## Changes to Document in NEWS.md

Based on git history since v0.2.0:

### Bug Fixes
- Fixed T/F shorthand to TRUE/FALSE in pca.R
- Converted \dontrun{} to \donttest{} for executable examples
- Fixed bootstrap.plsc() duplicate argument handling
- Fixed regress() PLS method dimension mismatch
- Fixed iris data frame to matrix conversion in examples
- Replaced non-ASCII characters with ASCII equivalents
- Fixed multiblock permutation tests
- Fixed composed projector composition and reconstruction
- Fixed singular covariance handling in discriminant projector
- Fixed preprocessing reversal for subset reconstruction
- Fixed intercept column handling in regress

### Internal Changes
- Registered S3 methods: classifier.projector, inverse_projection.projector, perm_ci.pca
- Added missing importFrom directives
- Removed unused tidyr from Imports
- Updated vignettes to use modern preprocessing API

### Deprecated
- prep() deprecated in favor of fit()
- perm_ci.pca() deprecated
- perm_test.plsc() deprecated

## Open Questions

1. **Vignette engine declarations**: The YAML headers appear correct but R CMD check reports "no recognized vignette engine". May need reformatting of the vignette header block. This could be a whitespace/newline issue in the YAML.

2. **albersdown dependency**: Used conditionally in all 14 vignettes. Acceptable to leave as-is with explanation, or could be made fully optional with fallback.

## Sources

### Primary (HIGH confidence)
- [R Packages 2e - Chapter 18: Other Markdown Files](https://r-pkgs.org/other-markdown.html) - NEWS.md format
- [R Packages 2e - Chapter 22: Releasing to CRAN](https://r-pkgs.org/release.html) - cran-comments.md, version bump, .Rbuildignore
- [usethis::use_cran_comments()](https://usethis.r-lib.org/reference/use_cran_comments.html) - cran-comments.md template
- [usethis::use_news_md()](https://rdrr.io/cran/usethis/man/use_news_md.html) - NEWS.md creation

### Secondary (MEDIUM confidence)
- [devtools NEWS.md](https://github.com/r-lib/devtools/blob/main/NEWS.md) - Example of well-formatted NEWS.md
- [ggplot2 NEWS.md](https://github.com/tidyverse/ggplot2/blob/main/NEWS.md) - Example from mature package

### Tertiary (LOW confidence)
- WebSearch results for CRAN submission practices - verified against primary sources

## Metadata

**Confidence breakdown:**
- NEWS.md format: HIGH - Documented in R Packages 2e, consistent across sources
- cran-comments.md structure: HIGH - Documented in R Packages 2e, usethis
- .Rbuildignore patterns: HIGH - Standard patterns, usethis handles escaping
- Version bump: HIGH - Standard semantic versioning
- NOTE explanations: MEDIUM - Based on common patterns; actual CRAN response may vary

**Research date:** 2026-01-20
**Valid until:** 2026-02-20 (30 days - documentation practices are stable)
