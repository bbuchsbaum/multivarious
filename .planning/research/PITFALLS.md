# Pitfalls Research: Common CRAN Submission Failures

**Domain:** R Package CRAN Submission (resubmission)
**Package:** multivarious
**Researched:** 2026-01-20
**Confidence:** HIGH (based on official CRAN policies and documented patterns)

## Executive Summary

CRAN submission failures fall into predictable categories. This research identifies pitfalls specific to the multivarious package context: 29 imports, 14 vignettes, deprecated functions, and prior CRAN presence. The most critical risks for this package are dependency count (NOTE threshold is 20), vignette build times, and proper deprecation handling.

---

## Critical Pitfalls

### Pitfall 1: Excessive Imports (29 Packages)

**What goes wrong:** CRAN triggers a NOTE when a package has 20+ non-default imports: "Imports includes [X] non-default packages. Importing from so many packages makes the package vulnerable to any of them becoming unavailable."

**Why it happens:** Organic growth of functionality over time; convenience of using full packages for small features.

**Consequences:** Automatic NOTE requires human review. If any dependency is archived, your package gets archived too. In 2024-2025, packages like `choroplethr` were archived simply because their dependency `acs` was archived.

**This package's status:** 29 imports significantly exceeds the 20-package threshold.

**Warning signs:**
- NOTE in R CMD check --as-cran about "Imports includes X non-default packages"
- Dependencies on packages with single maintainers or low activity

**Prevention strategy:**
1. Audit each import - is it used for just 1-2 functions?
2. Move heavy/rarely-used packages to Suggests with conditional loading
3. Reimplement trivial functions (e.g., if only using `capitalize()` from a heavy package)
4. Consider: `ggplot2`, `ggrepel`, `tibble`, `tidyr`, `dplyr`, `crayon` could potentially move to Suggests if used only in optional visualization/printing

**Fix approach:**
```r
# Before: In DESCRIPTION Imports
ggrepel

# After: In DESCRIPTION Suggests
ggrepel

# In code:
if (requireNamespace("ggrepel", quietly = TRUE)) {
  # use ggrepel
} else {
  # fallback or message
}
```

**Sources:**
- [R Packages (2e) - Dependencies in Practice](https://r-pkgs.org/dependencies-in-practice.html)
- [pkgndep vignette on CRAN](https://cran.r-project.org/web/packages/pkgndep/vignettes/suggestions.html)

---

### Pitfall 2: Using T/F Instead of TRUE/FALSE

**What goes wrong:** CRAN reviewers reject packages using `T` and `F` as boolean shorthand because these are not reserved words and can be redefined by users.

**Why it happens:** Developer convenience; old habits; copy-pasted code.

**Consequences:** CRAN will request changes; delays submission.

**This package's status:** FOUND - `R/pca.R` contains `drop = F` on lines 52, 53, 75.

**Warning signs:**
- `goodpractice::gp()` or `lintr::lint_package()` will flag these
- Manual grep: `grep -r "\bT\b\|\bF\b" R/`

**Prevention strategy:**
1. Run `lintr::lint_package()` before submission
2. Add lintr to CI pipeline
3. Search codebase for standalone `T` or `F` tokens

**Fix approach:**
```r
# Before:
scores[, (i + 1):ncomp, drop = F]

# After:
scores[, (i + 1):ncomp, drop = FALSE]
```

**Sources:**
- [CRAN Cookbook - Code Issues](https://contributor.r-project.org/cran-cookbook/code_issues.html)
- [goodpractice package](https://cran.r-project.org/web/packages/goodpractice/vignettes/goodpractice.html)

---

### Pitfall 3: Improper \dontrun{} Usage

**What goes wrong:** CRAN requests replacement of `\dontrun{}` with `\donttest{}` when examples are actually executable.

**Why it happens:** Developer wraps examples to avoid long run times, but uses wrong wrapper.

**Consequences:** CRAN will reject and request changes.

**This package's status:** FOUND - 3 instances in `R/cPCA.R`, `R/pca.R`, `R/all_generic.R`

**Warning signs:**
- Examples wrapped in `\dontrun{}` that could actually run
- CRAN feedback: "replace \dontrun with \donttest"

**Prevention strategy:**
1. Use `\dontrun{}` ONLY for examples that truly cannot execute (missing API keys, external services)
2. Use `\donttest{}` for long-running examples (>5 seconds)
3. Use `if(interactive()){}` for interactive-only functions (Shiny, plots requiring user input)
4. Keep unwrapped examples under 5 seconds per Rd file

**Fix approach:**
```r
# Before:
#' \dontrun{
#'   result <- slow_function(data)
#' }

# After (if executable but slow):
#' \donttest{
#'   result <- slow_function(data)
#' }

# After (if truly non-executable):
#' \dontrun{
#'   # Requires API key
#'   result <- api_function(key = Sys.getenv("API_KEY"))
#' }
```

**Sources:**
- [CRAN Cookbook - General Issues](https://contributor.r-project.org/cran-cookbook/general_issues.html)
- [R-hub blog on examples](https://blog.r-hub.io/2020/01/27/examples/)

---

### Pitfall 4: Deprecated Functions Without Proper Lifecycle Management

**What goes wrong:** Deprecated functions that still emit warnings clutter test output and may confuse CRAN reviewers. Functions deprecated without proper timeline can linger indefinitely.

**Why it happens:** Functions renamed/redesigned but old names kept for compatibility without clear removal timeline.

**Consequences:** Noise in R CMD check output; confusion about API stability; eventual need to remove anyway.

**This package's status:** FOUND - Multiple deprecated functions using `lifecycle::deprecate_warn()`:
- `prep()` -> `fit()`
- `init_transform()` -> `fit_transform()`
- `apply_transform()` -> `transform()`
- `reverse_transform()` -> `inverse_transform()`
- `perm_ci.pca()` -> `perm_test.pca()`
- `global_scores` argument in classifier functions

**Warning signs:**
- `.Deprecated()` or `lifecycle::deprecate_warn()` calls
- `warning()` about deprecated arguments
- No version specified for when deprecation started

**Prevention strategy:**
1. Use `lifecycle::deprecate_soft()` first (warns only for direct calls)
2. Graduate to `lifecycle::deprecate_warn()` after one release
3. Always specify the version when deprecation started
4. Document deprecation in NEWS.md
5. Plan removal timeline (typically 2 major versions)

**Fix approach:**
```r
# Good: Specify version
lifecycle::deprecate_warn(
  when = "0.2.0",
  what = "prep()",
  with = "fit()"
)

# Add to NEWS.md
## multivarious 0.2.0
### Deprecated
* `prep()` is deprecated in favor of `fit()`
```

**Sources:**
- [lifecycle package vignettes](https://cran.r-project.org/web/packages/lifecycle/vignettes/communicate.html)

---

## Moderate Pitfalls

### Pitfall 5: Vignette Build Time Exceeding Limits

**What goes wrong:** CRAN requires total check time under 10 minutes. With 14 vignettes, cumulative build time can easily exceed this.

**Why it happens:** Real-world examples with realistic data sizes; computationally intensive demonstrations.

**Consequences:** CRAN will ask to reduce check time; package rejection.

**This package's status:** HIGH RISK - 14 vignettes at 1.3MB total. Need to verify build times.

**Warning signs:**
- Vignette uses large datasets
- Vignettes take >30 seconds each to build
- Total package check exceeds 10 minutes

**Prevention strategy:**
1. Pre-compute expensive results and save as package data
2. Use `eval = FALSE` for expensive chunks with explanation
3. Create small toy datasets for demonstrations
4. Consider moving some vignettes to website-only (not built on CRAN)

**Fix approach:**
```r
# In vignette YAML:
---
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{Title}
  %\VignetteEngine{knitr::rmarkdown}
---

# In expensive chunks:
```{r eval=FALSE}
# This takes too long for CRAN checks
result <- expensive_computation(large_data)
```

# Or pre-compute:
```{r}
# Pre-computed result
data("precomputed_result", package = "multivarious")
```
```

**Sources:**
- [R Packages (2e) - Vignettes](https://r-pkgs.org/vignettes.html)
- [R-hub blog on vignette workflows](https://blog.r-hub.io/2020/06/03/vignettes/)

---

### Pitfall 6: Missing globalVariables() Declaration

**What goes wrong:** Using dplyr/tidyr with NSE creates "no visible binding for global variable" NOTEs.

**Why it happens:** Non-standard evaluation (NSE) in tidyverse packages references column names as symbols.

**Consequences:** NOTE in R CMD check; requires explanation or fix.

**This package's status:** Uses `dplyr::bind_rows`, `tibble`, `tidyr`. Need to verify if NSE columns are properly declared.

**Warning signs:**
- NOTE: "no visible binding for global variable 'column_name'"
- Using `dplyr::filter()`, `mutate()`, `select()` with unquoted column names

**Prevention strategy:**
1. Use `.data$column` pronoun from rlang
2. Or declare `utils::globalVariables(c("col1", "col2"))` in a file like `R/globals.R`

**Fix approach:**
```r
# Option 1: Use .data pronoun
#' @importFrom rlang .data
my_function <- function(df) {
  df %>% filter(.data$value > 0)
}

# Option 2: Declare globals (in R/globals.R)
utils::globalVariables(c("value", "name", "score"))
```

**Sources:**
- [dplyr in packages vignette](https://cran.r-project.org/web/packages/dplyr/vignettes/in-packages.html)
- [R-bloggers on global variables](https://www.r-bloggers.com/2019/08/no-visible-binding-for-global-variable/)

---

### Pitfall 7: Package Size Exceeding Limits

**What goes wrong:** CRAN issues NOTE at 5MB, hard limit at 10MB for source packages.

**Why it happens:** Large vignettes, data files, compiled code, or accidentally included build artifacts.

**Consequences:** NOTE requires justification; hard rejection at 10MB.

**This package's status:** Working directory is 26MB, but this includes git history and build artifacts. Need to check built package size.

**Warning signs:**
- NOTE about package size
- Large files in `inst/`, `data/`, `vignettes/`

**Prevention strategy:**
1. Check built package size: `devtools::build()` then check tarball size
2. Compress data with `tools::resaveRdaFiles()`
3. Move large data to separate data package
4. Ensure `.Rbuildignore` excludes unnecessary files

**Fix approach:**
```r
# Compress data files
tools::resaveRdaFiles("data/", compress = "xz")

# Check what's included
devtools::build()
# Inspect the .tar.gz
```

**Sources:**
- [R Packages (2e) - Release](https://r-pkgs.org/release.html)

---

### Pitfall 8: Examples Running Too Long

**What goes wrong:** Individual example files taking >5 seconds trigger review requests.

**Why it happens:** Realistic demonstrations with full datasets.

**Consequences:** CRAN asks to reduce example time; use `\donttest{}`.

**Warning signs:**
- R CMD check reports example timing
- Complex computations in examples

**Prevention strategy:**
1. Use minimal toy data in examples
2. Wrap slow examples in `\donttest{}`
3. Target <5 seconds per Rd file

**Sources:**
- [CRAN policies](https://cran.r-project.org/web/packages/policies.html)

---

## Minor Pitfalls

### Pitfall 9: DESCRIPTION Title/Description Issues

**What goes wrong:** CRAN has specific formatting requirements for Title and Description fields.

**Common issues:**
- Title not in title case
- Title starting with "A" or "The" or "This package"
- Title containing "with R" (redundant)
- Description not being 2+ complete sentences
- Missing period at end of Description

**This package's status:** Title and Description appear compliant but should verify:
- Title: "Extensible Data Structures for Multivariate Analysis" (good)
- Description: Appears adequate length

**Prevention strategy:**
1. Title should be title case, not start with package name or articles
2. Description should be 2+ sentences explaining what the package does
3. Don't say "This package..." - it's redundant
4. Spell out acronyms on first use

**Sources:**
- [CRAN Submission Checklist](https://cran.r-project.org/web/packages/submission_checklist.html)

---

### Pitfall 10: Not Cleaning Up in Examples/Tests

**What goes wrong:** Examples or tests that modify `par()`, `options()`, or `setwd()` without restoring.

**Why it happens:** Forgotten cleanup; not using `on.exit()`.

**Consequences:** CRAN rejection for "leaving the user in a different state."

**Prevention strategy:**
1. Always use `on.exit()` for state changes in functions
2. Use `withr` package for scoped state changes
3. In examples, manually reset any changes

**Fix approach:**
```r
# In functions:
my_function <- function() {
  old_par <- par(mfrow = c(2, 2))
  on.exit(par(old_par), add = TRUE)
  # ...
}

# Or use withr:
my_function <- function() {
  withr::with_par(list(mfrow = c(2, 2)), {
    # ...
  })
}
```

**Sources:**
- [CRAN Cookbook - Code Issues](https://contributor.r-project.org/cran-cookbook/code_issues.html)

---

### Pitfall 11: Insufficient cran-comments.md

**What goes wrong:** Not explaining NOTEs or test environment in submission comments.

**Why it happens:** Assuming CRAN knows context; minimal comments.

**Consequences:** Delayed review; back-and-forth questions.

**Prevention strategy:**
1. Create with `usethis::use_cran_comments()`
2. Include R CMD check results from multiple platforms
3. Explain any remaining NOTEs
4. For resubmissions, explain what changed
5. List reverse dependency check results if applicable

**Template:**
```markdown
## Test environments
* local macOS install, R 4.4.0
* win-builder (devel and release)
* R-hub (multiple platforms)

## R CMD check results
0 errors | 0 warnings | 1 note

* This is a resubmission. Changes:
  - Reduced imports from 29 to 22
  - Fixed \dontrun -> \donttest
  - Addressed NOTE about...

## Downstream dependencies
Checked 0 reverse dependencies.
```

**Sources:**
- [usethis::use_cran_comments documentation](https://usethis.r-lib.org/reference/use_cran_comments.html)

---

## Specific Risks for multivarious Package

Based on the package analysis, here are the prioritized risks:

| Risk | Severity | Status | Action Required |
|------|----------|--------|-----------------|
| 29 Imports (threshold: 20) | HIGH | PRESENT | Audit and reduce to <20 if possible |
| T/F instead of TRUE/FALSE | HIGH | FOUND in pca.R | Replace 4 instances |
| \dontrun misuse | MEDIUM | FOUND in 3 files | Review and convert to \donttest |
| Deprecated functions | MEDIUM | PRESENT | Document deprecation versions in NEWS |
| 14 vignettes build time | MEDIUM | UNKNOWN | Measure and optimize |
| Package size | LOW | UNKNOWN | Check built tarball size |
| globalVariables | LOW | POSSIBLY | Check for NSE NOTEs |

### Recommended Pre-Submission Checklist

1. [ ] Run `R CMD check --as-cran` locally with R-devel
2. [ ] Run `devtools::check_win_devel()` and `devtools::check_win_release()`
3. [ ] Run `rhub::check_for_cran()`
4. [ ] Replace all `F` with `FALSE` and `T` with `TRUE`
5. [ ] Convert `\dontrun` to `\donttest` where appropriate
6. [ ] Audit imports - move optional packages to Suggests
7. [ ] Verify vignette build time <10 minutes total
8. [ ] Create/update cran-comments.md
9. [ ] Update NEWS.md with deprecation notes
10. [ ] Run `goodpractice::gp()` for additional checks

---

## Sources

**Official CRAN Documentation:**
- [CRAN Repository Policy](https://cran.r-project.org/web/packages/policies.html)
- [CRAN Submission Checklist](https://cran.r-project.org/web/packages/submission_checklist.html)

**R Packages Book (2e):**
- [Releasing to CRAN](https://r-pkgs.org/release.html)
- [R CMD check](https://r-pkgs.org/R-CMD-check.html)
- [Dependencies](https://r-pkgs.org/dependencies-in-practice.html)
- [Vignettes](https://r-pkgs.org/vignettes.html)

**Community Resources:**
- [CRAN Cookbook](https://contributor.r-project.org/cran-cookbook/)
- [ThinkR prepare-for-cran](https://github.com/ThinkR-open/prepare-for-cran)
- [CRANhaven archiving stats](https://www.cranhaven.org/cran-archiving-stats.html)

**Tools:**
- [goodpractice package](https://cran.r-project.org/web/packages/goodpractice/vignettes/goodpractice.html)
- [pkgndep for dependency analysis](https://cran.r-project.org/web/packages/pkgndep/vignettes/pkgndep.html)
- [lifecycle package](https://cran.r-project.org/web/packages/lifecycle/)
