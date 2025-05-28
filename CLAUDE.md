# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

multivarious is an R package that provides extensible data structures for multivariate analysis, focusing on dimensionality reduction techniques and projection methods. The package implements a flexible framework built around the concept of "projectors" - objects that map high-dimensional data to lower-dimensional representations.

## Development Commands

### Building and Installing
```r
# From R console
devtools::install()        # Build and install the package
devtools::build()          # Build package tarball
devtools::load_all()       # Load all functions for development

# From command line
R CMD build .              # Build package
R CMD INSTALL .            # Install package
```

### Testing
```r
# From R console
devtools::test()                        # Run all tests
testthat::test_package("multivarious")  # Run tests directly
covr::package_coverage()                # Run tests with coverage

# Run a single test file
testthat::test_file("tests/testthat/test_pca.R")

# From command line
R CMD check .              # Run full package checks including tests
```

### Documentation and Checking
```r
# Generate/update documentation from roxygen2 comments
devtools::document()

# Run package checks
devtools::check()          # Full package check
R CMD check --as-cran .    # CRAN-style checks
rcmdcheck::rcmdcheck()     # Enhanced checking

# Build package website
pkgdown::build_site()
```

## Architecture Overview

### Core Class Hierarchy

The package is built around a hierarchical system of projector classes:

1. **`projector`** - Base class for all dimensionality reduction methods
   - Contains coefficient matrix `v` for projection
   - Includes preprocessing pipeline (`preproc`)
   - Implements caching for performance (`.cache` environment)

2. **`bi_projector`** - Extends projector for two-way mappings (e.g., PCA, SVD)
   - Adds scores matrix `s` and standard deviations `sdev`
   - Supports both row and column projections

3. **`cross_projector`** - For two-block methods (e.g., CCA)
   - Maintains separate coefficients for X and Y blocks
   - Enables cross-domain transfer operations

4. **`composed_projector`** - Chains multiple projectors sequentially

5. **`multiblock_projector`** - Handles concatenated blocks of variables

### Key Design Patterns

1. **S3 Generic Functions**: All major operations (`project`, `reconstruct`, `truncate`, etc.) are implemented as S3 generics with method dispatch

2. **Preprocessing Pipeline**: Sophisticated preprocessing system where each step has `forward()`, `apply()`, and `reverse()` methods

3. **Caching Strategy**: Expensive computations (inverse projections, etc.) are cached in each projector's `.cache` environment

4. **Partial Operations**: Framework supports projecting/reconstructing subsets of variables via `partial_project()` and related functions

### Important Implementation Details

- When modifying projectors, always clear the cache to ensure consistency
- Preprocessing must be reversible - all transformations need a reverse operation
- Use `chk::chk_*` functions for input validation
- Follow existing patterns for error messages using `rlang::abort()`
- Matrix operations should handle sparse matrices when possible

### Testing Patterns

Tests use testthat and follow these patterns:
- Test files are named `test_<component>.R`
- Use `expect_equal()` with tolerance for numeric comparisons
- Test both standard and edge cases (empty data, single observation, etc.)
- Verify that preprocessing is correctly reversed
- Check that caching doesn't affect results

### Common Development Tasks

When adding a new projector type:
1. Inherit from appropriate base class (`projector`, `bi_projector`, etc.)
2. Implement required methods (`project`, `reconstruct`, etc.)
3. Add appropriate S3 method implementations in relevant files
4. Include input validation using `chk` package
5. Add comprehensive tests
6. Update documentation with examples