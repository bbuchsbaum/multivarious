# Testing Patterns

**Analysis Date:** 2026-01-20

## Test Framework

**Runner:**
- testthat (version >= 3.0 implied by patterns)
- Config: `tests/testthat.R`

**Assertion Library:**
- testthat built-in expectations

**Run Commands:**
```bash
# From R console
devtools::test()                        # Run all tests
testthat::test_package("multivarious")  # Run tests directly
testthat::test_file("tests/testthat/test_pca.R")  # Single file

# With coverage
covr::package_coverage()

# From command line
R CMD check .                           # Full package check including tests
```

## Test File Organization

**Location:**
- All tests in `tests/testthat/`
- Test runner at `tests/testthat.R`

**Naming:**
- Pattern: `test_<component>.R`
- Examples: `test_pca.R`, `test_projector.R`, `test_preprocess.R`, `test_cross_projector.R`

**Structure:**
```
tests/
├── testthat.R                    # Test runner entry point
└── testthat/
    ├── test_pca.R               # PCA functionality tests
    ├── test_projector.R         # Base projector tests
    ├── test_bi_projector_union.R
    ├── test_bootstrap.R         # Bootstrap resampling tests
    ├── test_classifier.R
    ├── test_cpca.R
    ├── test_cross_projector.R
    ├── test_cv.R                # Cross-validation tests
    ├── test_discriminant_projector.R
    ├── test_geneig.R
    ├── test_geneig_methods.R
    ├── test-geneig-subspace.R
    ├── test_nystrom.R
    ├── test_plsc.R
    ├── test_preprocess.R        # Preprocessing pipeline tests
    ├── test_reconstruct_new_biprojector.R
    ├── test_regress.R
    └── test_svd.R
```

## Test Structure

**Suite Organization:**
```r
# Legacy context() style (still supported)
context("projector")
library(testthat)
library(multivarious)

test_that("can construct a projector object", {
  v <- matrix(rnorm(10*5), 10, 5)
  proj <- projector(v)
  expect_s3_class(proj, "projector")
  expect_equal(ncomp(proj), 5)
  expect_equal(shape(proj), c(10, 5))
})

# Modern style (no context block)
test_that("can run a simple pca analysis", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1)
  proj <- project(pres, mat1)
  s <- scores(pres)
  expect_equal(proj, s)
})
```

**Setup/Teardown Pattern:**
```r
# Test-level setup using set.seed for reproducibility
set.seed(123)

# Helper functions defined at file level
make_toy_pca <- function(n = 25L, p = 12L, noise = .15, k = 2L) {
  s1 <- rnorm(n, sd = 3)
  load1 <- runif(p, -1, 1)
  X <- outer(s1, load1) + matrix(rnorm(n * p, 0, noise), n, p)
  pca(X, ncomp = k, preproc = center(), method = "fast")
}

# Reusable test data at file scope
toy_pca <- make_toy_pca()
boot_res <- bootstrap_pca(toy_pca, nboot = 40L, k = 2L, seed = 999)
```

**Common Test Patterns:**
```r
# Dimension checks
expect_equal(dim(pdat), c(10, 5))
expect_equal(shape(proj), c(10, 5))

# Class checks
expect_s3_class(proj, "projector")
expect_s3_class(ptest, c("perm_test_pca", "perm_test"))

# Numeric equality with tolerance
expect_equal(recon, mat1, tolerance = 1e-3, ignore_attr = TRUE)
expect_equal(proj, s)

# Logical conditions
expect_true(ncomp(pres2) == 2)
expect_true(all.equal(dim(resid_vals), dim(mat1)))
expect_lt(mse_xy, 1e-10)  # Less than

# Error expectations
expect_error(bootstrap_pca(toy_pca, nboot = 0, k = k), "nboot must be a positive integer")
expect_error(reprocess(cp, dims$X[, 1:2], colind = c(1, 10), source = "X"))
```

## Mocking

**Framework:** No explicit mocking framework; uses helper functions and controlled data

**Patterns:**
```r
# Helper functions replace complex objects
perfect_fit <- function(train_data, ...) "identity"
perfect_eval <- function(model, test_data, ...) {
  Xrec <- test_data  # perfect reconstruction
  measure_reconstruction_error(test_data, Xrec, metrics = c("mse", "r2"))
}

# Deliberate failure simulation
error_fit <- function(train_data, ...) {
  stop("boom")
}

# Controlled random data generation
make_blocks <- function(n = 40L, pX = 6L, pY = 5L, d = 3L) {
  Vx <- qr.Q(qr(matrix(rnorm(pX * d), pX, d)))  # orthonormal
  Vy <- qr.Q(qr(matrix(rnorm(pY * d), pY, d)))
  F <- matrix(rnorm(n * d), n, d)
  list(X = F %*% t(Vx), Y = F %*% t(Vy), Vx = Vx, Vy = Vy, F = F)
}
```

**What to Mock:**
- Random data with controlled properties (orthonormal bases, known relationships)
- Perfect/identity models for baseline testing
- Deliberate failures for error handling tests

**What NOT to Mock:**
- Core matrix operations
- Package functions under test
- Preprocessing pipelines (test end-to-end)

## Fixtures and Factories

**Test Data Factories:**
```r
# PCA test data factory
make_toy_pca <- function(n = 25L, p = 12L, noise = .15, k = 2L) {
  s1 <- rnorm(n, sd = 3)
  load1 <- runif(p, -1, 1)
  X <- outer(s1, load1) + matrix(rnorm(n * p, 0, noise), n, p)
  pca(X, ncomp = k, preproc = center(), method = "fast")
}

# Cross-projector test data factory
make_blocks <- function(n = 40L, pX = 6L, pY = 5L, d = 3L) {
  Vx <- qr.Q(qr(matrix(rnorm(pX * d), pX, d)))
  Vy <- qr.Q(qr(matrix(rnorm(pY * d), pY, d)))
  F <- matrix(rnorm(n * d), n, d)
  list(X = F %*% t(Vx), Y = F %*% t(Vy), Vx = Vx, Vy = Vy, F = F)
}

# Discriminant analysis test data
set.seed(42)
n_per <- 25L
p_sig <- 2L
p_noise <- 8L
X_signal <- rbind(
  cbind(matrix(rnorm(n_per * p_sig, mean = -3), n_per, p_sig),
        matrix(rnorm(n_per * p_noise), n_per)),
  cbind(matrix(rnorm(n_per * p_sig, mean = 3), n_per, p_sig),
        matrix(rnorm(n_per * p_noise), n_per))
)
Y_signal <- factor(rep(c("A", "B"), each = n_per))
```

**Location:**
- Helper functions defined at top of test files
- No separate fixtures directory
- Inline data generation within tests for simple cases

## Coverage

**Requirements:** No enforced minimum; coverage measured via `covr`

**View Coverage:**
```r
# From R console
covr::package_coverage()

# Generate HTML report
covr::report()
```

**CI Configuration:**
- `codecov.yml` present in repository root
- Coverage uploaded to Codecov service

## Test Types

**Unit Tests:**
- Test individual functions in isolation
- Verify input/output dimensions
- Check class inheritance
- Validate numeric results against known values

```r
test_that("can construct a projector object", {
  v <- matrix(rnorm(10*5), 10, 5)
  proj <- projector(v)
  expect_s3_class(proj, "projector")
  expect_equal(ncomp(proj), 5)
})
```

**Integration Tests:**
- Test method chains (preprocessing -> fit -> project -> reconstruct)
- Verify round-trip transformations
- Test cross-component interactions

```r
test_that("can reconstruct a PCA and recover X after centering", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, preproc = center())
  recon <- reconstruct(pres)
  expect_equal(recon, mat1, tolerance = 1e-3, ignore_attr = TRUE)
})
```

**Statistical Tests:**
- Verify permutation tests yield expected p-values
- Test signal detection vs noise
- Validate bootstrap confidence intervals

```r
test_that("perm_test.discriminant_projector yields small p for signal and large p for noise", {
  # ... setup signal data ...
  pt_sig <- perm_test(dp_sig, X_signal, nperm = 150)
  expect_true(pt_sig$p.value < 0.05)

  # ... setup noise data ...
  pt_noise <- perm_test(dp_noise, X_noise, nperm = 150)
  expect_true(pt_noise$p.value > 0.10)
})
```

**E2E Tests:**
- Not formalized as separate test type
- Integration tests serve as end-to-end verification
- Full pipeline tests embedded in unit test files

## Common Patterns

**Reproducibility:**
```r
# Set seed at test level for reproducibility
set.seed(123)
set.seed(42)

# Pass seed to functions that use randomness
boot_res <- bootstrap_pca(toy_pca, nboot = 40L, k = k, seed = 999)
```

**Numeric Tolerance:**
```r
# Default tolerance for floating point comparisons
expect_equal(recon, mat1, tolerance = 1e-3, ignore_attr = TRUE)

# Stricter tolerance for exact relationships
expect_equal(proj, s)  # Default tolerance
expect_true(max(abs(F_hat - dims$F)) < 1e-12)

# Very loose tolerance for statistical properties
expect_true(abs(cor1) >= 0.9)
expect_true(acc_lda > .90)
```

**Error Testing:**
```r
# Expect specific error message
expect_error(bootstrap_pca(toy_pca, nboot = 0, k = k),
             "nboot must be a positive integer")

# Expect any error
expect_error(reprocess(cp, dims$X[, 1:2], colind = c(1, 10), source = "X"))

# Test error handling records errors gracefully
res <- cv_generic(X, folds, error_fit, dummy_eval)
expect_true(grepl("Fit failed", res$metrics[[1]]$error[1]))
```

**Matrix Property Verification:**
```r
# Check dimensions
expect_equal(dim(boot_res$E_Vb), c(p, k))
expect_equal(dim(result_apply), dim(expected_apply))

# Check statistical properties
expect_true(max(abs(colMeans(x_transformed))) < 1e-14)  # Centered
expect_true(max(abs(apply(x_scaled, 2, sd) - 1)) < 1e-14)  # Unit variance

# Check probability constraints
expect_true(all(abs(rowSums(probs_lda) - 1) < 1e-10))  # Sum to 1
```

**Preprocessing Integration:**
```r
# Initialize preprocessor before use in tests
preproc <- prep(pass())
Xp <- init_transform(preproc, X_signal)

# Test preprocessing round-trip
pp <- center() %>% prep()
X <- pp$init(mat1)
x2 <- pp$reverse_transform(X)
expect_equal(mat1, x2)
```

## Test Organization Best Practices

**One Concept Per Test:**
```r
test_that("can truncate a pca", {
  mat1 <- matrix(rnorm(10*15), 10, 15)
  pres <- pca(mat1, ncomp=4)
  pres2 <- truncate(pres, 2)

  expect_true(ncomp(pres2) == 2)
  expect_equal(length(pres2$sdev), 2)
  expect_equal(ncol(coef(pres2)), 2)
  expect_equal(ncol(scores(pres2)), 2)
})
```

**Descriptive Test Names:**
```r
test_that("can construct a projector object", ...)
test_that("can project data onto subspace", ...)
test_that("transfer converts X->Y (and Y->X) with low reconstruction error", ...)
test_that("bootstrap_pca returns object of correct class and shape", ...)
test_that("Z-scores equal mean divided by SD", ...)
```

**Helper Functions for Complex Setup:**
```r
# Define helpers at file level
make_blocks <- function(n = 40L, pX = 6L, pY = 5L, d = 3L) { ... }
make_blocks_nonortho <- function(n = 40L, pX = 6L, pY = 5L, d = 3L) { ... }
kfold_split <- function(n, k = 5) { ... }
```

---

*Testing analysis: 2026-01-20*
