# Architecture

**Analysis Date:** 2025-01-20

## Pattern Overview

**Overall:** S3 Object-Oriented Framework with Hierarchical Projector Classes

**Key Characteristics:**
- Inheritance-based class hierarchy using S3 generics
- Composition pattern for multi-stage projections
- Separation of preprocessing and projection concerns
- Caching strategy for expensive computations
- Two-way projections enabling both sample and variable transformations

## Layers

**Generic Functions Layer:**
- Purpose: Define the public API through S3 generic functions
- Location: `R/all_generic.R`, `R/preproc-generics.R`
- Contains: ~50 generic function definitions for projection, reconstruction, preprocessing, etc.
- Depends on: Nothing (pure interface definitions)
- Used by: All projector implementations

**Preprocessing Layer:**
- Purpose: Transform raw data before projection (centering, scaling, standardization)
- Location: `R/pre_process.R`, `R/preproc-generics.R`, `R/preproc-utils.R`
- Contains: `prepper`, `pre_processor` classes; `center()`, `colscale()`, `standardize()`, `pass()` functions
- Depends on: Generic functions
- Used by: Projector classes (embedded as `preproc` field)

**Base Projector Layer:**
- Purpose: Core projection functionality and coefficient management
- Location: `R/projector.R`
- Contains: `projector` class, `project()`, `reprocess()`, `partial_project()`, `inverse_projection()`
- Depends on: Preprocessing layer
- Used by: All derived projector classes

**Bi-Projector Layer:**
- Purpose: Two-way mapping between samples and variables (rows and columns)
- Location: `R/bi_projector.R`
- Contains: `bi_projector` class with scores (`s`), sdev, and variable projection
- Depends on: Base projector layer
- Used by: `pca`, `svd`, `discriminant_projector`, `multiblock_biprojector`

**Cross-Projector Layer:**
- Purpose: Two-block decompositions with separate X and Y coefficient matrices
- Location: `R/twoway_projector.R`
- Contains: `cross_projector` class for CCA-style methods
- Depends on: Base projector layer
- Used by: Canonical correlation analysis implementations

**Composed Projector Layer:**
- Purpose: Sequential chaining of multiple projectors
- Location: `R/composed_projector.R`
- Contains: `composed_projector`, `composed_partial_projector`, `%>>%` pipe operator
- Depends on: Base projector layer
- Used by: Multi-stage projection pipelines

**Multiblock Layer:**
- Purpose: Handling data with multiple variable blocks
- Location: `R/multiblock.R`
- Contains: `multiblock_projector`, `multiblock_biprojector`
- Depends on: Projector and bi-projector layers
- Used by: Multiblock analysis methods

**Algorithm Implementations:**
- Purpose: Specific dimensionality reduction algorithms
- Location: `R/pca.R`, `R/svd.R`, `R/plsc.R`, `R/regress.R`, `R/cPCA.R`, `R/geneig.R`, `R/discriminant_projector.R`
- Contains: `pca()`, `svd_wrapper()`, `plsc()`, `regress()`, `cPCAplus()`, `geneig()`
- Depends on: Bi-projector or cross-projector layers
- Used by: End users

**Utilities Layer:**
- Purpose: Cross-validation, bootstrap, classification wrappers
- Location: `R/cv.R`, `R/bootstrap.R`, `R/classifier.R`, `R/nystrom_embedding.R`
- Contains: `cv()`, `bootstrap()`, `classifier()`, `nystrom_approx()`
- Depends on: Projector classes
- Used by: End users for model evaluation

## Data Flow

**Standard Projection Flow:**

1. User creates preprocessing pipeline: `center() |> colscale()`
2. User fits model: `pca(X, ncomp=3, preproc=center())`
3. Internal: `fit_transform(preproc, X)` learns and applies preprocessing
4. Internal: `svd_wrapper()` computes decomposition on preprocessed data
5. Internal: `bi_projector()` packages results with preprocessing
6. User projects new data: `project(model, X_new)`
7. Internal: `reprocess(model, X_new)` applies learned preprocessing
8. Internal: Matrix multiplication with coefficient matrix `v`
9. User reconstructs: `reconstruct(model)` or `inverse_transform()`

**Preprocessing Data Flow:**

1. Create pipeline: `preproc <- center() |> standardize()`
2. Fit: `fitted <- fit(preproc, X_train)` - learns means, sds
3. Transform: `transform(fitted, X_test)` - applies learned parameters
4. Inverse: `inverse_transform(fitted, X_transformed)` - reverses transformation

**State Management:**
- Projector objects store preprocessing in `$preproc` field
- Cached computations stored in `.cache` environment attribute
- Preprocessing parameters stored in closure environments within `pre_processor`

## Key Abstractions

**Projector:**
- Purpose: Base abstraction for dimensionality reduction
- Examples: `R/projector.R`
- Pattern: S3 class with coefficient matrix `v` and preprocessing pipeline

**Bi-Projector:**
- Purpose: Two-way mapping (samples <-> components, variables <-> components)
- Examples: `R/bi_projector.R`, `R/pca.R`
- Pattern: Extends projector with scores `s`, singular values `sdev`

**Pre-Processor:**
- Purpose: Reversible data transformation pipeline
- Examples: `R/pre_process.R`
- Pattern: Chain of steps with `forward()`, `apply()`, `reverse()` methods

**Cross-Projector:**
- Purpose: Maps between two data domains (X and Y)
- Examples: `R/twoway_projector.R`
- Pattern: Dual coefficient matrices `vx`, `vy` with separate preprocessors

**Composed Projector:**
- Purpose: Sequential application of projectors
- Examples: `R/composed_projector.R`
- Pattern: List of projectors with combined coefficients computed lazily

## Entry Points

**Algorithm Constructors:**
- Location: `R/pca.R`, `R/svd.R`, `R/plsc.R`
- Triggers: User calls `pca()`, `svd_wrapper()`, `plsc()`
- Responsibilities: Validate input, preprocess, compute decomposition, return projector object

**Generic Methods:**
- Location: `R/all_generic.R`
- Triggers: User calls `project()`, `reconstruct()`, `scores()`, etc.
- Responsibilities: Dispatch to appropriate S3 method based on object class

**Preprocessing Functions:**
- Location: `R/pre_process.R`
- Triggers: User calls `center()`, `standardize()`, etc.
- Responsibilities: Create preprocessing pipeline (unfitted `prepper`)

## Error Handling

**Strategy:** Fail-fast with informative messages using `chk` and `cli` packages

**Patterns:**
- Input validation via `chk::chk_*()` functions at function entry
- Dimension mismatches caught early with `chk::chk_equal()`
- Range validation with `chk::chk_range()`, `chk::chk_subset()`
- Custom error messages via `cli::cli_abort()` for user-facing errors
- `tryCatch()` for external package calls (SVD solvers)

## Cross-Cutting Concerns

**Logging:** Minimal; uses `message()` for progress in permutation tests

**Validation:** Comprehensive input validation using `chk` package throughout

**Authentication:** Not applicable (local R package)

**Caching:**
- Per-object `.cache` environment for inverse projections
- Keyed by operation parameters (e.g., column indices)
- Cleared on truncation or modification

---

*Architecture analysis: 2025-01-20*
