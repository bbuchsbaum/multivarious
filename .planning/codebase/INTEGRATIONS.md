# External Integrations

**Analysis Date:** 2026-01-20

## APIs & External Services

**None:**
- This is a self-contained R package for multivariate analysis
- No external API calls or web services
- All computation is local

## Data Storage

**Databases:**
- None - Package operates on in-memory R objects (matrices, data frames)

**File Storage:**
- Local filesystem only
- Data loaded/saved using standard R mechanisms (RData, RDS)

**Caching:**
- In-memory caching via `.cache` environment within projector objects
- No persistent caching layer

## Authentication & Identity

**Auth Provider:**
- Not applicable (no external services)

## Monitoring & Observability

**Error Tracking:**
- None (standard R error handling)
- Errors raised via `rlang::abort()` and `stop()`

**Logs:**
- No formal logging framework
- Standard R `message()`, `warning()` for user communication
- `cli` package for formatted console output

## CI/CD & Deployment

**Hosting:**
- GitHub (source code): https://github.com/bbuchsbaum/multivarious
- GitHub Pages (documentation): https://bbuchsbaum.github.io/multivarious/
- CRAN (package distribution)

**CI Pipeline:**
- GitHub Actions
  - `check-standard.yaml`: R CMD check on macOS/Windows/Ubuntu
  - `test-coverage.yaml`: Coverage via covr, uploaded to Codecov
  - `pkgdown.yaml`: Documentation site build and deploy to gh-pages

**Deployment Process:**
1. Push to main/master triggers CI checks
2. pkgdown site auto-deployed on push/release
3. CRAN submission manual (see `cran-comments.md`)

## Environment Configuration

**Required env vars:**
- None required for package functionality

**Development env vars:**
- `GITHUB_PAT`: Used by GitHub Actions for authentication
- `_MULTIVARIOUS_DEV_COVERAGE`: Flag for development coverage mode

**Secrets location:**
- GitHub repository secrets (for Actions)
- No local secrets required

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Third-Party R Package Integrations

The package integrates with numerous CRAN packages for specialized computations.

**SVD/Eigenvalue Backends:**
| Package | Purpose | Used In |
|---------|---------|---------|
| `corpcor` | fast.svd | `R/svd.R` |
| `rsvd` | Randomized SVD | `R/svd.R` |
| `irlba` | Truncated SVD | `R/svd.R` |
| `RSpectra` | Spectral methods | `R/svd.R`, `R/geneig.R` |
| `svd` | propack.svd | `R/svd.R` |
| `geigen` | Generalized eigen | `R/geneig.R` |
| `PRIMME` | Iterative eigensolver | `R/geneig.R` |

**Statistical Methods:**
| Package | Purpose | Used In |
|---------|---------|---------|
| `glmnet` | Regularized regression | `R/regress.R` |
| `pls` | Partial Least Squares | `R/plsc.R` |
| `GPArotation` | Factor rotation | `R/pca.R` |

**Classification (optional):**
| Package | Purpose | Used In |
|---------|---------|---------|
| `randomForest` | RF classifiers | `R/classifier.R` |

## Code Coverage Service

**Codecov:**
- Configuration: `codecov.yml`
- Target: auto (informational)
- Threshold: 1%
- Integration via `covr::codecov()` in GitHub Actions

## Documentation Service

**pkgdown:**
- Template: `albersdown` (Bootstrap 5)
- URL: https://bbuchsbaum.github.io/multivarious/
- Build: Automated via GitHub Actions on push/release
- Configuration: `_pkgdown.yml`

---

*Integration audit: 2026-01-20*
