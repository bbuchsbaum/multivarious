## Submission summary

This is a maintenance release (0.3.2) of multivarious.

The most important change addresses an upcoming CRAN dependency archival:

* Removed the optional `PRIMME` backend from `geneig()` and `cPCAplus()`, and
  dropped `PRIMME` from `Suggests`, because the `PRIMME` package is scheduled
  for archival on CRAN. The iterative `RSpectra`/`subspace` backends and the
  dense `geigen`/`robust`/`sdiag` backends cover the same generalized
  eigenproblems, so no user-facing functionality is lost.

Other changes:

* Mixed-effect inference fixes: explicit term-scope and exchangeability
  overrides for `mixed_regress()`, corrected grouped row-metric
  whitening/unwhitening, and hardened effect-operator permutation and
  bootstrap behaviour for grouped designs.
* Added regression tests for the `PRIMME` removal path and the mixed-effect
  inference behaviour.

See NEWS.md for the complete list of changes.

## R CMD check results

0 errors | 0 warnings | 0 notes

## Reverse dependencies

There are currently no reverse dependencies for this package.

## Test environments

* Local: macOS Sonoma 14.3, R 4.5.1 (aarch64-apple-darwin20)
* win-builder: Windows Server 2022, R-release
* win-builder: Windows Server 2022, R-devel
* mac-builder: macOS, R-release (Apple Silicon)
