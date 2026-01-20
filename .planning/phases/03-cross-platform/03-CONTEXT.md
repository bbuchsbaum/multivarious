# Phase 3: Cross-Platform Verification - Context

**Gathered:** 2026-01-20
**Status:** Ready for planning

<domain>
## Phase Boundary

Verify package passes R CMD check on Windows, macOS, and R-devel via external build services before CRAN submission. Submit to win-builder and mac-builder, wait for results, address any issues, and update cran-comments.md with verification results.

</domain>

<decisions>
## Implementation Decisions

### Service selection
- Use win-builder and mac-builder (not rhub)
- Test on both R-release and R-devel via win-builder
- Submit to win-builder and mac-builder simultaneously
- Wait for results from all services before proceeding to submission

### Issue handling
- Errors and warnings are blocking — must fix before submission
- NOTEs are documented in cran-comments.md, not necessarily fixed
- New platform-specific NOTEs: investigate and document if benign
- Platform-specific warnings: must fix before submission (no tolerance)
- Dependency-related notes (e.g., PRIMME availability): acceptable, document only

### Claude's Discretion
- Exact format of platform results in cran-comments.md
- How to structure the verification workflow (submit commands, wait times)
- Technical investigation of any unexpected issues

</decisions>

<specifics>
## Specific Ideas

No specific requirements — standard verification workflow using CRAN-recommended services.

</specifics>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 03-cross-platform*
*Context gathered: 2026-01-20*
