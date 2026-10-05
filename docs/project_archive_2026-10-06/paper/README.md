# CoFiTok Paper Workspace

Current paper sources for:

```text
CoFiTok: Coarse-to-Fine Denoising Tokens for Pixel-Space Diffusion
```

## Entrypoints

- `venues/aaai27/main_aaai2027.tex`: canonical AAAI-27 anonymous main paper.
- `venues/aaai27/supplementary_aaai2027.tex`: canonical supplementary material.
- `latex/main.tex`: venue-neutral working version.

The compiled PDFs live beside these entrypoints. Experimental tables and
figures are sourced from versioned paths under `../CoFiTok-internal/artifacts/`;
large run outputs do not belong in this directory.

## Shared Files

- `references.bib`: bibliography used by both paper builds.
- `citation_audit.md`: current citation provenance, supported framing, and
  claim guardrails.
- `full_pdf_claim_audit_sources.json`: exact source-PDF provenance and SHA256
  values from the full-text audit.
- `latex/README.md` and `venues/aaai27/README.md`: build-specific instructions.

The obsolete Markdown manuscript, standalone figure plan, and overlapping
citation notes were removed after their content was superseded by the current
LaTeX sources and consolidated audit. Historical 2026-07-08 recovery artifacts
retain their original file inventory and checksums.

Before submission, verify the latest venue rules, author metadata,
supplement/checklist handling, page limits, and proceedings metadata.
