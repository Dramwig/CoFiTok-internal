# Paper Markdown cleanup (2026-07-12)

Status: `completed`.

- `paper/draft.md` and `paper/figures_and_tables.md` were removed after the
  current LaTeX sources superseded them.
- Four overlapping citation documents were consolidated into
  `paper/citation_audit.md`; exact PDF provenance remains in
  `paper/full_pdf_claim_audit_sources.json`.
- `paper/README.md`, `paper/latex/README.md`, the recovery bundle, and the paper
  artifact validator now point only to current sources and both AAAI entrypoints.
- The original files remain recoverable from
  `CoFiTok-internal/artifacts/recovery/remote_recovery_2026-07-08.tar.gz`.
- Verification: `229 passed`; venue-neutral, AAAI main, and supplementary PDFs
  build successfully; the real paper and venue artifact gates both report `ok`.
