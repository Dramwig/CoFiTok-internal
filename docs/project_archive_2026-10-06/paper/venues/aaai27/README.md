# CoFiTok AAAI-27 Submission Build

This directory contains an AAAI-27 anonymous-submission adaptation of the
venue-neutral CoFiTok LaTeX draft.

Official template source:

```text
https://aaai.org/authorkit27/
```

Local template files copied from the official author kit:

```text
aaai2027.sty
aaai2027.bst
ReproducibilityChecklist.tex
```

Canonical AAAI-27 entrypoints:

```text
main_aaai2027.tex
supplementary_aaai2027.tex
```

`main_aaai2027.tex` includes the established `main.tex` source so existing
links remain valid. The supplementary entrypoint imports the locked comparison
tables plus the versioned AAAI-27 component-ablation package.

Build both PDFs with:

```bash
make
```

Build either artifact with:

```bash
make main
make supplementary
```

Current status:

- Uses `\usepackage[submission]{aaai2027}` for anonymous submission.
- Keeps author metadata anonymous with `Anonymous Submission`.
- Uses the official AAAI-27 bibliography style through `aaai2027.sty`.
- Produces separate `main_aaai2027.pdf` and
  `supplementary_aaai2027.pdf` submission artifacts.
- Keeps the Experiments section to exactly `Setup`, `Results`, and `Ablations`.
- Reports matched two-seed component ablations and four hyperparameter sweeps
  from `CoFiTok-internal/artifacts/reports/aaai27_ablation_2026-07-11/`.
- Assembles Figure 2 in LaTeX from four standalone PNG subfigures; bitmap files
  contain only experimental grids, while letters and captions remain editable.
- Current build: 7 main-PDF pages and 6 supplementary pages; both logs are
  clean.
- Reproducibility checklist is copied locally but not input into `main.tex`
  until the exact submission requirement is finalized.

Before submission, verify the latest AAAI-27 instructions, page limit,
supplement/checklist handling, author metadata, and source-file naming rules.
