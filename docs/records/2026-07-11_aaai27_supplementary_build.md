# AAAI-27 Supplementary Build Completion

Date: 2026-07-11

## Outcome

The AAAI-27 paper directory now exposes two canonical entrypoints:

- `paper/venues/aaai27/main_aaai2027.tex`
- `paper/venues/aaai27/supplementary_aaai2027.tex`

The main entrypoint includes the established `main.tex` source for backward
compatibility. The supplementary entrypoint uses the same official AAAI-27
submission style and compiles independently.

## Supplementary Evidence

The supplementary PDF contains:

- An 8-dataset by 15-method evidence coverage matrix.
- Eight-dataset P0 generated-sample context for CoFiTok, parameter-matched
  direct dense epsilon, Improved DDPM, and EDM.
- Official-pretrained FlexTok and TiTok reconstruction context.
- Official ImageNet-256 50K eval-only rows for D-AR, MAR, and ReTok.
- P0 protocol settings, parameter counts, dataset-source boundaries, metric
  definitions, repeated evidence, and confirmatory generation controls.

Metric-bearing external tables are imported from the locked evidence package:

`CoFiTok-internal/artifacts/reports/paper_evidence_report_2026-07-11_final/latex_tables/paper_tables_appendix.tex`

## Verification

- Full local test suite passed: 215 tests.
- Generated LaTeX table wrapper compiled successfully.
- `main_aaai2027.pdf`: 7 letter-size pages.
- `supplementary_aaai2027.pdf`: 5 letter-size pages.
- Both final logs have zero undefined-reference, overfull-box, LaTeX-error,
  emergency-stop, or fatal-error matches.
- All 12 PDF pages were rendered with Poppler and visually checked for
  clipping, overlap, unreadable tables, and broken glyphs.
- PDF text checks confirmed the supplement contains EDM, Improved DDPM,
  FlexTok, TiTok, D-AR, MAR, and ReTok.

## SHA256

- `main_aaai2027.pdf`: `866d6a940b27f91e916bdd9e9386c7c770e3c28da4a90acc71df174a7646fb61`
- `supplementary_aaai2027.pdf`: `e37d0803ffb7bb48c5f4c75578d60f5f4975eb3496e68b6b97d7b9916f284132`
- `main_aaai2027.tex`: `7abcdf18134fa05c0c3b72a58ee6e59df9594f63f3e09d204229629507ec4657`
- `supplementary_aaai2027.tex`: `ef3cfede90f52d305b03e99cf1fd18a1e430178f75351ae0e250322f413e18b6`
