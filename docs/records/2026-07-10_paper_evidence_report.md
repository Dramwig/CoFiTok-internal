# Paper Evidence Report - 2026-07-10

## Generated Artifact

Unified evidence report:

```text
artifacts/reports/paper_evidence_report_2026-07-10/paper_evidence_report.md
```

Paper-ready claim pack:

```text
artifacts/reports/paper_evidence_report_2026-07-10/paper_claim_pack.md
```

Paper-ready LaTeX tables:

```text
artifacts/reports/paper_evidence_report_2026-07-10/latex_tables/paper_tables.tex
artifacts/reports/paper_evidence_report_2026-07-10/latex_tables/paper_tables_main.tex
artifacts/reports/paper_evidence_report_2026-07-10/latex_tables/paper_tables_appendix.tex
artifacts/reports/paper_evidence_report_2026-07-10/latex_tables/paper_tables_preview.md
artifacts/reports/paper_evidence_report_2026-07-10/latex_tables/paper_tables_manifest.json
```

Compile check:

```text
pdflatex -interaction=nonstopmode -halt-on-error paper_tables_compile_check.tex
```

Result: passed locally with TeX Live 2026; `paper_tables_compile_check.pdf` generated.

AAAI draft integration:

```text
paper/venues/aaai27/main.tex
```

Changes:

- Updated abstract, experimental setup, discussion, limitations, and conclusion to the current 8-dataset evidence boundary.
- Added `Current All-Dataset Evidence Audit`.
- Included:

```latex
\input{../../../CoFiTok-internal/artifacts/reports/paper_evidence_report_2026-07-10/latex_tables/paper_tables_main.tex}
```

Compile check:

```text
cd paper/venues/aaai27
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Result: passed locally with TeX Live 2026. The current main draft uses `paper_tables_main.tex` for the P0 generation and CoFiTok diagnostic tables, while `paper_tables_appendix.tex` retains P1 tokenizer reconstruction, official related-method eval-only rows, and claim-support tables. Current output `main.pdf` has 7 pages. Log scan found no `Undefined`, citation warning, `Overfull`, fatal, emergency, or error matches.

Inputs:

```text
artifacts/reports/p0_horizontal_table_2026-07-10_imagenet256_p0/p0_paper_table.json
artifacts/reports/baselines/p1_tokenizer_reconstruction_eval256_2026-07-10/p1_tokenizer_reconstruction_table.json
artifacts/reports/paper_comparison_matrix_2026-07-10_p1_eval256_guard_latest/paper_comparison_matrix_audit.json
artifacts/reports/baselines/official_related_methods_2026-07-10/official_related_methods_table.json
```

## Completion State

Current evidence matrix:

```text
completed=72
completed_eval_only=16
protocol_blocked=24
```

Interpretation:

- P0 runnable train/eval protocols are complete across current datasets.
- FlexTok and TiTok are complete only as eval-only tokenizer reconstruction rows.
- D-AR and ReTok now have completed official ImageNet-256 50K eval-only secondary rows. MAR has HF safetensors smoke-only evidence. D-AR, MAR, and ReTok still must not be reported as completed fair-training baselines in the P0 same-budget matrix.

## Claim Support

Supported:

- Ordered restricted dense-noise factorization.
- Prefix-controllable partial denoising.
- Diagnostic separation from channel-mask and deep-synthesis ablations.

Not supported:

- Broad unconditional generation SOTA.
- General superiority over decoder-based visual tokenizers.

Machine summary:

```text
CoFiTok best lowres Frechet rows: 0/8
CoFiTok best Inception Frechet rows: 1/8
CoFiTok average rank: 3.00 lowres, 2.62 Inception
CoFiTok path AUC lower than dense: 8/8 datasets
Mean CoFiTok PSNR gain over channel-mask ablation: 1.076 dB
Deep-S_k nonzero zero-token ratio: 8/8 datasets
```

## Paper Decision

The current evidence can support a scoped top-tier-style method paper only if the main claim is:

> CoFiTok factorizes pixel-space diffusion noise prediction into ordered restricted denoising components, enabling prefix-controllable partial denoising with strong diagnostics that the ordering and restricted synthesis matter.

The current evidence does not support presenting CoFiTok as a generation-quality winner against EDM or dense epsilon predictors.

## Highest-Value Remaining Work

1. Turn the evidence report into paper tables and captions with scoped language.
2. Add final visual panels for prefix denoising, zero/random/shuffle diagnostics, and deep-S_k failure.
3. If a stronger reviewer-facing baseline story is still needed, implement MAR HF-safetensors official 50K evaluation. Keep official/eval-only rows out of the P0 generation table.
