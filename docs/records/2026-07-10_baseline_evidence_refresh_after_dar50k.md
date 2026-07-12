# Baseline Evidence Refresh After D-AR 50K - 2026-07-10

## Scope

Refreshed the paper-facing baseline evidence after:

- D-AR official ImageNet-256 50K eval-only completion.
- FlexTok/TiTok P1 tokenizer reconstruction `eval256` completion on all eight datasets.

## Artifact Updates

Updated scripts:

```text
scripts/build_paper_comparison_matrix.py
scripts/build_paper_evidence_report.py
scripts/build_paper_latex_tables.py
```

Updated generated artifacts:

```text
artifacts/reports/paper_comparison_matrix_2026-07-10_p1_eval256_guard_latest/paper_comparison_matrix_audit.md
artifacts/reports/paper_evidence_report_2026-07-10/paper_evidence_report.md
artifacts/reports/paper_evidence_report_2026-07-10/paper_claim_pack.md
artifacts/reports/paper_evidence_report_2026-07-10/latex_tables/paper_tables.tex
artifacts/reports/paper_evidence_report_2026-07-10/latex_tables/paper_tables_main.tex
artifacts/reports/paper_evidence_report_2026-07-10/latex_tables/paper_tables_appendix.tex
```

## Matrix Rule

`build_paper_comparison_matrix.py` now prioritizes the audited P0 horizontal table:

```text
artifacts/reports/p0_horizontal_table_2026-07-10_imagenet256_p0/p0_paper_table.json
```

This avoids undercounting P0 completed cells when the older summary artifact is incomplete. The refreshed matrix count is:

```text
completed=72
completed_eval_only=16
protocol_blocked=24
```

## Related-Method Rule

D-AR official ImageNet-256 50K is complete only as a secondary eval-only related-method row:

```text
FID=2.6281
sFID=6.7195
IS=285.8914
Precision=0.8029
Recall=0.5884
samples=50000
```

It is not merged into the same-budget P0 matrix. Superseding update from the same day: MAR later passed HF-safetensors generation smoke, and ReTok later passed official GPT+VQ generation smoke; both remain 50K-metric pending and secondary-only.

## Verification

Local and remote checks passed:

```text
python -m py_compile scripts/build_paper_comparison_matrix.py scripts/build_paper_evidence_report.py scripts/build_paper_latex_tables.py
pdflatex -interaction=nonstopmode -halt-on-error paper_tables_compile_check.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

`main.pdf` is 7 pages. Log scan found no `Undefined`, citation warning, `Overfull`, fatal, emergency, or error matches.
