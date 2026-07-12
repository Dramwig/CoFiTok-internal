#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/root/autodl-tmp/CoFiTok
INTERNAL_ROOT="$PROJECT_ROOT/CoFiTok-internal"
CHECKPOINT_ROOT="$PROJECT_ROOT/checkpoints"
STATUS="$INTERNAL_ROOT/artifacts/runbooks/finalize_paper_evidence_2026-07-11.status"
LOG="$PROJECT_ROOT/checkpoints/finalize_paper_evidence_2026-07-11.log"

RELATED_DIR=artifacts/reports/baselines/official_related_methods_2026-07-11_final
P0_DIR=artifacts/reports/p0_horizontal_table_2026-07-11_dense_monolithic
MATRIX_DIR=artifacts/reports/paper_comparison_matrix_2026-07-11_final_guard
COVERAGE_DIR=artifacts/reports/protocol_coverage_2026-07-11_final
EVIDENCE_DIR=artifacts/reports/paper_evidence_report_2026-07-11_final
CLAIM_AUDIT_DIR=artifacts/reports/top_tier_claim_audit_2026-07-11_final
PARAM_AUDIT=artifacts/reports/baselines/p0_parameter_counts_2026-07-11.json
BASELINE_SUMMARY_DIR=artifacts/reports/baselines/summary_2026-07-11_final

exec > >(tee -a "$LOG") 2>&1
finish() {
  local code=$?
  if [[ "$code" == 0 ]]; then
    printf 'completed\n' > "$STATUS"
  else
    printf 'failed:%s\n' "$code" > "$STATUS"
  fi
  date --iso-8601=seconds
}
trap finish EXIT
printf 'running\n' > "$STATUS"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate pf-vlm
export PYTHONPATH="$INTERNAL_ROOT/src:$INTERNAL_ROOT:${PYTHONPATH:-}"
cd "$INTERNAL_ROOT"

python -m pytest

python scripts/baselines/build_official_related_methods_table.py \
  --project-root "$PROJECT_ROOT" \
  --output-dir "$RELATED_DIR"

python scripts/baselines/audit_p0_parameter_counts.py \
  --reports-root artifacts/reports/baselines \
  --output "$PARAM_AUDIT" \
  --strict

python scripts/baselines/summarize_baseline_reports.py \
  --reports-root artifacts/reports/baselines \
  --parameter-audit "$PARAM_AUDIT" \
  --output-dir "$BASELINE_SUMMARY_DIR"

python scripts/build_p0_paper_table.py \
  --summary artifacts/reports/summary_2026-07-11_dense_monolithic_p0/experiment_summary.json \
  --baseline-summary "$BASELINE_SUMMARY_DIR/baseline_summary.json" \
  --require-complete-primary \
  --output-dir "$P0_DIR"

python scripts/build_paper_comparison_matrix.py \
  --summary artifacts/reports/summary_2026-07-11_dense_monolithic_p0/experiment_summary.json \
  --p0-table "$P0_DIR/p0_paper_table.json" \
  --output-dir "$MATRIX_DIR"

python scripts/build_protocol_coverage_report.py \
  --matrix "$MATRIX_DIR/paper_comparison_matrix_audit.json" \
  --official-related "$RELATED_DIR/official_related_methods_table.json" \
  --require-complete-defined-protocols \
  --output-dir "$COVERAGE_DIR"

python scripts/build_long_budget_repeat_table.py \
  --reports-root "$CHECKPOINT_ROOT" \
  --output-dir artifacts/reports/long_budget_repeat_2026-07-11

python scripts/build_paper_evidence_report.py \
  --p0-table "$P0_DIR/p0_paper_table.json" \
  --p1-table artifacts/reports/baselines/p1_tokenizer_reconstruction_eval256_2026-07-10/p1_tokenizer_reconstruction_table.json \
  --matrix "$MATRIX_DIR/paper_comparison_matrix_audit.json" \
  --official-related "$RELATED_DIR/official_related_methods_table.json" \
  --long-budget artifacts/reports/long_budget_repeat_2026-07-11/long_budget_repeat_table.json \
  --imagenet256-confirmatory artifacts/reports/imagenet256_confirmatory_2026-07-11/imagenet256_confirmatory_report.json \
  --output-dir "$EVIDENCE_DIR"

python scripts/build_paper_latex_tables.py \
  --evidence "$EVIDENCE_DIR/paper_evidence_report.json" \
  --output-dir "$EVIDENCE_DIR/latex_tables"

python scripts/build_figure2_subfigures.py \
  --reports-root artifacts/reports \
  --output-dir artifacts/figures/final_visual_claim_2026-07-11

python scripts/build_top_tier_claim_audit.py \
  --evidence "$EVIDENCE_DIR/paper_evidence_report.json" \
  --coverage "$COVERAGE_DIR/protocol_coverage_report.json" \
  --output-dir "$CLAIM_AUDIT_DIR"

printf 'paper_evidence_finalized\n'
