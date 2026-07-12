from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_DATE_TAG = "2026-07-08"


@dataclass(frozen=True)
class ValidationRun:
    label: str
    config: str
    train_id: str
    order_eval: bool
    sample_eval: bool = True


NEXT_VALIDATION_RUNS = [
    ValidationRun(
        label="tiny_epsilononly_20k_seed2",
        config="configs/train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda.json",
        train_id="train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda",
        order_eval=False,
    ),
    ValidationRun(
        label="tiny_k8_light_20k_seed2",
        config="configs/train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda.json",
        train_id="train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda",
        order_eval=True,
    ),
    ValidationRun(
        label="imagenet_hf_epsilononly_20k_seed2",
        config="configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda.json",
        train_id="train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda",
        order_eval=False,
    ),
    ValidationRun(
        label="imagenet_hf_k8_light_20k_seed2",
        config="configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda.json",
        train_id="train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda",
        order_eval=True,
    ),
    ValidationRun(
        label="tiny_k8_light_multiscale_10k",
        config="configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda.json",
        train_id="train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda",
        order_eval=True,
    ),
    ValidationRun(
        label="imagenet_hf_k8_light_multiscale_10k",
        config="configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda.json",
        train_id="train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda",
        order_eval=True,
    ),
    ValidationRun(
        label="tiny_k8_light_nopathprefix_10k",
        config="configs/train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json",
        train_id="train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_cuda",
        order_eval=True,
        sample_eval=False,
    ),
    ValidationRun(
        label="tiny_k8_light_cleanmono_10k",
        config="configs/train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_10k_cuda.json",
        train_id="train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_10k_cuda",
        order_eval=True,
        sample_eval=False,
    ),
    ValidationRun(
        label="imagenet_hf_k8_light_nopathprefix_10k",
        config="configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json",
        train_id="train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_10k_cuda",
        order_eval=True,
        sample_eval=False,
    ),
    ValidationRun(
        label="imagenet_hf_k8_light_cleanmono_10k",
        config="configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_cuda.json",
        train_id="train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_cuda",
        order_eval=True,
        sample_eval=False,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write the next remote CoFiTok validation queue as a bash runbook.")
    parser.add_argument(
        "--output",
        default=f"artifacts/runbooks/next_validation_queue_{DEFAULT_DATE_TAG}.sh",
        help="Output bash script path, relative to CoFiTok-internal unless absolute.",
    )
    parser.add_argument("--date-tag", default=DEFAULT_DATE_TAG)
    parser.add_argument("--quality-images", type=int, default=1024)
    parser.add_argument("--quality-batches", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--real-count", type=int, default=8192)
    return parser.parse_args()


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")


def _bash_header(
    date_tag: str,
    quality_images: int,
    quality_batches: int,
    sample_count: int,
    sample_steps: int,
    real_count: int,
) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${{PROJECT_ROOT:-/root/autodl-tmp/CoFiTok}}"
CODE_DIR="${{CODE_DIR:-$PROJECT_ROOT/CoFiTok-internal}}"
CHECKPOINT_ROOT="${{CHECKPOINT_ROOT:-$PROJECT_ROOT/checkpoints}}"
PYTHON="${{PYTHON:-/root/autodl-tmp/conda/envs/pf-vlm/bin/python}}"
DATE_TAG="${{DATE_TAG:-{date_tag}}}"
QUALITY_IMAGES="${{QUALITY_IMAGES:-{quality_images}}}"
QUALITY_BATCHES="${{QUALITY_BATCHES:-{quality_batches}}}"
SAMPLE_COUNT="${{SAMPLE_COUNT:-{sample_count}}}"
SAMPLE_STEPS="${{SAMPLE_STEPS:-{sample_steps}}}"
REAL_COUNT="${{REAL_COUNT:-{real_count}}}"
SAMPLE_BATCH_SIZE="${{SAMPLE_BATCH_SIZE:-64}}"
QUALITY_BATCH_SIZE="${{QUALITY_BATCH_SIZE:-32}}"
REQUIRE_PUBLICATION_READY="${{REQUIRE_PUBLICATION_READY:-1}}"

cd "$CODE_DIR"
export PYTHONPATH=src
export TORCH_HOME="${{TORCH_HOME:-$CHECKPOINT_ROOT/torch_cache}}"
mkdir -p "$CHECKPOINT_ROOT" "$TORCH_HOME" artifacts/reports

if [[ ! -x "$PYTHON" ]]; then
  echo "Configured PYTHON is not executable: $PYTHON" >&2
  exit 2
fi

write_queue_status() {{
  local phase="$1"
  "$PYTHON" scripts/inspect_next_validation_queue.py \\
    --checkpoint-root "$CHECKPOINT_ROOT" \\
    --date-tag "$DATE_TAG" \\
    --quality-images "$QUALITY_IMAGES" \\
    --sample-count "$SAMPLE_COUNT" \\
    --sample-steps "$SAMPLE_STEPS" \\
    --output "artifacts/reports/next_validation_queue_status_${{phase}}_${{DATE_TAG}}.json"
}}

run_if_missing() {{
  local marker="$1"
  shift
  if [[ -f "$marker" ]]; then
    echo "[skip] $marker"
  else
    echo "[run] $*"
    "$@"
  fi
}}

run_suite() {{
  local label="$1"
  local config="$2"
  local train_id="$3"
  local order_eval="$4"
  local sample_eval="$5"
  local train_dir="$CHECKPOINT_ROOT/${{train_id}}_${{DATE_TAG}}"
  local checkpoint="$train_dir/checkpoint_final.pt"

  run_if_missing "$train_dir/report.json" \\
    "$PYTHON" scripts/train_short.py \\
      --config "$config" \\
      --output-dir "$train_dir"

  local quality_id="quality_${{label}}_${{QUALITY_IMAGES}}_t500_lpips_inception_${{DATE_TAG}}"
  run_if_missing "$CHECKPOINT_ROOT/$quality_id/quality_report.json" \\
    "$PYTHON" scripts/evaluate_quality.py \\
      --config "$config" \\
      --checkpoint "$checkpoint" \\
      --output-dir "$CHECKPOINT_ROOT/$quality_id" \\
      --split val \\
      --max-batches "$QUALITY_BATCHES" \\
      --max-images "$QUALITY_IMAGES" \\
      --timestep 500 \\
      --enable-lpips \\
      --enable-inception-fid

  if [[ "$order_eval" == "1" ]]; then
    for component_order in ordered random reverse; do
      local order_id="order_${{label}}_${{component_order}}_${{DATE_TAG}}"
      run_if_missing "$CHECKPOINT_ROOT/$order_id/report.json" \\
        "$PYTHON" scripts/evaluate_checkpoint.py \\
          --config "$config" \\
          --checkpoint "$checkpoint" \\
          --output-dir "$CHECKPOINT_ROOT/$order_id" \\
          --component-order "$component_order" \\
          --random-order-seed 0
    done
  fi

  if [[ "$sample_eval" == "1" ]]; then
    local sample_id="generated_${{label}}_${{SAMPLE_COUNT}}_ddim${{SAMPLE_STEPS}}_${{DATE_TAG}}"
    run_if_missing "$CHECKPOINT_ROOT/$sample_id/sample_report.json" \\
      "$PYTHON" scripts/sample_checkpoint.py \\
        --config "$config" \\
        --checkpoint "$checkpoint" \\
        --output-dir "$CHECKPOINT_ROOT/$sample_id" \\
        --num-samples "$SAMPLE_COUNT" \\
        --batch-size "$SAMPLE_BATCH_SIZE" \\
        --sample-steps "$SAMPLE_STEPS" \\
        --prefix-budgets 8 \\
        --save-images

    local generated_quality_id="generated_quality_${{label}}_${{SAMPLE_COUNT}}_ddim${{SAMPLE_STEPS}}_${{DATE_TAG}}"
    run_if_missing "$CHECKPOINT_ROOT/$generated_quality_id/generated_quality_report.json" \\
      "$PYTHON" scripts/evaluate_generated_samples.py \\
        --config "$config" \\
        --samples-dir "$CHECKPOINT_ROOT/$sample_id/samples_prefix_8" \\
        --output-dir "$CHECKPOINT_ROOT/$generated_quality_id" \\
        --split val \\
        --max-real-images "$REAL_COUNT" \\
        --max-sample-images "$SAMPLE_COUNT" \\
        --batch-size "$QUALITY_BATCH_SIZE" \\
        --enable-inception-fid
  fi
}}

write_queue_status "start"
"""


def _bash_footer() -> str:
    return """
"$PYTHON" scripts/summarize_experiments.py \\
  --reports-root "$CHECKPOINT_ROOT" \\
  --output-dir artifacts/reports/summary_2026-07-08

write_queue_status "final"

"$PYTHON" scripts/validate_dataset_conditions.py \\
  --conditions-dir docs/experiment_conditions \\
  --require-ok

"$PYTHON" scripts/validate_summary_consistency.py \\
  --summary-dir artifacts/reports/summary_2026-07-08 \\
  --doc docs/reports/cofitok_mvp_report_2026-07-08.md \\
  --doc docs/records/2026-07-08_completion_audit.md

"$PYTHON" scripts/validate_idea_requirements.py \\
  --project-root . \\
  --summary artifacts/reports/summary_2026-07-08/experiment_summary.json \\
  --queue-manifest artifacts/runbooks/next_validation_queue_2026-07-08.sh.manifest.json \\
  --output-json artifacts/reports/idea_requirements_2026-07-08.json \\
  --output-md docs/records/2026-07-08_idea_requirements_matrix.md \\
  --require-mvp-ok

"$PYTHON" scripts/validate_mvp_evidence.py \\
  --summary artifacts/reports/summary_2026-07-08/experiment_summary.json

if [[ "$REQUIRE_PUBLICATION_READY" == "1" ]]; then
  "$PYTHON" scripts/validate_publication_readiness.py \\
    --summary artifacts/reports/summary_2026-07-08/experiment_summary.json \\
    --output-json artifacts/reports/publication_readiness_2026-07-08.json \\
    --output-md docs/records/2026-07-08_publication_readiness_gap.md \\
    --require-ready
else
  "$PYTHON" scripts/validate_publication_readiness.py \\
    --summary artifacts/reports/summary_2026-07-08/experiment_summary.json \\
    --output-json artifacts/reports/publication_readiness_2026-07-08.json \\
    --output-md docs/records/2026-07-08_publication_readiness_gap.md
fi

"$PYTHON" scripts/validate_synthesis_contract.py \\
  --config-glob 'configs/*.json'

"$PYTHON" -m pytest -q
"""


def render_queue(
    date_tag: str = DEFAULT_DATE_TAG,
    quality_images: int = 1024,
    quality_batches: int = 64,
    sample_count: int = 2048,
    sample_steps: int = 50,
    real_count: int = 8192,
) -> str:
    for name, value in [
        ("quality_images", quality_images),
        ("quality_batches", quality_batches),
        ("sample_count", sample_count),
        ("sample_steps", sample_steps),
        ("real_count", real_count),
    ]:
        _validate_positive(name, value)

    lines = [
        _bash_header(
            date_tag=date_tag,
            quality_images=quality_images,
            quality_batches=quality_batches,
            sample_count=sample_count,
            sample_steps=sample_steps,
            real_count=real_count,
        ).rstrip(),
    ]
    for run in NEXT_VALIDATION_RUNS:
        order_flag = "1" if run.order_eval else "0"
        sample_flag = "1" if run.sample_eval else "0"
        lines.append(f'run_suite "{run.label}" "{run.config}" "{run.train_id}" "{order_flag}" "{sample_flag}"')
    lines.append(_bash_footer().rstrip())
    return "\n\n".join(lines) + "\n"


def write_queue(
    output_path: Path,
    date_tag: str = DEFAULT_DATE_TAG,
    quality_images: int = 1024,
    quality_batches: int = 64,
    sample_count: int = 2048,
    sample_steps: int = 50,
    real_count: int = 8192,
) -> dict[str, Any]:
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = render_queue(
        date_tag=date_tag,
        quality_images=quality_images,
        quality_batches=quality_batches,
        sample_count=sample_count,
        sample_steps=sample_steps,
        real_count=real_count,
    )
    output_path.write_text(script, encoding="utf-8", newline="\n")
    manifest = {
        "script": output_path.as_posix(),
        "date_tag": date_tag,
        "quality_images": quality_images,
        "quality_batches": quality_batches,
        "sample_count": sample_count,
        "sample_steps": sample_steps,
        "real_count": real_count,
        "run_count": len(NEXT_VALIDATION_RUNS),
        "train_count": len(NEXT_VALIDATION_RUNS),
        "quality_count": len(NEXT_VALIDATION_RUNS),
        "order_eval_count": sum(3 for run in NEXT_VALIDATION_RUNS if run.order_eval),
        "sampling_count": sum(1 for run in NEXT_VALIDATION_RUNS if run.sample_eval),
        "generated_quality_count": sum(1 for run in NEXT_VALIDATION_RUNS if run.sample_eval),
        "runs": [asdict(run) for run in NEXT_VALIDATION_RUNS],
    }
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**manifest, "manifest": manifest_path.as_posix()}


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    manifest = write_queue(
        output_path=output_path,
        date_tag=args.date_tag,
        quality_images=args.quality_images,
        quality_batches=args.quality_batches,
        sample_count=args.sample_count,
        sample_steps=args.sample_steps,
        real_count=args.real_count,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
