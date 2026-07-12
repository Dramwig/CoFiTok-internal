from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any


DEFAULT_INCLUDE_PATHS = [
    "CoFiTok-internal/pyproject.toml",
    "CoFiTok-internal/scripts/validate_summary_consistency.py",
    "CoFiTok-internal/scripts/validate_mvp_evidence.py",
    "CoFiTok-internal/scripts/validate_publication_readiness.py",
    "CoFiTok-internal/scripts/validate_idea_requirements.py",
    "CoFiTok-internal/scripts/validate_dataset_conditions.py",
    "CoFiTok-internal/scripts/validate_synthesis_contract.py",
    "CoFiTok-internal/scripts/validate_goal_completion.py",
    "CoFiTok-internal/scripts/summarize_experiments.py",
    "CoFiTok-internal/scripts/train_short.py",
    "CoFiTok-internal/scripts/evaluate_checkpoint.py",
    "CoFiTok-internal/scripts/evaluate_quality.py",
    "CoFiTok-internal/scripts/evaluate_generated_samples_stream.py",
    "CoFiTok-internal/scripts/export_official_fid_dirs.py",
    "CoFiTok-internal/scripts/evaluate_official_fid_dirs.py",
    "CoFiTok-internal/scripts/summarize_remote_queue_progress.py",
    "CoFiTok-internal/scripts/inspect_next_validation_queue.py",
    "CoFiTok-internal/scripts/remote_queue_status.py",
    "CoFiTok-internal/scripts/remote_recovery_launch.py",
    "CoFiTok-internal/scripts/make_remote_recovery_bundle.py",
    "CoFiTok-internal/scripts/make_next_validation_queue.py",
    "CoFiTok-internal/scripts/make_report_figures.py",
    "CoFiTok-internal/src/cofitok/configs.py",
    "CoFiTok-internal/src/cofitok/data/__init__.py",
    "CoFiTok-internal/src/cofitok/data/registry.py",
    "CoFiTok-internal/src/cofitok/models/__init__.py",
    "CoFiTok-internal/src/cofitok/models/cofitok.py",
    "CoFiTok-internal/src/cofitok/models/predictors.py",
    "CoFiTok-internal/src/cofitok/models/synthesis.py",
    "CoFiTok-internal/src/cofitok/training/losses.py",
    "CoFiTok-internal/tests/test_validate_summary_consistency.py",
    "CoFiTok-internal/tests/test_validate_mvp_evidence.py",
    "CoFiTok-internal/tests/test_validate_publication_readiness.py",
    "CoFiTok-internal/tests/test_validate_idea_requirements.py",
    "CoFiTok-internal/tests/test_validate_dataset_conditions.py",
    "CoFiTok-internal/tests/test_validate_synthesis_contract.py",
    "CoFiTok-internal/tests/test_validate_goal_completion.py",
    "CoFiTok-internal/tests/test_inspect_next_validation_queue.py",
    "CoFiTok-internal/tests/test_remote_queue_status.py",
    "CoFiTok-internal/tests/test_remote_recovery_launch.py",
    "CoFiTok-internal/tests/test_remote_recovery_bundle.py",
    "CoFiTok-internal/tests/test_next_validation_queue.py",
    "CoFiTok-internal/tests/test_report_figures.py",
    "CoFiTok-internal/tests/test_official_fid_protocol.py",
    "CoFiTok-internal/tests/test_configs.py",
    "CoFiTok-internal/tests/test_data_registry.py",
    "CoFiTok-internal/tests/test_forward.py",
    "CoFiTok-internal/tests/test_predictors.py",
    "CoFiTok-internal/tests/test_losses.py",
    "CoFiTok-internal/tests/test_train_short.py",
    "CoFiTok-internal/tests/test_summarize_experiments.py",
    "CoFiTok-internal/tests/test_summarize_remote_queue_progress.py",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_epsilononly_p150eval_20k_seed2_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_20k_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_cleanmono_10k_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_20k_seed2_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_10k_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k4_denoisepath_p150_light_5k_seed2_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k16_denoisepath_p150_light_5k_seed2_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_deepsk_5k_seed2_cuda.json",
    "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_simultaneous_5k_seed2_cuda.json",
    "CoFiTok-internal/configs/train_cifar10_k8_denoisepath_p150_light_decor_3k_cuda.json",
    "CoFiTok-internal/configs/train_cifar10_k8_denoisepath_p150_light_decorw0005_3k_cuda.json",
    "CoFiTok-internal/configs/train_cifar10_k8_denoisepath_p150_light_decorw001_3k_cuda.json",
    "CoFiTok-internal/configs/train_cifar10_k8_denoisepath_p150_light_decorw0005_sched1500w1000_3k_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_5k_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_sched2500w1500_5k_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_cuda.json",
    "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_seed2_cuda.json",
    "CoFiTok-internal/docs/ARCHITECTURE.md",
    "CoFiTok-internal/docs/experiment_conditions/datasets_2026-07-07.md",
    "CoFiTok-internal/docs/experiment_conditions/dataset_inventory_2026-07-08.md",
    "CoFiTok-internal/docs/experiment_conditions/downsampled_imagenet_64_plan_2026-07-08.md",
    "CoFiTok-internal/docs/experiment_conditions/downsampled_imagenet_64_tfds_inspect_2026-07-08.json",
    "CoFiTok-internal/docs/experiment_conditions/downsampled_imagenet_64_2026-07-09.md",
    "CoFiTok-internal/docs/experiment_conditions/downsampled_imagenet_64_manifest_summary_2026-07-09.json",
    "CoFiTok-internal/docs/experiment_conditions/imagenet_1k_64x64_hf_plan_2026-07-08.md",
    "CoFiTok-internal/docs/experiment_conditions/imagenet_1k_64x64_hf_manifest_summary_2026-07-08.json",
    "CoFiTok-internal/docs/experiment_conditions/quality_metrics_inception_2026-07-08.md",
    "CoFiTok-internal/docs/records/2026-07-08_ablation_seed_repeats.md",
    "CoFiTok-internal/docs/records/2026-07-08_token_scaling_seed_repeats.md",
    "CoFiTok-internal/docs/records/2026-07-08_quality_seed_slice.md",
    "CoFiTok-internal/docs/records/2026-07-08_next_validation_queue.md",
    "CoFiTok-internal/docs/records/2026-07-08_idea_requirements_matrix.md",
    "CoFiTok-internal/docs/records/2026-07-08_publication_readiness_gap.md",
    "CoFiTok-internal/docs/records/2026-07-08_remote_queue_launch.md",
    "CoFiTok-internal/docs/records/2026-07-08_remote_io_incident.md",
    "CoFiTok-internal/docs/records/2026-07-08_completion_audit.md",
    "CoFiTok-internal/docs/records/2026-07-08_goal_completion_audit.md",
    "CoFiTok-internal/docs/records/2026-07-08_next_step_decision.md",
    "CoFiTok-internal/docs/records/2026-07-08_official_fid_protocol.md",
    "CoFiTok-internal/docs/records/2026-07-08_full_validation_quality_sweep.md",
    "CoFiTok-internal/docs/records/2026-07-08_multiscale_20k_backbone_validation.md",
    "CoFiTok-internal/docs/records/2026-07-08_multiscale_fair_generation_protocol.md",
    "CoFiTok-internal/docs/records/2026-07-08_downsampled_imagenet64_source_recheck.md",
    "CoFiTok-internal/docs/records/2026-07-08_streamed_generated_quality_8192.md",
    "CoFiTok-internal/docs/records/2026-07-08_streamed_generated_quality_hf_50000.md",
    "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_probe.md",
    "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_weight_sweep.md",
    "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_tiny_probe.md",
    "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_scheduled_probe.md",
    "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_tiny_scheduled_probe.md",
    "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_tiny_conservative_schedule.md",
    "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_tiny_seed_confirmation.md",
    "CoFiTok-internal/docs/reports/cofitok_mvp_report_2026-07-08.md",
    "CoFiTok-internal/artifacts/runbooks/next_validation_queue_2026-07-08.sh",
    "CoFiTok-internal/artifacts/runbooks/next_validation_queue_2026-07-08.sh.manifest.json",
    "CoFiTok-internal/artifacts/runbooks/generated_quality_stream_8192_2026-07-08.sh",
    "CoFiTok-internal/artifacts/runbooks/generated_quality_stream_hf_50000_2026-07-08.sh",
    "CoFiTok-internal/artifacts/runbooks/multiscale_fair_generation_2026-07-08.sh",
    "CoFiTok-internal/artifacts/runbooks/official_fid_protocol_2026-07-08.sh",
    "CoFiTok-internal/artifacts/reports/idea_requirements_2026-07-08.json",
    "CoFiTok-internal/artifacts/reports/publication_readiness_2026-07-08.json",
    "CoFiTok-internal/artifacts/reports/goal_completion_audit_2026-07-08.json",
    "CoFiTok-internal/artifacts/reports/remote_queue_status_latest_2026-07-08.json",
    "CoFiTok-internal/artifacts/reports/remote_queue_progress_2026-07-08/completed_seed2_summary.json",
    "CoFiTok-internal/artifacts/reports/remote_queue_progress_2026-07-08/completed_seed2_summary.md",
    "CoFiTok-internal/artifacts/reports/summary_2026-07-08/experiment_summary.json",
    "CoFiTok-internal/artifacts/reports/summary_2026-07-08/core_summary.md",
    "CoFiTok-internal/artifacts/reports/summary_2026-07-08/train_summary.csv",
    "CoFiTok-internal/artifacts/reports/summary_2026-07-08/order_eval_summary.csv",
    "CoFiTok-internal/artifacts/reports/summary_2026-07-08/quality_summary.csv",
    "CoFiTok-internal/artifacts/reports/summary_2026-07-08/sample_summary.csv",
    "CoFiTok-internal/artifacts/reports/summary_2026-07-08/generated_quality_summary.csv",
    "CoFiTok-internal/artifacts/reports/summary_2026-07-08/official_fid_summary.csv",
    "CoFiTok-internal/artifacts/reports/multiscale_fair_generation_2026-07-08.log",
    "CoFiTok-internal/artifacts/reports/generated_quality_stream_tiny_epsilononly_multiscale_20k_10000_ddim50_2026-07-08/generated_quality_report.json",
    "CoFiTok-internal/artifacts/reports/generated_quality_stream_tiny_k8_light_multiscale_20k_10000_ddim50_2026-07-08/generated_quality_report.json",
    "CoFiTok-internal/artifacts/reports/generated_quality_stream_imagenet_hf_epsilononly_multiscale_20k_50000_ddim50_2026-07-08/generated_quality_report.json",
    "CoFiTok-internal/artifacts/reports/generated_quality_stream_imagenet_hf_k8_light_multiscale_20k_50000_ddim50_2026-07-08/generated_quality_report.json",
    "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_protocol_2026-07-09.log",
    "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_tiny_epsilononly_multiscale_20k_10000_ddim50_2026-07-08/official_fid_report.json",
    "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_tiny_epsilononly_multiscale_20k_10000_ddim50_2026-07-08/official_fid_export_manifest.json",
    "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_tiny_k8_light_multiscale_20k_10000_ddim50_2026-07-08/official_fid_report.json",
    "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_tiny_k8_light_multiscale_20k_10000_ddim50_2026-07-08/official_fid_export_manifest.json",
    "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_imagenet_hf_epsilononly_multiscale_20k_50000_ddim50_2026-07-08/official_fid_report.json",
    "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_imagenet_hf_epsilononly_multiscale_20k_50000_ddim50_2026-07-08/official_fid_export_manifest.json",
    "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_imagenet_hf_k8_light_multiscale_20k_50000_ddim50_2026-07-08/official_fid_report.json",
    "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_imagenet_hf_k8_light_multiscale_20k_50000_ddim50_2026-07-08/official_fid_export_manifest.json",
    "CoFiTok-internal/artifacts/reports/component_decorrelation_probe_2026-07-08/component_decorrelation_probe_summary.json",
    "CoFiTok-internal/artifacts/reports/component_decorrelation_weight_sweep_2026-07-08/component_decorrelation_weight_sweep_summary.json",
    "CoFiTok-internal/artifacts/reports/component_decorrelation_tiny_probe_2026-07-08/component_decorrelation_tiny_probe_summary.json",
    "CoFiTok-internal/artifacts/reports/component_decorrelation_scheduled_probe_2026-07-08/component_decorrelation_scheduled_probe_summary.json",
    "CoFiTok-internal/artifacts/reports/component_decorrelation_tiny_scheduled_probe_2026-07-08/component_decorrelation_tiny_scheduled_probe_summary.json",
    "CoFiTok-internal/artifacts/reports/component_decorrelation_tiny_conservative_schedule_2026-07-08/component_decorrelation_tiny_conservative_schedule_summary.json",
    "CoFiTok-internal/artifacts/reports/component_decorrelation_tiny_conservative_seed_confirm_2026-07-08/component_decorrelation_tiny_conservative_seed_confirm_summary.json",
    "paper/README.md",
    "paper/citation_audit.md",
    "paper/full_pdf_claim_audit_sources.json",
    "paper/references.bib",
    "paper/latex/README.md",
    "paper/latex/Makefile",
    "paper/latex/main.tex",
    "paper/venues/aaai27/README.md",
    "paper/venues/aaai27/Makefile",
    "paper/venues/aaai27/aaai2027.sty",
    "paper/venues/aaai27/aaai2027.bst",
    "paper/venues/aaai27/ReproducibilityChecklist.tex",
    "paper/venues/aaai27/main.tex",
    "paper/venues/aaai27/main_aaai2027.tex",
    "paper/venues/aaai27/supplementary_aaai2027.tex",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small tarball for recovering CoFiTok remote text artifacts.")
    parser.add_argument(
        "--project-root",
        default="..",
        help="Project root containing CoFiTok-internal/ and paper/. Default assumes the script runs from CoFiTok-internal/.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/recovery/remote_recovery_2026-07-08.tar.gz",
        help="Output tar.gz path. Relative paths are resolved from CoFiTok-internal/.",
    )
    parser.add_argument(
        "--max-bytes",
        type=int,
        default=5_000_000,
        help="Maximum allowed source payload bytes before compression.",
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_bytes(manifest: dict[str, Any]) -> bytes:
    return (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _add_bytes(tar: tarfile.TarFile, arcname: str, payload: bytes) -> None:
    info = tarfile.TarInfo(arcname)
    info.size = len(payload)
    info.mtime = 0
    tar.addfile(info, BytesIO(payload))


def create_recovery_bundle(
    project_root: Path,
    output_path: Path,
    include_paths: list[str] | None = None,
    max_bytes: int = 5_000_000,
) -> dict[str, Any]:
    include_paths = include_paths or DEFAULT_INCLUDE_PATHS
    project_root = project_root.resolve()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    total_bytes = 0
    for relative in include_paths:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError(f"Unsafe include path: {relative}")
        source = project_root / relative
        if not source.is_file():
            raise FileNotFoundError(f"Missing recovery-bundle source: {source}")
        size = source.stat().st_size
        total_bytes += size
        records.append(
            {
                "path": relative,
                "bytes": size,
                "sha256": _sha256(source),
            }
        )

    if total_bytes > max_bytes:
        raise ValueError(f"Recovery bundle source payload is {total_bytes} bytes, above max {max_bytes}")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": project_root.as_posix(),
        "remote_extract_root": "/root/autodl-tmp/CoFiTok",
        "file_count": len(records),
        "total_source_bytes": total_bytes,
        "files": records,
        "recommended_extract_command": (
            "tar -xzf remote_recovery_2026-07-08.tar.gz -C /root/autodl-tmp/CoFiTok"
        ),
        "post_extract_checks": [
            "cd /root/autodl-tmp/CoFiTok/CoFiTok-internal",
            "bash -n artifacts/runbooks/next_validation_queue_2026-07-08.sh",
            "bash -n artifacts/runbooks/generated_quality_stream_hf_50000_2026-07-08.sh",
            "bash -n artifacts/runbooks/multiscale_fair_generation_2026-07-08.sh",
            "bash -n artifacts/runbooks/official_fid_protocol_2026-07-08.sh",
            "PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python scripts/inspect_next_validation_queue.py --checkpoint-root /root/autodl-tmp/CoFiTok/checkpoints --output artifacts/reports/next_validation_queue_status_recovered_2026-07-08.json",
            "PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python scripts/validate_dataset_conditions.py --conditions-dir docs/experiment_conditions --require-ok",
            "PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python scripts/validate_summary_consistency.py --summary-dir artifacts/reports/summary_2026-07-08 --doc docs/reports/cofitok_mvp_report_2026-07-08.md --doc docs/records/2026-07-08_completion_audit.md",
            "PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python scripts/validate_idea_requirements.py --project-root . --summary artifacts/reports/summary_2026-07-08/experiment_summary.json --queue-manifest artifacts/runbooks/next_validation_queue_2026-07-08.sh.manifest.json --output-json artifacts/reports/idea_requirements_2026-07-08.json --output-md docs/records/2026-07-08_idea_requirements_matrix.md --require-mvp-ok",
            "PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python scripts/validate_mvp_evidence.py --summary artifacts/reports/summary_2026-07-08/experiment_summary.json",
            "PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python scripts/validate_publication_readiness.py --summary artifacts/reports/summary_2026-07-08/experiment_summary.json --output-json artifacts/reports/publication_readiness_2026-07-08.json --output-md docs/records/2026-07-08_publication_readiness_gap.md",
            "PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python scripts/validate_synthesis_contract.py --config-glob 'configs/*.json'",
            "PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python scripts/validate_goal_completion.py --require-complete --output-json artifacts/reports/goal_completion_audit_2026-07-08.json --output-md docs/records/2026-07-08_goal_completion_audit.md",
            "PYTHONPATH=src /root/autodl-tmp/conda/envs/pf-vlm/bin/python -m pytest -q",
        ],
    }

    with tarfile.open(output_path, "w:gz") as tar:
        for record in records:
            tar.add(project_root / record["path"], arcname=record["path"])
        _add_bytes(
            tar,
            "CoFiTok-internal/artifacts/recovery/remote_recovery_manifest.json",
            _manifest_bytes(manifest),
        )

    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    manifest_path.write_bytes(_manifest_bytes({**manifest, "bundle": output_path.as_posix()}))
    return {**manifest, "bundle": output_path.as_posix(), "manifest": manifest_path.as_posix()}


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    manifest = create_recovery_bundle(
        project_root=Path(args.project_root),
        output_path=output_path,
        max_bytes=args.max_bytes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
