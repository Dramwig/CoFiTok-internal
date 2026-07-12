import json
import tarfile
from pathlib import Path

import pytest

from scripts.make_remote_recovery_bundle import DEFAULT_INCLUDE_PATHS, create_recovery_bundle


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_create_recovery_bundle_writes_tar_and_manifest(tmp_path) -> None:
    project_root = tmp_path / "project"
    include_paths = [
        "CoFiTok-internal/scripts/example.py",
        "CoFiTok-internal/docs/records/record.md",
        "paper/citation_audit.md",
    ]
    for relative in include_paths:
        _write(project_root / relative, f"payload for {relative}\n")

    output = tmp_path / "bundle.tar.gz"
    manifest = create_recovery_bundle(project_root, output, include_paths=include_paths)

    assert manifest["file_count"] == len(include_paths)
    assert Path(manifest["bundle"]).exists()
    assert Path(manifest["manifest"]).exists()
    with tarfile.open(output, "r:gz") as tar:
        names = set(tar.getnames())
    for relative in include_paths:
        assert relative in names
    assert "CoFiTok-internal/artifacts/recovery/remote_recovery_manifest.json" in names

    persisted = json.loads(Path(manifest["manifest"]).read_text(encoding="utf-8"))
    assert persisted["file_count"] == len(include_paths)
    assert persisted["remote_extract_root"] == "/root/autodl-tmp/CoFiTok"
    assert any("bash -n" in check for check in persisted["post_extract_checks"])
    assert any("inspect_next_validation_queue.py" in check for check in persisted["post_extract_checks"])
    assert any("validate_idea_requirements.py" in check for check in persisted["post_extract_checks"])
    assert any("validate_dataset_conditions.py" in check for check in persisted["post_extract_checks"])
    assert any("validate_mvp_evidence.py" in check for check in persisted["post_extract_checks"])
    assert any("validate_publication_readiness.py" in check for check in persisted["post_extract_checks"])
    assert any("validate_synthesis_contract.py" in check for check in persisted["post_extract_checks"])
    assert any("validate_goal_completion.py" in check for check in persisted["post_extract_checks"])


def test_create_recovery_bundle_rejects_missing_file(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        create_recovery_bundle(
            tmp_path,
            tmp_path / "bundle.tar.gz",
            include_paths=["CoFiTok-internal/missing.txt"],
        )


def test_default_recovery_bundle_includes_remote_launcher() -> None:
    assert "CoFiTok-internal/pyproject.toml" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/scripts/remote_recovery_launch.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/tests/test_remote_recovery_launch.py" in DEFAULT_INCLUDE_PATHS


def test_default_recovery_bundle_includes_idea_requirements_matrix() -> None:
    assert "CoFiTok-internal/scripts/validate_idea_requirements.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/tests/test_validate_idea_requirements.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/artifacts/reports/idea_requirements_2026-07-08.json" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/docs/records/2026-07-08_idea_requirements_matrix.md" in DEFAULT_INCLUDE_PATHS


def test_default_recovery_bundle_includes_queue_progress_state() -> None:
    assert "CoFiTok-internal/scripts/summarize_remote_queue_progress.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/tests/test_summarize_remote_queue_progress.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/artifacts/reports/remote_queue_status_latest_2026-07-08.json" in DEFAULT_INCLUDE_PATHS
    assert (
        "CoFiTok-internal/artifacts/reports/remote_queue_progress_2026-07-08/completed_seed2_summary.json"
        in DEFAULT_INCLUDE_PATHS
    )


def test_default_recovery_bundle_includes_component_decorrelation_probe() -> None:
    assert "CoFiTok-internal/scripts/evaluate_quality.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/src/cofitok/training/losses.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/tests/test_losses.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/tests/test_train_short.py" in DEFAULT_INCLUDE_PATHS
    assert (
        "CoFiTok-internal/configs/train_cifar10_k8_denoisepath_p150_light_decor_3k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_probe.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/component_decorrelation_probe_2026-07-08/component_decorrelation_probe_summary.json"
        in DEFAULT_INCLUDE_PATHS
    )


def test_default_recovery_bundle_includes_component_decorrelation_sweep() -> None:
    assert (
        "CoFiTok-internal/configs/train_cifar10_k8_denoisepath_p150_light_decorw0005_3k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )


def test_default_recovery_bundle_includes_component_decorrelation_tiny_probe() -> None:
    assert (
        "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_5k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0005_sched2500w1500_5k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_decorw0003_sched3000w1500_5k_seed2_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_tiny_scheduled_probe.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_tiny_conservative_schedule.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_tiny_seed_confirmation.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/component_decorrelation_tiny_scheduled_probe_2026-07-08/component_decorrelation_tiny_scheduled_probe_summary.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/component_decorrelation_tiny_conservative_schedule_2026-07-08/component_decorrelation_tiny_conservative_schedule_summary.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/component_decorrelation_tiny_conservative_seed_confirm_2026-07-08/component_decorrelation_tiny_conservative_seed_confirm_summary.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert "paper/README.md" in DEFAULT_INCLUDE_PATHS
    assert "paper/citation_audit.md" in DEFAULT_INCLUDE_PATHS
    assert "paper/latex/README.md" in DEFAULT_INCLUDE_PATHS
    assert "paper/latex/Makefile" in DEFAULT_INCLUDE_PATHS
    assert "paper/latex/main.tex" in DEFAULT_INCLUDE_PATHS
    assert "paper/venues/aaai27/README.md" in DEFAULT_INCLUDE_PATHS
    assert "paper/venues/aaai27/Makefile" in DEFAULT_INCLUDE_PATHS
    assert "paper/venues/aaai27/aaai2027.sty" in DEFAULT_INCLUDE_PATHS
    assert "paper/venues/aaai27/aaai2027.bst" in DEFAULT_INCLUDE_PATHS
    assert "paper/venues/aaai27/ReproducibilityChecklist.tex" in DEFAULT_INCLUDE_PATHS
    assert "paper/venues/aaai27/main.tex" in DEFAULT_INCLUDE_PATHS
    assert "paper/venues/aaai27/main_aaai2027.tex" in DEFAULT_INCLUDE_PATHS
    assert "paper/venues/aaai27/supplementary_aaai2027.tex" in DEFAULT_INCLUDE_PATHS


def test_default_recovery_bundle_includes_component_decorrelation_scheduled_probe() -> None:
    assert (
        "CoFiTok-internal/configs/train_cifar10_k8_denoisepath_p150_light_decorw0005_sched1500w1000_3k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_scheduled_probe.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/component_decorrelation_scheduled_probe_2026-07-08/component_decorrelation_scheduled_probe_summary.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_tiny_probe.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/component_decorrelation_tiny_probe_2026-07-08/component_decorrelation_tiny_probe_summary.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/configs/train_cifar10_k8_denoisepath_p150_light_decorw001_3k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_component_decorrelation_weight_sweep.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/component_decorrelation_weight_sweep_2026-07-08/component_decorrelation_weight_sweep_summary.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/remote_queue_progress_2026-07-08/completed_seed2_summary.md"
        in DEFAULT_INCLUDE_PATHS
    )


def test_default_recovery_bundle_includes_goal_completion_audit() -> None:
    assert "CoFiTok-internal/scripts/validate_goal_completion.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/tests/test_validate_goal_completion.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/docs/records/2026-07-08_goal_completion_audit.md" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/docs/records/2026-07-08_next_step_decision.md" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/artifacts/reports/goal_completion_audit_2026-07-08.json" in DEFAULT_INCLUDE_PATHS
    assert "paper/citation_audit.md" in DEFAULT_INCLUDE_PATHS
    assert "paper/full_pdf_claim_audit_sources.json" in DEFAULT_INCLUDE_PATHS
    assert "paper/references.bib" in DEFAULT_INCLUDE_PATHS


def test_default_recovery_bundle_includes_full_validation_sweep_support() -> None:
    assert "CoFiTok-internal/src/cofitok/data/__init__.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/src/cofitok/data/registry.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/tests/test_data_registry.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/scripts/evaluate_quality.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/scripts/evaluate_generated_samples_stream.py" in DEFAULT_INCLUDE_PATHS
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_full_validation_quality_sweep.md"
        in DEFAULT_INCLUDE_PATHS
    )


def test_default_recovery_bundle_includes_multiscale_20k_backbone_validation() -> None:
    assert (
        "CoFiTok-internal/configs/train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/configs/train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_20k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/configs/train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_multiscale_20k_cuda.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_multiscale_20k_backbone_validation.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_multiscale_fair_generation_protocol.md"
        in DEFAULT_INCLUDE_PATHS
    )


def test_default_recovery_bundle_includes_downsampled_source_recheck() -> None:
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_downsampled_imagenet64_source_recheck.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert "CoFiTok-internal/docs/experiment_conditions/dataset_inventory_2026-07-08.md" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/docs/experiment_conditions/downsampled_imagenet_64_2026-07-09.md" in DEFAULT_INCLUDE_PATHS
    assert (
        "CoFiTok-internal/docs/experiment_conditions/downsampled_imagenet_64_manifest_summary_2026-07-09.json"
        in DEFAULT_INCLUDE_PATHS
    )


def test_default_recovery_bundle_includes_streamed_generated_quality_runbook() -> None:
    assert "CoFiTok-internal/scripts/evaluate_generated_samples_stream.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/scripts/export_official_fid_dirs.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/scripts/evaluate_official_fid_dirs.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/tests/test_official_fid_protocol.py" in DEFAULT_INCLUDE_PATHS
    assert (
        "CoFiTok-internal/artifacts/runbooks/generated_quality_stream_8192_2026-07-08.sh"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/runbooks/generated_quality_stream_hf_50000_2026-07-08.sh"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/runbooks/multiscale_fair_generation_2026-07-08.sh"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/runbooks/official_fid_protocol_2026-07-08.sh"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_streamed_generated_quality_8192.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_streamed_generated_quality_hf_50000.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/docs/records/2026-07-08_official_fid_protocol.md"
        in DEFAULT_INCLUDE_PATHS
    )
    assert "CoFiTok-internal/scripts/make_report_figures.py" in DEFAULT_INCLUDE_PATHS
    assert "CoFiTok-internal/tests/test_report_figures.py" in DEFAULT_INCLUDE_PATHS


def test_default_recovery_bundle_includes_multiscale_fair_generation_reports() -> None:
    assert (
        "CoFiTok-internal/artifacts/reports/multiscale_fair_generation_2026-07-08.log"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/generated_quality_stream_tiny_epsilononly_multiscale_20k_10000_ddim50_2026-07-08/generated_quality_report.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/generated_quality_stream_tiny_k8_light_multiscale_20k_10000_ddim50_2026-07-08/generated_quality_report.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/generated_quality_stream_imagenet_hf_epsilononly_multiscale_20k_50000_ddim50_2026-07-08/generated_quality_report.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/generated_quality_stream_imagenet_hf_k8_light_multiscale_20k_50000_ddim50_2026-07-08/generated_quality_report.json"
        in DEFAULT_INCLUDE_PATHS
    )


def test_default_recovery_bundle_includes_official_fid_reports() -> None:
    assert (
        "CoFiTok-internal/artifacts/reports/summary_2026-07-08/official_fid_summary.csv"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_protocol_2026-07-09.log"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_tiny_epsilononly_multiscale_20k_10000_ddim50_2026-07-08/official_fid_report.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_tiny_k8_light_multiscale_20k_10000_ddim50_2026-07-08/official_fid_report.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_imagenet_hf_epsilononly_multiscale_20k_50000_ddim50_2026-07-08/official_fid_report.json"
        in DEFAULT_INCLUDE_PATHS
    )
    assert (
        "CoFiTok-internal/artifacts/reports/official_fid_protocol_2026-07-09/official_fid_export_imagenet_hf_k8_light_multiscale_20k_50000_ddim50_2026-07-08/official_fid_report.json"
        in DEFAULT_INCLUDE_PATHS
    )
