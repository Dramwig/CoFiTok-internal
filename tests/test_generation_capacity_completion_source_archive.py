from __future__ import annotations

from pathlib import Path

from cofitok.generation.capacity_completion_decision import (
    build_capacity_completion_decision,
)
from cofitok.generation.capacity_scaling_result import (
    build_capacity_scaling_50k_result,
)
from cofitok.inference_replay import file_identity
from cofitok.reporting import file_sha256, write_json_report
from scripts.archive_generation_capacity_completion_sources import (
    build_source_archive,
)
from scripts.restore_generation_capacity_completion_sources import restore_sources
from test_generation_capacity_completion_decision import (
    DECISION_GIT,
    _decision_kwargs,
)
from test_generation_capacity_scaling_result import _kwargs as result_kwargs
from test_generation_capacity_scaling_training import _write_checkpoint


def test_capacity_completion_source_archive_survives_pruning_and_restores_names(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "capacity"
    result = build_capacity_scaling_50k_result(**result_kwargs())
    decision = build_capacity_completion_decision(
        **_decision_kwargs(result=result)
    )
    decision["selection"]["output_root"] = output_root.resolve().as_posix()
    for method in ("cofitok", "dense_identity"):
        run = output_root / f"base256_{method}"
        run.mkdir(parents=True)
        checkpoint = _write_checkpoint(
            run,
            step=50_000,
            environment_sha="3" * 64,
            dataset_sha="4" * 64,
        )
        checkpoint_path = run / checkpoint["checkpoint"]
        integrity_path = checkpoint_path.with_name(
            f"{checkpoint_path.name}.integrity.json"
        )
        decision["selection"]["resume_sources"][method] = {
            "checkpoint": file_identity(checkpoint_path),
            "checkpoint_integrity_manifest": file_identity(integrity_path),
        }
    decision_path = tmp_path / "decision.json"
    write_json_report(decision_path, decision)
    decision_sha = file_sha256(decision_path)
    report = build_source_archive(
        decision_path=decision_path,
        expected_decision_sha256=decision_sha,
        expected_decision_revision=DECISION_GIT["revision"],
        expected_decision_tree=DECISION_GIT["tree"],
        expected_decision_branch=DECISION_GIT["branch"],
        output_root=output_root,
        create_missing=True,
    )
    archive_report = output_root / "reports/archive.json"
    write_json_report(archive_report, report)
    archive_sha = file_sha256(archive_report)
    for method in ("cofitok", "dense_identity"):
        row = report["methods"][method]
        source = Path(row["source_checkpoint"]["path"])
        source_integrity = Path(row["source_integrity_manifest"]["path"])
        archive = Path(row["archive_checkpoint"]["path"])
        assert source.samefile(archive)
        source.unlink()
        source_integrity.unlink()
        assert not source.exists()
    restore_sources(
        archive_path=archive_report,
        expected_archive_sha256=archive_sha,
    )
    for method in ("cofitok", "dense_identity"):
        row = report["methods"][method]
        source = Path(row["source_checkpoint"]["path"])
        archive = Path(row["archive_checkpoint"]["path"])
        assert source.samefile(archive)
