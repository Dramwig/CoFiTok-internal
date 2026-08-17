from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cofitok.inference_replay import file_identity
from scripts import run_generation_capacity_terminal_uncertainty_waiter as waiter


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_identity(path)


def _args(tmp_path: Path) -> SimpleNamespace:
    control = tmp_path / "control"
    evaluator = tmp_path / "evaluator"
    output = tmp_path / "output"
    for path in (control, evaluator, output / "reports"):
        path.mkdir(parents=True, exist_ok=True)
    source_anchor = tmp_path / "capacity-result.json"
    _write(source_anchor, {"status": "completed"})
    real_cache = tmp_path / "real-cache.pt"
    real_cache.write_bytes(b"cache")
    return SimpleNamespace(
        project=control,
        evaluator_project=evaluator,
        source_kind="capacity_completion_100k",
        source_anchor=source_anchor,
        output_root=output,
        status_output=output / "reports" / "waiter_status.json",
        pid_file=output / "reports" / "waiter.pid",
        manifest_output=output / "reports" / "execution_manifest.json",
        qualification_output=output / "reports" / "qualification.json",
        gpu_slot_lock_target=tmp_path / "shared-gpu-slot",
        real_feature_cache_source=real_cache,
        expected_real_feature_cache_bytes=real_cache.stat().st_size,
        expected_real_feature_cache_sha256=file_identity(real_cache)["sha256"],
        python_executable=Path(__import__("sys").executable),
        expected_control_revision="1" * 40,
        expected_control_tree="2" * 40,
        expected_control_branch="analysis/capacity-uncertainty",
        expected_evaluator_revision="3" * 40,
        expected_evaluator_tree="4" * 40,
        expected_evaluator_branch="",
        expected_execution_revision="5" * 40,
        expected_execution_tree="6" * 40,
        expected_execution_branch="scale/capacity-completion",
        expected_training_revision="7" * 40,
        expected_training_tree="8" * 40,
        expected_training_branch="scale/generation-large-capacity",
        expected_result_revision="9" * 40,
        expected_result_tree="a" * 40,
        expected_result_branch="analysis/capacity-result",
        expected_source_evaluation_revision="",
        expected_source_evaluation_tree="",
        expected_source_evaluation_branch="",
        poll_seconds=0.001,
        required_idle_polls=1,
        timeout_seconds=60.0,
    )


@pytest.mark.parametrize("scientific_status", ["pass", "hold"])
def test_waiter_runs_one_capacity_audit_and_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_status: str,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        waiter.base_waiter,
        "verify_checkout",
        lambda _project, **kwargs: {
            "revision": kwargs["expected_revision"],
            "tree": kwargs["expected_tree"],
            "branch": kwargs["expected_branch"],
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        waiter.manifest_builder,
        "build_manifest",
        lambda **_kwargs: {
            "capacity_source": {
                "kind": args.source_kind,
                "anchor": file_identity(args.source_anchor),
            }
        },
    )
    manifest_record: dict[str, Any] = {}

    def validate_manifest(path: Path, **_kwargs) -> dict[str, Any]:
        manifest_record.update(
            identity=file_identity(path),
            stream_id="capacity_completion_100k_stream",
            report={
                "capacity_source": {
                    "kind": args.source_kind,
                    "anchor": file_identity(args.source_anchor),
                },
                "expected": {
                    "matched_sampling": {"sample_count": 10_000},
                },
            },
            output=args.output_root / "capacity_completion_100k" / "matched_uncertainty.json",
            cache_root=args.output_root / "feature_cache",
            cache_names={
                "real": "real-cache",
                "cofitok": "cofitok-cache",
                "dense_identity": "dense-cache",
            },
            bound_sources=[],
            start_index=0,
            end_index_exclusive=10_000,
            cofitok_checkpoint_sha256="b" * 64,
            dense_checkpoint_sha256="c" * 64,
            checkpoint_step=100_000,
            real_set_sha256="d" * 64,
        )
        return manifest_record

    monkeypatch.setattr(
        waiter.base_waiter,
        "validate_execution_manifest",
        validate_manifest,
    )
    monkeypatch.setattr(
        waiter.base_waiter,
        "prepare_real_feature_cache",
        lambda **_kwargs: {
            "status": "verified",
            "method": "hardlink",
            "claim_boundary": waiter.CLAIM_BOUNDARY,
        },
    )
    monkeypatch.setattr(waiter.base_waiter, "_process_rows", list)
    monkeypatch.setattr(waiter.base_waiter, "_gpu_compute_rows", list)
    monkeypatch.setattr(
        waiter.base_waiter,
        "audit_stage_spec",
        lambda **_kwargs: SimpleNamespace(name=args.source_kind),
    )
    audit_path = args.output_root / "audit.json"
    audit_identity = _write(audit_path, {"status": scientific_status})
    monkeypatch.setattr(
        waiter.base_waiter,
        "run_stage",
        lambda *_args, **_kwargs: {"status": "completed", "reused": False},
    )
    monkeypatch.setattr(
        waiter.base_waiter,
        "validate_audit_output",
        lambda *_args, **_kwargs: {
            "status": scientific_status,
            "decision": (
                "matched_relative_generation_advantage_supported"
                if scientific_status == "pass"
                else "matched_relative_generation_advantage_not_confirmed"
            ),
            "advantage_supported": scientific_status == "pass",
            "source": audit_identity,
            "claim_boundary": waiter.base_waiter.CLAIM_BOUNDARY,
        },
    )
    monkeypatch.setattr(
        waiter.qualification_builder,
        "build_qualification",
        lambda **_kwargs: {
            "status": scientific_status,
            "decision": (
                "matched_capacity_advantage_statistically_qualified"
                if scientific_status == "pass"
                else "matched_capacity_advantage_not_statistically_qualified"
            ),
            "claim_policy": {
                "matched_relative_generation_advantage_claim_allowed": (
                    scientific_status == "pass"
                ),
                "broad_generation_superiority_claim_allowed": False,
            },
            "claim_boundary": waiter.qualification_builder.CLAIM_BOUNDARY,
        },
    )

    assert waiter.run_waiter(args) == 0
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == scientific_status
    assert status["phase"] == "completed"
    assert status["execution_manifest"]["sample_count"] == 10_000
    assert status["stage"]["status"] == "completed"
    assert status["statistical_claim_qualification"]["status"] == scientific_status
    assert status["claim_boundary"]["broad_generation_superiority_claim_allowed"] is False
    assert not args.pid_file.exists()


def test_global_uncertainty_process_detection_is_cross_output() -> None:
    rows = [
        {
            "pid": 42,
            "command": (
                "python scripts/audit_generation_matched_uncertainty.py "
                "--output /another/output/report.json"
            ),
        },
        {"pid": 43, "command": "python unrelated.py"},
    ]
    detected = waiter.global_uncertainty_processes(rows)
    assert [row["pid"] for row in detected] == [42]


def test_claim_boundary_is_permanently_non_authorizing() -> None:
    assert waiter.CLAIM_BOUNDARY["diagnostic_non_authorizing"] is True
    assert waiter.CLAIM_BOUNDARY["training_launch_allowed"] is False
    assert waiter.CLAIM_BOUNDARY["full_300k_launch_allowed"] is False
    assert waiter.CLAIM_BOUNDARY["release_allowed"] is False
    assert waiter.CLAIM_BOUNDARY[
        "formal_large_scale_generation_advantage_claim_allowed"
    ] is False
