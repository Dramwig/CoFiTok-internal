from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from cofitok.inference_replay import file_identity
from scripts import run_generation_versioned_terminal_claim_chain as chain


CONTROL_GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/test-versioned-terminal-chain",
    "tracked_dirty": False,
}
QUALITY_GIT = {
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "scale/test-quality",
    "tracked_dirty": False,
}


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return file_identity(path)


def _source_sha(project: Path, name: str) -> str:
    return file_identity(project / "scripts" / name)["sha256"]


def _args(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Namespace:
    project = Path(__file__).resolve().parents[1]
    quality_project = tmp_path / "quality-project"
    quality_project.mkdir()
    quality_root = tmp_path / "quality"
    uncertainty_root = tmp_path / "uncertainty"
    uncertainty_root.mkdir()

    quality_path = quality_root / "reports" / "quality_bridge_result.json"
    quality_identity = _write(quality_path, {"status": "completed"})
    exposure_path = quality_root / "reports" / "exposure.json"
    expected_training_git = {
        **QUALITY_GIT,
        "path": quality_project.resolve().as_posix(),
    }
    exposure_identity = _write(
        exposure_path,
        {
            "schema_version": 1,
            "role": chain.EXPOSURE_ROLE,
            "status": "pass",
            "terminal_binding": {
                "terminal_result_binding_verified": True,
                "training_git": expected_training_git,
                "authoritative_terminal_verification": {
                    "status": "verified",
                    "quality_project": expected_training_git,
                    "verifier_output": {
                        "result": {"sha256": quality_identity["sha256"]}
                    },
                },
            },
        },
    )
    followup_path = quality_root / "reports" / "followup.json"
    followup_identity = _write(
        followup_path,
        {
            "schema_version": 2,
            "role": chain.FOLLOWUP_ROLE,
            "status": "completed",
            "source_reports": {"terminal_training_exposure": exposure_identity},
            "training_exposure": {"source_report": exposure_identity},
            "quality_bridge_execution_git": {
                "revision": QUALITY_GIT["revision"],
                "branch": QUALITY_GIT["branch"],
                "tracked_dirty": False,
            },
            "recommended_next_stage": {
                "execution_ready": False,
                "gpu_execution_allowed": False,
                "full_300k_launch_allowed": False,
            },
            "authorization_boundary": {
                "recommended_stage_execution_allowed": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "release_authorization_allowed": False,
            },
        },
    )
    manifest_path = uncertainty_root / "reports" / "manifest.json"
    manifest_identity = _write(manifest_path, {"role": "manifest"})
    uncertainty_path = uncertainty_root / "terminal" / "uncertainty.json"
    uncertainty_identity = _write(uncertainty_path, {"role": "uncertainty"})
    visual_path = quality_root / "reports" / "visual-status.json"
    visual_identity = _write(visual_path, {"status": "completed"})
    runtime_guard_path = quality_root / "reports" / "runtime-guard.json"
    runtime_guard_identity = _write(runtime_guard_path, {"status": "pass"})
    runtime_status_path = quality_root / "reports" / "runtime-status.json"
    runtime_status_identity = _write(
        runtime_status_path,
        {
            "schema_version": 1,
            "role": chain.RUNTIME_WAITER_ROLE,
            "status": "pass",
            "phase": "completed",
            "runtime_claim_guard": {"identity": runtime_guard_identity},
        },
    )

    monkeypatch.setattr(
        chain,
        "_git_identity",
        lambda path: QUALITY_GIT if path == quality_project.resolve() else CONTROL_GIT,
    )
    return Namespace(
        project=project,
        quality_project=quality_project,
        quality_output_root=quality_root,
        terminal_uncertainty_output_root=uncertainty_root,
        quality_result=quality_path,
        expected_quality_result_sha256=quality_identity["sha256"],
        authoritative_exposure=exposure_path,
        expected_authoritative_exposure_sha256=exposure_identity["sha256"],
        authoritative_followup=followup_path,
        expected_authoritative_followup_sha256=followup_identity["sha256"],
        execution_manifest=manifest_path,
        expected_execution_manifest_sha256=manifest_identity["sha256"],
        uncertainty_report=uncertainty_path,
        expected_uncertainty_report_sha256=uncertainty_identity["sha256"],
        visual_audit_waiter_status=visual_path,
        expected_visual_audit_waiter_status_sha256=visual_identity["sha256"],
        runtime_claim_guard=runtime_guard_path,
        expected_runtime_claim_guard_sha256=runtime_guard_identity["sha256"],
        runtime_waiter_status=runtime_status_path,
        expected_runtime_waiter_status_sha256=runtime_status_identity["sha256"],
        expected_control_revision=CONTROL_GIT["revision"],
        expected_control_tree=CONTROL_GIT["tree"],
        expected_control_branch=CONTROL_GIT["branch"],
        expected_runner_source_sha256=_source_sha(
            project, "run_generation_versioned_terminal_claim_chain.py"
        ),
        expected_qualification_builder_sha256=_source_sha(
            project,
            "build_generation_quality_bridge_statistical_claim_qualification.py",
        ),
        expected_language_builder_sha256=_source_sha(
            project, "build_generation_statistical_claim_language_guard.py"
        ),
        expected_terminal_builder_sha256=_source_sha(
            project, "build_generation_terminal_system_claim_guard.py"
        ),
        expected_quality_revision=QUALITY_GIT["revision"],
        expected_quality_tree=QUALITY_GIT["tree"],
        expected_quality_branch=QUALITY_GIT["branch"],
        expected_evaluator_revision="e" * 40,
        expected_evaluator_branch="",
    )


def _mock_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    qualification = {
        "schema_version": 1,
        "role": "generation_quality_bridge_statistical_claim_qualification",
        "status": "hold",
        "decision": "matched_quality_bridge_fid_advantage_not_statistically_qualified",
    }
    monkeypatch.setattr(
        chain.qualification_builder,
        "build_qualification",
        lambda **_kwargs: qualification,
    )
    monkeypatch.setattr(
        chain.qualification_builder,
        "build_fail_closed_qualification",
        lambda **_kwargs: qualification,
    )
    monkeypatch.setattr(
        chain.language_builder,
        "build_guard",
        lambda **_kwargs: {
            "schema_version": 1,
            "role": "generation_statistical_claim_language_guard",
            "status": "hold",
            "decision": "matched_distribution_quality_claim_not_supported",
        },
    )
    monkeypatch.setattr(
        chain.terminal_builder,
        "build_guard",
        lambda **_kwargs: {
            "schema_version": 1,
            "role": "generation_terminal_system_claim_guard",
            "status": "hold",
            "decision": (
                "terminal_system_evidence_complete_without_qualified_matched_advantage"
            ),
        },
    )


def test_chain_writes_new_versioned_outputs_and_preserves_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    _mock_builders(monkeypatch)

    assert chain.run_locked(args, require_detached=False) == 0

    paths = chain._output_paths(args.quality_output_root.resolve())
    status = json.loads(paths["terminal_status"].read_text(encoding="utf-8"))
    receipt = json.loads(paths["deployment"].read_text(encoding="utf-8"))
    assert paths["qualification"].is_file()
    assert paths["language"].is_file()
    assert paths["terminal"].is_file()
    assert status["status"] == "completed"
    assert status["guard_status"] == "hold"
    assert status["authorization_boundary"] == chain.TERMINAL_WAITER_BOUNDARY
    assert receipt["scientific_status"] == "hold"
    assert receipt["generation_advantage_proven"] is False
    assert all(
        value is False
        for name, value in receipt["scope"].items()
        if name.endswith("_allowed")
    )


def test_chain_rejects_an_executing_source_outside_the_bound_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    _mock_builders(monkeypatch)
    monkeypatch.setattr(chain, "__file__", str(tmp_path / "unbound_runner.py"))

    with pytest.raises(ValueError, match="executing source is outside"):
        chain.run_locked(args, require_detached=False)

    paths = chain._output_paths(args.quality_output_root.resolve())
    assert not paths["qualification"].exists()
    assert not paths["terminal"].exists()


def test_chain_uses_fail_closed_builder_when_uncertainty_report_is_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    _mock_builders(monkeypatch)
    args.uncertainty_report = None
    args.expected_uncertainty_report_sha256 = None
    called = {"fail_closed": 0}

    def build_fail_closed(**_kwargs):
        called["fail_closed"] += 1
        return {
            "schema_version": 1,
            "role": "generation_quality_bridge_statistical_claim_qualification",
            "status": "hold",
            "decision": (
                "matched_quality_bridge_fid_advantage_not_statistically_qualified"
            ),
        }

    monkeypatch.setattr(
        chain.qualification_builder,
        "build_fail_closed_qualification",
        build_fail_closed,
    )
    monkeypatch.setattr(
        chain.qualification_builder,
        "build_qualification",
        lambda **_kwargs: pytest.fail("full uncertainty builder must not run"),
    )

    assert chain.run_locked(args, require_detached=False) == 0
    assert called["fail_closed"] == 1


def test_chain_rejects_followup_that_does_not_bind_authoritative_exposure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    _mock_builders(monkeypatch)
    followup = json.loads(args.authoritative_followup.read_text(encoding="utf-8"))
    followup["training_exposure"]["source_report"]["sha256"] = "0" * 64
    args.expected_authoritative_followup_sha256 = _write(
        args.authoritative_followup,
        followup,
    )["sha256"]

    with pytest.raises(ValueError, match="follow-up decision binding differs"):
        chain.run_locked(args, require_detached=False)

    paths = chain._output_paths(args.quality_output_root.resolve())
    assert not paths["qualification"].exists()
    assert not paths["terminal"].exists()


def test_runtime_status_must_bind_exact_runtime_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path, monkeypatch)
    _mock_builders(monkeypatch)
    status = json.loads(args.runtime_waiter_status.read_text(encoding="utf-8"))
    status["runtime_claim_guard"]["identity"]["sha256"] = "f" * 64
    args.expected_runtime_waiter_status_sha256 = _write(
        args.runtime_waiter_status,
        status,
    )["sha256"]

    with pytest.raises(ValueError, match="runtime-strict waiter binding differs"):
        chain.run_locked(args, require_detached=False)
