from __future__ import annotations

import copy
import json
import subprocess
from argparse import Namespace
from pathlib import Path

import pytest

from cofitok.inference_replay import file_identity
from scripts.build_generation_terminal_runtime_strict_conjunct import (
    AUTHORIZATION_BOUNDARY,
    TERMINAL_BOUNDARY_REQUIRED,
    build_conjunct,
)
from scripts import wait_generation_terminal_runtime_strict_conjunct as waiter


ROOT = Path(__file__).resolve().parents[1]
TRAINING_REVISION = "c" * 40
TRAINING_BRANCH = "scale/generation-stability-quality-bridge-100k"


def _write(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_identity(path)


def _terminal_audit(path: Path, *, status: str = "pass", advantage: bool = True) -> dict:
    return _write(
        path,
        {
            "schema_version": 1,
            "role": "generation_quality_bridge_terminal_completion_audit",
            "status": "pass",
            "detail": "terminal_quality_bridge_evidence_physically_replayed",
            "terminal_status": status,
            "terminal_decision": (
                "matched_quality_advantage_qualified_with_terminal_system_evidence"
                if status == "pass"
                else "terminal_system_evidence_complete_without_qualified_matched_advantage"
            ),
            "generation_advantage_proven": advantage,
            "scope": {
                "training_git": {
                    "revision": TRAINING_REVISION,
                    "branch": TRAINING_BRANCH,
                    "tracked_dirty": False,
                }
            },
            "claim_policy": {
                "terminal_system_evidence_complete": True,
                "matched_distribution_quality_claim_allowed": advantage,
                "absolute_usability_claim_allowed": False,
                "broad_generation_superiority_claim_allowed": False,
                "sota_claim_allowed": False,
                "larger_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_or_release_allowed": False,
            },
            "authorization_boundary": copy.deepcopy(TERMINAL_BOUNDARY_REQUIRED),
        },
    )


def _terminal_status(
    path: Path,
    audit_identity: dict,
    *,
    status: str = "pass",
    advantage: bool = True,
) -> dict:
    decision = (
        "matched_quality_advantage_qualified_with_terminal_system_evidence"
        if status == "pass"
        else "terminal_system_evidence_complete_without_qualified_matched_advantage"
    )
    return _write(
        path,
        {
            "schema_version": 1,
            "role": "generation_quality_bridge_terminal_completion_audit_waiter",
            "status": "pass",
            "detail": "terminal_completion_audit_revalidated",
            "pid": 752882,
            "terminal_status": status,
            "generation_advantage_proven": advantage,
            "audit": {
                "identity": audit_identity,
                "terminal_status": status,
                "terminal_decision": decision,
                "generation_advantage_proven": advantage,
            },
            "scope": {
                "cpu_only_evidence_replay": True,
                "diagnostic_non_authorizing": True,
                "gpu_required": False,
                "training_launch_allowed": False,
                "sampling_launch_allowed": False,
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "promotion_authorization_allowed": False,
                "release_authorization_allowed": False,
                "inference_export_authorization_allowed": False,
                "process_signals_allowed": False,
                "upstream_decisions_modified": False,
            },
        },
    )


def _runtime_comparison(path: Path, *, canonical_trusted: bool = True) -> dict:
    decision = (
        "canonical_observational_runtime_claim_semantically_verified"
        if canonical_trusted
        else "canonical_runtime_claim_rejected_strict_guard_controls"
    )
    return _write(
        path,
        {
            "schema_version": 1,
            "role": "generation_runtime_claim_guard_strict_comparison",
            "status": "pass",
            "decision": decision,
            "training_identity": {
                "revision": TRAINING_REVISION,
                "branch": TRAINING_BRANCH,
            },
            "strict_policy": {
                "physical_lower_bound_methods": ["cofitok", "dense_identity"],
                "elapsed_metric_role": "observational_physical_lower_bounds_only",
                "direct_ranking_allowed": False,
            },
            "claim_policy": {
                "canonical_runtime_claim_trusted": canonical_trusted,
                "strict_runtime_claim_guard_required": True,
                "adjusted_elapsed_point_estimate_reporting_allowed": True,
                "physical_lower_bound_label_required": True,
                "observational_only_label_required": True,
                "training_wall_clock_direct_comparison_allowed": False,
                "training_throughput_direct_comparison_allowed": False,
                "cost_efficiency_ranking_allowed": False,
                "equal_wall_clock_budget_claim_allowed": False,
                "equal_gpu_hours_budget_claim_allowed": False,
                "equal_training_flops_budget_claim_allowed": False,
                "quality_or_generation_advantage_claim_allowed": False,
                "strict_recovery_binding_required": not canonical_trusted,
                "strict_recovery_binding_verified": not canonical_trusted,
            },
            "claim_boundary": {
                "diagnostic_non_authorizing": True,
                "gpu_execution_allowed": False,
                "training_launch_allowed": False,
                "sampling_launch_allowed": False,
                "inference_export_authorization_allowed": False,
                "release_authorization_allowed": False,
                "process_signals_allowed": False,
                "full_300k_launch_allowed": False,
            },
        },
    )


def _runtime_status(
    path: Path,
    comparison_identity: dict,
    *,
    canonical_trusted: bool = True,
) -> dict:
    decision = (
        "canonical_observational_runtime_claim_semantically_verified"
        if canonical_trusted
        else "canonical_runtime_claim_rejected_strict_guard_controls"
    )
    policy = {
        "canonical_runtime_claim_trusted": canonical_trusted,
        "cost_efficiency_ranking_allowed": False,
    }
    return _write(
        path,
        {
            "schema_version": 1,
            "role": "generation_runtime_claim_guard_strict_comparison_waiter",
            "status": "pass",
            "detail": decision,
            "pid": 895548,
            "comparison": {
                "identity": comparison_identity,
                "status": "pass",
                "decision": decision,
                "claim_policy": policy,
            },
            "scope": {
                "cpu_only": True,
                "non_authorizing": True,
                "gpu_execution_allowed": False,
                "checkpoint_payload_loading_allowed": False,
                "training_process_signals_allowed": False,
                "unrelated_process_signals_allowed": False,
                "source_waiter_signals_allowed": False,
                "promotion_authorization_allowed": False,
                "release_authorization_allowed": False,
                "full_300k_launch_allowed": False,
            },
        },
    )


def _sources(tmp_path: Path) -> dict:
    terminal_audit_path = tmp_path / "terminal_completion_audit.json"
    terminal_status_path = tmp_path / "terminal_completion_status.json"
    runtime_path = tmp_path / "runtime_claim_guard_comparison.json"
    runtime_status_path = tmp_path / "runtime_claim_guard_comparison_status.json"
    terminal_audit_identity = _terminal_audit(terminal_audit_path)
    terminal_status_identity = _terminal_status(
        terminal_status_path,
        terminal_audit_identity,
    )
    runtime_identity = _runtime_comparison(runtime_path)
    runtime_status_identity = _runtime_status(runtime_status_path, runtime_identity)
    return {
        "terminal_audit_path": terminal_audit_path,
        "terminal_audit_identity": terminal_audit_identity,
        "terminal_status_path": terminal_status_path,
        "terminal_status_identity": terminal_status_identity,
        "runtime_path": runtime_path,
        "runtime_identity": runtime_identity,
        "runtime_status_path": runtime_status_path,
        "runtime_status_identity": runtime_status_identity,
    }


def _build(source: dict) -> dict:
    return build_conjunct(
        terminal_status_path=source["terminal_status_path"],
        expected_terminal_status_sha256=source["terminal_status_identity"]["sha256"],
        terminal_audit_path=source["terminal_audit_path"],
        expected_terminal_audit_sha256=source["terminal_audit_identity"]["sha256"],
        runtime_status_path=source["runtime_status_path"],
        expected_runtime_status_sha256=source["runtime_status_identity"]["sha256"],
        runtime_comparison_path=source["runtime_path"],
        expected_runtime_comparison_sha256=source["runtime_identity"]["sha256"],
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
    )


def test_conjunct_requires_both_terminal_and_runtime_strict_pass(tmp_path: Path) -> None:
    report = _build(_sources(tmp_path))

    assert report["status"] == "pass"
    assert report["generation_advantage_proven"] is True
    assert report["claim_policy"]["runtime_strict_comparator_passed"] is True
    assert report["claim_policy"]["cost_efficiency_ranking_allowed"] is False
    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY


def test_conjunct_accepts_fail_closed_strict_guard_control(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    source["runtime_identity"] = _runtime_comparison(
        source["runtime_path"],
        canonical_trusted=False,
    )
    source["runtime_status_identity"] = _runtime_status(
        source["runtime_status_path"],
        source["runtime_identity"],
        canonical_trusted=False,
    )

    report = _build(source)

    assert report["status"] == "pass"
    assert report["replay"]["runtime_strict_comparison"][
        "canonical_runtime_claim_trusted"
    ] is False
    assert report["claim_policy"]["strict_runtime_guard_controls"] is True
    assert report["claim_policy"]["strict_recovery_binding_verified"] is True
    assert report["claim_policy"]["cost_efficiency_ranking_allowed"] is False


def test_conjunct_rejects_unbound_strict_guard_control(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    runtime = json.loads(source["runtime_path"].read_text(encoding="utf-8"))
    runtime["decision"] = "canonical_runtime_claim_rejected_strict_guard_controls"
    runtime["claim_policy"]["canonical_runtime_claim_trusted"] = False
    runtime["claim_policy"]["strict_recovery_binding_required"] = True
    runtime["claim_policy"]["strict_recovery_binding_verified"] = False
    source["runtime_identity"] = _write(source["runtime_path"], runtime)
    source["runtime_status_identity"] = _runtime_status(
        source["runtime_status_path"],
        source["runtime_identity"],
        canonical_trusted=False,
    )

    with pytest.raises(ValueError, match="recovery binding"):
        _build(source)


def test_conjunct_rejects_runtime_direct_ranking_drift(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    runtime = json.loads(source["runtime_path"].read_text(encoding="utf-8"))
    runtime["claim_policy"]["cost_efficiency_ranking_allowed"] = True
    source["runtime_identity"] = _write(source["runtime_path"], runtime)
    source["runtime_status_identity"] = _runtime_status(
        source["runtime_status_path"],
        source["runtime_identity"],
    )

    with pytest.raises(ValueError, match="runtime strict comparison contract"):
        _build(source)


def test_conjunct_rejects_runtime_physical_lower_bound_label_drift(
    tmp_path: Path,
) -> None:
    source = _sources(tmp_path)
    runtime = json.loads(source["runtime_path"].read_text(encoding="utf-8"))
    runtime["claim_policy"]["physical_lower_bound_label_required"] = False
    source["runtime_identity"] = _write(source["runtime_path"], runtime)
    source["runtime_status_identity"] = _runtime_status(
        source["runtime_status_path"],
        source["runtime_identity"],
    )

    with pytest.raises(ValueError, match="runtime strict comparison contract"):
        _build(source)


def test_conjunct_rejects_terminal_hold_advantage(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    source["terminal_audit_identity"] = _terminal_audit(
        source["terminal_audit_path"],
        status="hold",
        advantage=True,
    )

    with pytest.raises(ValueError, match="terminal hold cannot prove"):
        build_conjunct(
            terminal_status_path=source["terminal_status_path"],
            expected_terminal_status_sha256=source["terminal_status_identity"]["sha256"],
            terminal_audit_path=source["terminal_audit_path"],
            expected_terminal_audit_sha256=source["terminal_audit_identity"]["sha256"],
            runtime_status_path=source["runtime_status_path"],
            expected_runtime_status_sha256=source["runtime_status_identity"]["sha256"],
            runtime_comparison_path=source["runtime_path"],
            expected_runtime_comparison_sha256=source["runtime_identity"]["sha256"],
            expected_training_revision=TRAINING_REVISION,
            expected_training_branch=TRAINING_BRANCH,
        )


def test_conjunct_preserves_valid_terminal_hold(tmp_path: Path) -> None:
    source = _sources(tmp_path)
    source["terminal_audit_identity"] = _terminal_audit(
        source["terminal_audit_path"],
        status="hold",
        advantage=False,
    )
    source["terminal_status_identity"] = _terminal_status(
        source["terminal_status_path"],
        source["terminal_audit_identity"],
        status="hold",
        advantage=False,
    )

    report = _build(source)

    assert report["terminal_status"] == "hold"
    assert report["generation_advantage_proven"] is False
    assert report["claim_policy"]["generation_advantage_proven"] is False


def test_waiter_publishes_conjunct(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _sources(tmp_path)
    quality_root = tmp_path
    output_root = quality_root / "reports/terminal_runtime_strict_conjunct_v1"

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    runtime_pid = {"value": 1}
    monkeypatch.setattr(
        waiter,
        "_runtime_identity",
        lambda **_: {
            "pid": runtime_pid["value"],
            "parent_pid": 1,
            "start_ticks": runtime_pid["value"] * 100,
            "hostname": "test",
            "cwd": ROOT.as_posix(),
            "cmdline": "python waiter.py",
            "python_executable": "python",
            "cuda_visible_devices": "",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "nice": 10,
            "ionice": "idle",
        },
    )
    args = Namespace(
        project=ROOT,
        quality_output_root=quality_root,
        terminal_status=source["terminal_status_path"],
        terminal_audit=source["terminal_audit_path"],
        runtime_status=source["runtime_status_path"],
        runtime_comparison=source["runtime_path"],
        output_root=output_root,
        output=output_root / "terminal_runtime_strict_conjunct.json",
        status_output=output_root / "waiter_status.json",
        pid_file=output_root / "waiter.pid.json",
        deployment_receipt_output=output_root / "deployment_receipt.json",
        expected_control_revision=git("rev-parse", "HEAD"),
        expected_control_tree=git("rev-parse", "HEAD^{tree}"),
        expected_control_branch=git("branch", "--show-current"),
        expected_waiter_source_sha256=file_identity(
            ROOT / "scripts/wait_generation_terminal_runtime_strict_conjunct.py"
        )["sha256"],
        expected_builder_source_sha256=file_identity(
            ROOT / "scripts/build_generation_terminal_runtime_strict_conjunct.py"
        )["sha256"],
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )

    assert waiter.run_waiter(args, require_detached=False) == 0
    deployment_identity = file_identity(args.deployment_receipt_output)
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    report = json.loads(args.output.read_text(encoding="utf-8"))
    assert status["status"] == "pass"
    assert status["conjunct"]["identity"] == file_identity(args.output)
    assert report["claim_policy"]["runtime_strict_comparator_passed"] is True
    assert not args.pid_file.exists()

    runtime_pid["value"] = 2
    assert waiter.run_waiter(args, require_detached=False) == 0
    assert file_identity(args.deployment_receipt_output) == deployment_identity
    deployment = json.loads(args.deployment_receipt_output.read_text(encoding="utf-8"))
    assert deployment["runtime_policy"] == waiter.RUNTIME_POLICY
    assert not args.pid_file.exists()


def test_waiter_rejects_source_overlap_with_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _sources(tmp_path)
    output_root = tmp_path / "reports/terminal_runtime_strict_conjunct_v1"
    output_root.mkdir(parents=True)
    overlapping_terminal_status = output_root / "terminal_status.json"
    overlapping_terminal_status.write_bytes(source["terminal_status_path"].read_bytes())

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    monkeypatch.setattr(
        waiter,
        "_runtime_identity",
        lambda **_: {
            "pid": 1,
            "parent_pid": 1,
            "start_ticks": 100,
            "hostname": "test",
            "cwd": ROOT.as_posix(),
            "cmdline": "python waiter.py",
            "python_executable": "python",
            "cuda_visible_devices": "",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "nice": 10,
            "ionice": "idle",
        },
    )
    args = Namespace(
        project=ROOT,
        quality_output_root=tmp_path,
        terminal_status=overlapping_terminal_status,
        terminal_audit=source["terminal_audit_path"],
        runtime_status=source["runtime_status_path"],
        runtime_comparison=source["runtime_path"],
        output_root=output_root,
        output=output_root / "terminal_runtime_strict_conjunct.json",
        status_output=output_root / "waiter_status.json",
        pid_file=output_root / "waiter.pid.json",
        deployment_receipt_output=output_root / "deployment_receipt.json",
        expected_control_revision=git("rev-parse", "HEAD"),
        expected_control_tree=git("rev-parse", "HEAD^{tree}"),
        expected_control_branch=git("branch", "--show-current"),
        expected_waiter_source_sha256=file_identity(
            ROOT / "scripts/wait_generation_terminal_runtime_strict_conjunct.py"
        )["sha256"],
        expected_builder_source_sha256=file_identity(
            ROOT / "scripts/build_generation_terminal_runtime_strict_conjunct.py"
        )["sha256"],
        expected_training_revision=TRAINING_REVISION,
        expected_training_branch=TRAINING_BRANCH,
        poll_seconds=0.01,
        timeout_seconds=1.0,
    )

    with pytest.raises(ValueError, match="terminal_status overlaps output root"):
        waiter.run_waiter(args, require_detached=False)
