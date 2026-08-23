from __future__ import annotations

import json
import subprocess
from argparse import Namespace
from pathlib import Path
from typing import Any

import pytest

from scripts import wait_generation_quality_bridge_comparison as waiter

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = (
    ROOT
    / "artifacts"
    / "reports"
    / "baselines"
    / "official_related_methods_2026-07-11_final"
    / "official_related_methods_table.json"
)


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments],
        text=True,
    ).strip()


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _args(tmp_path: Path) -> Namespace:
    quality_root = tmp_path / "quality"
    quality_root.mkdir()
    terminal_root = quality_root / "reports" / "terminal_system_claim_guard_v1"
    output_dir = quality_root / "reports" / "quality_bridge_comparison_v1"
    source = ROOT / "scripts" / "wait_generation_quality_bridge_comparison.py"
    builder = ROOT / "scripts" / "build_generation_quality_bridge_comparison.py"
    return Namespace(
        project=ROOT,
        quality_output_root=quality_root,
        terminal_system_guard=terminal_root / "terminal_system_claim_guard.json",
        terminal_waiter_status=terminal_root / "waiter_status.json",
        official_related=OFFICIAL,
        expected_official_related_sha256=waiter.file_sha256(OFFICIAL),
        output_dir=output_dir,
        expected_output_dir_name="quality_bridge_comparison_v1",
        expected_terminal_dir_name="terminal_system_claim_guard_v1",
        status_output=output_dir / "waiter_status.json",
        deployment_receipt_output=output_dir / "deployment_receipt.json",
        lock=output_dir / "waiter.lock",
        expected_project_revision=_git("rev-parse", "HEAD"),
        expected_project_tree=_git("rev-parse", "HEAD^{tree}"),
        expected_project_branch=_git("branch", "--show-current"),
        expected_waiter_source_sha256=waiter.file_sha256(source),
        expected_builder_source_sha256=waiter.file_sha256(builder),
        poll_seconds=0.01,
        timeout_seconds=1.0,
        once=True,
    )


def _terminal_guard(args: Namespace, *, status: str) -> dict[str, Any]:
    assert status in {"pass", "hold"}
    return {
        "schema_version": 1,
        "role": waiter.TERMINAL_GUARD_ROLE,
        "status": status,
        "decision": (
            "matched_quality_advantage_qualified_with_terminal_system_evidence"
            if status == "pass"
            else "terminal_system_evidence_complete_without_qualified_matched_advantage"
        ),
    }


def _complete_terminal(args: Namespace, *, status: str) -> dict[str, Any]:
    guard = _terminal_guard(args, status=status)
    _write(args.terminal_system_guard, guard)
    guard_identity = waiter.file_identity(
        args.terminal_system_guard,
        name="test terminal guard",
    )
    _write(
        args.terminal_waiter_status,
        {
            "schema_version": 1,
            "role": waiter.TERMINAL_WAITER_ROLE,
            "status": "completed",
            "detail": "terminal_system_claim_guard_source_revalidated",
            "authorization_boundary": waiter.TERMINAL_WAITER_BOUNDARY,
            "expected": {"output": args.terminal_system_guard.resolve().as_posix()},
            "guard": guard_identity,
            "guard_status": status,
            "guard_decision": guard["decision"],
        },
    )
    return guard_identity


def _fake_comparison(
    args: Namespace,
    *,
    terminal_status: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": waiter.COMPARISON_ROLE,
        "status": terminal_status,
        "decision": _terminal_guard(args, status=terminal_status)["decision"],
        "authorization_boundary": waiter.COMPARISON_BOUNDARY,
        "comparison_policy": {
            "primary_direct_tier": "matched_training_direct",
            "external_context_tier": "official_pretrained_contextual",
            "cross_tier_numeric_ranking_allowed": False,
            "compute_matched_claim_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
        },
        "source_reports": {
            "terminal_system_claim_guard": waiter.file_identity(
                args.terminal_system_guard,
                name="test terminal guard",
            ),
            "official_related_methods": waiter.file_identity(
                OFFICIAL,
                name="test official table",
            ),
        },
        "matched_training_rows": [
            {
                "method": "CoFiTok K=8",
                "comparison_tier": "matched_training_direct",
            },
            {
                "method": "Dense identity",
                "comparison_tier": "matched_training_direct",
            },
        ],
        "official_context_rows": [
            {
                "method": method,
                "comparison_tier": "official_pretrained_contextual",
            }
            for method in ("D-AR", "MAR", "ReTok")
        ],
    }


def _install_fake_builder(
    monkeypatch: pytest.MonkeyPatch,
    args: Namespace,
    *,
    terminal_status: str,
    calls: list[list[str]] | None = None,
) -> None:
    original_run = subprocess.run

    def fake_run(
        command: list[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        if command[0] == "git":
            return original_run(command, **kwargs)
        if calls is not None:
            calls.append(command)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report = _fake_comparison(args, terminal_status=terminal_status)
        json_path = args.output_dir / "quality_bridge_comparison.json"
        payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if json_path.is_file() and json_path.read_text(encoding="utf-8") != payload:
            return subprocess.CompletedProcess(
                command,
                1,
                stdout="",
                stderr="existing comparison output differs",
            )
        json_path.write_text(payload, encoding="utf-8")
        for name, content in (
            ("quality_bridge_comparison.md", "# comparison\n"),
            ("quality_bridge_comparison.csv", "method\n"),
        ):
            path = args.output_dir / name
            if path.is_file() and path.read_text(encoding="utf-8") != content:
                return subprocess.CompletedProcess(
                    command,
                    1,
                    stdout="",
                    stderr="existing comparison output differs",
                )
            path.write_text(content, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout=str(json_path), stderr="")

    monkeypatch.setattr(waiter.subprocess, "run", fake_run)


def test_static_context_binds_git_tree_branch_and_sources(tmp_path: Path) -> None:
    args = _args(tmp_path)

    context = waiter.static_context(args)

    assert context["git"] == {
        "revision": args.expected_project_revision,
        "tree": args.expected_project_tree,
        "branch": args.expected_project_branch,
        "tracked_dirty": False,
    }
    assert context["waiter_source"]["sha256"] == args.expected_waiter_source_sha256
    assert context["builder_source"]["sha256"] == args.expected_builder_source_sha256
    assert context["official_source"]["sha256"] == (
        args.expected_official_related_sha256
    )


def test_static_context_accepts_explicit_versioned_output_and_terminal_dirs(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    reports = args.quality_output_root / "reports"
    terminal_root = reports / "terminal_system_claim_guard_v2_runtime_strict"
    output_dir = reports / "quality_bridge_comparison_v2_runtime_strict"
    args.expected_terminal_dir_name = terminal_root.name
    args.expected_output_dir_name = output_dir.name
    args.terminal_system_guard = terminal_root / "terminal_system_claim_guard.json"
    args.terminal_waiter_status = terminal_root / "waiter_status.json"
    args.output_dir = output_dir
    args.status_output = output_dir / "waiter_status.json"
    args.deployment_receipt_output = output_dir / "deployment_receipt.json"
    args.lock = output_dir / "waiter.lock"

    context = waiter.static_context(args)

    assert context["output_dir"] == output_dir.resolve()
    assert context["terminal_guard"] == args.terminal_system_guard.resolve()


def test_static_context_rejects_versioned_path_not_named_explicitly(
    tmp_path: Path,
) -> None:
    args = _args(tmp_path)
    args.output_dir = (
        args.quality_output_root
        / "reports"
        / "quality_bridge_comparison_v2_runtime_strict"
    )
    args.status_output = args.output_dir / "waiter_status.json"
    args.deployment_receipt_output = args.output_dir / "deployment_receipt.json"
    args.lock = args.output_dir / "waiter.lock"

    with pytest.raises(ValueError, match="canonical path contract differs"):
        waiter.static_context(args)


@pytest.mark.parametrize(
    "name",
    ["../quality_bridge_comparison_v2", "quality_bridge_comparison_v2/child"],
)
def test_static_context_rejects_unsafe_explicit_versioned_dir_name(
    tmp_path: Path,
    name: str,
) -> None:
    args = _args(tmp_path)
    args.expected_output_dir_name = name

    with pytest.raises(ValueError, match="directory name is invalid"):
        waiter.static_context(args)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_project_revision", "0" * 40),
        ("expected_project_tree", "0" * 40),
        ("expected_project_branch", "analysis/wrong-branch"),
        ("expected_waiter_source_sha256", "0" * 64),
        ("expected_builder_source_sha256", "0" * 64),
    ],
)
def test_static_context_rejects_git_or_source_drift(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    args = _args(tmp_path)
    setattr(args, field, value)

    with pytest.raises(ValueError, match="differs"):
        waiter.static_context(args)


def test_waiter_records_waiting_without_terminal_guard(tmp_path: Path) -> None:
    args = _args(tmp_path)

    assert waiter.run_waiter(args, enforce_runtime=False) == 0

    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    receipt = json.loads(args.deployment_receipt_output.read_text(encoding="utf-8"))
    assert status["status"] == "waiting"
    assert status["detail"] == "waiting_for_terminal_guard_waiter"
    assert status["comparison"] is None
    assert receipt["status"] == "pass"
    assert receipt["scope"] == waiter.SCOPE


def test_waiter_fails_closed_when_terminal_waiter_failed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    _write(
        args.terminal_waiter_status,
        {
            "schema_version": 1,
            "role": waiter.TERMINAL_WAITER_ROLE,
            "status": "failed",
            "detail": "upstream_terminal_evidence_failed",
            "authorization_boundary": waiter.TERMINAL_WAITER_BOUNDARY,
            "expected": {"output": args.terminal_system_guard.resolve().as_posix()},
        },
    )

    assert waiter.run_waiter(args, enforce_runtime=False) == 1

    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["detail"] == "terminal_system_claim_guard_waiter_failed"
    assert not (args.output_dir / "quality_bridge_comparison.json").exists()


@pytest.mark.parametrize("terminal_status", ["pass", "hold"])
def test_waiter_builds_for_pass_or_hold_and_operationally_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_status: str,
) -> None:
    args = _args(tmp_path)
    _complete_terminal(args, status=terminal_status)
    calls: list[list[str]] = []
    _install_fake_builder(
        monkeypatch,
        args,
        terminal_status=terminal_status,
        calls=calls,
    )

    assert waiter.run_waiter(args, enforce_runtime=False) == 0

    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "pass"
    assert status["terminal_status"] == terminal_status
    assert status["scope"] == waiter.SCOPE
    assert len(calls) == 1
    command = calls[0]
    assert command[command.index("--expected-terminal-system-guard-sha256") + 1]
    assert command[command.index("--expected-official-related-sha256") + 1] == (
        args.expected_official_related_sha256
    )


def test_waiter_rejects_official_table_sha_drift(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.expected_official_related_sha256 = "0" * 64

    assert waiter.run_waiter(args, enforce_runtime=False) == 1

    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "official related-method table SHA256 differs" in status["detail"]


def test_waiter_replays_existing_outputs_without_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    _complete_terminal(args, status="hold")
    calls: list[list[str]] = []
    _install_fake_builder(
        monkeypatch,
        args,
        terminal_status="hold",
        calls=calls,
    )

    assert waiter.run_waiter(args, enforce_runtime=False) == 0
    first = {
        name: waiter.file_identity(path, name=name)
        for name, path in waiter.static_context(args)["outputs"].items()
    }
    assert waiter.run_waiter(args, enforce_runtime=False) == 0
    second = {
        name: waiter.file_identity(path, name=name)
        for name, path in waiter.static_context(args)["outputs"].items()
    }

    assert first == second
    assert len(calls) == 2


def test_waiter_rejects_existing_output_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    _complete_terminal(args, status="hold")
    _install_fake_builder(monkeypatch, args, terminal_status="hold")
    assert waiter.run_waiter(args, enforce_runtime=False) == 0
    (args.output_dir / "quality_bridge_comparison.md").write_text(
        "drift\n",
        encoding="utf-8",
    )

    assert waiter.run_waiter(args, enforce_runtime=False) == 1

    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert "existing comparison output differs" in status["detail"]


def test_waiter_duplicate_lock_fails_without_touching_status(tmp_path: Path) -> None:
    args = _args(tmp_path)
    with waiter.exclusive_waiter_lock(args.lock):
        assert waiter.run_waiter(args, enforce_runtime=False) == 75

    assert not args.status_output.exists()


def test_waiter_scope_is_permanently_non_authorizing() -> None:
    assert waiter.SCOPE["diagnostic_non_authorizing"] is True
    assert waiter.SCOPE["cpu_only"] is True
    for field in (
        "gpu_execution_allowed",
        "training_launch_allowed",
        "sampling_launch_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "inference_export_authorization_allowed",
        "promotion_authorization_allowed",
        "release_authorization_allowed",
        "process_signals_allowed",
        "training_process_signals_allowed",
        "unrelated_process_signals_allowed",
        "cross_tier_numeric_ranking_allowed",
        "broad_generation_superiority_claim_allowed",
        "sota_claim_allowed",
    ):
        assert waiter.SCOPE[field] is False
