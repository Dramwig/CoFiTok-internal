from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cofitok.inference_replay import file_identity
from scripts import (
    run_generation_quality_bridge_terminal_uncertainty_waiter as waiter,
)


PRECEDING_PID = 809569
PRECEDING_GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/generation-matched-uncertainty-v1",
    "tracked_dirty": False,
}


def _preceding_status(
    output_root: Path,
    *,
    status: str = "waiting",
) -> dict[str, Any]:
    return {
        "schema_version": waiter.base_waiter.WAITER_SCHEMA_VERSION,
        "role": waiter.base_waiter.WAITER_ROLE,
        "status": status,
        "detail": "done" if status in {"pass", "hold"} else "waiting",
        "phase": (
            "completed"
            if status in {"pass", "hold"}
            else "failed" if status == "failed" else "quality_bridge"
        ),
        "pid": PRECEDING_PID,
        "expected": {
            "output_root": output_root.resolve().as_posix(),
            "control_git": PRECEDING_GIT,
        },
        "claim_boundary": waiter.base_waiter.CLAIM_BOUNDARY,
        "updated_at": "2026-08-18T00:00:00+00:00",
        "summary": None,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_preceding_status_binds_exact_pid_checkout_and_output(tmp_path: Path) -> None:
    output_root = (tmp_path / "preceding").resolve()
    validated = waiter.validate_preceding_waiter_status(
        _preceding_status(output_root, status="hold"),
        expected_output_root=output_root,
        expected_pid=PRECEDING_PID,
        expected_control_git=PRECEDING_GIT,
    )

    assert validated["terminal_status_published"] is True
    changed = _preceding_status(output_root, status="hold")
    changed["pid"] = PRECEDING_PID + 1
    with pytest.raises(ValueError, match="contract differs"):
        waiter.validate_preceding_waiter_status(
            changed,
            expected_output_root=output_root,
            expected_pid=PRECEDING_PID,
            expected_control_git=PRECEDING_GIT,
        )


def test_preceding_waiter_is_terminal_only_after_pid_and_children_exit(
    tmp_path: Path,
) -> None:
    output_root = (tmp_path / "preceding").resolve()
    status_path = output_root / "reports" / "waiter_status.json"
    pid_path = output_root / "reports" / "waiter.pid"
    _write_json(status_path, _preceding_status(output_root, status="pass"))
    _write_json(
        pid_path,
        {
            "schema_version": 1,
            "role": waiter.base_waiter.WAITER_ROLE,
            "pid": PRECEDING_PID,
            "expected_control_revision": PRECEDING_GIT["revision"],
        },
    )

    with_pid = waiter.preceding_waiter_state(
        output_root=output_root,
        expected_pid=PRECEDING_PID,
        expected_control_git=PRECEDING_GIT,
        process_rows=[],
    )
    assert with_pid["terminal"] is False

    pid_path.unlink()
    with_child = waiter.preceding_waiter_state(
        output_root=output_root,
        expected_pid=PRECEDING_PID,
        expected_control_git=PRECEDING_GIT,
        process_rows=[
            {
                "pid": 42,
                "command": (
                    "python audit_generation_matched_uncertainty.py "
                    + output_root.as_posix()
                ),
            }
        ],
    )
    assert with_child["terminal"] is False

    terminal = waiter.preceding_waiter_state(
        output_root=output_root,
        expected_pid=PRECEDING_PID,
        expected_control_git=PRECEDING_GIT,
        process_rows=[],
    )
    assert terminal["terminal"] is True
    assert terminal["status_source"] == file_identity(status_path)


def _quality_status(revision: str, branch: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": waiter.base_waiter.QUALITY_EXECUTION_ROLE,
        "status": "completed",
        "detail": waiter.base_waiter.QUALITY_EXECUTION_COMPLETED_DETAIL,
        "exit_code": 0,
        "git": {
            "revision": revision,
            "branch": branch,
            "tracked_dirty": False,
        },
        "quality_bridge_only": True,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "report_is_promotion_gate": False,
        "updated_at": "2026-08-18T00:00:00+00:00",
    }


def _args(tmp_path: Path) -> SimpleNamespace:
    control = tmp_path / "control"
    evaluator = tmp_path / "evaluator"
    quality = tmp_path / "quality"
    quality_output = tmp_path / "quality-output"
    preceding = tmp_path / "preceding"
    output = tmp_path / "terminal-output"
    for path in (control, evaluator, quality, quality_output / "reports", preceding):
        path.mkdir(parents=True, exist_ok=True)
    revision = "c" * 40
    branch = "scale/generation-stability-quality-bridge-100k"
    quality_paths = waiter.base_waiter._quality_paths(quality_output)
    _write_json(
        quality_paths["execution_status"],
        _quality_status(revision, branch),
    )
    _write_json(quality_paths["result"], {})
    real_cache = tmp_path / "real-cache.pt"
    real_cache.write_bytes(b"cache")
    return SimpleNamespace(
        project=control,
        evaluator_project=evaluator,
        quality_project=quality,
        quality_output_root=quality_output,
        preceding_output_root=preceding,
        expected_preceding_pid=PRECEDING_PID,
        expected_preceding_control_revision=PRECEDING_GIT["revision"],
        expected_preceding_control_tree=PRECEDING_GIT["tree"],
        expected_preceding_control_branch=PRECEDING_GIT["branch"],
        output_root=output,
        status_output=output / "reports" / "waiter_status.json",
        pid_file=output / "reports" / "waiter.pid",
        manifest_output=output / "reports" / "execution_manifest.json",
        real_feature_cache_source=real_cache,
        expected_real_feature_cache_bytes=real_cache.stat().st_size,
        expected_real_feature_cache_sha256=file_identity(real_cache)["sha256"],
        python_executable=Path(__import__("sys").executable),
        expected_control_revision="d" * 40,
        expected_control_tree="e" * 40,
        expected_control_branch="analysis/terminal-uncertainty",
        expected_evaluator_revision="f" * 40,
        expected_evaluator_tree="1" * 40,
        expected_evaluator_branch="",
        expected_quality_revision=revision,
        expected_quality_tree="2" * 40,
        expected_quality_branch=branch,
        poll_seconds=0.001,
        required_idle_polls=1,
        timeout_seconds=60.0,
    )


@pytest.mark.parametrize("scientific_status", ["pass", "hold"])
def test_waiter_runs_one_terminal_audit_and_accepts_pass_or_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scientific_status: str,
) -> None:
    args = _args(tmp_path)
    quality_paths = waiter.base_waiter._quality_paths(args.quality_output_root)
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
    monkeypatch.setattr(waiter.base_waiter, "quality_processes", lambda *_a, **_k: [])
    monkeypatch.setattr(waiter.base_waiter, "_process_rows", list)
    monkeypatch.setattr(
        waiter.base_waiter,
        "verify_quality_bridge_result",
        lambda **_kwargs: {
            "status": "verified",
            "execution_status": file_identity(quality_paths["execution_status"]),
            "result": file_identity(quality_paths["result"]),
            "source_files": {},
            "quality_screen": {"status": "hold"},
            "authorization_boundary": {
                "full_training_launch_allowed": False,
                "full_300k_launch_allowed": False,
                "report_is_promotion_gate": False,
            },
        },
    )
    monkeypatch.setattr(
        waiter.manifest_builder,
        "build_manifest",
        lambda **_kwargs: {"manifest": True},
    )

    def prepare_manifest(path: Path, *_args, **_kwargs) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("{}", encoding="utf-8")
        return file_identity(path)

    monkeypatch.setattr(waiter.manifest_builder, "prepare_manifest", prepare_manifest)

    manifest_record: dict[str, Any] = {}

    def validate_manifest(path: Path, **_kwargs) -> dict[str, Any]:
        manifest_record.update(
            identity=file_identity(path),
            stream_id=waiter.EXPECTED_STREAM_ID,
            report={},
            output=args.output_root / "terminal_100k" / "matched_uncertainty.json",
            cache_root=args.output_root / "feature_cache",
            cache_names={
                "real": "real-cache",
                "cofitok": "cofitok-cache",
                "dense_identity": "dense-cache",
            },
            bound_sources=[],
            start_index=0,
            end_index_exclusive=10_000,
            cofitok_checkpoint_sha256="3" * 64,
            dense_checkpoint_sha256="4" * 64,
            checkpoint_step=100_000,
            real_set_sha256="5" * 64,
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
    preceding_source = {"path": "preceding.json", "bytes": 1, "sha256": "6" * 64}
    monkeypatch.setattr(
        waiter,
        "preceding_waiter_state",
        lambda **_kwargs: {
            "terminal": True,
            "status": "hold",
            "status_source": preceding_source,
            "active_processes": [],
        },
    )
    monkeypatch.setattr(waiter.base_waiter, "uncertainty_processes", lambda *_a, **_k: [])
    monkeypatch.setattr(waiter.base_waiter, "_gpu_compute_rows", list)
    monkeypatch.setattr(
        waiter.base_waiter,
        "audit_stage_spec",
        lambda **_kwargs: SimpleNamespace(name="terminal_100k"),
    )
    monkeypatch.setattr(
        waiter.base_waiter,
        "run_stage",
        lambda *_args, **_kwargs: {
            "status": "completed",
            "reused": False,
        },
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
            "source": {"path": "audit.json"},
            "claim_boundary": waiter.CLAIM_BOUNDARY,
        },
    )

    assert waiter.run_waiter(args) == 0
    status = json.loads(args.status_output.read_text(encoding="utf-8"))
    assert status["status"] == scientific_status
    assert status["phase"] == "completed"
    assert status["execution_manifest"]["stream_id"] == waiter.EXPECTED_STREAM_ID
    assert status["stage"]["status"] == "completed"
    assert status["claim_boundary"]["full_300k_launch_allowed"] is False
    assert not args.pid_file.exists()


def test_claim_boundary_remains_non_authorizing() -> None:
    assert waiter.CLAIM_BOUNDARY["diagnostic_non_authorizing"] is True
    assert waiter.CLAIM_BOUNDARY["broad_generation_superiority_claim_allowed"] is False
    assert waiter.CLAIM_BOUNDARY["training_launch_allowed"] is False
    assert waiter.CLAIM_BOUNDARY["full_300k_launch_allowed"] is False
    assert waiter.CLAIM_BOUNDARY["release_allowed"] is False
