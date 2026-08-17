from __future__ import annotations

import errno
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import file_sha256
from scripts import run_generation_matched_uncertainty_waiter as waiter


def _quality_status(*, status: str = "completed") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": waiter.QUALITY_EXECUTION_ROLE,
        "status": status,
        "detail": (
            waiter.QUALITY_EXECUTION_COMPLETED_DETAIL
            if status == "completed"
            else "quality bridge running"
        ),
        "exit_code": None,
        "git": {
            "revision": "c" * 40,
            "branch": "scale/generation-stability-quality-bridge-100k",
            "tracked_dirty": False,
        },
        "quality_bridge_only": True,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "report_is_promotion_gate": False,
        "updated_at": "2026-08-18T00:00:00+00:00",
    }


def _manifest_record(
    tmp_path: Path,
    *,
    name: str,
    output_root: Path,
    start_index: int,
) -> dict[str, Any]:
    manifest = tmp_path / f"{name}.json"
    manifest.write_text("{}", encoding="utf-8")
    output = output_root / name / "matched_uncertainty.json"
    return {
        "identity": waiter._source(manifest),
        "stream_id": name,
        "report": {},
        "output": output,
        "cache_root": output_root / "feature_cache",
        "cache_names": {
            "real": "real-cache",
            "cofitok": f"{name}-cofitok",
            "dense_identity": f"{name}-dense",
        },
        "bound_sources": [],
        "start_index": start_index,
        "end_index_exclusive": start_index + 10_000,
        "cofitok_checkpoint_sha256": "1" * 64,
        "dense_checkpoint_sha256": "2" * 64,
        "checkpoint_step": 50_000,
        "real_set_sha256": "3" * 64,
    }


def _waiter_args(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, Any]]:
    control = tmp_path / "control"
    evaluator = tmp_path / "evaluator"
    quality = tmp_path / "quality"
    quality_output = tmp_path / "quality-output"
    output = tmp_path / "uncertainty-output"
    for path in (control, evaluator, quality, quality_output / "reports"):
        path.mkdir(parents=True)
    quality_paths = waiter._quality_paths(quality_output)
    quality_paths["execution_status"].write_text(
        json.dumps(_quality_status()),
        encoding="utf-8",
    )
    quality_paths["result"].write_text("{}", encoding="utf-8")
    formal_path = tmp_path / "formal.json"
    confirmation_path = tmp_path / "confirmation.json"
    formal_path.write_text("{}", encoding="utf-8")
    confirmation_path.write_text("{}", encoding="utf-8")
    cache_source = tmp_path / "real-cache.pt"
    cache_source.write_bytes(b"cache")
    args = SimpleNamespace(
        project=control,
        evaluator_project=evaluator,
        quality_project=quality,
        quality_output_root=quality_output,
        output_root=output,
        status_output=output / "waiter_status.json",
        pid_file=output / "waiter.pid",
        formal_manifest=formal_path,
        confirmation_manifest=confirmation_path,
        expected_formal_manifest_sha256=file_sha256(formal_path),
        expected_confirmation_manifest_sha256=file_sha256(confirmation_path),
        real_feature_cache_source=cache_source,
        expected_real_feature_cache_bytes=cache_source.stat().st_size,
        expected_real_feature_cache_sha256=file_sha256(cache_source),
        python_executable=Path(__import__("sys").executable),
        expected_control_revision="a" * 40,
        expected_control_tree="a" * 40,
        expected_control_branch="analysis/generation-matched-uncertainty-v1",
        expected_evaluator_revision="b" * 40,
        expected_evaluator_tree="b" * 40,
        expected_evaluator_branch="",
        expected_quality_revision="c" * 40,
        expected_quality_tree="c" * 40,
        expected_quality_branch="scale/generation-stability-quality-bridge-100k",
        poll_seconds=0.001,
        required_idle_polls=2,
        timeout_seconds=60.0,
    )
    manifests = {
        "formal": _manifest_record(
            tmp_path,
            name="formal_00000000_00010000",
            output_root=output,
            start_index=0,
        ),
        "confirmation": _manifest_record(
            tmp_path,
            name="confirmation_00010000_00020000",
            output_root=output,
            start_index=10_000,
        ),
    }
    return args, manifests


def test_quality_execution_status_contract_is_non_authorizing() -> None:
    result = waiter.validate_quality_execution_status(
        _quality_status(),
        expected_revision="c" * 40,
        expected_branch="scale/generation-stability-quality-bridge-100k",
    )

    assert result["status"] == "completed"

    changed = _quality_status()
    changed["full_300k_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        waiter.validate_quality_execution_status(
            changed,
            expected_revision="c" * 40,
            expected_branch="scale/generation-stability-quality-bridge-100k",
        )


def test_checkout_contract_requires_exact_clean_revision_tree_and_branch(
    tmp_path,
) -> None:
    project = tmp_path / "checkout"
    subprocess.run(
        ["git", "init", "-b", "analysis/test-waiter", str(project)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "config", "user.name", "Test"],
        check=True,
    )
    tracked = project / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(project), "add", "tracked.txt"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(project), "commit", "-m", "initial"],
        check=True,
        capture_output=True,
        text=True,
    )
    identity = waiter._git_identity(project)

    assert (
        waiter.verify_checkout(
            project,
            expected_revision=identity["revision"],
            expected_tree=identity["tree"],
            expected_branch=identity["branch"],
            label="test",
        )
        == identity
    )

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        waiter.verify_checkout(
            project,
            expected_revision=identity["revision"],
            expected_tree=identity["tree"],
            expected_branch=identity["branch"],
            label="test",
        )


def test_verify_quality_bridge_result_replays_every_terminal_source(
    tmp_path,
    monkeypatch,
) -> None:
    quality_project = tmp_path / "quality"
    output_root = tmp_path / "output"
    (quality_project / "scripts").mkdir(parents=True)
    paths = waiter._quality_paths(output_root)
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(_quality_status()) if name == "execution_status" else "{}",
            encoding="utf-8",
        )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        boundary = {
            "quality_bridge_evidence_complete": True,
            "quality_bridge_execution_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "report_is_promotion_gate": False,
            "release_authorization_allowed": False,
            "new_gate_required": True,
        }
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(
                {
                    "status": "verified",
                    "result": waiter._source(paths["result"]),
                    "quality_screen": {"status": "hold"},
                    "authorization_boundary": boundary,
                }
            ),
            stderr="",
        )

    monkeypatch.setattr(waiter.subprocess, "run", fake_run)

    result = waiter.verify_quality_bridge_result(
        python_executable=Path(__import__("sys").executable),
        quality_project=quality_project,
        quality_output_root=output_root,
        expected_revision="c" * 40,
        expected_branch="scale/generation-stability-quality-bridge-100k",
    )

    command = captured["command"]
    for option in (
        "--cofitok-training",
        "--dense-training",
        "--training-pair-validation",
        "--cofitok-training-audit",
        "--dense-training-audit",
        "--milestone-50000",
        "--milestone-100000",
        "--cofitok-sampling-preflight",
        "--dense-sampling-preflight",
        "--cofitok-generation",
        "--dense-generation",
        "--cofitok-checkpoint-eval",
        "--dense-checkpoint-eval",
        "--class-fidelity-qualification",
        "--cofitok-class-fidelity",
        "--dense-class-fidelity",
    ):
        assert option in command
    assert captured["kwargs"]["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert result["status"] == "verified"
    assert result["authorization_boundary"]["full_300k_launch_allowed"] is False


def test_real_feature_cache_hardlink_is_verified_and_reused(tmp_path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "cache" / "destination.pt"
    receipt = tmp_path / "reports" / "cache.json"
    source.write_bytes(b"verified-cache")
    expected_sha = file_sha256(source)

    first = waiter.prepare_real_feature_cache(
        source=source,
        destination=destination,
        receipt_path=receipt,
        expected_bytes=source.stat().st_size,
        expected_sha256=expected_sha,
    )
    second = waiter.prepare_real_feature_cache(
        source=source,
        destination=destination,
        receipt_path=receipt,
        expected_bytes=source.stat().st_size,
        expected_sha256=expected_sha,
    )

    assert first == second
    assert first["method"] == "hardlink"
    assert first["same_inode"] is True
    assert file_sha256(destination) == expected_sha


def test_real_feature_cache_copies_only_for_cross_device_link_failure(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "cache" / "destination.pt"
    receipt = tmp_path / "reports" / "cache.json"
    source.write_bytes(b"cross-device-cache")

    def cross_device(*_args, **_kwargs):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(waiter.os, "link", cross_device)
    result = waiter.prepare_real_feature_cache(
        source=source,
        destination=destination,
        receipt_path=receipt,
        expected_bytes=source.stat().st_size,
        expected_sha256=file_sha256(source),
    )

    assert result["method"] == "cross_device_copy"
    assert result["same_inode"] is False
    assert destination.read_bytes() == source.read_bytes()


def test_real_feature_cache_identity_drift_is_rejected(tmp_path) -> None:
    source = tmp_path / "source.pt"
    destination = tmp_path / "cache" / "destination.pt"
    receipt = tmp_path / "reports" / "cache.json"
    source.write_bytes(b"cache")
    expected_sha = file_sha256(source)
    waiter.prepare_real_feature_cache(
        source=source,
        destination=destination,
        receipt_path=receipt,
        expected_bytes=source.stat().st_size,
        expected_sha256=expected_sha,
    )
    destination.write_bytes(b"drift")

    with pytest.raises(ValueError, match="identity differs|receipt differs"):
        waiter.prepare_real_feature_cache(
            source=source,
            destination=destination,
            receipt_path=receipt,
            expected_bytes=source.stat().st_size,
            expected_sha256=expected_sha,
        )


def test_quality_and_uncertainty_process_filters_are_scoped(tmp_path) -> None:
    quality_project = tmp_path / "quality"
    quality_output = tmp_path / "quality-output"
    uncertainty_output = tmp_path / "uncertainty-output"
    rows = [
        {
            "pid": 1,
            "cwd": quality_project.as_posix(),
            "command": "python scripts/train_generation.py",
        },
        {
            "pid": 2,
            "cwd": "/other",
            "command": (
                "python scripts/evaluate_generation_metrics.py "
                + quality_output.as_posix()
            ),
        },
        {
            "pid": 3,
            "cwd": "/other",
            "command": (
                "python scripts/audit_generation_matched_uncertainty.py "
                + uncertainty_output.as_posix()
            ),
        },
        {"pid": 4, "cwd": "/other", "command": "python unrelated.py"},
    ]

    assert [
        row["pid"]
        for row in waiter.quality_processes(
            rows,
            quality_project=quality_project,
            quality_output_root=quality_output,
        )
    ] == [1, 2]
    assert [
        row["pid"]
        for row in waiter.uncertainty_processes(
            rows,
            output_root=uncertainty_output,
        )
    ] == [3]


def test_consecutive_idle_counter_resets_for_gpu_or_competing_process() -> None:
    count = waiter.next_idle_count(0, gpu_rows=[], competing_processes=[])
    assert count == 1
    count = waiter.next_idle_count(
        count,
        gpu_rows=[{"pid": 10}],
        competing_processes=[],
    )
    assert count == 0
    count = waiter.next_idle_count(1, gpu_rows=[], competing_processes=[{"pid": 11}])
    assert count == 0


def test_stage_specs_bind_inputs_outputs_and_keep_hold_nonfatal(tmp_path) -> None:
    evaluator = tmp_path / "evaluator"
    output = tmp_path / "output"
    evaluator.mkdir()
    manifest = _manifest_record(
        tmp_path,
        name="formal",
        output_root=output,
        start_index=0,
    )
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    manifest["bound_sources"] = [waiter._source(source)]
    real_cache = tmp_path / "real.pt"
    real_cache.write_bytes(b"real")

    spec = waiter.audit_stage_spec(
        name="formal",
        evaluator_project=evaluator,
        python_executable=Path(__import__("sys").executable),
        manifest=manifest,
        real_cache=real_cache,
        stage_root=output / "stage_receipts",
    )

    assert spec.gpu_required is True
    assert Path(manifest["identity"]["path"]) in spec.input_files
    assert source.resolve() in spec.input_files
    assert manifest["output"] in spec.output_files
    assert "--require-advantage" not in spec.command


def test_waiter_execution_manifest_binds_sources_and_output_scope(tmp_path) -> None:
    output_root = tmp_path / "uncertainty"
    source = tmp_path / "sampling.json"
    source.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "role": waiter.EXECUTION_MANIFEST_ROLE,
        "stream_id": "formal_00000000_00010000",
        "claim_boundary": {
            key: waiter.CLAIM_BOUNDARY[key]
            for key in (
                "diagnostic_non_authorizing",
                "replaces_fid_point_estimates",
                "replaces_promotion_gate",
                "broad_generation_superiority_claim_allowed",
                "full_training_launch_allowed",
                "full_300k_launch_allowed",
            )
        },
        "arguments": {
            "output": (output_root / "formal" / "matched_uncertainty.json").as_posix(),
            "cache_root": (output_root / "feature_cache").as_posix(),
            "real_cache_name": "real",
            "cpu": False,
        },
        "source_files": {"sampling": waiter._source(source)},
        "expected": {
            "matched_sampling": {
                "start_index": 0,
                "end_index_exclusive": 10_000,
                "sample_count": 10_000,
            },
            "real_set": {"sha256": "3" * 64},
            "sample_sets": {
                "cofitok": {
                    "sha256": "1" * 64,
                    "checkpoint_sha256": "4" * 64,
                    "checkpoint_step": 50_000,
                },
                "dense_identity": {
                    "sha256": "2" * 64,
                    "checkpoint_sha256": "5" * 64,
                    "checkpoint_step": 50_000,
                },
            },
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    result = waiter.validate_execution_manifest(
        manifest_path,
        expected_sha256=file_sha256(manifest_path),
        output_root=output_root.resolve(),
    )

    assert (
        result["output"]
        == (output_root / "formal" / "matched_uncertainty.json").resolve()
    )
    assert result["cache_names"]["cofitok"] == "matched_uncertainty_cofitok_" + "1" * 16
    assert result["bound_sources"] == [waiter._source(source)]

    manifest["arguments"]["output"] = (tmp_path / "outside.json").as_posix()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="output scope differs"):
        waiter.validate_execution_manifest(
            manifest_path,
            expected_sha256=file_sha256(manifest_path),
            output_root=output_root.resolve(),
        )


def test_run_stage_uses_stage_once_and_accepts_reused_completion(
    tmp_path,
    monkeypatch,
) -> None:
    evaluator = tmp_path / "evaluator"
    evaluator.mkdir()
    spec = waiter.StageSpec(
        name="formal",
        state=tmp_path / "formal.stage.json",
        command=("python", "audit.py"),
        input_files=(tmp_path / "input.json",),
        output_files=(tmp_path / "output.json",),
        gpu_required=True,
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(
            returncode=0,
            stdout='worker log\n{"status":"completed","reused":true}\n',
            stderr="",
        )

    monkeypatch.setattr(waiter.subprocess, "run", fake_run)
    result = waiter.run_stage(
        spec,
        evaluator_project=evaluator,
        python_executable=Path(__import__("sys").executable),
    )

    assert result["status"] == "completed"
    assert result["reused"] is True
    assert "run_generation_stage_once.py" in " ".join(captured["command"])
    assert "--input-file" in captured["command"]
    assert "--output-file" in captured["command"]


def _patch_waiter_runtime(
    monkeypatch,
    *,
    args,
    manifests,
    summary_status: str,
) -> list[list[dict[str, Any]]]:
    monkeypatch.setattr(
        waiter,
        "verify_checkout",
        lambda project, **kwargs: {
            "revision": kwargs["expected_revision"],
            "tree": kwargs["expected_tree"],
            "branch": kwargs["expected_branch"],
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        waiter,
        "validate_execution_manifest",
        lambda path, **_kwargs: (
            manifests["formal"]
            if Path(path).resolve() == args.formal_manifest.resolve()
            else manifests["confirmation"]
        ),
    )
    monkeypatch.setattr(waiter, "validate_manifest_pair", lambda *_args: None)
    quality_rows = [[{"pid": 100}], []]
    monkeypatch.setattr(
        waiter,
        "quality_processes",
        lambda *_args, **_kwargs: quality_rows.pop(0) if quality_rows else [],
    )
    monkeypatch.setattr(waiter, "_process_rows", list)
    monkeypatch.setattr(waiter, "uncertainty_processes", lambda *_a, **_k: [])
    quality_paths = waiter._quality_paths(args.quality_output_root)
    monkeypatch.setattr(
        waiter,
        "verify_quality_bridge_result",
        lambda **_kwargs: {
            "status": "verified",
            "execution_status": waiter._source(quality_paths["execution_status"]),
            "result": waiter._source(quality_paths["result"]),
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
        waiter,
        "prepare_real_feature_cache",
        lambda **_kwargs: {
            "status": "verified",
            "method": "hardlink",
            "claim_boundary": waiter.CLAIM_BOUNDARY,
        },
    )
    gpu_sequence = [
        [],
        [{"pid": 200, "process_name": "capacity", "used_memory_mib": 1}],
        [],
        [],
        [],
        [],
        [],
        [],
    ]
    observed_gpu: list[list[dict[str, Any]]] = []

    def next_gpu():
        value = gpu_sequence.pop(0) if gpu_sequence else []
        observed_gpu.append(value)
        return value

    monkeypatch.setattr(waiter, "_gpu_compute_rows", next_gpu)
    monkeypatch.setattr(waiter, "_sleep_until_next_poll", lambda **_kwargs: True)
    monkeypatch.setattr(
        waiter,
        "run_stage",
        lambda spec, **_kwargs: {
            "status": "completed",
            "reused": spec.name == "confirmation",
            "state": spec.state.as_posix(),
        },
    )
    monkeypatch.setattr(
        waiter,
        "validate_audit_output",
        lambda path, **_kwargs: {
            "status": "pass",
            "decision": "matched_relative_generation_advantage_supported",
            "advantage_supported": True,
            "source": {"path": Path(path).as_posix()},
            "claim_boundary": waiter.CLAIM_BOUNDARY,
        },
    )
    monkeypatch.setattr(
        waiter,
        "validate_summary_output",
        lambda path, **_kwargs: {
            "status": summary_status,
            "decision": (
                "repeated_cross_protocol_relative_advantage_supported"
                if summary_status == "pass"
                else "repeated_relative_advantage_not_confirmed"
            ),
            "repeated_advantage_supported": summary_status == "pass",
            "checks": {},
            "source": {"path": Path(path).as_posix()},
            "claim_boundary": waiter.CLAIM_BOUNDARY,
        },
    )
    return observed_gpu


@pytest.mark.parametrize("summary_status", ["pass", "hold"])
def test_waiter_waits_resets_idle_and_closes_pass_or_hold(
    tmp_path,
    monkeypatch,
    summary_status,
) -> None:
    args, manifests = _waiter_args(tmp_path)
    observed_gpu = _patch_waiter_runtime(
        monkeypatch,
        args=args,
        manifests=manifests,
        summary_status=summary_status,
    )

    exit_code = waiter.run_waiter(args)
    status = json.loads(args.status_output.read_text(encoding="utf-8"))

    assert exit_code == 0
    assert status["status"] == summary_status
    assert status["phase"] == "completed"
    assert status["claim_boundary"]["full_300k_launch_allowed"] is False
    assert status["claim_boundary"]["release_allowed"] is False
    assert set(status["stages"]) == {"formal", "confirmation", "summary"}
    assert status["stages"]["confirmation"]["reused"] is True
    assert any(rows and rows[0]["pid"] == 200 for rows in observed_gpu)
    assert not args.pid_file.exists()


def test_duplicate_waiter_is_rejected_by_os_output_lock(tmp_path) -> None:
    args, _ = _waiter_args(tmp_path)

    with (
        exclusive_output_lock(args.output_root, role="test-owner"),
        pytest.raises(OutputLockError, match="already locked"),
    ):
        waiter.run_waiter(args)


def test_claim_boundary_is_permanently_non_authorizing() -> None:
    assert waiter.CLAIM_BOUNDARY == {
        "diagnostic_non_authorizing": True,
        "replaces_fid_point_estimates": False,
        "replaces_promotion_gate": False,
        "broad_generation_superiority_claim_allowed": False,
        "training_launch_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "release_allowed": False,
        "training_process_signals_allowed": False,
        "unrelated_process_signals_allowed": False,
    }
