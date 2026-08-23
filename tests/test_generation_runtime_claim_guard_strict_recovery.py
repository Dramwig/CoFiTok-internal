from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.inference_replay import file_identity
from scripts.build_generation_runtime_claim_guard_comparison import _validate_wrapper
from scripts.recover_generation_runtime_claim_guard_strict_replay import (
    BEHAVIOR,
    SCOPE,
    build_wrapper_receipt,
)


STRICT_REVISION = "0" * 40


def _identity(path: str) -> dict:
    return {"path": path, "bytes": 1, "sha256": "a" * 64}


def _receipt() -> dict:
    return build_wrapper_receipt(
        strict_control={
            "path": "/strict",
            "revision": STRICT_REVISION,
            "tree": "1" * 40,
            "branch": "analysis/strict",
            "tracked_dirty": False,
        },
        wrapper_runtime={
            "pid": 200,
            "ppid": 1,
            "start_ticks": 123,
            "cwd": "/strict",
            "cmdline_sha256": "b" * 64,
            "cuda_visible_devices": "",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "nice": 10,
            "ionice": "idle",
        },
        canonical_runtime={"pid": 100},
        source_runtime={"pid": 50},
        targets={
            "output_root": "/output/v3",
            "status_output": "/output/v3/waiter_status.json",
            "guard_output": "/output/v3/runtime_compute_claim_guard.json",
        },
        recovery_control={
            "path": "/recovery",
            "revision": "2" * 40,
            "tree": "3" * 40,
            "branch": "analysis/recovery",
            "tracked_dirty": False,
        },
        recovery_source=_identity("/recovery/recover.py"),
        original_wrapper=_identity("/output/v2/wrapper.json"),
        failure_log=_identity("/tmp/wrapper.log"),
        strict_runner=_identity("/strict/scripts/runner.py"),
        strict_builder=_identity("/strict/scripts/builder.py"),
        strict_runner_argv_sha256="c" * 64,
        failed_targets={
            "status_output": {"path": "/output/v2/status.json", "absent": True},
            "guard_output": {"path": "/output/v2/guard.json", "absent": True},
        },
        prior_failure_status=_identity("/output/v2/status.json"),
    )


def test_recovery_receipt_preserves_existing_comparison_contract() -> None:
    receipt = _receipt()
    _validate_wrapper(
        receipt,
        canonical_waiter={"pid": 100},
        strict_waiter={
            "pid": 200,
            "control_git": {
                "revision": STRICT_REVISION,
                "tree": "1" * 40,
                "branch": "analysis/strict",
                "tracked_dirty": False,
            },
            "source_waiter": {"pid": 50},
        },
        expected_strict_control_revision=STRICT_REVISION,
    )
    assert receipt["scope"] == SCOPE
    assert receipt["behavior"] == BEHAVIOR


def test_recovery_receipt_binds_failed_attempt_without_overwriting_it() -> None:
    recovery = _receipt()["recovery"]
    assert recovery["reason"] == "original_wrapper_missing_pythonpath_import_failure"
    assert recovery["new_output_version"] == "runtime_compute_claim_guard_strict_replay_v3"
    assert recovery["old_outputs_overwritten"] is False
    assert recovery["old_process_signaled"] is False
    assert recovery["failed_targets_absent_before_recovery"]["status_output"][
        "absent"
    ] is True


def test_recovery_receipt_pid_drift_is_rejected() -> None:
    receipt = _receipt()
    receipt["wrapper"]["pid"] = 201
    try:
        _validate_wrapper(
            receipt,
            canonical_waiter={"pid": 100},
            strict_waiter={
                "pid": 200,
                "control_git": {
                    "revision": STRICT_REVISION,
                    "tree": "1" * 40,
                    "branch": "analysis/strict",
                    "tracked_dirty": False,
                },
                "source_waiter": {"pid": 50},
            },
            expected_strict_control_revision=STRICT_REVISION,
        )
    except ValueError as error:
        assert "wrapper deployment contract differs" in str(error)
    else:
        raise AssertionError("wrapper PID drift was accepted")


def _write(path: Path, payload: str) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    return file_identity(path)


def test_required_recovery_binding_physically_reopens_sources(tmp_path: Path) -> None:
    receipt = _receipt()
    original = tmp_path / "original_wrapper.json"
    original.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": "generation_runtime_claim_guard_strict_replay_wrapper_deployment",
                "status": "pass",
            }
        ),
        encoding="utf-8",
    )
    failure = tmp_path / "failure.log"
    receipt["recovery"]["original_wrapper_deployment"] = file_identity(original)
    receipt["recovery"]["original_failure_log"] = _write(
        failure,
        "run_generation_quality_bridge_runtime_claim_guard_waiter.py\n"
        "ModuleNotFoundError: No module named 'scripts'\n",
    )
    for field in (
        "recovery_source",
        "strict_runner_source",
        "strict_builder_source",
        "canonical_status",
    ):
        receipt["recovery"][field] = _write(tmp_path / f"{field}.txt", field)
    strict_status = tmp_path / "v3/waiter_status.json"
    strict_guard = tmp_path / "v3/runtime_compute_claim_guard.json"
    receipt["targets"]["status_output"] = strict_status.resolve().as_posix()
    receipt["targets"]["guard_output"] = strict_guard.resolve().as_posix()
    receipt["recovery"]["failed_targets_absent_before_recovery"] = {
        "status_output": {
            "path": (tmp_path / "v2/status.json").resolve().as_posix(),
            "absent": True,
        },
        "guard_output": {
            "path": (tmp_path / "v2/guard.json").resolve().as_posix(),
            "absent": True,
        },
    }
    failure_status = tmp_path / "v2/status.json"
    failure_status.parent.mkdir(parents=True, exist_ok=True)
    failure_status.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "role": (
                    "generation_runtime_claim_guard_strict_replay_wrapper_"
                    "failure_record"
                ),
                "status": "failed",
                "detail": "original_wrapper_missing_pythonpath_import_failure",
                "original_wrapper_deployment": file_identity(original),
                "original_failure_log": file_identity(failure),
                "scope": copy.deepcopy(SCOPE),
            }
        ),
        encoding="utf-8",
    )
    receipt["recovery"]["prior_failure_status"] = file_identity(failure_status)

    _validate_wrapper(
        receipt,
        canonical_waiter={"pid": 100},
        strict_waiter={
            "pid": 200,
            "control_git": {
                "revision": STRICT_REVISION,
                "tree": "1" * 40,
                "branch": "analysis/strict",
                "tracked_dirty": False,
            },
            "source_waiter": {"pid": 50},
        },
        expected_strict_control_revision=STRICT_REVISION,
        strict_status_path=strict_status,
        strict_guard_path=strict_guard,
        require_recovery_binding=True,
    )

    failure.write_text("different failure\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failure log identity differs"):
        _validate_wrapper(
            receipt,
            canonical_waiter={"pid": 100},
            strict_waiter={
                "pid": 200,
                "control_git": {
                    "revision": STRICT_REVISION,
                    "tree": "1" * 40,
                    "branch": "analysis/strict",
                    "tracked_dirty": False,
                },
                "source_waiter": {"pid": 50},
            },
            expected_strict_control_revision=STRICT_REVISION,
            strict_status_path=strict_status,
            strict_guard_path=strict_guard,
            require_recovery_binding=True,
        )
