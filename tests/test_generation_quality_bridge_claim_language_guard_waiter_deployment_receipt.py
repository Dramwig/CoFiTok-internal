from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from scripts import (
    build_generation_quality_bridge_claim_language_guard_waiter_deployment_receipt as receipt,
)
from scripts import (
    run_generation_quality_bridge_claim_language_guard_waiter as waiter,
)


CONTROL_GIT = {
    "path": "/checkouts/control",
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/claim-language-guard-waiter",
    "tracked_dirty": False,
}
SOURCE_GIT = {
    "path": "/checkouts/source",
    "revision": "c" * 40,
    "tree": "d" * 40,
    "branch": "analysis/quality-claim-waiter",
    "tracked_dirty": False,
}
FORMAL_GIT = {
    "path": "/formal",
    "revision": "e" * 40,
    "tree": "f" * 40,
    "branch": "scale/generative-system",
    "tracked_dirty": False,
}
PATHS = {
    "project": CONTROL_GIT["path"],
    "source_project": SOURCE_GIT["path"],
    "formal_project": FORMAL_GIT["path"],
    "source_pid": 123,
    "source_output_root": "/outputs/source",
    "source_status": "/outputs/source/reports/waiter_status.json",
    "source_pid_file": "/outputs/source/reports/waiter.pid",
    "source_report": "/outputs/source/reports/qualification.json",
    "output_root": "/outputs/guard",
    "status_output": "/outputs/guard/reports/waiter_status.json",
    "pid_file": "/outputs/guard/reports/waiter.pid",
    "guard_output": "/outputs/guard/reports/claim_language_guard.json",
}
PYTHON = "/env/bin/python3.10"


def _identity(path: str, *, empty: bool = False) -> dict[str, Any]:
    return {
        "path": path,
        "bytes": 0 if empty else 100,
        "sha256": "1" * 64,
    }


def _expected() -> dict[str, Any]:
    return {
        "control_git": receipt._git_contract(CONTROL_GIT),
        "source_kind": waiter.SOURCE_KIND,
        "source_waiter": {
            "pid": PATHS["source_pid"],
            "control_git": receipt._git_contract(SOURCE_GIT),
            "output_root": PATHS["source_output_root"],
            "status": PATHS["source_status"],
            "pid_file": PATHS["source_pid_file"],
        },
        "source_report": PATHS["source_report"],
        "output_root": PATHS["output_root"],
        "guard_output": PATHS["guard_output"],
    }


def _process(pid: int) -> dict[str, Any]:
    expected = _expected()
    argv = [
        PYTHON,
        "scripts/run_generation_quality_bridge_claim_language_guard_waiter.py",
        "--project",
        PATHS["project"],
        "--source-output-root",
        PATHS["source_output_root"],
        "--source-status",
        PATHS["source_status"],
        "--source-pid-file",
        PATHS["source_pid_file"],
        "--source-report",
        PATHS["source_report"],
        "--output-root",
        PATHS["output_root"],
        "--status-output",
        PATHS["status_output"],
        "--pid-file",
        PATHS["pid_file"],
        "--guard-output",
        PATHS["guard_output"],
        "--expected-source-pid",
        str(PATHS["source_pid"]),
        "--expected-control-revision",
        CONTROL_GIT["revision"],
        "--expected-control-tree",
        CONTROL_GIT["tree"],
        "--expected-control-branch",
        CONTROL_GIT["branch"],
        "--expected-source-control-revision",
        SOURCE_GIT["revision"],
        "--expected-source-control-tree",
        SOURCE_GIT["tree"],
        "--expected-source-control-branch",
        SOURCE_GIT["branch"],
        "--poll-seconds",
        "60",
        "--timeout-seconds",
        "2592000",
    ]
    assert expected["source_kind"] == "quality_bridge_100k"
    return {
        "pid": pid,
        "ppid": 1,
        "start_ticks": 123456,
        "cwd": PATHS["project"],
        "exe": PYTHON,
        "argv": argv,
        "environment": {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": f"{PATHS['project']}:{PATHS['project']}/src",
        },
    }


def _source_process() -> dict[str, Any]:
    return {
        "pid": PATHS["source_pid"],
        "ppid": 1,
        "start_ticks": 654321,
        "cwd": PATHS["source_project"],
        "exe": PYTHON,
        "argv": [
            PYTHON,
            "scripts/run_generation_quality_bridge_claim_qualification_waiter.py",
            "--output-root",
            PATHS["source_output_root"],
        ],
        "environment": {
            "CUDA_VISIBLE_DEVICES": "-1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    }


def _inputs() -> dict[str, Any]:
    expected = _expected()
    source_state = {
        "status": "waiting",
        "detail": "waiting_for_terminal_quality_and_uncertainty_sources",
        "phase": "source",
        "pid": PATHS["source_pid"],
        "terminal": False,
        "qualification": None,
        "updated_at": "2026-08-18T00:00:00+00:00",
    }
    waiter_pid = 456
    return {
        "expected_control_git": CONTROL_GIT,
        "expected_source_git": SOURCE_GIT,
        "expected_formal_git": FORMAL_GIT,
        "expected_python": PYTHON,
        "expected_paths": PATHS,
        "bundle": {
            "identity": _identity("/tmp/waiter.bundle"),
            "heads": [
                {
                    "revision": CONTROL_GIT["revision"],
                    "ref": f"refs/heads/{CONTROL_GIT['branch']}",
                }
            ],
            "prerequisites": [SOURCE_GIT["revision"]],
            "verification": {"status": "pass", "stdout": "", "stderr": ""},
        },
        "control_checkout": CONTROL_GIT,
        "source_checkout": SOURCE_GIT,
        "formal_checkout": FORMAL_GIT,
        "waiter_status": {
            "schema_version": waiter.WAITER_SCHEMA_VERSION,
            "role": waiter.WAITER_ROLE,
            "status": "waiting",
            "detail": "waiting_for_quality_claim_source",
            "phase": "source",
            "pid": waiter_pid,
            "expected": expected,
            "source_waiter": source_state,
            "source": None,
            "claim_language_guard": None,
            "claim_boundary": waiter.CLAIM_BOUNDARY,
        },
        "waiter_status_identity": _identity(PATHS["status_output"]),
        "waiter_pid": {
            "schema_version": 1,
            "role": waiter.WAITER_ROLE,
            "pid": waiter_pid,
            "expected_control_revision": CONTROL_GIT["revision"],
            "expected_source_pid": PATHS["source_pid"],
        },
        "waiter_pid_identity": _identity(PATHS["pid_file"]),
        "waiter_log_identity": _identity("/outputs/guard/reports/waiter.log", empty=True),
        "waiter_process": _process(waiter_pid),
        "source_status": {
            "status": "waiting",
            "role": waiter.source_waiter.WAITER_ROLE,
        },
        "source_status_identity": _identity(PATHS["source_status"]),
        "source_state": source_state,
        "source_pid": {
            "role": waiter.source_waiter.WAITER_ROLE,
            "pid": PATHS["source_pid"],
            "expected_control_revision": SOURCE_GIT["revision"],
        },
        "source_pid_identity": _identity(PATHS["source_pid_file"]),
        "source_process": _source_process(),
        "source_report_present": False,
        "guard_output_present": False,
        "gpu_compute_rows": [
            {
                "pid": 999,
                "process_name": "/env/bin/python",
                "used_memory_mib": 85000,
            }
        ],
        "hostname": "server",
        "created_at": "2026-08-18T00:00:00+00:00",
    }


def test_build_receipt_binds_waiter_source_and_gpu_isolation() -> None:
    report = receipt.build_receipt(**_inputs())

    assert report["status"] == "pass"
    assert report["waiter"]["process"]["environment"]["CUDA_VISIBLE_DEVICES"] == "-1"
    assert report["source"]["claim_report_present"] is False
    assert report["initial_state"]["guard_output_present"] is False
    assert report["claim_boundary"]["waiter_launch_authorization_allowed"] is False


def test_build_receipt_allows_source_heartbeat_timestamp_to_advance() -> None:
    inputs = _inputs()
    inputs["source_state"] = copy.deepcopy(inputs["source_state"])
    inputs["source_state"]["updated_at"] = "2026-08-18T00:01:00+00:00"

    report = receipt.build_receipt(**inputs)

    assert report["status"] == "pass"


def test_build_receipt_rejects_authorizing_waiter_boundary() -> None:
    inputs = _inputs()
    inputs["waiter_status"]["claim_boundary"] = dict(waiter.CLAIM_BOUNDARY)
    inputs["waiter_status"]["claim_boundary"]["training_launch_allowed"] = True

    with pytest.raises(ValueError, match="initial status differs"):
        receipt.build_receipt(**inputs)


def test_build_receipt_rejects_bundle_prerequisite_drift() -> None:
    inputs = _inputs()
    inputs["bundle"]["prerequisites"] = ["9" * 40]

    with pytest.raises(ValueError, match="bundle contract differs"):
        receipt.build_receipt(**inputs)


def test_build_receipt_rejects_visible_cuda_waiter() -> None:
    inputs = _inputs()
    inputs["waiter_process"]["environment"]["CUDA_VISIBLE_DEVICES"] = "0"

    with pytest.raises(ValueError, match="process identity differs"):
        receipt.build_receipt(**inputs)


def test_build_receipt_rejects_waiter_in_gpu_compute_table() -> None:
    inputs = _inputs()
    inputs["gpu_compute_rows"].append(
        {
            "pid": inputs["waiter_status"]["pid"],
            "process_name": PYTHON,
            "used_memory_mib": 100,
        }
    )

    with pytest.raises(ValueError, match="unexpectedly allocated GPU"):
        receipt.build_receipt(**inputs)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("source_report_present", "source state differs"),
        ("guard_output_present", "source state differs"),
    ],
)
def test_build_receipt_rejects_nonempty_terminal_state(
    field: str,
    message: str,
) -> None:
    inputs = _inputs()
    inputs[field] = True

    with pytest.raises(ValueError, match=message):
        receipt.build_receipt(**inputs)


def test_build_receipt_rejects_dirty_formal_checkout() -> None:
    inputs = _inputs()
    dirty = copy.deepcopy(FORMAL_GIT)
    dirty["tracked_dirty"] = True
    inputs["formal_checkout"] = dirty

    with pytest.raises(ValueError, match="Git identity is invalid"):
        receipt.build_receipt(**inputs)


def test_build_receipt_rejects_source_pid_drift() -> None:
    inputs = _inputs()
    inputs["source_pid"]["pid"] = PATHS["source_pid"] + 1

    with pytest.raises(ValueError, match="source waiter process differs"):
        receipt.build_receipt(**inputs)
