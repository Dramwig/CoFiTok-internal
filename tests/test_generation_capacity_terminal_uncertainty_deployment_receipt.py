from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts import (
    build_generation_capacity_terminal_uncertainty_deployment_receipt as receipt,
)
from scripts import run_generation_capacity_terminal_uncertainty_waiter as waiter


CONTROL = {
    "path": "/checkouts/control",
    "revision": "1" * 40,
    "tree": "2" * 40,
    "branch": "analysis/generation-capacity-terminal-uncertainty-v1",
    "tracked_dirty": False,
}
EVALUATOR = {
    "path": "/checkouts/evaluator",
    "revision": "3" * 40,
    "tree": "4" * 40,
    "branch": "",
    "tracked_dirty": False,
}
OUTPUT = "/checkpoints/capacity-terminal-uncertainty"
PID = 12345


def _identity(path: str, *, size: int = 1) -> dict[str, Any]:
    return {"path": path, "bytes": size, "sha256": "a" * 64}


def _status(source_kind: str = "capacity_completion_100k") -> dict[str, Any]:
    source_git = {
        "execution": {
            "revision": "5" * 40,
            "tree": "6" * 40,
            "branch": "scale/capacity-completion",
        },
        "training": {
            "revision": "7" * 40,
            "tree": "8" * 40,
            "branch": "scale/capacity-training",
        },
        "result": {
            "revision": "9" * 40,
            "tree": "b" * 40,
            "branch": "analysis/capacity-result",
        },
        "evaluation": {"revision": "", "tree": "", "branch": ""},
    }
    return {
        "schema_version": waiter.WAITER_SCHEMA_VERSION,
        "role": waiter.WAITER_ROLE,
        "status": "waiting",
        "detail": "waiting_for_capacity_terminal_source_anchor",
        "phase": "source",
        "hostname": "host",
        "pid": PID,
        "expected": {
            "control_git": {key: CONTROL[key] for key in CONTROL if key != "path"},
            "evaluator_git": {
                key: EVALUATOR[key] for key in EVALUATOR if key != "path"
            },
            "source_kind": source_kind,
            "source_anchor_path": "/checkpoints/future-source.json",
            "source_git": source_git,
            "output_root": OUTPUT,
            "manifest_output": f"{OUTPUT}/reports/execution_manifest.json",
            "qualification_output": f"{OUTPUT}/reports/qualification.json",
            "gpu_slot_lock_target": "/checkpoints/shared-gpu-slot",
            "real_feature_cache_source": {
                "path": "/checkpoints/real-cache.pt",
                "bytes": 409601577,
                "sha256": "c" * 64,
            },
            "required_idle_polls": 5,
        },
        "source_anchor": None,
        "execution_manifest": None,
        "real_feature_cache": None,
        "stage": None,
        "audit": None,
        "statistical_claim_qualification": None,
        "gpu_compute_rows": [],
        "competing_uncertainty_processes": [],
        "idle_polls": 0,
        "required_idle_polls": 5,
        "claim_boundary": copy.deepcopy(waiter.CLAIM_BOUNDARY),
        "updated_at": "2026-08-18T00:00:00+00:00",
    }


def _build(**overrides: Any) -> dict[str, Any]:
    status = overrides.pop("waiter_status", _status())
    source_kind = status["expected"]["source_kind"]
    process = {
        "pid": PID,
        "ppid": 1,
        "elapsed_seconds": 10,
        "cwd": CONTROL["path"],
        "command": (
            "python scripts/run_generation_capacity_terminal_uncertainty_waiter.py "
            f"--source-kind {source_kind} "
            f"--source-anchor {status['expected']['source_anchor_path']} "
            f"--output-root {OUTPUT} "
            f"--expected-control-revision {CONTROL['revision']}"
        ),
    }
    arguments = {
        "source_kind": source_kind,
        "output_root": OUTPUT,
        "source_anchor_present": False,
        "control_checkout": CONTROL,
        "evaluator_checkout": EVALUATOR,
        "receipt_builder_checkout": {
            "path": "/checkouts/receipt-builder",
            "revision": "e" * 40,
            "tree": "f" * 40,
            "branch": "",
            "tracked_dirty": False,
        },
        "waiter_bundle": {
            "identity": _identity("/tmp/control.bundle", size=100),
            "heads": [
                {
                    "revision": CONTROL["revision"],
                    "ref": f"refs/heads/{CONTROL['branch']}",
                }
            ],
            "prerequisites": ["d" * 40],
        },
        "receipt_builder_bundle": {
            "identity": _identity("/tmp/receipt-builder.bundle", size=50),
            "heads": [
                {
                    "revision": "e" * 40,
                    "ref": "refs/heads/analysis/receipt-builder",
                }
            ],
            "prerequisites": [CONTROL["revision"]],
        },
        "waiter_status": status,
        "waiter_status_identity": _identity(f"{OUTPUT}/reports/waiter_status.json"),
        "pid_payload": {
            "schema_version": 1,
            "role": waiter.WAITER_ROLE,
            "pid": PID,
            "hostname": "host",
            "source_kind": source_kind,
            "expected_control_revision": CONTROL["revision"],
            "started_at": "2026-08-18T00:00:00+00:00",
        },
        "pid_identity": _identity(f"{OUTPUT}/reports/waiter.pid"),
        "log_identity": _identity(f"{OUTPUT}/reports/waiter.log", size=0),
        "process": process,
        "gpu_compute_rows": [{"pid": 999, "process_name": "trainer", "used_memory_mib": 1}],
        "hostname": "host",
        "created_at": "2026-08-18T00:00:01+00:00",
    }
    arguments.update(overrides)
    return receipt.build_receipt(**arguments)


@pytest.mark.parametrize(
    "source_kind",
    ["capacity_completion_100k", "capacity_full_300k"],
)
def test_builds_non_authorizing_active_receipt(source_kind: str) -> None:
    status = _status(source_kind)
    built = _build(waiter_status=status)
    assert built["status"] == "active"
    assert built["source_kind"] == source_kind
    assert built["waiter"]["pid"] == PID
    assert built["source_contract"]["anchor_present_at_deployment"] is False
    assert built["authorization_boundary"]["training_launch_allowed"] is False
    assert built["authorization_boundary"]["full_300k_launch_allowed"] is False
    assert built["authorization_boundary"]["release_allowed"] is False
    assert built["validation"]["waiter_pid_not_in_gpu_compute"] is True


def test_rejects_authorizing_claim_boundary() -> None:
    status = _status()
    status["claim_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="claim boundary"):
        _build(waiter_status=status)


def test_rejects_process_pid_mismatch() -> None:
    process = {
        "pid": PID + 1,
        "cwd": CONTROL["path"],
        "command": "scripts/run_generation_capacity_terminal_uncertainty_waiter.py",
    }
    with pytest.raises(ValueError, match="PID binding"):
        _build(process=process)


def test_rejects_source_anchor_present_at_deployment() -> None:
    with pytest.raises(ValueError, match="already existed"):
        _build(source_anchor_present=True)
