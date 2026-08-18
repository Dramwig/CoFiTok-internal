from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_generation_quality_bridge_ipc_recovery_supervisor_v2.py"
)


def load_supervisor() -> ModuleType:
    if sys.platform == "win32" and "fcntl" not in sys.modules:
        fcntl = ModuleType("fcntl")
        fcntl.LOCK_EX = 2
        fcntl.LOCK_NB = 4
        fcntl.flock = lambda *_args, **_kwargs: None
        sys.modules["fcntl"] = fcntl
    name = "cofitok_quality_bridge_ipc_recovery_supervisor_v2_test"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pin_watchdog() -> dict[str, object]:
    return {
        "report": {
            "status": "failed",
            "reason": "child_failed",
            "watchdog_exit_code": 1,
            "child_exit_code": 1,
        }
    }


def test_exact_pin_memory_resource_sharer_failure_is_bounded() -> None:
    module = load_supervisor()
    log = "\n".join(module.PIN_MEMORY_FAILURE_MARKERS)

    assert module.classify_retryable_exit(
        1,
        watchdog=pin_watchdog(),
        monitor={
            "status": "failed",
            "issues": ["matched queue incomplete without active trainer/runbook"],
        },
        controller_log_text=log,
    ) == (True, "bounded_pin_memory_resource_sharer_failure")


def test_exit_one_requires_all_exact_failure_markers() -> None:
    module = load_supervisor()

    retryable, reason = module.classify_retryable_exit(
        1,
        watchdog=pin_watchdog(),
        monitor={"status": "failed", "issues": []},
        controller_log_text="RuntimeError: Pin memory thread exited unexpectedly",
    )

    assert retryable is False
    assert reason == "exit_1_lacks_exact_pin_memory_ipc_signature"


def test_integrity_issue_is_never_retried_as_ipc_failure() -> None:
    module = load_supervisor()
    log = "\n".join(module.PIN_MEMORY_FAILURE_MARKERS)

    retryable, reason = module.classify_retryable_exit(
        1,
        watchdog=pin_watchdog(),
        monitor={"status": "failed", "issues": ["checkpoint integrity mismatch"]},
        controller_log_text=log,
    )

    assert retryable is False
    assert reason == "exit_1_has_nontransient_monitor_issue"


def test_latest_controller_log_uses_latest_supervisor_segment(
    tmpdir: object,
) -> None:
    module = load_supervisor()
    tmp_path = Path(str(tmpdir))
    direct = tmp_path / "direct.log"
    supervisor = tmp_path / "supervisor.log"
    direct.write_text("direct failure", encoding="utf-8")
    supervisor.write_text(
        "old failure\n"
        "[time] launching bounded recovery controller\n"
        "latest child output\n",
        encoding="utf-8",
    )
    module.DIRECT_CONTROLLER_LOG = direct
    module.SUPERVISOR_LOG = supervisor

    observed = module.latest_controller_log_text()

    assert observed.startswith("launching bounded recovery controller")
    assert "latest child output" in observed
    assert "old failure" not in observed


def test_recovery_is_limited_and_uses_distinct_v2_artifacts() -> None:
    module = load_supervisor()

    assert module.MAX_RECOVERY_ATTEMPTS == 3
    assert module.SUPERVISOR_STATUS.name == "recovery_supervisor_v2_status.json"
    assert module.SUPERVISOR_LOCK.name == "recovery_supervisor_v2.lock"
    assert module.RECOVERY_CONTROLLER_RECEIPT.name == (
        "recovery_controller_v2_launch.json"
    )
    assert (
        "COFITOK_QUALITY_BRIDGE_RECOVERY_SUPERVISOR_V2_SHA256"
        in SCRIPT.read_text(encoding="utf-8")
    )
