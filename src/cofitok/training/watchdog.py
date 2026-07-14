from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

from cofitok.reporting import write_json_report


WATCHDOG_SCHEMA_VERSION = 1
MONITOR_FAILURE_EXIT_CODE = 86
MONITOR_STARTUP_EXIT_CODE = 87
MONITOR_SILENCE_EXIT_CODE = 88
MONITOR_PROCESS_EXIT_CODE = 89
TERMINAL_MONITOR_FAILURES = {"failed", "stalled"}


@dataclass(frozen=True)
class TrainingWatchdogConfig:
    monitor_report: Path
    monitor_pid_file: Path
    expected_monitor_name: str
    status_output: Path
    command: tuple[str, ...]
    poll_seconds: float = 30.0
    startup_grace_seconds: float = 600.0
    monitor_silence_seconds: float = 900.0
    monitor_process_grace_seconds: float = 120.0
    termination_grace_seconds: float = 60.0

    def validate(self) -> None:
        if not self.expected_monitor_name.strip():
            raise ValueError("expected monitor name must not be empty")
        if not self.command:
            raise ValueError("watchdog child command must not be empty")
        for name, value in (
            ("poll_seconds", self.poll_seconds),
            ("startup_grace_seconds", self.startup_grace_seconds),
            ("monitor_silence_seconds", self.monitor_silence_seconds),
            ("monitor_process_grace_seconds", self.monitor_process_grace_seconds),
            ("termination_grace_seconds", self.termination_grace_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return payload if isinstance(payload, dict) else None


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _process_exists(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    if os.name == "nt":
        # On Windows, os.kill(pid, 0) emits CTRL_C_EVENT instead of performing
        # the POSIX existence probe. Query the process handle without signaling.
        import ctypes
        from ctypes import wintypes

        process_query_limited_information = 0x1000
        still_active = 259
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD,
            wintypes.BOOL,
            wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetExitCodeProcess.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(wintypes.DWORD),
        )
        kernel32.GetExitCodeProcess.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information,
            False,
            pid,
        )
        if not handle:
            return False
        try:
            exit_code = wintypes.DWORD()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == still_active
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _read_monitor_pid(path: Path) -> int | None:
    try:
        value = int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if value > 0 else None


def _monitor_summary(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if report is None:
        return None
    runs = {}
    for method, row in report.get("runs", {}).items():
        if isinstance(row, dict):
            runs[str(method)] = {
                "complete": row.get("complete"),
                "last_step": row.get("last_step"),
                "progress_fraction": row.get("progress_fraction"),
            }
    return {
        "monitor": report.get("monitor"),
        "status": report.get("status"),
        "stage": report.get("stage"),
        "updated_at": report.get("updated_at"),
        "issues": report.get("issues"),
        "runs": runs,
    }


def _existing_history(path: Path) -> tuple[int, list[dict[str, Any]]]:
    previous = _read_json(path)
    if previous is None:
        return 1, []
    invocation = int(previous.get("invocation", 0)) + 1
    history = previous.get("history", [])
    if not isinstance(history, list):
        history = []
    prior = {
        key: previous.get(key)
        for key in (
            "invocation",
            "status",
            "reason",
            "launched_at",
            "finished_at",
            "child_exit_code",
            "watchdog_exit_code",
        )
    }
    history = [row for row in history if isinstance(row, dict)]
    history.append(prior)
    return invocation, history[-32:]


def _write_status(
    config: TrainingWatchdogConfig,
    *,
    invocation: int,
    history: list[dict[str, Any]],
    status: str,
    reason: str,
    launched_at: datetime,
    child_pid: int,
    monitor_report: dict[str, Any] | None,
    monitor_pid: int | None,
    child_exit_code: int | None = None,
    watchdog_exit_code: int | None = None,
    finished_at: datetime | None = None,
) -> None:
    now = datetime.now(timezone.utc)
    write_json_report(
        config.status_output,
        {
            "schema_version": WATCHDOG_SCHEMA_VERSION,
            "role": "generation_training_watchdog",
            "status": status,
            "reason": reason,
            "invocation": invocation,
            "history": history,
            "launched_at": launched_at.isoformat(),
            "updated_at": now.isoformat(),
            "finished_at": finished_at.isoformat() if finished_at else None,
            "hostname": os.environ.get("HOSTNAME", ""),
            "pid": os.getpid(),
            "child_pid": child_pid,
            "child_exit_code": child_exit_code,
            "watchdog_exit_code": watchdog_exit_code,
            "command": list(config.command),
            "monitor_report_path": config.monitor_report.resolve().as_posix(),
            "monitor_pid_file": config.monitor_pid_file.resolve().as_posix(),
            "expected_monitor_name": config.expected_monitor_name,
            "monitor_pid": monitor_pid,
            "monitor_process_alive": _process_exists(monitor_pid),
            "monitor": _monitor_summary(monitor_report),
            "timing": {
                "poll_seconds": config.poll_seconds,
                "startup_grace_seconds": config.startup_grace_seconds,
                "monitor_silence_seconds": config.monitor_silence_seconds,
                "monitor_process_grace_seconds": config.monitor_process_grace_seconds,
                "termination_grace_seconds": config.termination_grace_seconds,
            },
        },
    )


def _normalize_child_exit_code(return_code: int) -> int:
    return return_code if return_code >= 0 else 128 + abs(return_code)


def _terminate_child(child: subprocess.Popen[Any], grace_seconds: float) -> int:
    return_code = child.poll()
    if return_code is not None:
        return return_code
    try:
        if os.name == "posix":
            os.killpg(child.pid, signal.SIGTERM)
        else:
            child.terminate()
    except ProcessLookupError:
        pass
    try:
        return child.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        except ProcessLookupError:
            pass
        return child.wait()


def run_training_watchdog(config: TrainingWatchdogConfig) -> int:
    config.validate()
    invocation, history = _existing_history(config.status_output)
    launched_at = datetime.now(timezone.utc)
    launched_monotonic = time.monotonic()
    popen_options: dict[str, Any] = {}
    if os.name == "posix":
        popen_options["start_new_session"] = True
    child = subprocess.Popen(config.command, **popen_options)
    requested_signal: int | None = None

    def request_shutdown(signum: int, _frame: Any) -> None:
        nonlocal requested_signal
        requested_signal = signum

    previous_handlers: dict[int, Any] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            previous_handlers[signum] = signal.signal(signum, request_shutdown)
        except (OSError, RuntimeError, ValueError):
            # Signal handlers are unavailable outside the main thread and on
            # some Python/platform combinations. Child monitoring still works.
            pass

    last_monitor_signature: tuple[Any, ...] | None = None
    last_monitor_change = launched_monotonic
    saw_fresh_monitor = False
    monitor_dead_since: float | None = None
    latest_monitor: dict[str, Any] | None = None
    latest_monitor_pid: int | None = None
    abort_reason: str | None = None
    watchdog_exit_code: int | None = None

    try:
        while child.poll() is None:
            now_monotonic = time.monotonic()
            if requested_signal is not None:
                abort_reason = f"watchdog_received_signal_{requested_signal}"
                watchdog_exit_code = 128 + requested_signal
                break

            latest_monitor = _read_json(config.monitor_report)
            latest_monitor_pid = _read_monitor_pid(config.monitor_pid_file)
            monitor_updated_at = _parse_timestamp(
                latest_monitor.get("updated_at") if latest_monitor else None
            )
            monitor_name_matches = (
                latest_monitor is not None
                and latest_monitor.get("monitor") == config.expected_monitor_name
            )
            monitor_is_fresh = (
                monitor_name_matches
                and monitor_updated_at is not None
                and monitor_updated_at >= launched_at - timedelta(seconds=5)
            )
            if monitor_is_fresh:
                saw_fresh_monitor = True
                signature = (
                    latest_monitor.get("updated_at"),
                    latest_monitor.get("status"),
                    latest_monitor.get("stage"),
                )
                if signature != last_monitor_signature:
                    last_monitor_signature = signature
                    last_monitor_change = now_monotonic
                if latest_monitor.get("status") in TERMINAL_MONITOR_FAILURES:
                    abort_reason = (
                        "monitor_reported_" + str(latest_monitor.get("status"))
                    )
                    watchdog_exit_code = MONITOR_FAILURE_EXIT_CODE
                    break

            monitor_process_alive = _process_exists(latest_monitor_pid)
            monitor_status = latest_monitor.get("status") if latest_monitor else None
            if monitor_process_alive or monitor_status == "pass":
                monitor_dead_since = None
            elif monitor_dead_since is None:
                monitor_dead_since = now_monotonic
            elif (
                now_monotonic - monitor_dead_since
                > config.monitor_process_grace_seconds
            ):
                abort_reason = "monitor_process_missing"
                watchdog_exit_code = MONITOR_PROCESS_EXIT_CODE
                break

            elapsed = now_monotonic - launched_monotonic
            if not saw_fresh_monitor and elapsed > config.startup_grace_seconds:
                abort_reason = "monitor_did_not_publish_fresh_status"
                watchdog_exit_code = MONITOR_STARTUP_EXIT_CODE
                break
            if (
                saw_fresh_monitor
                and now_monotonic - last_monitor_change
                > config.monitor_silence_seconds
            ):
                abort_reason = "monitor_status_stopped_updating"
                watchdog_exit_code = MONITOR_SILENCE_EXIT_CODE
                break

            _write_status(
                config,
                invocation=invocation,
                history=history,
                status="running",
                reason="child_and_monitor_active",
                launched_at=launched_at,
                child_pid=child.pid,
                monitor_report=latest_monitor,
                monitor_pid=latest_monitor_pid,
            )
            try:
                child.wait(timeout=config.poll_seconds)
            except subprocess.TimeoutExpired:
                pass

        if abort_reason is not None and watchdog_exit_code is not None:
            _write_status(
                config,
                invocation=invocation,
                history=history,
                status="terminating",
                reason=abort_reason,
                launched_at=launched_at,
                child_pid=child.pid,
                monitor_report=latest_monitor,
                monitor_pid=latest_monitor_pid,
                watchdog_exit_code=watchdog_exit_code,
            )
            child_exit_code = _terminate_child(child, config.termination_grace_seconds)
            finished_at = datetime.now(timezone.utc)
            _write_status(
                config,
                invocation=invocation,
                history=history,
                status="failed",
                reason=abort_reason,
                launched_at=launched_at,
                child_pid=child.pid,
                monitor_report=latest_monitor,
                monitor_pid=latest_monitor_pid,
                child_exit_code=child_exit_code,
                watchdog_exit_code=watchdog_exit_code,
                finished_at=finished_at,
            )
            return watchdog_exit_code

        child_exit_code = child.wait()
        normalized_exit_code = _normalize_child_exit_code(child_exit_code)
        finished_at = datetime.now(timezone.utc)
        _write_status(
            config,
            invocation=invocation,
            history=history,
            status="passed" if child_exit_code == 0 else "failed",
            reason="child_completed" if child_exit_code == 0 else "child_failed",
            launched_at=launched_at,
            child_pid=child.pid,
            monitor_report=latest_monitor,
            monitor_pid=latest_monitor_pid,
            child_exit_code=child_exit_code,
            watchdog_exit_code=normalized_exit_code,
            finished_at=finished_at,
        )
        return normalized_exit_code
    finally:
        if child.poll() is None:
            _terminate_child(child, config.termination_grace_seconds)
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except (OSError, RuntimeError, ValueError):
                pass
