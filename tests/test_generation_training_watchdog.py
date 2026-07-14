from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from cofitok.training.watchdog import (
    MONITOR_FAILURE_EXIT_CODE,
    MONITOR_PROCESS_EXIT_CODE,
    MONITOR_SILENCE_EXIT_CODE,
    MONITOR_STARTUP_EXIT_CODE,
    TrainingWatchdogConfig,
    run_training_watchdog,
)


MONITOR_NAME = "test_generation_pair"


def _write_monitor(path: Path, *, status: str, updated_at: datetime) -> None:
    path.write_text(
        json.dumps(
            {
                "monitor": MONITOR_NAME,
                "status": status,
                "stage": "cofitok_training",
                "updated_at": updated_at.isoformat(),
                "issues": ["test issue"] if status in {"failed", "stalled"} else [],
                "runs": {
                    "cofitok": {
                        "complete": False,
                        "last_step": 10,
                        "progress_fraction": 0.1,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _config(tmp_path: Path, command: tuple[str, ...]) -> TrainingWatchdogConfig:
    monitor = tmp_path / "monitor.json"
    pid_file = tmp_path / "monitor.pid"
    pid_file.write_text(str(os.getpid()), encoding="ascii")
    return TrainingWatchdogConfig(
        monitor_report=monitor,
        monitor_pid_file=pid_file,
        expected_monitor_name=MONITOR_NAME,
        status_output=tmp_path / "watchdog.json",
        command=command,
        poll_seconds=0.05,
        startup_grace_seconds=1.0,
        monitor_silence_seconds=1.0,
        monitor_process_grace_seconds=1.0,
        termination_grace_seconds=1.0,
    )


def test_watchdog_passes_through_successful_child(tmp_path: Path) -> None:
    config = _config(tmp_path, (sys.executable, "-c", "print('ok')"))
    _write_monitor(
        config.monitor_report,
        status="running",
        updated_at=datetime.now(timezone.utc),
    )

    exit_code = run_training_watchdog(config)

    assert exit_code == 0
    report = json.loads(config.status_output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"
    assert report["reason"] == "child_completed"
    assert report["child_exit_code"] == 0


def test_watchdog_terminates_child_on_fresh_stalled_monitor(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        (sys.executable, "-c", "import time; time.sleep(60)"),
    )
    _write_monitor(
        config.monitor_report,
        status="stalled",
        updated_at=datetime.now(timezone.utc),
    )

    started = time.monotonic()
    exit_code = run_training_watchdog(config)

    assert time.monotonic() - started < 5.0
    assert exit_code == MONITOR_FAILURE_EXIT_CODE
    report = json.loads(config.status_output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["reason"] == "monitor_reported_stalled"
    assert report["watchdog_exit_code"] == MONITOR_FAILURE_EXIT_CODE


def test_watchdog_ignores_stale_failure_if_child_completes(tmp_path: Path) -> None:
    config = _config(
        tmp_path,
        (sys.executable, "-c", "import time; time.sleep(0.1)"),
    )
    _write_monitor(
        config.monitor_report,
        status="failed",
        updated_at=datetime.now(timezone.utc) - timedelta(days=1),
    )

    exit_code = run_training_watchdog(config)

    assert exit_code == 0
    report = json.loads(config.status_output.read_text(encoding="utf-8"))
    assert report["status"] == "passed"


def test_watchdog_terminates_child_when_fresh_monitor_stops_updating(
    tmp_path: Path,
) -> None:
    base = _config(
        tmp_path,
        (sys.executable, "-c", "import time; time.sleep(60)"),
    )
    config = replace(base, monitor_silence_seconds=0.2)
    _write_monitor(
        config.monitor_report,
        status="running",
        updated_at=datetime.now(timezone.utc),
    )

    exit_code = run_training_watchdog(config)

    assert exit_code == MONITOR_SILENCE_EXIT_CODE
    report = json.loads(config.status_output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["reason"] == "monitor_status_stopped_updating"


def test_watchdog_terminates_child_when_monitor_never_publishes(
    tmp_path: Path,
) -> None:
    base = _config(
        tmp_path,
        (sys.executable, "-c", "import time; time.sleep(60)"),
    )
    config = replace(base, startup_grace_seconds=0.2)

    exit_code = run_training_watchdog(config)

    assert exit_code == MONITOR_STARTUP_EXIT_CODE
    report = json.loads(config.status_output.read_text(encoding="utf-8"))
    assert report["reason"] == "monitor_did_not_publish_fresh_status"


def test_watchdog_terminates_child_when_monitor_process_disappears(
    tmp_path: Path,
) -> None:
    base = _config(
        tmp_path,
        (sys.executable, "-c", "import time; time.sleep(60)"),
    )
    config = replace(base, monitor_process_grace_seconds=0.2)
    config.monitor_pid_file.write_text(str(2**31 - 1), encoding="ascii")
    _write_monitor(
        config.monitor_report,
        status="running",
        updated_at=datetime.now(timezone.utc),
    )

    exit_code = run_training_watchdog(config)

    assert exit_code == MONITOR_PROCESS_EXIT_CODE
    report = json.loads(config.status_output.read_text(encoding="utf-8"))
    assert report["reason"] == "monitor_process_missing"


def test_watchdog_passes_through_child_failure(tmp_path: Path) -> None:
    config = _config(tmp_path, (sys.executable, "-c", "raise SystemExit(7)"))
    _write_monitor(
        config.monitor_report,
        status="running",
        updated_at=datetime.now(timezone.utc),
    )

    exit_code = run_training_watchdog(config)

    assert exit_code == 7
    report = json.loads(config.status_output.read_text(encoding="utf-8"))
    assert report["status"] == "failed"
    assert report["reason"] == "child_failed"
    assert report["child_exit_code"] == 7
