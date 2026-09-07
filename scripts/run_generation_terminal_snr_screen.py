"""Run the authorized terminal-SNR four-arm screen through the proven core.

The capacity-screen controller already implements the exact fresh/resume,
checkpoint, sampling, evaluator, and process-isolation sequence shared by this
screen.  This entrypoint binds that core in a fresh Python process to the
terminal-SNR arm set, seed, authorization schemas, and result builder.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
)
from cofitok.generation.terminal_snr_screen import (
    ARM_NAMES,
    ARM_SPECS,
    EFFECTIVE_BATCH,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    SAMPLE_STEPS,
    STOP_STEP,
)
from cofitok.generation.terminal_snr_screen_arm import (
    build_terminal_snr_screen_arm_validation,
)
from cofitok.generation.terminal_snr_screen_execution import (
    LIVE_SNAPSHOT_ROLE,
    LIVE_SNAPSHOT_SCHEMA,
    terminal_snr_screen_benchmark_root,
    terminal_snr_screen_control_root,
    terminal_snr_screen_execution_lock_path,
    validate_terminal_snr_screen_execution_authorization,
    validate_terminal_snr_screen_launch_receipt_physical,
)
from cofitok.generation.terminal_snr_screen_result import (
    build_terminal_snr_screen_result,
    validate_terminal_snr_screen_result_contract,
)
from cofitok.reporting import write_json_report
try:
    from scripts import run_generation_capacity_screen as core
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    import run_generation_capacity_screen as core  # type: ignore[no-redef]


ROLE = "generation_terminal_snr_screen_controller"
STATUS_SCHEMA = 1
RESULT_FILENAME = "terminal_snr_screen_result.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the authorized terminal-SNR four-arm screen to 10K."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--expected-runtime-selection-sha256", required=True)
    parser.add_argument("--live-snapshot", type=Path, required=True)
    parser.add_argument("--expected-live-snapshot-sha256", required=True)
    for arm in ARM_NAMES:
        flag = arm.replace("_", "-")
        parser.add_argument(
            f"--{flag}-config",
            dest=f"{arm}_config",
            type=Path,
            required=True,
        )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--log-output", type=Path, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _write_status(
    path: Path,
    *,
    status: str,
    stage: str,
    detail: str,
    authorization: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    exit_code: int | None = None,
) -> None:
    payload = {
        "schema_version": STATUS_SCHEMA,
        "role": ROLE,
        "status": status,
        "stage": stage,
        "detail": detail,
        "exit_code": exit_code,
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "hostname": socket.gethostname(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "authorization": dict(authorization),
        "launch_receipt": dict(launch_receipt),
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "frozen_confirmation_allowed": False,
        "full_training_launch_allowed": False,
        "full_300k_launch_allowed": False,
        "promotion_allowed": False,
        "export_allowed": False,
        "release_allowed": False,
        "process_signals_allowed": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _validate_root_for_resume(
    output_root: Path,
    *,
    status_path: Path,
    authorization_identity: Mapping[str, Any],
    launch_identity: Mapping[str, Any],
) -> None:
    if not output_root.is_dir() or output_root.is_symlink():
        raise ValueError("terminal-SNR resume root is not a directory")
    if not status_path.is_file() or status_path.is_symlink():
        raise ValueError("terminal-SNR resume status is missing")
    status = core._read(status_path, "terminal-SNR controller status")
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("role") != ROLE
        or status.get("status") not in {"running", "failed"}
        or status.get("authorization") != dict(authorization_identity)
        or status.get("launch_receipt") != dict(launch_identity)
        or status.get("terminal_status") != "hold"
        or status.get("generation_advantage_proven") is not False
    ):
        raise ValueError("terminal-SNR resume status is not bound to this launch")
    for field in (
        "frozen_confirmation_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_allowed",
        "export_allowed",
        "release_allowed",
        "process_signals_allowed",
    ):
        if status.get(field) is not False:
            raise ValueError(f"terminal-SNR resume status permits {field}")
    if (output_root / RESULT_FILENAME).exists():
        raise ValueError("terminal-SNR result already exists; use a fresh audit")
    allowed = {"training", "evaluations", "arm_validations"}
    unexpected = sorted(
        child.name for child in output_root.iterdir() if child.name not in allowed
    )
    if unexpected:
        raise ValueError(
            "terminal-SNR resume root contains unexpected entries: "
            + ", ".join(unexpected)
        )


def _build_result(
    *,
    project_root: Path,
    output_root: Path,
    preparation: Mapping[str, Any],
    preparation_id: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_id: Mapping[str, Any],
    launch_path: Path,
    launch_sha: str,
    configs: Mapping[str, Path],
) -> Path:
    del launch_path, launch_sha, configs
    arms: dict[str, dict[str, Any]] = {}
    arm_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        path = output_root / "arm_validations" / f"{arm}.json"
        arms[arm] = core._read(path, f"{arm} terminal-SNR arm validation")
        arm_ids[arm] = identity(path)
    report = build_terminal_snr_screen_result(
        preparation=dict(preparation),
        preparation_identity=dict(preparation_id),
        launch_receipt=dict(launch_receipt),
        launch_receipt_identity=dict(launch_id),
        arm_validations=arms,
        arm_validation_identities=arm_ids,
        result_git=checkout_identity(project_root),
    )
    validate_terminal_snr_screen_result_contract(report)
    path = output_root / RESULT_FILENAME
    if path.exists():
        if core._read(path, "terminal-SNR result") != report:
            raise ValueError("existing terminal-SNR result differs from replay")
    else:
        write_json_report(path, report)
    return path


def _bindings() -> dict[str, Any]:
    return {
        "ROLE": ROLE,
        "STATUS_SCHEMA": STATUS_SCHEMA,
        "ARM_NAMES": ARM_NAMES,
        "ARM_SPECS": ARM_SPECS,
        "EFFECTIVE_BATCH": EFFECTIVE_BATCH,
        "STOP_STEP": STOP_STEP,
        "SAMPLE_COUNT": SAMPLE_COUNT,
        "SAMPLE_STEPS": SAMPLE_STEPS,
        "SAMPLE_SEED": SAMPLE_SEED,
        "LIVE_SNAPSHOT_ROLE": LIVE_SNAPSHOT_ROLE,
        "LIVE_SNAPSHOT_SCHEMA": LIVE_SNAPSHOT_SCHEMA,
        "capacity_screen_execution_lock_path": (
            terminal_snr_screen_execution_lock_path
        ),
        "capacity_screen_control_root": terminal_snr_screen_control_root,
        "capacity_screen_benchmark_root": terminal_snr_screen_benchmark_root,
        "validate_capacity_screen_execution_authorization": (
            validate_terminal_snr_screen_execution_authorization
        ),
        "validate_capacity_screen_launch_receipt_contract": (
            validate_terminal_snr_screen_launch_receipt_physical
        ),
        "build_capacity_screen_arm_validation": (
            build_terminal_snr_screen_arm_validation
        ),
        "build_capacity_screen_result": build_terminal_snr_screen_result,
        "validate_capacity_screen_result_contract": (
            validate_terminal_snr_screen_result_contract
        ),
        "parse_args": parse_args,
        "_write_status": _write_status,
        "_validate_root_for_resume": _validate_root_for_resume,
        "_build_result": _build_result,
    }


def _configure_core() -> dict[str, Any]:
    replacements = _bindings()
    saved = {name: getattr(core, name) for name in replacements}
    for name, value in replacements.items():
        setattr(core, name, value)
    return saved


def _restore_core(saved: Mapping[str, Any]) -> None:
    for name, value in saved.items():
        setattr(core, name, value)


def main(argv: Sequence[str] | None = None) -> int:
    saved = _configure_core()
    try:
        return core.main(argv)
    finally:
        _restore_core(saved)


if __name__ == "__main__":
    raise SystemExit(main())
