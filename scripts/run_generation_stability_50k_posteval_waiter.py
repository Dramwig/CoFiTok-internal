from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report


ROLE = "generation_stability_50k_posteval_waiter"
EXPECTED_PAIR_STEPS = 50_000
EXPECTED_PAIR_IMAGES = 3_200_000
EXPECTED_EFFECTIVE_BATCH = 64


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError("monitor timestamp is missing")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("monitor timestamp is not timezone-aware")
    return parsed.astimezone(timezone.utc)


def validate_monitor(
    monitor: dict[str, Any],
    *,
    expected_name: str,
    expected_training_revision: str,
    expected_training_branch: str,
    now: datetime | None = None,
    silence_seconds: float = 900.0,
) -> dict[str, Any]:
    if monitor.get("monitor") != expected_name:
        raise ValueError("stability 50K monitor name mismatch")
    git = monitor.get("git")
    if (
        not isinstance(git, dict)
        or git.get("revision") != expected_training_revision
        or git.get("branch") != expected_training_branch
        or git.get("tracked_dirty") is not False
    ):
        raise ValueError("stability 50K monitor Git identity mismatch")
    status = str(monitor.get("status", ""))
    stage = str(monitor.get("stage", ""))
    if status not in {"waiting", "running", "pass", "failed", "stalled"}:
        raise ValueError(f"unsupported stability 50K monitor status: {status}")
    updated_at = _parse_timestamp(monitor.get("updated_at"))
    age_seconds = max(
        0.0,
        ((now or _utc_now()).astimezone(timezone.utc) - updated_at).total_seconds(),
    )
    if status in {"waiting", "running"} and age_seconds > silence_seconds:
        raise RuntimeError(
            f"stability 50K monitor is stale by {age_seconds:.1f} seconds"
        )
    issues = monitor.get("issues")
    if not isinstance(issues, list):
        raise ValueError("stability 50K monitor issues are malformed")
    return {
        "status": status,
        "stage": stage,
        "updated_at": updated_at.isoformat(),
        "age_seconds": age_seconds,
        "issues": list(issues),
    }


def validate_pair_summary(
    summary: dict[str, Any],
    *,
    expected_training_revision: str,
    expected_training_branch: str,
) -> dict[str, Any]:
    git = summary.get("git")
    if (
        summary.get("schema_version") != 1
        or summary.get("status") != "completed"
        or summary.get("stage") != "stability_matched_50k"
        or not isinstance(git, dict)
        or git.get("revision") != expected_training_revision
        or git.get("branch") != expected_training_branch
        or int(summary.get("completed_steps_per_method", -1))
        != EXPECTED_PAIR_STEPS
        or int(summary.get("images_seen_per_method", -1))
        != EXPECTED_PAIR_IMAGES
        or int(summary.get("effective_batch_size", -1))
        != EXPECTED_EFFECTIVE_BATCH
        or summary.get("formal_300k_authorization_allowed") is not False
        or summary.get("formal_ema_sampling_gate_required") is not True
    ):
        raise ValueError("stability 50K pair summary contract mismatch")
    pair = summary.get("training_pair")
    if not isinstance(pair, dict) or pair.get("status") != "pass":
        raise ValueError("stability 50K training pair did not pass")
    sources = summary.get("sources")
    if not isinstance(sources, dict) or set(sources) != {
        "cofitok_training",
        "dense_training",
        "decision_validation",
        "config_validation",
    }:
        raise ValueError("stability 50K pair summary sources are incomplete")
    return {
        "status": "verified",
        "revision": expected_training_revision,
        "branch": expected_training_branch,
        "completed_steps_per_method": EXPECTED_PAIR_STEPS,
        "images_seen_per_method": EXPECTED_PAIR_IMAGES,
    }


def _git_identity(project: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _verify_evaluation_checkout(
    project: Path,
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    identity = _git_identity(project)
    if identity != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise RuntimeError("post-evaluation checkout Git identity mismatch")
    return identity


def _gpu_compute_pids() -> list[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pids: list[int] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if stripped:
            pids.append(int(stripped))
    return pids


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _status(
    *,
    status: str,
    detail: str,
    expected: dict[str, Any],
    monitor: dict[str, Any] | None = None,
    pair_summary: dict[str, Any] | None = None,
    child_pid: int | None = None,
    child_exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "child_pid": child_pid,
        "child_exit_code": child_exit_code,
        "expected": expected,
        "monitor": monitor,
        "pair_summary": pair_summary,
        "updated_at": _utc_now().isoformat(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for a bound matched stability 50K pair, then run formal EMA "
            "post-evaluation from a separate clean evaluation revision."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--monitor-report", type=Path, required=True)
    parser.add_argument("--pair-summary", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--posteval-runbook", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--expected-monitor-name", required=True)
    parser.add_argument("--expected-stability-decision-sha256", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-evaluation-revision", required=True)
    parser.add_argument("--expected-evaluation-branch", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=604_800.0)
    parser.add_argument("--poll-seconds", type=float, default=300.0)
    parser.add_argument("--monitor-silence-seconds", type=float, default=900.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if (
        args.timeout_seconds <= 0.0
        or args.poll_seconds <= 0.0
        or args.monitor_silence_seconds <= 0.0
    ):
        raise ValueError("waiter timing values must be positive")
    project = args.project.resolve()
    runbook = args.posteval_runbook.resolve()
    if not runbook.is_file():
        raise FileNotFoundError(runbook)
    expected = {
        "monitor_name": args.expected_monitor_name,
        "stability_decision_sha256": args.expected_stability_decision_sha256,
        "training_revision": args.expected_training_revision,
        "training_branch": args.expected_training_branch,
        "evaluation_revision": args.expected_evaluation_revision,
        "evaluation_branch": args.expected_evaluation_branch,
        "formal_300k_allowed": False,
    }
    _verify_evaluation_checkout(
        project,
        expected_revision=args.expected_evaluation_revision,
        expected_branch=args.expected_evaluation_branch,
    )
    deadline = time.monotonic() + args.timeout_seconds
    monitor_observation: dict[str, Any] | None = None
    summary_observation: dict[str, Any] | None = None
    while True:
        if args.monitor_report.is_file():
            monitor = _read_object(args.monitor_report)
            monitor_observation = validate_monitor(
                monitor,
                expected_name=args.expected_monitor_name,
                expected_training_revision=args.expected_training_revision,
                expected_training_branch=args.expected_training_branch,
                silence_seconds=args.monitor_silence_seconds,
            )
            if monitor_observation["status"] in {"failed", "stalled"}:
                raise RuntimeError(
                    "stability 50K training monitor reached "
                    f"{monitor_observation['status']}"
                )
            pair_ready = (
                monitor_observation["status"] == "pass"
                and monitor_observation["stage"] == "complete"
                and not monitor_observation["issues"]
                and args.pair_summary.is_file()
            )
            if pair_ready:
                summary_observation = validate_pair_summary(
                    _read_object(args.pair_summary),
                    expected_training_revision=args.expected_training_revision,
                    expected_training_branch=args.expected_training_branch,
                )
                gpu_pids = _gpu_compute_pids()
                if not gpu_pids:
                    break
                detail = "pair_complete_waiting_for_idle_gpu"
            else:
                detail = "waiting_for_completed_training_pair"
        else:
            detail = "waiting_for_training_monitor"
        write_json_report(
            args.status_output,
            _status(
                status="waiting",
                detail=detail,
                expected=expected,
                monitor=monitor_observation,
                pair_summary=summary_observation,
            ),
        )
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the matched stability 50K pair")
        time.sleep(args.poll_seconds)

    _verify_evaluation_checkout(
        project,
        expected_revision=args.expected_evaluation_revision,
        expected_branch=args.expected_evaluation_branch,
    )
    pair_source = _source(args.pair_summary)
    environment = os.environ.copy()
    environment.update(
        {
            "PYTHON": sys.executable,
            "CHECKPOINT_ROOT": str(args.checkpoint_root.resolve()),
            "EXPECTED_STABILITY_DECISION_SHA256": (
                args.expected_stability_decision_sha256
            ),
            "EXPECTED_TRAINING_REVISION": args.expected_training_revision,
            "EXPECTED_TRAINING_BRANCH": args.expected_training_branch,
            "EXPECTED_TARGET_REVISION": args.expected_evaluation_revision,
            "EXPECTED_TARGET_BRANCH": args.expected_evaluation_branch,
        }
    )
    child = subprocess.Popen(
        ["bash", str(runbook)],
        cwd=project,
        env=environment,
    )
    write_json_report(
        args.status_output,
        _status(
            status="running",
            detail="formal_ema_postevaluation_running",
            expected=expected,
            monitor=monitor_observation,
            pair_summary={**(summary_observation or {}), "source": pair_source},
            child_pid=child.pid,
        ),
    )
    exit_code = child.wait()
    final_status = "pass" if exit_code == 0 else "failed"
    write_json_report(
        args.status_output,
        _status(
            status=final_status,
            detail=(
                "formal_ema_postevaluation_completed"
                if exit_code == 0
                else "formal_ema_postevaluation_failed"
            ),
            expected=expected,
            monitor=monitor_observation,
            pair_summary={**(summary_observation or {}), "source": pair_source},
            child_pid=child.pid,
            child_exit_code=exit_code,
        ),
    )
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        try:
            parsed = _parse_args()
            expected_error = {
                "monitor_name": parsed.expected_monitor_name,
                "stability_decision_sha256": (
                    parsed.expected_stability_decision_sha256
                ),
                "training_revision": parsed.expected_training_revision,
                "training_branch": parsed.expected_training_branch,
                "evaluation_revision": parsed.expected_evaluation_revision,
                "evaluation_branch": parsed.expected_evaluation_branch,
                "formal_300k_allowed": False,
            }
            write_json_report(
                parsed.status_output,
                _status(
                    status="failed",
                    detail=f"{type(error).__name__}: {error}",
                    expected=expected_error,
                ),
            )
        finally:
            raise
