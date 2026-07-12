from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


DEFAULT_HOST = "pro6000"
DEFAULT_REMOTE_CODE_DIR = "/root/autodl-tmp/CoFiTok/CoFiTok-internal"
DEFAULT_CHECKPOINT_ROOT = "/root/autodl-tmp/CoFiTok/checkpoints"
DEFAULT_REMOTE_PYTHON = "/root/autodl-tmp/conda/envs/pf-vlm/bin/python"
DEFAULT_LOG = "artifacts/logs/next_validation_queue_2026-07-08.log"
DEFAULT_PID_FILE = "artifacts/logs/next_validation_queue_2026-07-08.pid"
DEFAULT_REMOTE_STATUS_JSON = "artifacts/reports/next_validation_queue_status_poll_2026-07-08.json"
SSH_OPTIONS = [
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=8",
    "-o",
    "ConnectionAttempts=1",
    "-o",
    "ServerAliveInterval=4",
    "-o",
    "ServerAliveCountMax=1",
]


REMOTE_STATUS_PROGRAM = r"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

CONFIG = __CONFIG_JSON__


def run_command(command: list[str], timeout: int = 20) -> dict:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except Exception as error:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": repr(error),
        }
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def compact_queue(queue: dict) -> dict:
    first_incomplete = None
    complete_labels = []
    for run in queue.get("runs", []):
        if run.get("status") == "complete":
            complete_labels.append(run.get("label"))
        elif first_incomplete is None:
            first_incomplete = {
                "label": run.get("label"),
                "complete_count": run.get("complete_count"),
                "expected_count": run.get("expected_count"),
                "missing_count": run.get("missing_count"),
            }
    return {
        "status": queue.get("status"),
        "counts": queue.get("counts"),
        "by_type": queue.get("by_type"),
        "complete_run_count": len(complete_labels),
        "complete_labels": complete_labels,
        "first_incomplete": first_incomplete,
    }


def tail_lines(path: Path, line_count: int) -> list[str]:
    if not path.is_file():
        return []
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    except Exception as error:
        return [f"<failed to read log: {error!r}>"]


def main() -> None:
    code_dir = Path(CONFIG["code_dir"])
    os.chdir(code_dir)
    sys.path.insert(0, str(code_dir))
    sys.path.insert(0, str(code_dir / "src"))

    from scripts.inspect_next_validation_queue import inspect_queue

    queue = inspect_queue(
        checkpoint_root=Path(CONFIG["checkpoint_root"]),
        date_tag=CONFIG["date_tag"],
        quality_images=CONFIG["quality_images"],
        sample_count=CONFIG["sample_count"],
        sample_steps=CONFIG["sample_steps"],
    )
    remote_status_json = Path(CONFIG["remote_status_json"])
    remote_status_json.parent.mkdir(parents=True, exist_ok=True)
    remote_status_json.write_text(json.dumps(queue, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pid_file = Path(CONFIG["pid_file"])
    pid = pid_file.read_text(encoding="utf-8").strip() if pid_file.is_file() else ""
    process = run_command(["ps", "-p", pid, "-o", "pid,ppid,stat,etime,cmd"], timeout=10) if pid else {
        "returncode": None,
        "stdout": "",
        "stderr": "pid file missing",
    }
    pgrep = run_command(
        [
            "pgrep",
            "-af",
            "train_short.py|evaluate_quality.py|evaluate_checkpoint.py|sample_checkpoint.py|evaluate_generated_samples.py|next_validation_queue",
        ],
        timeout=10,
    )
    gpu = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader",
        ],
        timeout=15,
    )
    gpu_apps = run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        timeout=15,
    )

    queue_brief = compact_queue(queue)
    pgrep_stdout = pgrep.get("stdout") or ""
    process_stdout = process.get("stdout") or ""
    runbook_alive = (
        "next_validation_queue_2026-07-08.sh" in pgrep_stdout
        or (bool(pid) and process.get("returncode") == 0 and "next_validation_queue_2026-07-08.sh" in process_stdout)
    )
    if queue.get("status") == "complete":
        status = "complete"
    elif runbook_alive:
        status = "running"
    else:
        status = "stopped_incomplete"

    payload = {
        "status": status,
        "host_code_dir": str(code_dir),
        "pid": pid or None,
        "queue": queue_brief,
        "process": process,
        "active_processes": pgrep,
        "gpu": gpu,
        "gpu_apps": gpu_apps,
        "log_path": CONFIG["log_path"],
        "log_tail": tail_lines(Path(CONFIG["log_path"]), CONFIG["log_tail_lines"]),
        "remote_status_json": str(remote_status_json),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
"""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Poll the remote CoFiTok next-validation queue.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-code-dir", default=DEFAULT_REMOTE_CODE_DIR)
    parser.add_argument("--checkpoint-root", default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--remote-python", default=DEFAULT_REMOTE_PYTHON)
    parser.add_argument("--date-tag", default="2026-07-08")
    parser.add_argument("--quality-images", type=int, default=1024)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--log-path", default=DEFAULT_LOG)
    parser.add_argument("--pid-file", default=DEFAULT_PID_FILE)
    parser.add_argument("--remote-status-json", default=DEFAULT_REMOTE_STATUS_JSON)
    parser.add_argument("--log-tail-lines", type=int, default=60)
    parser.add_argument("--timeout", type=int, default=45)
    parser.add_argument("--output-json", default="")
    parser.add_argument("--allow-unreachable", action="store_true")
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args(argv)


def build_remote_command(host: str, remote_code_dir: str, remote_python: str) -> list[str]:
    remote = f"cd {shlex.quote(remote_code_dir)} && {shlex.quote(remote_python)} -"
    return ["ssh", *SSH_OPTIONS, host, remote]


def build_program(args: argparse.Namespace) -> str:
    config = {
        "code_dir": args.remote_code_dir,
        "checkpoint_root": args.checkpoint_root,
        "date_tag": args.date_tag,
        "quality_images": args.quality_images,
        "sample_count": args.sample_count,
        "sample_steps": args.sample_steps,
        "log_path": args.log_path,
        "pid_file": args.pid_file,
        "remote_status_json": args.remote_status_json,
        "log_tail_lines": args.log_tail_lines,
    }
    return REMOTE_STATUS_PROGRAM.replace("__CONFIG_JSON__", json.dumps(config, sort_keys=True))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _failure(reason: str, **extra: Any) -> dict[str, Any]:
    return {"status": "unreachable", "reason": reason, **extra}


def poll_remote(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    command = build_remote_command(args.host, args.remote_code_dir, args.remote_python)
    program = build_program(args)
    try:
        completed = subprocess.run(
            command,
            input=program,
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
    except subprocess.TimeoutExpired as error:
        payload = _failure("timeout", timeout_seconds=error.timeout, command=error.cmd)
        return payload, 0 if args.allow_unreachable else 1

    if completed.returncode != 0:
        payload = _failure(
            "command_failed",
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            command=command,
        )
        return payload, 0 if args.allow_unreachable else completed.returncode

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        payload = _failure(
            "invalid_json",
            error=str(error),
            stdout=completed.stdout,
            stderr=completed.stderr,
            command=command,
        )
        return payload, 1

    if args.require_complete and payload.get("status") != "complete":
        return payload, 1
    return payload, 0


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload, exit_code = poll_remote(args)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json:
        _write_json(Path(args.output_json), payload)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
