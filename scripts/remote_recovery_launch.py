from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence


DEFAULT_HOST = "pro6000"
DEFAULT_REMOTE_ROOT = "/root/autodl-tmp/CoFiTok"
DEFAULT_BUNDLE = "artifacts/recovery/remote_recovery_2026-07-08.tar.gz"
DEFAULT_MANIFEST = "artifacts/recovery/remote_recovery_2026-07-08.tar.gz.manifest.json"
DEFAULT_RUNBOOK = "artifacts/runbooks/next_validation_queue_2026-07-08.sh"
DEFAULT_SESSION = "cofitok_next_validation_20260708"
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


@dataclass(frozen=True)
class RemoteRecoveryPlan:
    host: str
    remote_root: str
    code_dir: str
    bundle: str
    manifest: str
    probe_command: list[str]
    extract_command: list[str]
    post_extract_commands: list[list[str]]
    launch_command: list[str] | None


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe pro6000, extract the recovery bundle, run checks, and optionally launch the next queue."
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--remote-root", default=DEFAULT_REMOTE_ROOT)
    parser.add_argument("--bundle", default=DEFAULT_BUNDLE)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--runbook", default=DEFAULT_RUNBOOK)
    parser.add_argument("--session", default=DEFAULT_SESSION)
    parser.add_argument("--dry-run", action="store_true", help="Print the command plan without executing SSH.")
    parser.add_argument("--probe-only", action="store_true", help="Only check whether the remote responds.")
    parser.add_argument("--skip-transfer", action="store_true", help="Do not stream/extract the recovery bundle.")
    parser.add_argument("--skip-post-checks", action="store_true", help="Do not run manifest post-extract checks.")
    parser.add_argument("--start-queue", action="store_true", help="Start the next-validation runbook in tmux/nohup.")
    parser.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="Allow --start-queue even if nvidia-smi reports active compute processes.",
    )
    parser.add_argument("--timeout", type=int, default=600, help="Timeout in seconds for each SSH command.")
    return parser.parse_args(argv)


def _quote(value: str) -> str:
    return shlex.quote(value)


def ssh_command(host: str, remote_command: str) -> list[str]:
    return ["ssh", *SSH_OPTIONS, host, remote_command]


def build_probe_remote_command(remote_root: str) -> str:
    root = _quote(remote_root)
    return f"echo ok && hostname && date && mkdir -p {root} && cd {root} && pwd"


def build_extract_remote_command(remote_root: str) -> str:
    root = _quote(remote_root)
    return f"mkdir -p {root} && tar -xzf - -C {root}"


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def post_extract_checks_from_manifest(manifest: dict[str, Any], code_dir: str) -> list[str]:
    raw_checks = manifest.get("post_extract_checks", [])
    if not isinstance(raw_checks, list):
        raise ValueError("manifest post_extract_checks must be a list")

    checks = []
    for raw in raw_checks:
        if not isinstance(raw, str):
            raise ValueError("manifest post_extract_checks entries must be strings")
        stripped = raw.strip()
        if not stripped or stripped.startswith("cd "):
            continue
        checks.append(f"cd {_quote(code_dir)} && {stripped}")
    return checks


def build_launch_remote_command(code_dir: str, runbook: str, session: str, require_idle_gpu: bool = True) -> str:
    quoted_code_dir = _quote(code_dir)
    quoted_runbook = _quote(runbook)
    quoted_session = _quote(session)
    log_path = "artifacts/logs/next_validation_queue_2026-07-08.log"
    pid_path = "artifacts/logs/next_validation_queue_2026-07-08.pid"
    quoted_log = _quote(log_path)
    quoted_pid = _quote(pid_path)
    lines = [
        "set -euo pipefail",
        f"cd {quoted_code_dir}",
        "mkdir -p artifacts/logs artifacts/reports",
    ]
    if require_idle_gpu:
        lines.extend(
            [
                "if ! command -v nvidia-smi >/dev/null 2>&1; then",
                "  echo 'nvidia-smi not found; refusing to start queue' >&2",
                "  exit 3",
                "fi",
                "nvidia-smi",
                "gpu_pids=\"$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')\"",
                "if [ -n \"$gpu_pids\" ]; then",
                "  echo 'GPU compute processes are active; refusing to start queue' >&2",
                "  exit 3",
                "fi",
            ]
        )
    lines.extend(
        [
            "if command -v tmux >/dev/null 2>&1; then",
            f"  if tmux has-session -t {quoted_session} 2>/dev/null; then",
            f"    echo tmux_session_exists={quoted_session}",
            "  else",
            f"    tmux new-session -d -s {quoted_session} \"bash {quoted_runbook} > {quoted_log} 2>&1\"",
            f"    echo tmux_session_started={quoted_session}",
            "  fi",
            "else",
            f"  if [ -f {quoted_pid} ] && kill -0 \"$(cat {quoted_pid})\" 2>/dev/null; then",
            f"    echo nohup_pid_exists=$(cat {quoted_pid})",
            "  else",
            f"    nohup bash {quoted_runbook} > {quoted_log} 2>&1 < /dev/null &",
            f"    echo $! > {quoted_pid}",
            f"    echo nohup_pid_started=$(cat {quoted_pid})",
            "  fi",
            "fi",
            f"echo queue_log={log_path}",
        ]
    )
    return "\n".join(lines)


def build_plan(
    host: str,
    remote_root: str,
    bundle_path: Path,
    manifest_path: Path,
    runbook: str,
    session: str,
    include_post_checks: bool = True,
    include_launch: bool = False,
    require_idle_gpu: bool = True,
) -> RemoteRecoveryPlan:
    code_dir = f"{remote_root.rstrip('/')}/CoFiTok-internal"
    manifest = load_manifest(manifest_path)
    post_checks = post_extract_checks_from_manifest(manifest, code_dir) if include_post_checks else []
    launch_remote = build_launch_remote_command(code_dir, runbook, session, require_idle_gpu) if include_launch else None
    return RemoteRecoveryPlan(
        host=host,
        remote_root=remote_root,
        code_dir=code_dir,
        bundle=bundle_path.as_posix(),
        manifest=manifest_path.as_posix(),
        probe_command=ssh_command(host, build_probe_remote_command(remote_root)),
        extract_command=ssh_command(host, build_extract_remote_command(remote_root)),
        post_extract_commands=[ssh_command(host, check) for check in post_checks],
        launch_command=ssh_command(host, launch_remote) if launch_remote else None,
    )


def _run(command: list[str], input_bytes: bytes | None = None, timeout: int = 600) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(command, input=input_bytes, check=True, timeout=timeout)


def _command_text(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def plan_as_json(plan: RemoteRecoveryPlan) -> str:
    payload = asdict(plan)
    payload["probe_command_text"] = _command_text(plan.probe_command)
    payload["extract_command_text"] = _command_text(plan.extract_command)
    payload["post_extract_command_texts"] = [_command_text(command) for command in plan.post_extract_commands]
    payload["launch_command_text"] = _command_text(plan.launch_command) if plan.launch_command else None
    return json.dumps(payload, indent=2, sort_keys=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    bundle_path = Path(args.bundle)
    manifest_path = Path(args.manifest)
    if not bundle_path.is_absolute():
        bundle_path = Path.cwd() / bundle_path
    if not manifest_path.is_absolute():
        manifest_path = Path.cwd() / manifest_path

    plan = build_plan(
        host=args.host,
        remote_root=args.remote_root,
        bundle_path=bundle_path,
        manifest_path=manifest_path,
        runbook=args.runbook,
        session=args.session,
        include_post_checks=not args.skip_post_checks,
        include_launch=args.start_queue,
        require_idle_gpu=not args.allow_busy_gpu,
    )

    if args.dry_run:
        print(plan_as_json(plan))
        return

    try:
        _run(plan.probe_command, timeout=args.timeout)
        if args.probe_only:
            return

        if not args.skip_transfer:
            _run(plan.extract_command, input_bytes=bundle_path.read_bytes(), timeout=args.timeout)

        for command in plan.post_extract_commands:
            _run(command, timeout=args.timeout)

        if plan.launch_command:
            _run(plan.launch_command, timeout=args.timeout)
    except subprocess.TimeoutExpired as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "timeout",
                    "timeout_seconds": error.timeout,
                    "command": error.cmd,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    except subprocess.CalledProcessError as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "reason": "command_failed",
                    "returncode": error.returncode,
                    "command": error.cmd,
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        raise SystemExit(error.returncode) from None


if __name__ == "__main__":
    main()
