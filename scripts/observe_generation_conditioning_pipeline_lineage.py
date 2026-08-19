from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

from cofitok.generation_conditioning_lineage import (
    REPORT_ROLE,
    build_conditioning_lineage_report,
    inspect_conditioning_stage,
    load_conditioning_lineage_plan,
)
from cofitok.inference_replay import file_identity
from cofitok.reporting import file_sha256, write_json_report

DEFAULT_PLAN = Path(
    "configs/generation/diagnostics/conditioning_generation_pipeline_lineage_v1.json"
)


def _git(project: Path) -> dict[str, Any]:
    def run(*arguments: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(project), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    return {
        "path": project.resolve().as_posix(),
        "revision": run("rev-parse", "HEAD"),
        "tree": run("rev-parse", "HEAD^{tree}"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _process(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    try:
        stat = (root / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
        cmdline = (root / "cmdline").read_bytes()
        return {
            "pid": pid,
            "start_ticks": int(stat[19]),
            "cwd": (root / "cwd").resolve().as_posix(),
            "cmdline_sha256": hashlib.sha256(cmdline).hexdigest(),
            "alive": True,
        }
    except (FileNotFoundError, IndexError, OSError, PermissionError, ValueError):
        return {"pid": pid, "alive": False}


def _gpu_compute_processes() -> tuple[list[dict[str, Any]], bool]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return [], False
    rows = []
    complete = True
    for line in result.stdout.splitlines():
        values = [value.strip() for value in line.split(",", 2)]
        if len(values) != 3:
            complete = False
            continue
        try:
            pid = int(values[0])
            memory = int(values[2])
            process = _process(pid)
            rows.append(
                {
                    "pid": pid,
                    "process_name": values[1],
                    "used_memory_mib": memory,
                    "start_ticks": process.get("start_ticks"),
                    "cwd": process.get("cwd"),
                    "cmdline_sha256": process.get("cmdline_sha256"),
                }
            )
        except ValueError:
            complete = False
    return rows, complete


def _load_history(output: Path, *, plan_sha256: str) -> dict[str, Any]:
    if not output.is_file():
        return {}
    try:
        with output.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("existing conditioning lineage report is invalid") from error
    if (
        not isinstance(payload, dict)
        or payload.get("role") != REPORT_ROLE
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("plan"), dict)
        or payload["plan"].get("sha256") != plan_sha256
        or not isinstance(payload.get("history"), dict)
    ):
        raise ValueError("existing conditioning lineage report cannot be resumed")
    return dict(payload["history"])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously audit the exact five-stage conditioning-repair lineage "
            "without launching or signaling any process."
        )
    )
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("/root/autodl-tmp/CoFiTok/checkpoints/generation"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--stale-seconds", type=float, default=300.0)
    parser.add_argument("--timeout-seconds", type=float, default=31_536_000.0)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if (
        args.poll_seconds <= 0
        or args.stale_seconds <= args.poll_seconds
        or args.timeout_seconds <= 0
        or len(args.expected_plan_sha256) != 64
    ):
        parser.error("conditioning lineage thresholds or plan hash are invalid")
    return args


def main() -> int:
    args = _parse_args()
    project = args.project.resolve()
    plan_path = (
        args.plan.resolve()
        if args.plan.is_absolute()
        else (project / args.plan).resolve()
    )
    checkpoint_root = args.checkpoint_root.resolve()
    output = args.output.resolve()
    expected_git = {
        "path": project.as_posix(),
        "revision": args.expected_revision,
        "tree": args.expected_tree,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if _git(project) != expected_git:
        raise ValueError("conditioning lineage observer checkout differs")
    if file_sha256(plan_path) != args.expected_plan_sha256:
        raise ValueError("conditioning lineage plan SHA256 differs")
    plan = load_conditioning_lineage_plan(plan_path)
    if plan["identity"]["sha256"] != args.expected_plan_sha256:
        raise ValueError("conditioning lineage plan identity differs")
    histories = _load_history(output, plan_sha256=args.expected_plan_sha256)
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        observed_at = time.time()
        external_issues = []
        observer_git = _git(project)
        if observer_git != expected_git:
            external_issues.append("observer_checkout_changed")
        if file_sha256(plan_path) != args.expected_plan_sha256:
            external_issues.append("conditioning_lineage_plan_changed")
        standing_path = Path(plan["standing_authorization"]["path"])
        if not standing_path.is_file():
            external_issues.append("standing_authorization_missing")
            standing_actual = None
        else:
            standing_actual = file_identity(standing_path)
            if standing_actual != plan["standing_authorization"]:
                external_issues.append("standing_authorization_identity_changed")

        stages = []
        next_histories = {}
        for spec in plan["stages"]:
            stage, history = inspect_conditioning_stage(
                spec,
                now_unix=observed_at,
                stale_seconds=args.stale_seconds,
                standing_authorization=plan["standing_authorization"],
                previous_history=histories.get(spec["name"]),
                git_inspector=_git,
                process_inspector=_process,
            )
            stages.append(stage)
            next_histories[spec["name"]] = history
        histories = next_histories
        gpu, gpu_complete = _gpu_compute_processes()
        usage = shutil.disk_usage(checkpoint_root)
        report = build_conditioning_lineage_report(
            stages=stages,
            histories=histories,
            observed_at_unix=observed_at,
            stale_seconds=args.stale_seconds,
            plan_identity=plan["identity"],
            observer_git=observer_git,
            standing_authorization={
                "expected": plan["standing_authorization"],
                "actual": standing_actual,
                "valid": standing_actual == plan["standing_authorization"],
            },
            gpu_compute_processes=gpu,
            gpu_query_complete=gpu_complete,
            disk={
                "path": checkpoint_root.as_posix(),
                "total_bytes": usage.total,
                "used_bytes": usage.used,
                "free_bytes": usage.free,
            },
            hostname=socket.gethostname(),
            external_issues=external_issues,
        )
        report["observer"] = {
            "pid": os.getpid(),
            "poll_seconds": args.poll_seconds,
            "stale_seconds": args.stale_seconds,
            "timeout_seconds": args.timeout_seconds,
            "output": output.as_posix(),
        }
        write_json_report(output, report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "current_stage": report["progress"]["current_stage"],
                    "detail": report["detail"],
                    "issues": report["issues"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if args.once or report["status"] == "pass":
            return 1 if report["status"] in {"failed", "stalled"} else 0
        if time.monotonic() >= deadline:
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
