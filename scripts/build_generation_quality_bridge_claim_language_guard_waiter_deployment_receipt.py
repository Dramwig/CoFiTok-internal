from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    reject_symlink_chain,
)

try:
    import run_generation_quality_bridge_claim_language_guard_waiter as waiter
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        run_generation_quality_bridge_claim_language_guard_waiter as waiter,
    )


SCHEMA_VERSION = 1
ROLE = "generation_quality_bridge_claim_language_guard_waiter_deployment"
CLAIM_BOUNDARY = {
    **waiter.CLAIM_BOUNDARY,
    "deployment_receipt_only": True,
    "waiter_launch_authorization_allowed": False,
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value.lower())
    )


def _require_identity(identity: Mapping[str, Any], *, label: str) -> None:
    if (
        not isinstance(identity.get("path"), str)
        or int(identity.get("bytes", -1)) < 0
        or not _is_hex(identity.get("sha256"), 64)
    ):
        raise ValueError(f"{label} identity is invalid")


def _require_git(identity: Mapping[str, Any], *, label: str) -> None:
    if (
        not _is_hex(identity.get("revision"), 40)
        or not _is_hex(identity.get("tree"), 40)
        or not isinstance(identity.get("branch"), str)
        or identity.get("tracked_dirty") is not False
        or not isinstance(identity.get("path"), str)
    ):
        raise ValueError(f"{label} Git identity is invalid")


def _git_contract(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: identity.get(key)
        for key in ("revision", "tree", "branch", "tracked_dirty")
    }


def _source_state_contract(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: identity.get(key)
        for key in (
            "status",
            "detail",
            "phase",
            "pid",
            "terminal",
            "qualification",
        )
    }


def _git_identity(project: Path) -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "path": project.resolve().as_posix(),
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(
            git("status", "--porcelain", "--untracked-files=no")
        ),
    }


def _stable_json_snapshot(
    path: Path,
    *,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw = path.read_bytes()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return payload, {
        "path": path.resolve().as_posix(),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _bundle_prerequisites(path: Path) -> list[str]:
    prerequisites: list[str] = []
    with path.open("rb") as handle:
        header = handle.readline()
        if not header.startswith(b"# v") or b"git bundle" not in header:
            raise ValueError("claim-language waiter bundle header is invalid")
        for raw_line in handle:
            line = raw_line.rstrip(b"\r\n")
            if not line:
                break
            if not line.startswith(b"-"):
                continue
            token = line[1:].split(b" ", 1)[0].decode("ascii").lower()
            if not _is_hex(token, 40):
                raise ValueError(
                    "claim-language waiter bundle prerequisite is invalid"
                )
            prerequisites.append(token)
    return sorted(set(prerequisites))


def _bundle_heads(path: Path) -> list[dict[str, str]]:
    completed = subprocess.run(
        ["git", "bundle", "list-heads", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    heads: list[dict[str, str]] = []
    for line in completed.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2 or not _is_hex(fields[0], 40):
            raise ValueError("claim-language waiter bundle head is invalid")
        heads.append({"revision": fields[0].lower(), "ref": fields[1]})
    if not heads:
        raise ValueError("claim-language waiter bundle has no advertised head")
    return heads


def _bundle_contract(path: Path, *, project: Path) -> dict[str, Any]:
    verified = subprocess.run(
        ["git", "bundle", "verify", str(path)],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    )
    return {
        "identity": file_identity(path),
        "heads": _bundle_heads(path),
        "prerequisites": _bundle_prerequisites(path),
        "verification": {
            "status": "pass",
            "stdout": verified.stdout.strip(),
            "stderr": verified.stderr.strip(),
        },
    }


def _process_identity(pid: int) -> dict[str, Any]:
    proc = Path("/proc") / str(pid)
    if not proc.is_dir():
        raise ValueError(f"process {pid} is not active")
    argv = [
        field.decode("utf-8", errors="replace")
        for field in (proc / "cmdline").read_bytes().split(b"\0")
        if field
    ]
    environ: dict[str, str] = {}
    for raw in (proc / "environ").read_bytes().split(b"\0"):
        if b"=" not in raw:
            continue
        key, value = raw.split(b"=", 1)
        name = key.decode("utf-8", errors="replace")
        if name in {
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "PYTHONPATH",
        }:
            environ[name] = value.decode("utf-8", errors="replace")
    stat_tail = (proc / "stat").read_text(encoding="utf-8").rsplit(")", 1)[1].split()
    return {
        "pid": pid,
        "ppid": int(stat_tail[1]),
        "start_ticks": int(stat_tail[19]),
        "cwd": (proc / "cwd").resolve().as_posix(),
        "exe": (proc / "exe").resolve().as_posix(),
        "argv": argv,
        "environment": environ,
    }


def _gpu_compute_rows() -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[dict[str, Any]] = []
    for raw in completed.stdout.splitlines():
        fields = [field.strip() for field in raw.split(",", 2)]
        if len(fields) != 3 or not fields[0]:
            continue
        rows.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_memory_mib": int(fields[2]),
            }
        )
    return rows


def _flag(argv: Sequence[str], name: str) -> str:
    positions = [index for index, value in enumerate(argv) if value == name]
    if len(positions) != 1 or positions[0] + 1 >= len(argv):
        raise ValueError(f"waiter process flag is missing or repeated: {name}")
    return str(argv[positions[0] + 1])


def _require_waiter_process(
    process: Mapping[str, Any],
    *,
    expected_pid: int,
    expected_python: str,
    expected_project: str,
    expected: Mapping[str, Any],
) -> None:
    argv = process.get("argv")
    environment = process.get("environment")
    if (
        int(process.get("pid", -1)) != expected_pid
        or process.get("cwd") != expected_project
        or process.get("exe") != expected_python
        or not isinstance(argv, Sequence)
        or isinstance(argv, (str, bytes))
        or not isinstance(environment, Mapping)
        or environment.get("CUDA_VISIBLE_DEVICES") != "-1"
        or environment.get("OMP_NUM_THREADS") != "1"
        or environment.get("MKL_NUM_THREADS") != "1"
        or environment.get("PYTHONPATH")
        != f"{expected_project}:{expected_project}/src"
        or "scripts/run_generation_quality_bridge_claim_language_guard_waiter.py"
        not in argv
    ):
        raise ValueError("claim-language guard waiter process identity differs")
    source_waiter = expected["source_waiter"]
    exact_flags = {
        "--project": expected_project,
        "--source-output-root": source_waiter["output_root"],
        "--source-status": source_waiter["status"],
        "--source-pid-file": source_waiter["pid_file"],
        "--source-report": expected["source_report"],
        "--output-root": expected["output_root"],
        "--status-output": expected["status_output"],
        "--pid-file": expected["pid_file"],
        "--guard-output": expected["guard_output"],
        "--expected-source-pid": str(source_waiter["pid"]),
        "--expected-control-revision": expected["control_git"]["revision"],
        "--expected-control-tree": expected["control_git"]["tree"],
        "--expected-control-branch": expected["control_git"]["branch"],
        "--expected-source-control-revision": source_waiter["control_git"][
            "revision"
        ],
        "--expected-source-control-tree": source_waiter["control_git"]["tree"],
        "--expected-source-control-branch": source_waiter["control_git"]["branch"],
    }
    for name, value in exact_flags.items():
        if _flag(argv, name) != str(value):
            raise ValueError(f"claim-language guard waiter process {name} differs")
    if (
        float(_flag(argv, "--poll-seconds")) <= 0.0
        or float(_flag(argv, "--timeout-seconds")) <= 0.0
    ):
        raise ValueError("claim-language guard waiter timing differs")


def build_receipt(
    *,
    expected_control_git: Mapping[str, Any],
    expected_source_git: Mapping[str, Any],
    expected_formal_git: Mapping[str, Any],
    expected_python: str,
    expected_paths: Mapping[str, Any],
    bundle: Mapping[str, Any],
    control_checkout: Mapping[str, Any],
    source_checkout: Mapping[str, Any],
    formal_checkout: Mapping[str, Any],
    waiter_status: Mapping[str, Any],
    waiter_status_identity: Mapping[str, Any],
    waiter_pid: Mapping[str, Any],
    waiter_pid_identity: Mapping[str, Any],
    waiter_log_identity: Mapping[str, Any],
    waiter_process: Mapping[str, Any],
    source_status: Mapping[str, Any],
    source_status_identity: Mapping[str, Any],
    source_state: Mapping[str, Any],
    source_pid: Mapping[str, Any],
    source_pid_identity: Mapping[str, Any],
    source_process: Mapping[str, Any],
    source_report_present: bool,
    guard_output_present: bool,
    gpu_compute_rows: Sequence[Mapping[str, Any]],
    hostname: str,
    created_at: str,
) -> dict[str, Any]:
    for label, identity in (
        ("expected control", expected_control_git),
        ("expected source", expected_source_git),
        ("expected formal", expected_formal_git),
        ("control checkout", control_checkout),
        ("source checkout", source_checkout),
        ("formal checkout", formal_checkout),
    ):
        _require_git(identity, label=label)
    if (
        _git_contract(control_checkout) != _git_contract(expected_control_git)
        or _git_contract(source_checkout) != _git_contract(expected_source_git)
        or _git_contract(formal_checkout) != _git_contract(expected_formal_git)
    ):
        raise ValueError("claim-language guard deployment Git identity differs")
    if (
        control_checkout["path"] != expected_paths["project"]
        or source_checkout["path"] != expected_paths["source_project"]
        or formal_checkout["path"] != expected_paths["formal_project"]
    ):
        raise ValueError("claim-language guard deployment checkout path differs")

    bundle_identity = bundle.get("identity")
    if not isinstance(bundle_identity, Mapping):
        raise ValueError("claim-language guard deployment bundle identity is missing")
    _require_identity(bundle_identity, label="claim-language guard deployment bundle")
    if (
        bundle.get("heads")
        != [
            {
                "revision": expected_control_git["revision"],
                "ref": f"refs/heads/{expected_control_git['branch']}",
            }
        ]
        or bundle.get("prerequisites") != [expected_source_git["revision"]]
        or not isinstance(bundle.get("verification"), Mapping)
        or bundle["verification"].get("status") != "pass"
    ):
        raise ValueError("claim-language guard deployment bundle contract differs")

    for label, identity in (
        ("waiter status", waiter_status_identity),
        ("waiter PID", waiter_pid_identity),
        ("waiter log", waiter_log_identity),
        ("source status", source_status_identity),
        ("source PID", source_pid_identity),
    ):
        _require_identity(identity, label=label)
    expected = {
        "control_git": _git_contract(expected_control_git),
        "source_kind": waiter.SOURCE_KIND,
        "source_waiter": {
            "pid": int(expected_paths["source_pid"]),
            "control_git": _git_contract(expected_source_git),
            "output_root": expected_paths["source_output_root"],
            "status": expected_paths["source_status"],
            "pid_file": expected_paths["source_pid_file"],
        },
        "source_report": expected_paths["source_report"],
        "output_root": expected_paths["output_root"],
        "guard_output": expected_paths["guard_output"],
    }
    embedded_source_state = waiter_status.get("source_waiter")
    if (
        waiter_status.get("schema_version") != waiter.WAITER_SCHEMA_VERSION
        or waiter_status.get("role") != waiter.WAITER_ROLE
        or waiter_status.get("status") != "waiting"
        or waiter_status.get("phase") != "source"
        or waiter_status.get("detail") != "waiting_for_quality_claim_source"
        or waiter_status.get("expected") != expected
        or not isinstance(embedded_source_state, Mapping)
        or _source_state_contract(embedded_source_state)
        != _source_state_contract(source_state)
        or waiter_status.get("source") is not None
        or waiter_status.get("claim_language_guard") is not None
        or waiter_status.get("claim_boundary") != waiter.CLAIM_BOUNDARY
    ):
        raise ValueError("claim-language guard waiter initial status differs")
    waiter_pid_value = int(waiter_status.get("pid", -1))
    if (
        waiter_pid_value < 1
        or waiter_pid.get("schema_version") != 1
        or waiter_pid.get("role") != waiter.WAITER_ROLE
        or int(waiter_pid.get("pid", -1)) != waiter_pid_value
        or waiter_pid.get("expected_control_revision")
        != expected_control_git["revision"]
        or int(waiter_pid.get("expected_source_pid", -1))
        != int(expected_paths["source_pid"])
    ):
        raise ValueError("claim-language guard waiter PID binding differs")

    process_expected = {
        **expected,
        "status_output": expected_paths["status_output"],
        "pid_file": expected_paths["pid_file"],
    }
    _require_waiter_process(
        waiter_process,
        expected_pid=waiter_pid_value,
        expected_python=expected_python,
        expected_project=expected_paths["project"],
        expected=process_expected,
    )
    if (
        source_state.get("status") != "waiting"
        or source_state.get("terminal") is not False
        or source_status.get("status") != "waiting"
        or source_report_present
        or guard_output_present
    ):
        raise ValueError("claim-language guard deployment source state differs")
    source_pid_value = int(expected_paths["source_pid"])
    if (
        source_pid.get("role") != waiter.source_waiter.WAITER_ROLE
        or int(source_pid.get("pid", -1)) != source_pid_value
        or source_pid.get("expected_control_revision")
        != expected_source_git["revision"]
        or int(source_process.get("pid", -1)) != source_pid_value
        or source_process.get("cwd") != expected_paths["source_project"]
        or "scripts/run_generation_quality_bridge_claim_qualification_waiter.py"
        not in source_process.get("argv", [])
    ):
        raise ValueError("claim-language guard source waiter process differs")
    if any(int(row.get("pid", -1)) == waiter_pid_value for row in gpu_compute_rows):
        raise ValueError("claim-language guard waiter unexpectedly allocated GPU")

    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "pass",
        "decision": "claim_language_guard_waiter_deployed_non_authorizing",
        "git": {
            "control": dict(control_checkout),
            "source_waiter": dict(source_checkout),
            "formal": dict(formal_checkout),
        },
        "bundle": dict(bundle),
        "waiter": {
            "status": dict(waiter_status),
            "status_identity": dict(waiter_status_identity),
            "pid": dict(waiter_pid),
            "pid_identity": dict(waiter_pid_identity),
            "log": dict(waiter_log_identity),
            "process": dict(waiter_process),
        },
        "source": {
            "status": dict(source_status),
            "status_identity": dict(source_status_identity),
            "validated_state": dict(source_state),
            "pid": dict(source_pid),
            "pid_identity": dict(source_pid_identity),
            "process": dict(source_process),
            "claim_report_present": source_report_present,
        },
        "initial_state": {
            "guard_output_present": guard_output_present,
            "waiting_for_quality_claim_source": True,
        },
        "gpu_compute_rows": [dict(row) for row in gpu_compute_rows],
        "python_executable": expected_python,
        "claim_boundary": CLAIM_BOUNDARY,
        "hostname": hostname,
        "created_at": created_at,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an integrity-bound deployment receipt for the CPU-only "
            "quality-bridge claim-language guard waiter."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--formal-project", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-output-root", type=Path, required=True)
    parser.add_argument("--source-status", type=Path, required=True)
    parser.add_argument("--source-pid-file", type=Path, required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--guard-output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-waiter-pid", type=int, required=True)
    parser.add_argument("--expected-source-pid", type=int, required=True)
    parser.add_argument("--expected-python", type=Path, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-source-control-revision", required=True)
    parser.add_argument("--expected-source-control-tree", required=True)
    parser.add_argument("--expected-source-control-branch", required=True)
    parser.add_argument("--expected-formal-revision", required=True)
    parser.add_argument("--expected-formal-tree", required=True)
    parser.add_argument("--expected-formal-branch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = reject_symlink_chain(
        args.project,
        name="claim-language guard deployment control checkout",
    ).resolve()
    source_project = reject_symlink_chain(
        args.source_project,
        name="claim-language guard deployment source checkout",
    ).resolve()
    formal_project = reject_symlink_chain(
        args.formal_project,
        name="claim-language guard deployment formal checkout",
    ).resolve()
    bundle_path = reject_symlink_chain(
        args.bundle,
        name="claim-language guard deployment bundle",
    ).resolve()
    source_output_root = reject_symlink_chain(
        args.source_output_root,
        name="claim-language guard deployment source output root",
    ).resolve()
    source_status_path = reject_symlink_chain(
        args.source_status,
        name="claim-language guard deployment source status",
    ).resolve()
    source_pid_path = reject_symlink_chain(
        args.source_pid_file,
        name="claim-language guard deployment source PID",
    ).resolve()
    source_report_path = reject_symlink_chain(
        args.source_report,
        name="claim-language guard deployment source report",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="claim-language guard deployment output root",
    ).resolve()
    status_path = reject_symlink_chain(
        args.status_output,
        name="claim-language guard deployment waiter status",
    ).resolve()
    pid_path = reject_symlink_chain(
        args.pid_file,
        name="claim-language guard deployment waiter PID",
    ).resolve()
    guard_path = reject_symlink_chain(
        args.guard_output,
        name="claim-language guard deployment guard output",
    ).resolve()
    log_path = reject_symlink_chain(
        args.log,
        name="claim-language guard deployment waiter log",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="claim-language guard deployment receipt",
    ).resolve()
    if not (
        status_path.is_file()
        and pid_path.is_file()
        and log_path.is_file()
        and source_status_path.is_file()
        and source_pid_path.is_file()
    ):
        raise ValueError("claim-language guard deployment source files are incomplete")
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError(
            "claim-language guard deployment receipt is outside output root"
        ) from error

    expected_control_git = {
        "path": project.as_posix(),
        "revision": args.expected_control_revision,
        "tree": args.expected_control_tree,
        "branch": args.expected_control_branch,
        "tracked_dirty": False,
    }
    expected_source_git = {
        "path": source_project.as_posix(),
        "revision": args.expected_source_control_revision,
        "tree": args.expected_source_control_tree,
        "branch": args.expected_source_control_branch,
        "tracked_dirty": False,
    }
    expected_formal_git = {
        "path": formal_project.as_posix(),
        "revision": args.expected_formal_revision,
        "tree": args.expected_formal_tree,
        "branch": args.expected_formal_branch,
        "tracked_dirty": False,
    }
    waiter_status, waiter_status_identity = _stable_json_snapshot(
        status_path,
        name="claim-language guard waiter status",
    )
    waiter_pid, waiter_pid_identity = _stable_json_snapshot(
        pid_path,
        name="claim-language guard waiter PID",
    )
    source_status, source_status_identity = _stable_json_snapshot(
        source_status_path,
        name="claim-language guard source waiter status",
    )
    source_pid, source_pid_identity = _stable_json_snapshot(
        source_pid_path,
        name="claim-language guard source waiter PID",
    )
    source_state = waiter.validate_source_status(
        source_status,
        expected_pid=args.expected_source_pid,
        expected_output_root=source_output_root,
        expected_source_report=source_report_path,
        expected_control_git=_git_contract(expected_source_git),
    )
    expected_paths = {
        "project": project.as_posix(),
        "source_project": source_project.as_posix(),
        "formal_project": formal_project.as_posix(),
        "source_pid": args.expected_source_pid,
        "source_output_root": source_output_root.as_posix(),
        "source_status": source_status_path.as_posix(),
        "source_pid_file": source_pid_path.as_posix(),
        "source_report": source_report_path.as_posix(),
        "output_root": output_root.as_posix(),
        "status_output": status_path.as_posix(),
        "pid_file": pid_path.as_posix(),
        "guard_output": guard_path.as_posix(),
    }
    receipt = build_receipt(
        expected_control_git=expected_control_git,
        expected_source_git=expected_source_git,
        expected_formal_git=expected_formal_git,
        expected_python=args.expected_python.resolve().as_posix(),
        expected_paths=expected_paths,
        bundle=_bundle_contract(bundle_path, project=project),
        control_checkout=_git_identity(project),
        source_checkout=_git_identity(source_project),
        formal_checkout=_git_identity(formal_project),
        waiter_status=waiter_status,
        waiter_status_identity=waiter_status_identity,
        waiter_pid=waiter_pid,
        waiter_pid_identity=waiter_pid_identity,
        waiter_log_identity=file_identity(log_path),
        waiter_process=_process_identity(args.expected_waiter_pid),
        source_status=source_status,
        source_status_identity=source_status_identity,
        source_state=source_state,
        source_pid=source_pid,
        source_pid_identity=source_pid_identity,
        source_process=_process_identity(args.expected_source_pid),
        source_report_present=source_report_path.exists(),
        guard_output_present=guard_path.exists(),
        gpu_compute_rows=_gpu_compute_rows(),
        hostname=socket.gethostname(),
        created_at=_utc_now(),
    )
    if source_report_path.exists() or guard_path.exists():
        raise ValueError("claim-language guard deployment terminal state changed")
    identity = prepare_manifest(
        output,
        receipt,
        resume=output.is_file(),
        overwrite=False,
    )
    if os.name != "nt":
        output.chmod(0o444)
    print(json.dumps({"status": "pass", "receipt": identity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
