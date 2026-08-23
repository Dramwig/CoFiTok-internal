from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA_VERSION = 1
RECEIPT_ROLE = "generation_runtime_claim_guard_strict_replay_wrapper_deployment"
RECOVERY_SCHEMA_VERSION = 1
RECOVERY_ROLE = "generation_runtime_claim_guard_strict_replay_recovery"
FAILURE_RECORD_ROLE = "generation_runtime_claim_guard_strict_replay_wrapper_failure_record"
SCOPE = {
    "cpu_only": True,
    "non_authorizing": True,
    "gpu_execution_allowed": False,
    "training_process_signals_allowed": False,
    "unrelated_process_signals_allowed": False,
    "old_waiter_signals_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "full_300k_launch_allowed": False,
}
BEHAVIOR = {
    "waits_for_exact_old_pid_start_ticks_and_cmdline": True,
    "executes_only_after_old_waiter_exits_or_identity_changes": True,
    "writes_independent_output": True,
    "replaces_canonical_guard": False,
    "comparison_required_before_runtime_claim_trust": True,
}
EXPECTED_FAILURE_MARKERS = (
    "ModuleNotFoundError: No module named 'scripts'",
    "run_generation_quality_bridge_runtime_claim_guard_waiter.py",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_identity(path: Path) -> dict[str, Any]:
    source = path.resolve(strict=True)
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"source is not a regular non-symlink file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": source.as_posix(),
        "bytes": source.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not readable JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    return payload


def git_identity(project: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
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
        "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _proc_cmdline(pid: int) -> bytes | None:
    path = Path(f"/proc/{pid}/cmdline")
    try:
        return path.read_bytes()
    except OSError:
        return None


def _process_start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8").split()
    return int(fields[21])


def _ionice_class(pid: int) -> str:
    output = subprocess.run(
        ["ionice", "-p", str(pid)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()
    return "idle" if "idle" in output else output


def runtime_identity(*, cwd: Path) -> dict[str, Any]:
    pid = os.getpid()
    command = _proc_cmdline(pid)
    if command is None:
        raise ValueError("recovery wrapper command line is unavailable")
    return {
        "pid": pid,
        "ppid": os.getppid(),
        "start_ticks": _process_start_ticks(pid),
        "cwd": cwd.resolve().as_posix(),
        "cmdline_sha256": _sha256_bytes(command),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "nice": os.getpriority(os.PRIO_PROCESS, pid),
        "ionice": _ionice_class(pid),
    }


def _assert_identity(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            raise ValueError(f"{label} {key} differs")


def _assert_file_sha(path: Path, expected_sha256: str, *, label: str) -> dict[str, Any]:
    identity = file_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return identity


def _assert_absent(path: Path, *, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"{label} already exists")


def _strict_runner_duplicates(*, runner: Path, output_root: Path) -> list[int]:
    runner_token = runner.resolve().as_posix().encode()
    output_token = output_root.resolve().as_posix().encode()
    duplicates: list[int] = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        if pid == os.getpid():
            continue
        command = _proc_cmdline(pid)
        if command is not None and runner_token in command and output_token in command:
            duplicates.append(pid)
    return sorted(duplicates)


def build_wrapper_receipt(
    *,
    strict_control: Mapping[str, Any],
    wrapper_runtime: Mapping[str, Any],
    canonical_runtime: Mapping[str, Any],
    source_runtime: Mapping[str, Any],
    targets: Mapping[str, Any],
    recovery_control: Mapping[str, Any],
    recovery_source: Mapping[str, Any],
    original_wrapper: Mapping[str, Any],
    failure_log: Mapping[str, Any],
    strict_runner: Mapping[str, Any],
    strict_builder: Mapping[str, Any],
    strict_runner_argv_sha256: str,
    failed_targets: Mapping[str, Any],
    prior_failure_status: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "role": RECEIPT_ROLE,
        "status": "pass",
        "scope": dict(SCOPE),
        "strict_control": dict(strict_control),
        "wrapper": dict(wrapper_runtime),
        "old_canonical_waiter": dict(canonical_runtime),
        "runtime_fairness_source_waiter": dict(source_runtime),
        "behavior": dict(BEHAVIOR),
        "targets": dict(targets),
        "recovery": {
            "schema_version": RECOVERY_SCHEMA_VERSION,
            "role": RECOVERY_ROLE,
            "reason": "original_wrapper_missing_pythonpath_import_failure",
            "recovery_control": dict(recovery_control),
            "recovery_source": dict(recovery_source),
            "original_wrapper_deployment": dict(original_wrapper),
            "original_failure_log": dict(failure_log),
            "strict_runner_source": dict(strict_runner),
            "strict_builder_source": dict(strict_builder),
            "strict_runner_argv_sha256": strict_runner_argv_sha256,
            "failed_targets_absent_before_recovery": dict(failed_targets),
            "prior_failure_status": dict(prior_failure_status),
            "new_output_version": "runtime_compute_claim_guard_strict_replay_v3",
            "old_outputs_overwritten": False,
            "old_process_signaled": False,
        },
    }


def build_failure_record(
    *,
    original_wrapper: Mapping[str, Any],
    failure_log: Mapping[str, Any],
    failed_pid: int,
    guard_output: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": FAILURE_RECORD_ROLE,
        "status": "failed",
        "detail": "original_wrapper_missing_pythonpath_import_failure",
        "pid": failed_pid,
        "original_wrapper_deployment": dict(original_wrapper),
        "original_failure_log": dict(failure_log),
        "guard_output": {
            "path": guard_output.resolve().as_posix(),
            "absent": not guard_output.exists() and not guard_output.is_symlink(),
        },
        "scope": dict(SCOPE),
    }


def _atomic_write_new(
    path: Path,
    payload: Mapping[str, Any],
    *,
    label: str,
) -> None:
    _assert_absent(path, label=label)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    _assert_absent(temporary, label=f"{label} temporary file")
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recover one failed CUDA-hidden strict runtime-claim replay by writing "
            "a versioned deployment receipt and execing the pinned strict runner."
        )
    )
    parser.add_argument("--recovery-project", type=Path, required=True)
    parser.add_argument("--strict-project", type=Path, required=True)
    parser.add_argument("--original-wrapper-receipt", type=Path, required=True)
    parser.add_argument("--expected-original-wrapper-receipt-sha256", required=True)
    parser.add_argument("--failure-log", type=Path, required=True)
    parser.add_argument("--expected-failure-log-sha256", required=True)
    parser.add_argument("--canonical-status", type=Path, required=True)
    parser.add_argument("--expected-canonical-status-sha256", required=True)
    parser.add_argument("--strict-output-root", type=Path, required=True)
    parser.add_argument("--wrapper-receipt-output", type=Path, required=True)
    parser.add_argument("--strict-status-output", type=Path, required=True)
    parser.add_argument("--strict-pid-file", type=Path, required=True)
    parser.add_argument("--strict-deployment-receipt-output", type=Path, required=True)
    parser.add_argument("--strict-guard-output", type=Path, required=True)
    parser.add_argument("--expected-recovery-revision", required=True)
    parser.add_argument("--expected-recovery-tree", required=True)
    parser.add_argument("--expected-recovery-branch", required=True)
    parser.add_argument("--expected-recovery-source-sha256", required=True)
    parser.add_argument("--expected-original-wrapper-pid", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    parser.add_argument("--detach-timeout-seconds", type=float, default=30.0)
    return parser.parse_args()


def _wait_for_detach(timeout_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while os.getppid() != 1:
        if time.monotonic() >= deadline:
            raise ValueError("recovery wrapper did not detach to parent PID 1")
        time.sleep(0.05)


def _validate_runtime_policy(runtime: Mapping[str, Any]) -> None:
    if (
        runtime.get("ppid") != 1
        or runtime.get("cuda_visible_devices") != ""
        or runtime.get("omp_num_threads") != "1"
        or runtime.get("mkl_num_threads") != "1"
        or runtime.get("nice") != 10
        or runtime.get("ionice") != "idle"
    ):
        raise ValueError("recovery wrapper runtime policy differs")


def _runner_arguments(
    *,
    strict_project: Path,
    canonical_status: Mapping[str, Any],
    strict_output_root: Path,
    status_output: Path,
    pid_file: Path,
    deployment_output: Path,
    guard_output: Path,
    poll_seconds: float,
    timeout_seconds: float,
) -> list[str]:
    expected = canonical_status["expected"]
    source = expected["source_waiter"]
    pair = expected["pair_monitor"]
    control = git_identity(strict_project)
    source_control = source["control_git"]
    return [
        os.environ.get("PYTHON_EXECUTABLE", "/root/autodl-tmp/conda/envs/pf-vlm/bin/python3.10"),
        (strict_project / "scripts/run_generation_quality_bridge_runtime_claim_guard_waiter.py").as_posix(),
        "--project",
        strict_project.as_posix(),
        "--quality-output-root",
        str(Path(expected["output_root"]).parents[1]),
        "--source-status",
        source["status"],
        "--source-final-report",
        source["final_report"],
        "--source-deployment-receipt",
        source["deployment_receipt"]["path"],
        "--expected-source-deployment-receipt-sha256",
        source["deployment_receipt"]["sha256"],
        "--pair-monitor",
        pair["path"],
        "--output-root",
        strict_output_root.as_posix(),
        "--status-output",
        status_output.as_posix(),
        "--pid-file",
        pid_file.as_posix(),
        "--deployment-receipt-output",
        deployment_output.as_posix(),
        "--guard-output",
        guard_output.as_posix(),
        "--expected-source-pid",
        str(source["pid"]),
        "--expected-control-revision",
        control["revision"],
        "--expected-control-tree",
        control["tree"],
        "--expected-control-branch",
        control["branch"],
        "--expected-source-control-revision",
        source_control["revision"],
        "--expected-source-control-tree",
        source_control["tree"],
        "--expected-source-control-branch",
        source_control["branch"],
        "--expected-training-revision",
        pair["training_revision"],
        "--expected-training-branch",
        pair["training_branch"],
        "--expected-monitor-name",
        pair["monitor_name"],
        "--poll-seconds",
        str(poll_seconds),
        "--timeout-seconds",
        str(timeout_seconds),
    ]


def main() -> int:
    args = _parse_args()
    _wait_for_detach(args.detach_timeout_seconds)
    recovery_project = args.recovery_project.resolve(strict=True)
    strict_project = args.strict_project.resolve(strict=True)
    recovery_control = git_identity(recovery_project)
    _assert_identity(
        recovery_control,
        {
            "revision": args.expected_recovery_revision,
            "tree": args.expected_recovery_tree,
            "branch": args.expected_recovery_branch,
            "tracked_dirty": False,
        },
        label="recovery control checkout",
    )
    recovery_source = _assert_file_sha(
        Path(__file__),
        args.expected_recovery_source_sha256,
        label="recovery launcher source",
    )
    original_wrapper_identity = _assert_file_sha(
        args.original_wrapper_receipt,
        args.expected_original_wrapper_receipt_sha256,
        label="original wrapper deployment receipt",
    )
    original_wrapper = read_json(
        args.original_wrapper_receipt,
        label="original wrapper deployment receipt",
    )
    if (
        original_wrapper.get("role") != RECEIPT_ROLE
        or original_wrapper.get("status") != "pass"
        or int(original_wrapper.get("wrapper", {}).get("pid", -1))
        != args.expected_original_wrapper_pid
    ):
        raise ValueError("original wrapper deployment contract differs")
    if _proc_cmdline(args.expected_original_wrapper_pid) is not None:
        raise ValueError("original failed wrapper process is still active")

    failure_log_identity = _assert_file_sha(
        args.failure_log,
        args.expected_failure_log_sha256,
        label="original wrapper failure log",
    )
    failure_text = args.failure_log.read_text(encoding="utf-8", errors="replace")
    if any(marker not in failure_text for marker in EXPECTED_FAILURE_MARKERS):
        raise ValueError("original wrapper failure reason differs")

    canonical_status_identity = _assert_file_sha(
        args.canonical_status,
        args.expected_canonical_status_sha256,
        label="canonical runtime claim status",
    )
    canonical_status = read_json(args.canonical_status, label="canonical runtime claim status")
    if (
        canonical_status.get("role")
        != "generation_quality_bridge_runtime_claim_guard_waiter"
        or canonical_status.get("status") != "pass"
        or canonical_status.get("phase") != "completed"
        or int(canonical_status.get("pid", -1))
        != int(original_wrapper.get("old_canonical_waiter", {}).get("pid", -2))
    ):
        raise ValueError("canonical runtime claim status contract differs")

    strict_control = git_identity(strict_project)
    _assert_identity(
        strict_control,
        {
            key: original_wrapper["strict_control"][key]
            for key in ("revision", "tree", "branch", "tracked_dirty")
        },
        label="strict control checkout",
    )
    runner = strict_project / "scripts/run_generation_quality_bridge_runtime_claim_guard_waiter.py"
    builder = strict_project / "scripts/build_generation_runtime_compute_claim_guard.py"
    strict_runner_identity = _assert_file_sha(
        runner,
        original_wrapper["strict_control"]["runner"]["sha256"],
        label="strict runner source",
    )
    strict_builder_identity = _assert_file_sha(
        builder,
        original_wrapper["strict_control"]["builder"]["sha256"],
        label="strict builder source",
    )

    failed_targets = original_wrapper["targets"]
    for key in ("status_output", "guard_output"):
        _assert_absent(Path(failed_targets[key]), label=f"failed v2 {key}")
    failure_record = build_failure_record(
        original_wrapper=original_wrapper_identity,
        failure_log=failure_log_identity,
        failed_pid=args.expected_original_wrapper_pid,
        guard_output=Path(failed_targets["guard_output"]),
    )
    _atomic_write_new(
        Path(failed_targets["status_output"]),
        failure_record,
        label="strict v2 wrapper failure status",
    )
    failure_status_identity = file_identity(Path(failed_targets["status_output"]))
    new_targets = {
        "output_root": args.strict_output_root.resolve().as_posix(),
        "status_output": args.strict_status_output.resolve().as_posix(),
        "pid_file": args.strict_pid_file.resolve().as_posix(),
        "deployment_receipt": args.strict_deployment_receipt_output.resolve().as_posix(),
        "guard_output": args.strict_guard_output.resolve().as_posix(),
        "wrapper_receipt": args.wrapper_receipt_output.resolve().as_posix(),
    }
    for key in ("status_output", "pid_file", "deployment_receipt", "guard_output"):
        _assert_absent(Path(new_targets[key]), label=f"strict v3 {key}")
    if _strict_runner_duplicates(runner=runner, output_root=args.strict_output_root):
        raise ValueError("duplicate strict runtime claim runner exists")

    os.chdir(strict_project)
    wrapper_runtime = runtime_identity(cwd=strict_project)
    _validate_runtime_policy(wrapper_runtime)
    runner_arguments = _runner_arguments(
        strict_project=strict_project,
        canonical_status=canonical_status,
        strict_output_root=args.strict_output_root.resolve(),
        status_output=args.strict_status_output.resolve(),
        pid_file=args.strict_pid_file.resolve(),
        deployment_output=args.strict_deployment_receipt_output.resolve(),
        guard_output=args.strict_guard_output.resolve(),
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_seconds,
    )
    runner_argv_sha256 = _sha256_bytes(b"\0".join(x.encode() for x in runner_arguments))
    strict_control_receipt = dict(strict_control)
    strict_control_receipt["runner"] = strict_runner_identity
    strict_control_receipt["builder"] = strict_builder_identity
    receipt = build_wrapper_receipt(
        strict_control=strict_control_receipt,
        wrapper_runtime=wrapper_runtime,
        canonical_runtime=original_wrapper["old_canonical_waiter"],
        source_runtime=original_wrapper["runtime_fairness_source_waiter"],
        targets=new_targets,
        recovery_control=recovery_control,
        recovery_source=recovery_source,
        original_wrapper=original_wrapper_identity,
        failure_log=failure_log_identity,
        strict_runner=strict_runner_identity,
        strict_builder=strict_builder_identity,
        strict_runner_argv_sha256=runner_argv_sha256,
        failed_targets={
            key: {"path": failed_targets[key], "absent": True}
            for key in ("status_output", "guard_output")
        },
        prior_failure_status=failure_status_identity,
    )
    receipt["recovery"]["canonical_status"] = canonical_status_identity
    _atomic_write_new(
        args.wrapper_receipt_output.resolve(),
        receipt,
        label="recovery wrapper receipt",
    )

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONPATH": f"{strict_project / 'src'}:{strict_project}",
        }
    )
    os.execve(runner_arguments[0], runner_arguments, environment)
    raise AssertionError("os.execve returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
