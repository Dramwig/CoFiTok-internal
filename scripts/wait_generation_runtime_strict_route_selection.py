from __future__ import annotations

import argparse
import copy
import os
import socket
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.output_lock import OutputLockError, exclusive_output_lock
from cofitok.reporting import write_json_report

try:
    import build_generation_runtime_strict_route_selection as builder
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import build_generation_runtime_strict_route_selection as builder


SCHEMA_VERSION = 1
ROLE = "generation_terminal_runtime_strict_route_selection_waiter"
DEPLOYMENT_ROLE = "generation_terminal_runtime_strict_route_selection_waiter_deployment"
OUTPUT_DIR_NAME = "terminal_runtime_strict_route_selection_v1"
RECEIPT_NAME = "route_selection_receipt.json"
STATUS_NAME = "waiter_status.json"
DEPLOYMENT_NAME = "deployment_receipt.json"

SCOPE = {
    "cpu_only": True,
    "permanently_non_authorizing": True,
    "route_selection_only": True,
    "gpu_execution_allowed": False,
    "checkpoint_payload_loading_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "inference_export_authorization_allowed": False,
    "process_signals_allowed": False,
    "source_waiter_signals_allowed": False,
    "legacy_interlocks_modified": False,
    "upstream_decisions_modified": False,
}
RUNTIME_POLICY = {
    "detached_parent_pid": 1,
    "cuda_visible_devices": "",
    "omp_num_threads": "1",
    "mkl_num_threads": "1",
    "minimum_nice": 10,
    "ionice": "idle",
}

HASH_ARGUMENTS = {
    "standing_authorization": "expected_standing_authorization_sha256",
    "pair_monitor": "expected_pair_monitor_sha256",
    "runtime_guard": "expected_runtime_guard_sha256",
    "exposure_deployment_receipt": "expected_exposure_deployment_sha256",
    "comparison_deployment_receipt": "expected_comparison_deployment_sha256",
    "completion_deployment_receipt": "expected_completion_deployment_sha256",
    "conjunct_deployment_receipt": "expected_conjunct_deployment_sha256",
    "interlock_receipt": "expected_interlock_receipt_sha256",
    "factorization_marker": "expected_factorization_marker_sha256",
    "conditioning_marker": "expected_conditioning_marker_sha256",
    "random_token_marker": "expected_random_token_marker_sha256",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_identity(project: Path) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", *arguments],
            cwd=project,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "tree": git("rev-parse", "HEAD^{tree}"),
        "branch": git("branch", "--show-current"),
        "tracked_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
    }


def _wait_for_detach(seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while os.getppid() != RUNTIME_POLICY["detached_parent_pid"]:
        if time.monotonic() >= deadline:
            raise ValueError("route selector did not detach to parent PID 1")
        time.sleep(0.05)


def _runtime_identity(
    *, require_detached: bool, detach_wait_seconds: float
) -> dict[str, Any]:
    if require_detached:
        _wait_for_detach(detach_wait_seconds)
    pid = os.getpid()
    parent_pid = os.getppid()
    nice = os.getpriority(os.PRIO_PROCESS, 0)
    ionice = subprocess.run(
        ["ionice", "-p", str(pid)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    proc = Path("/proc") / str(pid)
    stat_fields = proc.joinpath("stat").read_text(encoding="utf-8").split()
    gpu_rows = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .splitlines()
    )
    gpu_pids = {
        int(row.split(",", 1)[0].strip())
        for row in gpu_rows
        if row.strip() and row.split(",", 1)[0].strip().isdigit()
    }
    runtime = {
        "pid": pid,
        "parent_pid": parent_pid,
        "start_ticks": int(stat_fields[21]),
        "hostname": socket.gethostname(),
        "cwd": os.readlink(proc / "cwd"),
        "cmdline": (
            proc.joinpath("cmdline")
            .read_bytes()
            .replace(b"\0", b" ")
            .decode("utf-8", "replace")
            .strip()
        ),
        "python_executable": os.path.realpath(sys.executable),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
        "nice": nice,
        "ionice": ionice,
        "gpu_processes": gpu_rows,
        "self_is_gpu_process": pid in gpu_pids,
    }
    if (
        (require_detached and parent_pid != RUNTIME_POLICY["detached_parent_pid"])
        or runtime["cuda_visible_devices"] != RUNTIME_POLICY["cuda_visible_devices"]
        or runtime["omp_num_threads"] != RUNTIME_POLICY["omp_num_threads"]
        or runtime["mkl_num_threads"] != RUNTIME_POLICY["mkl_num_threads"]
        or nice < RUNTIME_POLICY["minimum_nice"]
        or ionice != RUNTIME_POLICY["ionice"]
        or runtime["self_is_gpu_process"] is not False
    ):
        raise ValueError("runtime-strict route selector runtime policy differs")
    return runtime


def _within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _expected_hashes(args: argparse.Namespace) -> dict[str, str]:
    return {
        source: str(getattr(args, argument))
        for source, argument in HASH_ARGUMENTS.items()
    }


def _source_paths(args: argparse.Namespace) -> builder.RouteSourcePaths:
    return builder.RouteSourcePaths(
        quality_root=reject_symlink_chain(
            args.quality_output_root, name="route selector quality root"
        ).resolve(),
        standing_authorization=reject_symlink_chain(
            args.standing_authorization, name="route selector standing authorization"
        ).resolve(),
        factorization_marker=reject_symlink_chain(
            args.factorization_marker, name="route selector factorization marker"
        ).resolve(),
        conditioning_marker=reject_symlink_chain(
            args.conditioning_marker, name="route selector conditioning marker"
        ).resolve(),
        random_token_marker=reject_symlink_chain(
            args.random_token_marker, name="route selector random-token marker"
        ).resolve(),
    )


def _canonical_context(
    args: argparse.Namespace,
    *,
    require_detached: bool,
) -> dict[str, Any]:
    project = reject_symlink_chain(
        args.project, name="route selector project"
    ).resolve()
    git = _git_identity(project)
    expected_git = {
        "revision": args.expected_control_revision,
        "tree": args.expected_control_tree,
        "branch": args.expected_control_branch,
        "tracked_dirty": False,
    }
    if git != expected_git:
        raise ValueError("runtime-strict route selector Git identity differs")
    waiter_source = file_identity(project / "scripts" / Path(__file__).name)
    builder_source = file_identity(
        project / "scripts" / "build_generation_runtime_strict_route_selection.py"
    )
    if waiter_source["sha256"] != args.expected_waiter_source_sha256:
        raise ValueError("runtime-strict route selector waiter SHA256 differs")
    if builder_source["sha256"] != args.expected_builder_source_sha256:
        raise ValueError("runtime-strict route selector builder SHA256 differs")
    paths = _source_paths(args)
    reports = paths.quality_root / "reports"
    output_dir = reject_symlink_chain(
        args.output_dir, name="route selector output directory"
    ).resolve()
    expected_output_dir = reports / OUTPUT_DIR_NAME
    if output_dir != expected_output_dir:
        raise ValueError("runtime-strict route selector output directory differs")
    targets = {
        "receipt": output_dir / RECEIPT_NAME,
        "status": output_dir / STATUS_NAME,
        "deployment": output_dir / DEPLOYMENT_NAME,
    }
    generation_root = paths.quality_root.parent
    expected_markers = {
        "factorization": generation_root
        / "stability_full_data_100k_factorization_quality_regression_v1.lock"
        / "supersession_marker.json",
        "conditioning": generation_root
        / "conditioning_ranking_four_arm_probe1k_v1.lock"
        / "supersession_marker.json",
        "random_token": generation_root
        / "stability_full_data_100k_random_token_semantic_visual_v1",
    }
    observed_markers = {
        "factorization": paths.factorization_marker,
        "conditioning": paths.conditioning_marker,
        "random_token": paths.random_token_marker,
    }
    if observed_markers != expected_markers:
        raise ValueError("runtime-strict route selector marker paths differ")
    for target in targets.values():
        if not _within(target, output_dir):
            raise ValueError(
                "runtime-strict route selector target escapes output directory"
            )
    static_paths = paths.json_paths()
    hashes = _expected_hashes(args)
    static_identities = {}
    for name, expected_sha256 in hashes.items():
        identity = file_identity(static_paths[name])
        if identity["sha256"] != expected_sha256:
            raise ValueError(f"runtime-strict route selector {name} SHA256 differs")
        static_identities[name] = identity
    runtime = _runtime_identity(
        require_detached=require_detached,
        detach_wait_seconds=args.detach_wait_seconds,
    )
    if runtime["cwd"] != project.as_posix():
        raise ValueError("runtime-strict route selector cwd differs")
    return {
        "project": project,
        "git": git,
        "waiter_source": waiter_source,
        "builder_source": builder_source,
        "paths": paths,
        "json_paths": static_paths,
        "output_dir": output_dir,
        "targets": targets,
        "expected_hashes": hashes,
        "static_identities": static_identities,
        "runtime": runtime,
    }


def _static_signature(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "git": context["git"],
        "waiter_source": context["waiter_source"],
        "builder_source": context["builder_source"],
        "expected_hashes": context["expected_hashes"],
        "static_identities": context["static_identities"],
        "targets": {name: path.as_posix() for name, path in context["targets"].items()},
    }


def _deployment_payload(context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": DEPLOYMENT_ROLE,
        "status": "pass",
        "control_git": copy.deepcopy(dict(context["git"])),
        "sources": {
            "waiter": copy.deepcopy(dict(context["waiter_source"])),
            "builder": copy.deepcopy(dict(context["builder_source"])),
        },
        "runtime_policy": copy.deepcopy(RUNTIME_POLICY),
        "expected_source_hashes": copy.deepcopy(dict(context["expected_hashes"])),
        "static_source_identities": copy.deepcopy(dict(context["static_identities"])),
        "targets": {name: path.as_posix() for name, path in context["targets"].items()},
        "scope": copy.deepcopy(SCOPE),
    }


def _prepare_exact_json(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    expected = dict(payload)
    if path.is_file():
        existing = read_json_object(path, name="route selector deployment receipt")
        if existing != expected:
            raise ValueError("route selector deployment receipt differs")
    else:
        write_json_report(path, expected)
    return file_identity(path)


def observe_upstreams(context: Mapping[str, Any]) -> dict[str, Any]:
    paths = context["json_paths"]
    required_outputs = {
        "quality_bridge_result": paths["quality_bridge_result"],
        "followup_decision": paths["followup_decision"],
        "terminal_guard": paths["terminal_guard"],
        "comparison": paths["comparison"],
        "completion": paths["completion"],
        "conjunct": paths["conjunct"],
    }
    missing = [name for name, path in required_outputs.items() if not path.is_file()]
    status_contracts = {
        "exposure": (paths["exposure_waiter_status"], "completed"),
        "terminal_guard": (paths["terminal_guard_status"], "completed"),
        "comparison": (paths["comparison_status"], "pass"),
        "completion": (paths["completion_status"], "pass"),
        "conjunct": (paths["conjunct_status"], "pass"),
    }
    statuses = {}
    waiting = []
    for name, (path, ready_status) in status_contracts.items():
        if not path.is_file():
            waiting.append(f"{name}_status_missing")
            continue
        report = read_json_object(path, name=f"{name} waiter status")
        statuses[name] = {
            "identity": file_identity(path),
            "status": report.get("status"),
            "detail": report.get("detail"),
        }
        if report.get("status") in {"failed", "error"}:
            return {
                "state": "failed",
                "detail": f"{name}_waiter_failed",
                "statuses": statuses,
                "missing": missing,
            }
        if report.get("status") != ready_status:
            waiting.append(f"{name}_not_{ready_status}")
    if missing or waiting:
        return {
            "state": "waiting",
            "detail": "waiting_for_runtime_strict_terminal_route_sources",
            "missing": missing,
            "waiting": waiting,
            "statuses": statuses,
        }
    return {
        "state": "ready",
        "detail": "runtime_strict_terminal_route_sources_ready",
        "missing": [],
        "waiting": [],
        "statuses": statuses,
    }


def _status_payload(
    *,
    status: str,
    detail: str,
    context: Mapping[str, Any],
    deployment: Mapping[str, Any],
    observation: Mapping[str, Any] | None = None,
    receipt: Mapping[str, Any] | None = None,
    selection: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "updated_at": _utc_now(),
        "git": copy.deepcopy(dict(context["git"])),
        "runtime": copy.deepcopy(dict(context["runtime"])),
        "waiter_source": copy.deepcopy(dict(context["waiter_source"])),
        "builder_source": copy.deepcopy(dict(context["builder_source"])),
        "deployment_receipt": copy.deepcopy(dict(deployment)),
        "observation": copy.deepcopy(dict(observation)) if observation else None,
        "receipt": copy.deepcopy(dict(receipt)) if receipt else None,
        "selection": copy.deepcopy(dict(selection)) if selection else None,
        "generation_advantage_proven": (
            bool(selection.get("generation_advantage_proven", False))
            if selection
            else False
        ),
        "error_type": type(error).__name__ if error is not None else None,
        "error": str(error) if error is not None else None,
        "scope": copy.deepcopy(SCOPE),
    }


def _publish(path: Path, **kwargs: Any) -> None:
    write_json_report(path, _status_payload(**kwargs))


def run_locked(
    args: argparse.Namespace,
    *,
    require_detached: bool = True,
) -> int:
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("route selector timing values must be positive")
    context = _canonical_context(args, require_detached=require_detached)
    context["output_dir"].mkdir(parents=True, exist_ok=True)
    deployment = _prepare_exact_json(
        context["targets"]["deployment"], _deployment_payload(context)
    )
    signature = _static_signature(context)
    deadline = time.monotonic() + args.timeout_seconds
    while True:
        refreshed = _canonical_context(args, require_detached=require_detached)
        if _static_signature(refreshed) != signature:
            raise ValueError("runtime-strict route selector static context changed")
        context = refreshed
        observation = observe_upstreams(context)
        if observation["state"] == "failed":
            _publish(
                context["targets"]["status"],
                status="failed",
                detail=observation["detail"],
                context=context,
                deployment=deployment,
                observation=observation,
            )
            return 1
        if observation["state"] != "ready":
            _publish(
                context["targets"]["status"],
                status="waiting",
                detail=observation["detail"],
                context=context,
                deployment=deployment,
                observation=observation,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError("runtime-strict route selector timed out")
            time.sleep(min(args.poll_seconds, remaining))
            continue
        _publish(
            context["targets"]["status"],
            status="running",
            detail="replaying_runtime_strict_terminal_route_sources",
            context=context,
            deployment=deployment,
            observation=observation,
        )
        report = builder.build_route_selection(
            paths=context["paths"],
            expected_hashes=context["expected_hashes"],
            selector_git=context["git"],
            selector_sources={
                "waiter": context["waiter_source"],
                "builder": context["builder_source"],
            },
        )
        receipt_identity = prepare_manifest(
            context["targets"]["receipt"],
            report,
            resume=context["targets"]["receipt"].is_file(),
            overwrite=False,
        )
        _publish(
            context["targets"]["status"],
            status="pass",
            detail="runtime_strict_terminal_route_selected_non_authorizing",
            context=context,
            deployment=deployment,
            observation=observation,
            receipt=receipt_identity,
            selection={
                **copy.deepcopy(dict(report["selection"])),
                "terminal_status": report["terminal_status"],
                "generation_advantage_proven": report["generation_advantage_proven"],
            },
        )
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the corrected runtime-strict terminal evidence chain and publish "
            "one CPU-only, permanently non-authorizing route-selection receipt."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--factorization-marker", type=Path, required=True)
    parser.add_argument("--conditioning-marker", type=Path, required=True)
    parser.add_argument("--random-token-marker", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-control-revision", required=True)
    parser.add_argument("--expected-control-tree", required=True)
    parser.add_argument("--expected-control-branch", required=True)
    parser.add_argument("--expected-waiter-source-sha256", required=True)
    parser.add_argument("--expected-builder-source-sha256", required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--expected-pair-monitor-sha256", required=True)
    parser.add_argument("--expected-runtime-guard-sha256", required=True)
    parser.add_argument("--expected-exposure-deployment-sha256", required=True)
    parser.add_argument("--expected-comparison-deployment-sha256", required=True)
    parser.add_argument("--expected-completion-deployment-sha256", required=True)
    parser.add_argument("--expected-conjunct-deployment-sha256", required=True)
    parser.add_argument("--expected-interlock-receipt-sha256", required=True)
    parser.add_argument("--expected-factorization-marker-sha256", required=True)
    parser.add_argument("--expected-conditioning-marker-sha256", required=True)
    parser.add_argument("--expected-random-token-marker-sha256", required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    parser.add_argument("--detach-wait-seconds", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = reject_symlink_chain(
        args.output_dir, name="route selector output directory"
    ).resolve()
    try:
        with exclusive_output_lock(output_dir, role=ROLE):
            return run_locked(args)
    except OutputLockError as error:
        print(str(error), file=sys.stderr)
        return 75
    except Exception as error:  # noqa: BLE001 - fail-closed terminal boundary
        try:
            context = _canonical_context(args, require_detached=False)
            context["output_dir"].mkdir(parents=True, exist_ok=True)
            deployment = (
                file_identity(context["targets"]["deployment"])
                if context["targets"]["deployment"].is_file()
                else _prepare_exact_json(
                    context["targets"]["deployment"], _deployment_payload(context)
                )
            )
            _publish(
                context["targets"]["status"],
                status="failed",
                detail=f"{type(error).__name__}:{error}",
                context=context,
                deployment=deployment,
                error=error,
            )
        except Exception as status_error:  # noqa: BLE001
            print(
                f"unable to publish route selector failure: {status_error}",
                file=sys.stderr,
            )
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
