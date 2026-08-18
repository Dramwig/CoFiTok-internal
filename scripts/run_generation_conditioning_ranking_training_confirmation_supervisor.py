from __future__ import annotations

import argparse
import copy
import os
import socket
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    validate_standing_experiment_authorization,
)
from cofitok.generation.conditioning_ranking_training_confirmation import (
    CONFIG_RELATIVE_PATHS,
    EXECUTION_AUTHORIZATION_BOUNDARY,
    EXPECTED_OUTPUT_ROOT,
    IDLE_GPU_EVIDENCE_ROLE,
    RUNBOOK_RELATIVE_PATH,
    RUN_NAMES,
    STAGE,
    build_training_confirmation_execution_receipt,
    build_training_confirmation_preparation,
    sampling_validation_route,
    validate_idle_gpu_evidence,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report
from scripts.build_generation_conditioning_ranking_sampling_validation import (
    replay_sampling_validation,
)
from scripts.prepare_generation_conditioning_ranking_training_confirmation import (
    _parameter_count,
)


ROLE = "generation_conditioning_ranking_training_confirmation_supervisor"
LAUNCH_RECEIPT_ROLE = (
    "generation_conditioning_ranking_training_confirmation_runbook_launch_receipt"
)
TRAINING_STATUS_RELATIVE_PATH = Path("reports") / "training_status.json"
SUPERVISOR_BOUNDARY = {
    **copy.deepcopy(EXECUTION_AUTHORIZATION_BOUNDARY),
    "source_not_selected_gpu_work_allowed": False,
    "failed_or_malformed_source_gpu_work_allowed": False,
}


def _bound_json(
    path: Path,
    *,
    expected_sha256: str | None,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if expected_sha256 is not None and identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return read_json_object(source, name=label), identity


def _full_git_status(project: Path) -> str:
    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=project,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _assert_exact_clean_git(
    project: Path,
    *,
    expected_git: Mapping[str, Any],
) -> None:
    if git_provenance(project) != dict(expected_git) or _full_git_status(project):
        raise ValueError(
            "training confirmation supervisor requires the exact fully clean checkout"
        )


def parse_gpu_process_pids(output: str) -> list[int]:
    pids: list[int] = []
    for line in output.splitlines():
        value = line.strip()
        if not value or value.lower().startswith("no running"):
            continue
        try:
            pids.append(int(value.split(",", 1)[0].strip()))
        except ValueError as error:
            raise ValueError(
                f"unparseable nvidia-smi compute process row: {value}"
            ) from error
    return sorted(set(pids))


def gpu_compute_pids() -> list[int]:
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
    return parse_gpu_process_pids(result.stdout)


def build_idle_gpu_evidence(
    *,
    git: Mapping[str, Any],
    output_root: str,
    observations: list[Mapping[str, Any]],
    required_polls: int,
) -> dict[str, Any]:
    if required_polls != 5:
        raise ValueError("training confirmation requires exactly five idle polls")
    report = {
        "schema_version": 1,
        "role": IDLE_GPU_EVIDENCE_ROLE,
        "status": "pass",
        "stage": STAGE,
        "output_root": output_root,
        "git": dict(git),
        "required_consecutive_idle_polls": required_polls,
        "hostname": socket.gethostname(),
        "observations": [copy.deepcopy(dict(row)) for row in observations],
    }
    validate_idle_gpu_evidence(
        report,
        expected_git=git,
        expected_output_root=output_root,
        required_polls=required_polls,
    )
    return report


def build_launch_receipt(
    *,
    git: Mapping[str, Any],
    output_root: str,
    sampling_identity: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    standing_identity: Mapping[str, Any],
    idle_identity: Mapping[str, Any],
    execution_receipt_identity: Mapping[str, Any],
    runbook_identity: Mapping[str, Any],
    config_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "role": LAUNCH_RECEIPT_ROLE,
        "status": "launch_reserved",
        "stage": STAGE,
        "output_root": output_root,
        "git": dict(git),
        "sources": {
            "sampling_validation": dict(sampling_identity),
            "preparation": dict(preparation_identity),
            "standing_authorization": dict(standing_identity),
            "idle_gpu_evidence": dict(idle_identity),
            "execution_receipt": dict(execution_receipt_identity),
            "runbook": dict(runbook_identity),
            "configs": {
                name: dict(config_identities[name]) for name in RUN_NAMES
            },
        },
        "command": ["bash", str(runbook_identity["path"])],
        "authorization_boundary": copy.deepcopy(SUPERVISOR_BOUNDARY),
    }


def _status(
    *,
    status: str,
    detail: str,
    project: Path,
    output_root: Path,
    idle_polls: int,
    child_pid: int | None = None,
    sources: Mapping[str, Any] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "role": ROLE,
        "status": status,
        "detail": detail,
        "updated_at_unix": time.time(),
        "pid": os.getpid(),
        "child_pid": child_pid,
        "project": project.resolve().as_posix(),
        "output_root": output_root.resolve().as_posix(),
        "idle_gpu_polls": idle_polls,
        "sources": dict(sources or {}),
        "authorization_boundary": copy.deepcopy(SUPERVISOR_BOUNDARY),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _write_status(path: Path, **kwargs: Any) -> None:
    write_json_report(path, _status(**kwargs))


def _config_paths(project: Path) -> dict[str, Path]:
    paths = {
        name: reject_symlink_chain(
            project / relative,
            name=f"{name} config",
        ).resolve()
        for name, relative in CONFIG_RELATIVE_PATHS.items()
    }
    if set(paths) != set(RUN_NAMES) or any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("training confirmation config set is incomplete")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wait for the exact shared-pass 5K sampling validation, collect "
            "five idle-GPU observations, and launch the fresh four-arm 5K "
            "training confirmation at most once."
        )
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--sampling-validation", type=Path, required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--idle-gpu-evidence", type=Path, required=True)
    parser.add_argument("--execution-receipt", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--status-output", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--required-idle-polls", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=float, default=2_592_000.0)
    return parser.parse_args()


def _run(args: argparse.Namespace) -> int:
    project = reject_symlink_chain(
        args.project,
        name="training confirmation supervisor project",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="training confirmation output root",
    ).resolve()
    sampling_path = reject_symlink_chain(
        args.sampling_validation,
        name="conditioning-ranking 5K sampling validation",
    ).resolve()
    standing_path = reject_symlink_chain(
        args.standing_authorization,
        name="standing experiment authorization",
    ).resolve()
    preparation_path = reject_symlink_chain(
        args.preparation,
        name="training confirmation preparation",
    ).resolve()
    idle_path = reject_symlink_chain(
        args.idle_gpu_evidence,
        name="training confirmation idle GPU evidence",
    ).resolve()
    execution_receipt_path = reject_symlink_chain(
        args.execution_receipt,
        name="training confirmation execution receipt",
    ).resolve()
    launch_receipt_path = reject_symlink_chain(
        args.launch_receipt,
        name="training confirmation launch receipt",
    ).resolve()
    status_output = reject_symlink_chain(
        args.status_output,
        name="training confirmation supervisor status",
    ).resolve()
    pid_file = reject_symlink_chain(
        args.pid_file,
        name="training confirmation supervisor pid",
    ).resolve()
    runbook = reject_symlink_chain(
        args.runbook,
        name="training confirmation runbook",
    ).resolve()
    python = reject_symlink_chain(
        args.python,
        name="training confirmation Python",
    ).resolve()
    expected_git = {
        "revision": args.expected_revision,
        "branch": args.expected_branch,
        "tracked_dirty": False,
    }
    if output_root.as_posix() != EXPECTED_OUTPUT_ROOT:
        raise ValueError("training confirmation output root differs")
    if args.required_idle_polls != 5:
        raise ValueError("training confirmation requires exactly five idle polls")
    if args.poll_seconds <= 0.0 or args.timeout_seconds <= 0.0:
        raise ValueError("training confirmation supervisor timing values must be positive")
    if runbook != (project / RUNBOOK_RELATIVE_PATH).resolve():
        raise ValueError("training confirmation runbook path differs")
    if not project.is_dir() or not runbook.is_file() or not python.is_file():
        raise FileNotFoundError(
            "training confirmation checkout, runbook, or Python is missing"
        )
    control_paths = (
        preparation_path,
        idle_path,
        execution_receipt_path,
        launch_receipt_path,
        status_output,
        pid_file,
    )
    control_parent = preparation_path.parent
    if any(path.parent != control_parent for path in control_paths):
        raise ValueError(
            "training confirmation control artifacts must share one directory"
        )
    if any(output_root == path or output_root in path.parents for path in control_paths):
        raise ValueError("training confirmation control artifacts cannot be in output root")
    _assert_exact_clean_git(project, expected_git=expected_git)
    standing, standing_identity = _bound_json(
        standing_path,
        expected_sha256=args.expected_standing_authorization_sha256,
        label="standing experiment authorization",
    )
    validate_standing_experiment_authorization(standing)
    configs = _config_paths(project)
    config_identities = {name: file_identity(path) for name, path in configs.items()}
    runbook_identity = file_identity(runbook)
    if launch_receipt_path.exists():
        raise FileExistsError(
            "training confirmation launch was already reserved; relaunch is forbidden"
        )
    if idle_path.exists() or execution_receipt_path.exists():
        raise FileExistsError(
            "stale training confirmation execution-control artifact exists"
        )
    if output_root.exists() or Path(f"{output_root}.lock").exists():
        raise FileExistsError("training confirmation output or lock already exists")
    prepare_manifest(
        pid_file,
        {
            "schema_version": 1,
            "role": f"{ROLE}_pid",
            "pid": os.getpid(),
            "project": project.as_posix(),
            "expected_revision": args.expected_revision,
            "expected_branch": args.expected_branch,
        },
        resume=False,
        overwrite=False,
    )
    sources: dict[str, Any] = {
        "standing_authorization": standing_identity,
        "runbook": runbook_identity,
        "configs": config_identities,
    }
    started = time.monotonic()
    observations: list[dict[str, Any]] = []
    preparation_identity: dict[str, Any] | None = None
    sampling_identity: dict[str, Any] | None = None
    sampling: dict[str, Any] | None = None
    while True:
        if time.monotonic() - started > args.timeout_seconds:
            _write_status(
                status_output,
                status="failed",
                detail="timeout_before_training_confirmation_launch",
                project=project,
                output_root=output_root,
                idle_polls=len(observations),
                sources=sources,
            )
            return 4
        if preparation_identity is None:
            if not sampling_path.is_file():
                observations = []
                _write_status(
                    status_output,
                    status="waiting",
                    detail="waiting_for_shared_pass_5k_sampling_validation",
                    project=project,
                    output_root=output_root,
                    idle_polls=0,
                    sources=sources,
                )
                time.sleep(args.poll_seconds)
                continue
            sampling, sampling_identity = replay_sampling_validation(
                sampling_path,
            )
            sources["sampling_validation"] = sampling_identity
            route = sampling_validation_route(sampling)
            if route == "not_selected":
                _write_status(
                    status_output,
                    status="completed",
                    detail="training_confirmation_not_selected_by_5k_sampling",
                    project=project,
                    output_root=output_root,
                    idle_polls=0,
                    sources=sources,
                )
                return 0
            if route != "selected":
                raise ValueError(
                    "conditioning-ranking 5K sampling validation route is malformed"
                )
            loaded_configs = {
                name: config_to_dict(load_config(path))
                for name, path in configs.items()
            }
            preparation = build_training_confirmation_preparation(
                sampling_validation=sampling,
                sampling_validation_identity=sampling_identity,
                standing_authorization=standing,
                standing_authorization_identity=standing_identity,
                control_cofitok=loaded_configs["control_cofitok"],
                control_dense=loaded_configs["control_dense_identity"],
                ranked_cofitok=loaded_configs["ranked_cofitok"],
                ranked_dense=loaded_configs["ranked_dense_identity"],
                config_identities=config_identities,
                parameter_counts={
                    name: _parameter_count(path) for name, path in configs.items()
                },
                builder_git=git_provenance(project),
                expected_revision=args.expected_revision,
                expected_branch=args.expected_branch,
                expected_output_root=output_root.as_posix(),
            )
            preparation_identity = prepare_manifest(
                preparation_path,
                preparation,
                resume=preparation_path.is_file(),
                overwrite=False,
            )
            sources["preparation"] = preparation_identity
        if sampling is None or sampling_identity is None or preparation_identity is None:
            raise RuntimeError("training confirmation preparation state was not retained")
        if output_root.exists() or Path(f"{output_root}.lock").exists():
            raise FileExistsError("training confirmation output or lock appeared")
        pids = gpu_compute_pids()
        if pids:
            observations = []
            _write_status(
                status_output,
                status="waiting",
                detail="gpu_busy",
                project=project,
                output_root=output_root,
                idle_polls=0,
                sources={**sources, "gpu_compute_pids": pids},
            )
            time.sleep(args.poll_seconds)
            continue
        observed_at = time.time()
        if observations and observed_at <= observations[-1]["observed_at_unix"]:
            observed_at = float(observations[-1]["observed_at_unix"]) + 1e-6
        observations.append(
            {
                "poll_index": len(observations) + 1,
                "observed_at_unix": observed_at,
                "gpu_compute_pids": [],
            }
        )
        if len(observations) < args.required_idle_polls:
            _write_status(
                status_output,
                status="waiting",
                detail="waiting_for_consecutive_idle_gpu_polls",
                project=project,
                output_root=output_root,
                idle_polls=len(observations),
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue
        if gpu_compute_pids():
            observations = []
            _write_status(
                status_output,
                status="waiting",
                detail="gpu_became_busy_before_launch_reservation",
                project=project,
                output_root=output_root,
                idle_polls=0,
                sources=sources,
            )
            time.sleep(args.poll_seconds)
            continue
        idle = build_idle_gpu_evidence(
            git=expected_git,
            output_root=output_root.as_posix(),
            observations=observations,
            required_polls=args.required_idle_polls,
        )
        idle_identity = prepare_manifest(
            idle_path,
            idle,
            resume=False,
            overwrite=False,
        )
        sources["idle_gpu_evidence"] = idle_identity
        _assert_exact_clean_git(project, expected_git=expected_git)
        if file_identity(sampling_path) != sampling_identity:
            raise ValueError("conditioning-ranking sampling validation changed")
        if file_identity(preparation_path) != preparation_identity:
            raise ValueError("training confirmation preparation changed")
        if file_identity(standing_path) != standing_identity:
            raise ValueError("standing experiment authorization changed")
        if file_identity(runbook) != runbook_identity or {
            name: file_identity(path) for name, path in configs.items()
        } != config_identities:
            raise ValueError("training confirmation code-bound inputs changed")
        receipt = build_training_confirmation_execution_receipt(
            preparation=read_json_object(
                preparation_path,
                name="training confirmation preparation",
            ),
            preparation_identity=preparation_identity,
            sampling_validation=sampling,
            sampling_validation_identity=sampling_identity,
            standing_authorization=standing,
            standing_authorization_identity=standing_identity,
            idle_gpu_evidence=idle,
            idle_gpu_evidence_identity=idle_identity,
            config_identities=config_identities,
            runbook_identity=runbook_identity,
            receipt_git=git_provenance(project),
            expected_revision=args.expected_revision,
            expected_branch=args.expected_branch,
            expected_output_root=output_root.as_posix(),
        )
        execution_identity = prepare_manifest(
            execution_receipt_path,
            receipt,
            resume=False,
            overwrite=False,
        )
        sources["execution_receipt"] = execution_identity
        if gpu_compute_pids():
            raise RuntimeError(
                "GPU became busy after execution receipt; launch was not reserved"
            )
        launch_receipt = build_launch_receipt(
            git=expected_git,
            output_root=output_root.as_posix(),
            sampling_identity=sampling_identity,
            preparation_identity=preparation_identity,
            standing_identity=standing_identity,
            idle_identity=idle_identity,
            execution_receipt_identity=execution_identity,
            runbook_identity=runbook_identity,
            config_identities=config_identities,
        )
        launch_identity = prepare_manifest(
            launch_receipt_path,
            launch_receipt,
            resume=False,
            overwrite=False,
        )
        sources["runbook_launch_receipt"] = launch_identity
        environment = os.environ.copy()
        environment.update(
            {
                "PROJECT": project.as_posix(),
                "PYTHON": python.as_posix(),
                "EXPECTED_REVISION": args.expected_revision,
                "EXPECTED_BRANCH": args.expected_branch,
                "SAMPLING_VALIDATION": sampling_path.as_posix(),
                "EXPECTED_SAMPLING_VALIDATION_SHA256": sampling_identity["sha256"],
                "STANDING_AUTHORIZATION": standing_path.as_posix(),
                "EXPECTED_STANDING_AUTHORIZATION_SHA256": standing_identity["sha256"],
                "PREPARATION_REPORT": preparation_path.as_posix(),
                "EXPECTED_PREPARATION_SHA256": preparation_identity["sha256"],
                "IDLE_GPU_EVIDENCE": idle_path.as_posix(),
                "EXPECTED_IDLE_GPU_EVIDENCE_SHA256": idle_identity["sha256"],
                "EXECUTION_RECEIPT": execution_receipt_path.as_posix(),
                "EXPECTED_EXECUTION_RECEIPT_SHA256": execution_identity["sha256"],
                "EXPECTED_RUNBOOK_SHA256": runbook_identity["sha256"],
                "OUTPUT_ROOT": output_root.as_posix(),
            }
        )
        child = subprocess.Popen(
            ["bash", runbook.as_posix()],
            cwd=project,
            env=environment,
        )
        while child.poll() is None:
            _write_status(
                status_output,
                status="running",
                detail="conditioning_ranking_four_arm_train5k_confirmation_running",
                project=project,
                output_root=output_root,
                idle_polls=len(observations),
                child_pid=child.pid,
                sources=sources,
            )
            time.sleep(min(args.poll_seconds, 60.0))
        if child.returncode != 0:
            _write_status(
                status_output,
                status="failed",
                detail="training_confirmation_runbook_failed",
                project=project,
                output_root=output_root,
                idle_polls=len(observations),
                child_pid=child.pid,
                sources=sources,
                error=f"runbook exit code {child.returncode}",
            )
            return int(child.returncode or 1)
        training_status = output_root / TRAINING_STATUS_RELATIVE_PATH
        if not training_status.is_file():
            raise FileNotFoundError("training confirmation status is missing")
        sources["training_status"] = file_identity(training_status)
        _write_status(
            status_output,
            status="completed",
            detail="conditioning_ranking_four_arm_train5k_confirmation_completed",
            project=project,
            output_root=output_root,
            idle_polls=len(observations),
            child_pid=child.pid,
            sources=sources,
        )
        return 0


def main() -> int:
    args = parse_args()
    try:
        return _run(args)
    except Exception as error:
        try:
            _write_status(
                Path(args.status_output).resolve(),
                status="failed",
                detail="training_confirmation_supervisor_exception",
                project=Path(args.project).resolve(),
                output_root=Path(args.output_root).resolve(),
                idle_polls=0,
                sources={},
                error=f"{type(error).__name__}: {error}",
            )
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
