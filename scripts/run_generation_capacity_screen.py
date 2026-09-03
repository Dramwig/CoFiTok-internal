"""Execute the explicitly authorized four-arm capacity screen.

The screen is intentionally bounded at an exact 10K stop for each fresh arm.
It is a diagnostic intervention only: it can train and evaluate the four
predeclared arms, but it cannot authorize confirmation, 300K training,
promotion, export, or release.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - formal execution is POSIX-only
    fcntl = None  # type: ignore[assignment]

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.capacity_qualification_training import (
    validate_capacity_qualification_partial_training,
)
from cofitok.generation.capacity_screen import (
    ARM_NAMES,
    ARM_SPECS,
    EFFECTIVE_BATCH,
    STOP_STEP,
)
from cofitok.generation.capacity_screen_arm import (
    build_capacity_screen_arm_validation,
)
from cofitok.generation.capacity_screen_execution import (
    LIVE_SNAPSHOT_ROLE,
    LIVE_SNAPSHOT_SCHEMA,
    capacity_screen_benchmark_root,
    capacity_screen_control_root,
    capacity_screen_execution_lock_path,
    validate_capacity_screen_execution_authorization,
    validate_capacity_screen_launch_receipt_contract,
)
from cofitok.generation.capacity_screen_result import (
    build_capacity_screen_result,
    validate_capacity_screen_result_contract,
)
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.checkpointing import (
    resolve_latest_checkpoint,
    verify_training_checkpoint,
)


ROLE = "generation_capacity_screen_controller"
STATUS_SCHEMA = 1
SAMPLE_COUNT = 1_000
SAMPLE_STEPS = 100
SAMPLE_SEED = 0
ROLLOUT_SEED = 2029
ROLLOUT_TEACHER_TIMESTEPS = [999, 900, 750, 500, 250, 100, 10]
MIN_FREE_BYTES = 120 * 1024**3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute the source-bound four-arm capacity screen to 10K."
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
    parser.add_argument(
        "--base128-cofitok-config",
        dest="base128_cofitok_config",
        type=Path,
        required=True,
        help="Fresh base-128 CoFiTok capacity-screen configuration.",
    )
    parser.add_argument(
        "--base128-dense-identity-config",
        dest="base128_dense_identity_config",
        type=Path,
        required=True,
        help="Fresh base-128 dense-identity capacity-screen configuration.",
    )
    parser.add_argument(
        "--base256-cofitok-config",
        dest="base256_cofitok_config",
        type=Path,
        required=True,
        help="Fresh base-256 CoFiTok capacity-screen configuration.",
    )
    parser.add_argument(
        "--base256-dense-identity-config",
        dest="base256_dense_identity_config",
        type=Path,
        required=True,
        help="Fresh base-256 dense-identity capacity-screen configuration.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--status-output", type=Path, default=None)
    parser.add_argument("--log-output", type=Path, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Recover only a verified incomplete capacity-screen output root.",
    )
    return parser.parse_args(argv)


def _read(path: Path, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    payload = read_object(source, name=name)
    return payload


def _stable_load(
    path: Path,
    *,
    expected_sha256: str,
    name: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = identity(path)
    if before["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    payload = _read(path, name)
    after = identity(path)
    if after != before:
        raise ValueError(f"{name} changed while being read")
    return payload, before


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
        "capacity_confirmation_allowed": False,
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
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


@contextmanager
def _execution_lock(path: Path) -> Iterator[None]:
    if fcntl is None:
        raise RuntimeError("capacity screen execution requires POSIX flock")
    path = reject_symlink_chain(path, name="capacity screen execution lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another capacity screen controller holds the sibling lock"
            ) from error
        yield
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _gpu_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inventory_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory: list[dict[str, Any]] = []
    for line in inventory_result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 5)]
        if len(fields) != 6:
            raise ValueError("capacity-screen GPU inventory is malformed")
        inventory.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "memory_used_mib": int(fields[3]),
                "memory_total_mib": int(fields[4]),
                "utilization_percent": int(fields[5]),
            }
        )
    compute_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    compute: list[dict[str, Any]] = []
    for line in compute_result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise ValueError("capacity-screen compute inventory is malformed")
        compute.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": int(fields[2]),
            }
        )
    return inventory, compute


def _matching_processes(output_root: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    markers = (
        "run_generation_capacity_screen.py",
        "train_generation.py",
        "generate_samples.py",
        "evaluate_generation_metrics.py",
        "evaluate_generation_class_fidelity.py",
        "evaluate_generation_checkpoint.py",
        "evaluate_generation_rollout_stability.py",
    )
    root_text = output_root.as_posix()
    current = os.getpid()
    conflicts: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        fields = stripped.split(maxsplit=2)
        try:
            pid = int(fields[0])
        except (IndexError, ValueError):
            continue
        if pid == current:
            continue
        command = fields[-1]
        if root_text in command and any(marker in command for marker in markers):
            conflicts.append({"pid": pid, "command": stripped})
    return conflicts


def _assert_idle(output_root: Path) -> None:
    inventory, compute = _gpu_inventory()
    if len(inventory) != 1:
        raise RuntimeError("capacity screen requires exactly one target GPU")
    if compute:
        raise RuntimeError("capacity screen refuses a busy target GPU")
    if any(
        row["memory_used_mib"] > 16 or row["utilization_percent"] > 5
        for row in inventory
    ):
        raise RuntimeError("capacity screen target GPU is not idle")
    conflicts = _matching_processes(output_root)
    if conflicts:
        raise RuntimeError("matching capacity-screen process is already active")


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_handle: Any,
) -> None:
    log_handle.write("\n$ " + shlex.join(str(value) for value in command) + "\n")
    log_handle.flush()
    subprocess.run(
        [str(value) for value in command],
        cwd=cwd,
        env=dict(env),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        check=True,
    )


def _common_env(project_root: Path) -> dict[str, str]:
    environment = dict(os.environ)
    entries = [project_root.as_posix(), (project_root / "src").as_posix()]
    if environment.get("PYTHONPATH"):
        entries.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(entries)
    return environment


def _resolved_config(config_path: Path, *, micro: int, accumulation: int) -> dict[str, Any]:
    config = load_config(config_path)
    from dataclasses import replace

    resolved = replace(
        config,
        data=replace(config.data, batch_size=micro),
        optimization=replace(
            config.optimization,
            gradient_accumulation_steps=accumulation,
        ),
    )
    return config_to_dict(resolved)


def _run_has_entries(run_dir: Path) -> bool:
    if not run_dir.exists():
        return False
    run_dir = reject_symlink_chain(run_dir, name="capacity screen training run")
    if not run_dir.is_dir():
        raise ValueError(f"capacity screen training run is not a directory: {run_dir}")
    return any(run_dir.iterdir())


def _training_state(
    *,
    run_dir: Path,
    config_path: Path,
    arm: str,
    execution_git: Mapping[str, Any],
    runtime_environment_sha256: str,
    dataset_identity_sha256: str,
    micro_batch: int,
    accumulation: int,
) -> tuple[str, int]:
    """Return ``(action, step)`` where action is fresh/resume/skip."""

    if not _run_has_entries(run_dir):
        return "fresh", 0
    latest_path = run_dir / "latest.json"
    if not latest_path.is_file() or latest_path.is_symlink():
        raise ValueError(f"{arm} non-empty run lacks a safe latest.json")
    checkpoint = resolve_latest_checkpoint(run_dir)
    integrity = verify_training_checkpoint(checkpoint)
    step = int(integrity.get("step", -1))
    if step < 1 or step > STOP_STEP:
        raise ValueError(f"{arm} checkpoint step {step} is outside the bounded 10K stop")
    if (
        integrity.get("git_revision") != execution_git["revision"]
        or integrity.get("git_branch") != execution_git["branch"]
        or integrity.get("git_dirty") is not False
        or integrity.get("runtime_environment_sha256") != runtime_environment_sha256
        or integrity.get("dataset_identity_sha256") != dataset_identity_sha256
    ):
        raise ValueError(f"{arm} checkpoint identity differs from the launch receipt")
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError(f"{arm} run manifest is missing or unsafe")
    manifest = _read(manifest_path, f"{arm} run manifest")
    expected_config = _resolved_config(
        config_path,
        micro=micro_batch,
        accumulation=accumulation,
    )
    if (
        manifest.get("config") != expected_config
        or manifest.get("git", {}).get("revision") != execution_git["revision"]
        or manifest.get("git", {}).get("branch") != execution_git["branch"]
        or manifest.get("git", {}).get("dirty") is not False
        or manifest.get("runtime_environment_sha256") != runtime_environment_sha256
        or manifest.get("dataset_provenance", {}).get("identity_sha256") is None
        or int(manifest.get("parameter_count", -1))
        != int(ARM_SPECS[arm]["parameter_count"])
    ):
        raise ValueError(f"{arm} run manifest identity/config differs")
    if step == STOP_STEP:
        report_path = run_dir / "training_report.json"
        if not report_path.is_file():
            raise ValueError(f"{arm} reached 10K without a training report")
        validate_capacity_qualification_partial_training(
            report_path=report_path,
            config_path=config_path,
            expected_revision=str(execution_git["revision"]),
            expected_branch=str(execution_git["branch"]),
            expected_parameter_count=int(ARM_SPECS[arm]["parameter_count"]),
            expected_micro_batch_size=micro_batch,
            expected_gradient_accumulation_steps=accumulation,
            expected_stage=str(ARM_SPECS[arm]["recipe_stage"]),
            expected_base_channels=int(ARM_SPECS[arm]["base_channels"]),
            allow_exact_resume=True,
        )
        return "skip", step
    return "resume", step


def _validate_resume_rollout(
    *,
    report_path: Path,
    checkpoint: Path,
    config_path: Path,
    arm: str,
    execution_git: Mapping[str, Any],
) -> None:
    report = _read(report_path, f"{arm} rollout report")
    if report.get("status") != "completed":
        raise RuntimeError(
            f"{arm} rollout output is partial; refusing implicit rerun"
        )
    integrity = verify_training_checkpoint(checkpoint)
    expected_protocol = {
        "num_images": 64,
        "batch_size": 2,
        "teacher_timesteps": ROLLOUT_TEACHER_TIMESTEPS,
        "sample_steps": SAMPLE_STEPS,
        "guidance_scale": 1.5,
        "teacher_guidance_scale": 1.0,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "clip_x0": True,
        "precision": "bf16",
        "seed": ROLLOUT_SEED,
    }
    if (
        report.get("checkpoint") != checkpoint.resolve().as_posix()
        or report.get("checkpoint_sha256") != integrity.get("checkpoint_sha256")
        or report.get("checkpoint_step") != STOP_STEP
        or report.get("weights") != "ema"
        or report.get("protocol") != expected_protocol
        or report.get("config") != _resolved_config(
            config_path,
            micro=int(report.get("config", {}).get("data", {}).get("batch_size", -1)),
            accumulation=int(
                report.get("config", {})
                .get("optimization", {})
                .get("gradient_accumulation_steps", -1)
            ),
        )
        or report.get("git", {}).get("revision") != execution_git["revision"]
        or report.get("git", {}).get("branch") != execution_git["branch"]
        or report.get("git", {}).get("tracked_dirty") is not False
    ):
        raise ValueError(f"{arm} completed rollout binding differs")


def _layout(output_root: Path, arm: str) -> dict[str, Path]:
    evaluation = output_root / "evaluations" / arm
    run = output_root / "training" / arm
    prefix = int(ARM_SPECS[arm]["prefix_budget"])
    sampling = evaluation / "sampling"
    return {
        "run": run,
        "training_report": run / "training_report.json",
        "sampling": sampling,
        "sampling_report": sampling / "sampling_report.json",
        "generated": sampling / f"prefix_{prefix}",
        "metrics": evaluation / "metrics",
        "metrics_report": evaluation / "metrics" / "generation_metrics_report.json",
        "class_fidelity": evaluation / "class_fidelity",
        "class_report": evaluation / "class_fidelity" / "class_fidelity_report.json",
        "mechanism": evaluation / "checkpoint_eval",
        "checkpoint_report": evaluation / "checkpoint_eval" / "checkpoint_evaluation_report.json",
        "rollout": evaluation / "rollout",
        "rollout_report": evaluation / "rollout" / "rollout_stability_report.json",
    }


def _validate_root_for_resume(
    output_root: Path,
    *,
    status_path: Path,
    authorization_identity: Mapping[str, Any],
    launch_identity: Mapping[str, Any],
) -> None:
    if not output_root.is_dir() or output_root.is_symlink():
        raise ValueError("capacity screen resume root is not a directory")
    if not status_path.is_file() or status_path.is_symlink():
        raise ValueError("capacity screen resume status is missing")
    status = _read(status_path, "capacity screen controller status")
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("role") != ROLE
        or status.get("status") not in {"running", "failed"}
        or status.get("authorization") != dict(authorization_identity)
        or status.get("launch_receipt") != dict(launch_identity)
        or status.get("terminal_status") != "hold"
        or status.get("generation_advantage_proven") is not False
    ):
        raise ValueError("capacity screen resume status is not bound to this launch")
    for field in (
        "capacity_confirmation_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_allowed",
        "export_allowed",
        "release_allowed",
        "process_signals_allowed",
    ):
        if status.get(field) is not False:
            raise ValueError(f"capacity screen resume status permits {field}")
    if (output_root / "capacity_screen_result.json").exists():
        raise ValueError("capacity screen result already exists; use a fresh audit")
    allowed = {"training", "evaluations", "arm_validations"}
    unexpected = sorted(
        child.name for child in output_root.iterdir() if child.name not in allowed
    )
    if unexpected:
        raise ValueError(
            "capacity screen resume root contains unexpected entries: "
            + ", ".join(unexpected)
        )


def _assert_fresh_root_absent(output_root: Path, *, control_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"capacity screen output root already exists; pass --resume: {output_root}"
        )
    if control_root.exists() and any(control_root.iterdir()):
        raise FileExistsError(
            "capacity screen control directory already contains state; pass --resume"
        )


def _build_arm_validation(
    *,
    project_root: Path,
    output_root: Path,
    arm: str,
    launch_receipt: Path,
    launch_sha: str,
    config_path: Path,
) -> dict[str, Any]:
    layout = _layout(output_root, arm)
    report_path = output_root / "arm_validations" / f"{arm}.json"
    report = build_capacity_screen_arm_validation(
        arm=arm,
        launch_receipt_path=launch_receipt,
        expected_launch_receipt_sha256=launch_sha,
        config_path=config_path,
        training_report_path=layout["training_report"],
        sampling_report_path=layout["sampling_report"],
        metrics_report_path=layout["metrics_report"],
        class_fidelity_report_path=layout["class_report"],
        checkpoint_evaluation_report_path=layout["checkpoint_report"],
        rollout_report_path=layout["rollout_report"],
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        existing = _read(report_path, f"{arm} arm validation")
        if existing != report:
            raise ValueError(f"{arm} existing arm validation differs")
    else:
        write_json_report(report_path, report)
    return _read(report_path, f"{arm} arm validation")


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
    arm_reports: dict[str, dict[str, Any]] = {}
    arm_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        report_path = output_root / "arm_validations" / f"{arm}.json"
        arm_reports[arm] = _read(report_path, f"{arm} arm validation")
        arm_ids[arm] = identity(report_path)
    result = build_capacity_screen_result(
        preparation=dict(preparation),
        preparation_identity=dict(preparation_id),
        launch_receipt=dict(launch_receipt),
        launch_receipt_identity=dict(launch_id),
        arm_validations=arm_reports,
        arm_validation_identities=arm_ids,
        result_git=checkout_identity(project_root),
    )
    validate_capacity_screen_result_contract(result)
    path = output_root / "capacity_screen_result.json"
    if path.exists():
        existing = _read(path, "capacity screen result")
        if existing != result:
            raise ValueError("existing capacity screen result differs from replay")
    else:
        write_json_report(path, result)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = reject_symlink_chain(
        args.project_root,
        name="capacity screen execution checkout",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="capacity screen output root",
    ).resolve()
    exact_lock = Path(capacity_screen_execution_lock_path(output_root.as_posix()))
    supplied_lock = reject_symlink_chain(
        exact_lock,
        name="capacity screen execution lock",
    ).resolve()
    if supplied_lock != exact_lock.resolve():
        raise ValueError("capacity screen execution lock is not the exact sibling lock")
    control_root = Path(capacity_screen_control_root(output_root.as_posix()))
    status_path = (
        reject_symlink_chain(args.status_output, name="capacity screen status")
        if args.status_output is not None
        else control_root / "controller_status.json"
    ).resolve()
    log_path = (
        reject_symlink_chain(args.log_output, name="capacity screen log")
        if args.log_output is not None
        else control_root / "controller.log"
    ).resolve()
    if output_root.as_posix() in status_path.as_posix() or output_root.as_posix() in log_path.as_posix():
        raise ValueError("capacity screen control artifacts must stay outside the result root")

    execution_git = checkout_identity(project_root)
    if execution_git["tracked_dirty"] is not False:
        raise ValueError("capacity screen execution checkout has tracked changes")
    preparation, preparation_id = _stable_load(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        name="capacity screen preparation",
    )
    authorization, authorization_id = _stable_load(
        args.authorization,
        expected_sha256=args.expected_authorization_sha256,
        name="capacity screen execution authorization",
    )
    launch, launch_id = _stable_load(
        args.launch_receipt,
        expected_sha256=args.expected_launch_receipt_sha256,
        name="capacity screen launch receipt",
    )
    runtime_selection, runtime_id = _stable_load(
        args.runtime_selection,
        expected_sha256=args.expected_runtime_selection_sha256,
        name="capacity screen runtime selection",
    )
    live_snapshot, live_id = _stable_load(
        args.live_snapshot,
        expected_sha256=args.expected_live_snapshot_sha256,
        name="capacity screen live snapshot",
    )
    validated_authorization = validate_capacity_screen_execution_authorization(
        authorization,
        preparation_identity=preparation_id,
        expected_execution_checkout=execution_git,
        expected_output_root=output_root.as_posix(),
    )
    validated_launch = validate_capacity_screen_launch_receipt_contract(
        launch,
        expected_execution_checkout=execution_git,
    )
    if (
        validated_launch.get("output_root") != output_root.as_posix()
        or validated_launch.get("execution_lock") != exact_lock.as_posix()
        or validated_launch.get("source_evidence", {}).get("execution_authorization")
        != authorization_id
        or validated_launch.get("source_evidence", {}).get("runtime_selection")
        != runtime_id
        or validated_launch.get("source_evidence", {}).get("live_snapshot")
        != live_id
        or validated_launch.get("runtime_selection", {}).get("effective_batch_size")
        != EFFECTIVE_BATCH
        or validated_launch.get("runtime_selection", {}).get("dataset_identity_sha256")
        != live_snapshot.get("dataset_identity_sha256")
        or validated_launch.get("runtime_selection", {}).get("runtime_environment_sha256")
        != live_snapshot.get("runtime_environment_sha256")
    ):
        raise ValueError("capacity screen launch sources differ from supplied evidence")
    launch_auth = validated_launch.get("authorization", {})
    if launch_auth.get("preparation") != preparation_id:
        raise ValueError("capacity screen launch authorization binds another preparation")
    runtime = validated_launch["runtime_selection"]
    micro_batch = int(runtime["micro_batch_size"])
    accumulation = int(runtime["gradient_accumulation_steps"])
    if micro_batch * accumulation != EFFECTIVE_BATCH:
        raise ValueError("capacity screen runtime selection changes effective batch")
    configs = {
        arm: reject_symlink_chain(
            getattr(args, f"{arm}_config", None),
            name=f"{arm} config",
        ).resolve()
        for arm in ARM_NAMES
    }
    launch_configs = validated_launch["source_evidence"]["configs"]
    for arm, path in configs.items():
        if identity(path) != launch_configs[arm]:
            raise ValueError(f"{arm} config differs from launch receipt")
    if not args.python.is_file():
        raise FileNotFoundError(f"capacity screen Python executable is missing: {args.python}")
    auth_summary = authorization_id
    launch_summary = launch_id
    completed = False
    try:
        with _execution_lock(supplied_lock):
            if args.resume:
                _validate_root_for_resume(
                    output_root,
                    status_path=status_path,
                    authorization_identity=auth_summary,
                    launch_identity=launch_summary,
                )
            else:
                _assert_fresh_root_absent(output_root, control_root=control_root)
                output_root.parent.mkdir(parents=True, exist_ok=True)
                output_root.mkdir(parents=True, exist_ok=False)
            _assert_idle(output_root)
            control_root.mkdir(parents=True, exist_ok=True)
            env = _common_env(project_root)
            _write_status(
                status_path,
                status="running",
                stage="preflight",
                detail="capacity-screen authorization and launch receipt validated",
                authorization=auth_summary,
                launch_receipt=launch_summary,
            )
            with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                for arm in ARM_NAMES:
                    layout = _layout(output_root, arm)
                    config_path = configs[arm]
                    action, step = _training_state(
                        run_dir=layout["run"],
                        config_path=config_path,
                        arm=arm,
                        execution_git=execution_git,
                        runtime_environment_sha256=str(
                            runtime["runtime_environment_sha256"]
                        ),
                        dataset_identity_sha256=str(runtime["dataset_identity_sha256"]),
                        micro_batch=micro_batch,
                        accumulation=accumulation,
                    )
                    if action == "skip":
                        _write_status(
                            status_path,
                            status="running",
                            stage=f"training_{arm}",
                            detail=f"reusing verified completed {arm} 10K stop",
                            authorization=auth_summary,
                            launch_receipt=launch_summary,
                        )
                    else:
                        _assert_idle(output_root)
                        remaining = STOP_STEP if action == "fresh" else STOP_STEP - step
                        command: list[str | Path] = [
                            args.python,
                            project_root / "scripts" / "train_generation.py",
                            "--config",
                            config_path,
                            "--output-dir",
                            layout["run"],
                            "--micro-batch-size",
                            str(micro_batch),
                            "--gradient-accumulation-steps",
                            str(accumulation),
                            "--stop-after-steps",
                            str(remaining),
                        ]
                        if action == "resume":
                            command.extend(["--resume", "auto"])
                        _write_status(
                            status_path,
                            status="running",
                            stage=f"training_{arm}",
                            detail=f"{action} training {arm} to exact 10K stop",
                            authorization=auth_summary,
                            launch_receipt=launch_summary,
                        )
                        _run(command, cwd=project_root, env=env, log_handle=log_handle)
                    training_report = layout["training_report"]
                    if not training_report.is_file():
                        raise FileNotFoundError(f"{arm} training report is missing after run")
                    validate_capacity_qualification_partial_training(
                        report_path=training_report,
                        config_path=config_path,
                        expected_revision=str(execution_git["revision"]),
                        expected_branch=str(execution_git["branch"]),
                        expected_parameter_count=int(ARM_SPECS[arm]["parameter_count"]),
                        expected_micro_batch_size=micro_batch,
                        expected_gradient_accumulation_steps=accumulation,
                        expected_stage=str(ARM_SPECS[arm]["recipe_stage"]),
                        expected_base_channels=int(ARM_SPECS[arm]["base_channels"]),
                        allow_exact_resume=True,
                    )
                    checkpoint = layout["run"] / f"checkpoint_step_{STOP_STEP:08d}.pt"
                    _assert_idle(output_root)
                    prefix = int(ARM_SPECS[arm]["prefix_budget"])
                    resume_flag = ["--resume"] if args.resume else []
                    _write_status(
                        status_path,
                        status="running",
                        stage=f"sampling_{arm}",
                        detail=f"EMA DDIM-100 sampling for {arm}",
                        authorization=auth_summary,
                        launch_receipt=launch_summary,
                    )
                    _run(
                        [
                            args.python,
                            project_root / "scripts" / "generate_samples.py",
                            "--checkpoint",
                            checkpoint,
                            "--output-dir",
                            layout["sampling"],
                            "--num-samples",
                            str(SAMPLE_COUNT),
                            "--batch-size",
                            "4",
                            "--sample-steps",
                            str(SAMPLE_STEPS),
                            "--prefix-budgets",
                            str(prefix),
                            "--guidance-scale",
                            "1.5",
                            "--guidance-rescale",
                            "0.0",
                            "--cfg-batch-mode",
                            "batched",
                            "--eta",
                            "0.0",
                            "--seed",
                            str(SAMPLE_SEED),
                            "--weights",
                            "ema",
                            "--precision",
                            "bf16",
                            *resume_flag,
                        ],
                        cwd=project_root,
                        env=env,
                        log_handle=log_handle,
                    )
                    _write_status(
                        status_path,
                        status="running",
                        stage=f"metrics_{arm}",
                        detail=f"FID/IS/precision/recall for {arm}",
                        authorization=auth_summary,
                        launch_receipt=launch_summary,
                    )
                    _run(
                        [
                            args.python,
                            project_root / "scripts" / "evaluate_generation_metrics.py",
                            "--real-dir",
                            args.real_dir,
                            "--generated-dir",
                            layout["generated"],
                            "--sampling-report",
                            layout["sampling_report"],
                            "--output-dir",
                            layout["metrics"],
                            "--batch-size",
                            "64",
                            "--prc-batch-size",
                            str(SAMPLE_COUNT),
                            "--min-samples",
                            str(SAMPLE_COUNT),
                            "--seed",
                            str(SAMPLE_SEED),
                            "--cache-root",
                            args.cache_root,
                            "--real-cache-name",
                            "imagenet256_val_50k_torch_fidelity_v04",
                            *resume_flag,
                        ],
                        cwd=project_root,
                        env=env,
                        log_handle=log_handle,
                    )
                    _write_status(
                        status_path,
                        status="running",
                        stage=f"class_fidelity_{arm}",
                        detail=f"class fidelity for {arm}",
                        authorization=auth_summary,
                        launch_receipt=launch_summary,
                    )
                    _run(
                        [
                            args.python,
                            project_root / "scripts" / "evaluate_generation_class_fidelity.py",
                            "--generated-dir",
                            layout["generated"],
                            "--sampling-report",
                            layout["sampling_report"],
                            "--output-dir",
                            layout["class_fidelity"],
                            "--classifier-checkpoint",
                            args.classifier_checkpoint,
                            "--batch-size",
                            "64",
                            "--num-workers",
                            "8",
                            "--min-samples",
                            str(SAMPLE_COUNT),
                            *resume_flag,
                        ],
                        cwd=project_root,
                        env=env,
                        log_handle=log_handle,
                    )
                    _write_status(
                        status_path,
                        status="running",
                        stage=f"mechanism_{arm}",
                        detail=f"checkpoint mechanism evaluation for {arm}",
                        authorization=auth_summary,
                        launch_receipt=launch_summary,
                    )
                    _run(
                        [
                            args.python,
                            project_root / "scripts" / "evaluate_generation_checkpoint.py",
                            "--checkpoint",
                            checkpoint,
                            "--output-dir",
                            layout["mechanism"],
                            "--num-images",
                            "256",
                            "--timestep",
                            "500",
                            "--random-orders",
                            str(4 if ARM_SPECS[arm]["method"] == "cofitok" else 0),
                            "--seed",
                            str(SAMPLE_SEED),
                            "--weights",
                            "ema",
                            "--precision",
                            "bf16",
                            *resume_flag,
                        ],
                        cwd=project_root,
                        env=env,
                        log_handle=log_handle,
                    )
                    rollout_dir = layout["rollout"]
                    if args.resume and rollout_dir.exists():
                        report_path = layout["rollout_report"]
                        if not report_path.is_file() or report_path.is_symlink():
                            raise RuntimeError(
                                f"{arm} rollout is partial; explicit repair is required"
                            )
                        _validate_resume_rollout(
                            report_path=report_path,
                            checkpoint=checkpoint,
                            config_path=config_path,
                            arm=arm,
                            execution_git=execution_git,
                        )
                    else:
                        _write_status(
                            status_path,
                            status="running",
                            stage=f"rollout_{arm}",
                            detail=f"DDIM-100 rollout stability for {arm}",
                            authorization=auth_summary,
                            launch_receipt=launch_summary,
                        )
                        _run(
                            [
                                args.python,
                                project_root / "scripts" / "evaluate_generation_rollout_stability.py",
                                "--checkpoint",
                                checkpoint,
                                "--output-dir",
                                rollout_dir,
                                "--num-images",
                                "64",
                                "--batch-size",
                                "2",
                                "--sample-steps",
                                str(SAMPLE_STEPS),
                                "--seed",
                                str(ROLLOUT_SEED),
                                "--weights",
                                "ema",
                                "--precision",
                                "bf16",
                                "--guidance-scale",
                                "1.5",
                                "--guidance-rescale",
                                "0.0",
                                "--teacher-guidance-scale",
                                "1.0",
                                "--cfg-batch-mode",
                                "batched",
                                "--clip-x0",
                            ],
                            cwd=project_root,
                            env=env,
                            log_handle=log_handle,
                        )
                    _write_status(
                        status_path,
                        status="running",
                        stage=f"arm_validation_{arm}",
                        detail=f"physical validation for {arm}",
                        authorization=auth_summary,
                        launch_receipt=launch_summary,
                    )
                    _build_arm_validation(
                        project_root=project_root,
                        output_root=output_root,
                        arm=arm,
                        launch_receipt=args.launch_receipt,
                        launch_sha=args.expected_launch_receipt_sha256,
                        config_path=config_path,
                    )
                _write_status(
                    status_path,
                    status="running",
                    stage="result",
                    detail="building and physically validating capacity screen result",
                    authorization=auth_summary,
                    launch_receipt=launch_summary,
                )
                result_path = _build_result(
                    project_root=project_root,
                    output_root=output_root,
                    preparation=preparation,
                    preparation_id=preparation_id,
                    launch_receipt=launch,
                    launch_id=launch_id,
                    launch_path=args.launch_receipt,
                    launch_sha=args.expected_launch_receipt_sha256,
                    configs=configs,
                )
                validate_capacity_screen_result_contract(
                    _read(result_path, "capacity screen result")
                )
                _write_status(
                    status_path,
                    status="completed",
                    stage="complete",
                    detail=(
                        "capacity screen completed; result is diagnostic and terminal hold remains"
                    ),
                    authorization=auth_summary,
                    launch_receipt=launch_summary,
                )
                completed = True
        return 0
    except BaseException as error:
        if output_root.exists() or args.status_output is not None:
            try:
                _write_status(
                    status_path,
                    status="failed",
                    stage="failed",
                    detail=f"{type(error).__name__}: {error}",
                    authorization=auth_summary,
                    launch_receipt=launch_summary,
                    exit_code=1,
                )
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
