"""Execute the authorized frozen-checkpoint capacity confirmation.

This controller never trains. It samples and evaluates the four exact step-10K
checkpoints frozen by a passing capacity screen, builds physical arm reports,
and emits the fail-closed confirmation result while preserving terminal hold.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - formal execution is POSIX-only
    fcntl = None  # type: ignore[assignment]

from cofitok.generation.capacity_confirmation import (
    SAMPLE_BATCH_SIZE,
    SAMPLE_COUNT,
    SAMPLE_SEED,
    SAMPLE_STEPS,
    validate_capacity_confirmation_preparation_contract,
)
from cofitok.generation.capacity_confirmation_arm import (
    build_capacity_confirmation_arm_validation,
)
from cofitok.generation.capacity_confirmation_execution import (
    capacity_confirmation_control_root,
    capacity_confirmation_execution_lock_path,
    validate_capacity_confirmation_execution_authorization,
    validate_capacity_confirmation_launch_receipt_contract,
    validate_capacity_confirmation_live_snapshot,
)
from cofitok.generation.capacity_confirmation_result import (
    build_capacity_confirmation_result,
    validate_capacity_confirmation_result_contract,
)
from cofitok.generation.capacity_screen import ARM_NAMES, ARM_SPECS, STOP_STEP
from cofitok.generation.capacity_screen_arm import (
    validate_capacity_screen_arm_validation,
)
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import write_json_report
from cofitok.training.checkpointing import verify_training_checkpoint


ROLE = "generation_capacity_confirmation_controller"
STATUS_SCHEMA = 1
MIN_FREE_BYTES = 120 * 1024**3


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate four frozen capacity-screen checkpoints with 10K samples."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--expected-launch-receipt-sha256", required=True)
    parser.add_argument("--live-snapshot", type=Path, required=True)
    parser.add_argument("--expected-live-snapshot-sha256", required=True)
    for arm in ARM_NAMES:
        option = arm.replace("_", "-")
        parser.add_argument(
            f"--{option}-screen-validation",
            dest=f"{arm}_screen_validation",
            type=Path,
            required=True,
        )
        parser.add_argument(
            f"--expected-{option}-screen-validation-sha256",
            dest=f"expected_{arm}_screen_validation_sha256",
            required=True,
        )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--real-dir", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--status-output", type=Path)
    parser.add_argument("--log-output", type=Path)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Recover only a verified incomplete confirmation output root.",
    )
    return parser.parse_args(argv)


def _read(path: Path, name: str) -> dict[str, Any]:
    source = reject_symlink_chain(path, name=name)
    if not source.is_file():
        raise FileNotFoundError(f"{name} is missing: {source}")
    return read_object(source, name=name)


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
        "frozen_checkpoint_training_allowed": False,
        "large_capacity_readiness_launch_allowed": False,
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
        raise RuntimeError("capacity confirmation requires POSIX flock")
    path = reject_symlink_chain(path, name="capacity confirmation execution lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "another capacity confirmation controller holds the sibling lock"
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
            raise ValueError("capacity-confirmation GPU inventory is malformed")
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
            raise ValueError("capacity-confirmation compute inventory is malformed")
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
        "run_generation_capacity_confirmation.py",
        "generate_samples.py",
        "evaluate_generation_metrics.py",
        "evaluate_generation_class_fidelity.py",
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
        command = fields[-1]
        if (
            pid != current
            and root_text in command
            and any(marker in command for marker in markers)
        ):
            conflicts.append({"pid": pid, "command": stripped})
    return conflicts


def _assert_idle(output_root: Path) -> None:
    inventory, compute = _gpu_inventory()
    if len(inventory) != 1:
        raise RuntimeError("capacity confirmation requires exactly one target GPU")
    if compute:
        raise RuntimeError("capacity confirmation refuses a busy target GPU")
    if any(
        row["memory_used_mib"] > 16 or row["utilization_percent"] > 5
        for row in inventory
    ):
        raise RuntimeError("capacity confirmation target GPU is not idle")
    if _matching_processes(output_root):
        raise RuntimeError("matching capacity-confirmation process is already active")


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


def _layout(output_root: Path, arm: str) -> dict[str, Path]:
    arm_root = output_root / arm
    sampling = arm_root / "samples_10000_ddim100_cfg15"
    prefix = int(ARM_SPECS[arm]["prefix_budget"])
    return {
        "root": arm_root,
        "sampling": sampling,
        "sampling_report": sampling / "sampling_report.json",
        "generated": sampling / f"prefix_{prefix}",
        "metrics": arm_root / "metrics",
        "metrics_report": arm_root / "metrics" / "generation_metrics_report.json",
        "class_fidelity": arm_root / "class_fidelity",
        "class_report": arm_root
        / "class_fidelity"
        / "class_fidelity_report.json",
    }


def _validate_root_for_resume(
    output_root: Path,
    *,
    status_path: Path,
    authorization_identity: Mapping[str, Any],
    launch_identity: Mapping[str, Any],
) -> None:
    if not output_root.is_dir() or output_root.is_symlink():
        raise ValueError("capacity confirmation resume root is not a directory")
    if not status_path.is_file() or status_path.is_symlink():
        raise ValueError("capacity confirmation resume status is missing")
    status = _read(status_path, "capacity confirmation controller status")
    if (
        status.get("schema_version") != STATUS_SCHEMA
        or status.get("role") != ROLE
        or status.get("status") not in {"running", "failed"}
        or status.get("authorization") != dict(authorization_identity)
        or status.get("launch_receipt") != dict(launch_identity)
        or status.get("terminal_status") != "hold"
        or status.get("generation_advantage_proven") is not False
    ):
        raise ValueError("capacity confirmation resume status differs")
    for field in (
        "frozen_checkpoint_training_allowed",
        "large_capacity_readiness_launch_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_allowed",
        "export_allowed",
        "release_allowed",
        "process_signals_allowed",
    ):
        if status.get(field) is not False:
            raise ValueError(f"capacity confirmation resume status permits {field}")
    if (output_root / "capacity_confirmation_result.json").exists():
        raise ValueError("capacity confirmation result already exists")
    allowed = set(ARM_NAMES) | {"arm_validations"}
    unexpected = sorted(
        child.name for child in output_root.iterdir() if child.name not in allowed
    )
    if unexpected:
        raise ValueError(
            "capacity confirmation resume root contains unexpected entries: "
            + ", ".join(unexpected)
        )


def _assert_fresh_root_absent(output_root: Path, *, control_root: Path) -> None:
    if output_root.exists():
        raise FileExistsError(
            f"capacity confirmation output root already exists; pass --resume: {output_root}"
        )
    if control_root.exists() and any(control_root.iterdir()):
        raise FileExistsError(
            "capacity confirmation control directory already contains state; pass --resume"
        )


def _verify_frozen_checkpoint(
    screen_validation: Mapping[str, Any],
    *,
    arm: str,
) -> Path:
    checkpoint = dict(screen_validation["training"]["checkpoint"])
    path = reject_symlink_chain(checkpoint["path"], name=f"{arm} frozen checkpoint")
    if not path.is_file():
        raise FileNotFoundError(f"{arm} frozen checkpoint is missing: {path}")
    verified = verify_training_checkpoint(path)
    sidecar = reject_symlink_chain(
        checkpoint["integrity_manifest"]["path"],
        name=f"{arm} frozen checkpoint integrity",
    )
    if (
        int(checkpoint.get("step", -1)) != STOP_STEP
        or path.stat().st_size != int(checkpoint.get("bytes", -1))
        or verified.get("checkpoint_sha256") != checkpoint.get("sha256")
        or int(verified.get("step", -1)) != STOP_STEP
        or identity(sidecar) != checkpoint.get("integrity_manifest")
    ):
        raise ValueError(f"{arm} frozen checkpoint integrity differs")
    return path.resolve()


def _build_arm_validation(
    *,
    output_root: Path,
    arm: str,
    launch_receipt: Path,
    launch_sha: str,
    screen_validation: Path,
    screen_validation_sha: str,
) -> dict[str, Any]:
    layout = _layout(output_root, arm)
    report_path = output_root / "arm_validations" / f"{arm}.json"
    report = build_capacity_confirmation_arm_validation(
        arm=arm,
        launch_receipt_path=launch_receipt,
        expected_launch_receipt_sha256=launch_sha,
        screen_arm_validation_path=screen_validation,
        expected_screen_arm_validation_sha256=screen_validation_sha,
        sampling_report_path=layout["sampling_report"],
        metrics_report_path=layout["metrics_report"],
        class_fidelity_report_path=layout["class_report"],
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    if report_path.exists():
        if _read(report_path, f"{arm} confirmation validation") != report:
            raise ValueError(f"{arm} existing confirmation validation differs")
    else:
        write_json_report(report_path, report)
    return _read(report_path, f"{arm} confirmation validation")


def _build_result(
    *,
    project_root: Path,
    output_root: Path,
    preparation: Mapping[str, Any],
    preparation_id: Mapping[str, Any],
    launch_receipt: Mapping[str, Any],
    launch_id: Mapping[str, Any],
) -> Path:
    reports: dict[str, dict[str, Any]] = {}
    report_ids: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        path = output_root / "arm_validations" / f"{arm}.json"
        reports[arm] = _read(path, f"{arm} confirmation validation")
        report_ids[arm] = identity(path)
    result = build_capacity_confirmation_result(
        preparation=dict(preparation),
        preparation_identity=dict(preparation_id),
        launch_receipt=dict(launch_receipt),
        launch_receipt_identity=dict(launch_id),
        arm_validations=reports,
        arm_validation_identities=report_ids,
        result_git=checkout_identity(project_root),
    )
    validate_capacity_confirmation_result_contract(result)
    path = output_root / "capacity_confirmation_result.json"
    if path.exists():
        if _read(path, "capacity confirmation result") != result:
            raise ValueError("existing capacity confirmation result differs")
    else:
        write_json_report(path, result)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = reject_symlink_chain(
        args.project_root,
        name="capacity-confirmation execution checkout",
    ).resolve()
    output_root = reject_symlink_chain(
        args.output_root,
        name="capacity-confirmation output root",
    ).resolve()
    exact_lock = Path(capacity_confirmation_execution_lock_path(output_root.as_posix()))
    control_root = Path(capacity_confirmation_control_root(output_root.as_posix()))
    status_path = (
        reject_symlink_chain(args.status_output, name="capacity confirmation status")
        if args.status_output is not None
        else control_root / "controller_status.json"
    ).resolve()
    log_path = (
        reject_symlink_chain(args.log_output, name="capacity confirmation log")
        if args.log_output is not None
        else control_root / "controller.log"
    ).resolve()
    if (
        output_root.as_posix() in status_path.as_posix()
        or output_root.as_posix() in log_path.as_posix()
    ):
        raise ValueError("capacity confirmation control artifacts must stay outside result root")

    execution_git = checkout_identity(project_root)
    if execution_git["tracked_dirty"] is not False:
        raise ValueError("capacity-confirmation execution checkout has tracked changes")
    preparation, preparation_id = _stable_load(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        name="capacity-confirmation preparation",
    )
    validate_capacity_confirmation_preparation_contract(
        preparation,
        expected_output_root=output_root.as_posix(),
    )
    authorization, authorization_id = _stable_load(
        args.authorization,
        expected_sha256=args.expected_authorization_sha256,
        name="capacity-confirmation execution authorization",
    )
    validate_capacity_confirmation_execution_authorization(
        authorization,
        preparation_identity=preparation_id,
        expected_execution_checkout=execution_git,
        expected_output_root=output_root.as_posix(),
    )
    launch, launch_id = _stable_load(
        args.launch_receipt,
        expected_sha256=args.expected_launch_receipt_sha256,
        name="capacity-confirmation launch receipt",
    )
    validated_launch = validate_capacity_confirmation_launch_receipt_contract(
        launch,
        expected_execution_checkout=execution_git,
    )
    live, live_id = _stable_load(
        args.live_snapshot,
        expected_sha256=args.expected_live_snapshot_sha256,
        name="capacity-confirmation live snapshot",
    )
    validate_capacity_confirmation_live_snapshot(
        live,
        expected_output_root=output_root.as_posix(),
        expected_execution_checkout=execution_git,
        expected_execution_lock=exact_lock.as_posix(),
    )
    sources = validated_launch["source_evidence"]
    if (
        sources.get("preparation") != preparation_id
        or sources.get("execution_authorization") != authorization_id
        or sources.get("live_snapshot") != live_id
        or validated_launch.get("output_root") != output_root.as_posix()
        or validated_launch.get("execution_lock") != exact_lock.as_posix()
    ):
        raise ValueError("capacity-confirmation launch sources differ")
    if not args.python.is_file():
        raise FileNotFoundError(
            f"capacity-confirmation Python executable is missing: {args.python}"
        )
    if not args.real_dir.is_dir():
        raise FileNotFoundError(f"capacity-confirmation real directory is missing: {args.real_dir}")
    if not args.classifier_checkpoint.is_file():
        raise FileNotFoundError(
            f"capacity-confirmation classifier is missing: {args.classifier_checkpoint}"
        )

    screen_reports: dict[str, dict[str, Any]] = {}
    screen_paths: dict[str, Path] = {}
    screen_ids = sources["capacity_screen_arm_validations"]
    for arm in ARM_NAMES:
        path = reject_symlink_chain(
            getattr(args, f"{arm}_screen_validation"),
            name=f"{arm} screen validation",
        ).resolve()
        report, descriptor = _stable_load(
            path,
            expected_sha256=getattr(
                args,
                f"expected_{arm}_screen_validation_sha256",
            ),
            name=f"{arm} screen validation",
        )
        validated = validate_capacity_screen_arm_validation(report)
        if (
            descriptor != screen_ids[arm]
            or descriptor
            != preparation["source_evidence"]["capacity_screen_arm_validations"][arm]
            or validated.get("arm") != arm
            or validated.get("execution_git") != execution_git
        ):
            raise ValueError(f"{arm} screen evidence differs")
        screen_reports[arm] = validated
        screen_paths[arm] = path

    try:
        with _execution_lock(exact_lock):
            if args.resume:
                _validate_root_for_resume(
                    output_root,
                    status_path=status_path,
                    authorization_identity=authorization_id,
                    launch_identity=launch_id,
                )
            else:
                _assert_fresh_root_absent(output_root, control_root=control_root)
                output_root.parent.mkdir(parents=True, exist_ok=True)
                output_root.mkdir(parents=True, exist_ok=False)
            _assert_idle(output_root)
            if (
                int(validated_launch["storage_capacity"]["filesystem"]["free_bytes"])
                < MIN_FREE_BYTES
                or shutil.disk_usage(output_root.parent).free < MIN_FREE_BYTES
            ):
                raise RuntimeError("capacity-confirmation launch storage is below reserve")
            control_root.mkdir(parents=True, exist_ok=True)
            _write_status(
                status_path,
                status="running",
                stage="preflight",
                detail="frozen confirmation authorization and physical checkpoints validated",
                authorization=authorization_id,
                launch_receipt=launch_id,
            )
            env = _common_env(project_root)
            resume_flag = ["--resume"] if args.resume else []
            with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
                for arm in ARM_NAMES:
                    layout = _layout(output_root, arm)
                    checkpoint = _verify_frozen_checkpoint(
                        screen_reports[arm],
                        arm=arm,
                    )
                    prefix = int(ARM_SPECS[arm]["prefix_budget"])
                    _assert_idle(output_root)
                    _write_status(
                        status_path,
                        status="running",
                        stage=f"sampling_{arm}",
                        detail=f"10K EMA DDIM-100 sampling for frozen {arm}",
                        authorization=authorization_id,
                        launch_receipt=launch_id,
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
                            str(SAMPLE_BATCH_SIZE),
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
                        detail=f"FID/IS/precision/recall for frozen {arm}",
                        authorization=authorization_id,
                        launch_receipt=launch_id,
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
                        detail=f"class fidelity for frozen {arm}",
                        authorization=authorization_id,
                        launch_receipt=launch_id,
                    )
                    _run(
                        [
                            args.python,
                            project_root
                            / "scripts"
                            / "evaluate_generation_class_fidelity.py",
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
                        stage=f"arm_validation_{arm}",
                        detail=f"physical confirmation validation for {arm}",
                        authorization=authorization_id,
                        launch_receipt=launch_id,
                    )
                    _build_arm_validation(
                        output_root=output_root,
                        arm=arm,
                        launch_receipt=args.launch_receipt,
                        launch_sha=args.expected_launch_receipt_sha256,
                        screen_validation=screen_paths[arm],
                        screen_validation_sha=getattr(
                            args,
                            f"expected_{arm}_screen_validation_sha256",
                        ),
                    )
                _write_status(
                    status_path,
                    status="running",
                    stage="result",
                    detail="building fail-closed capacity confirmation result",
                    authorization=authorization_id,
                    launch_receipt=launch_id,
                )
                result_path = _build_result(
                    project_root=project_root,
                    output_root=output_root,
                    preparation=preparation,
                    preparation_id=preparation_id,
                    launch_receipt=launch,
                    launch_id=launch_id,
                )
                validate_capacity_confirmation_result_contract(
                    _read(result_path, "capacity confirmation result")
                )
                _write_status(
                    status_path,
                    status="completed",
                    stage="complete",
                    detail="capacity confirmation completed; terminal scientific hold preserved",
                    authorization=authorization_id,
                    launch_receipt=launch_id,
                )
        return 0
    except BaseException as error:
        if output_root.exists() or args.status_output is not None:
            try:
                _write_status(
                    status_path,
                    status="failed",
                    stage="failed",
                    detail=f"{type(error).__name__}: {error}",
                    authorization=authorization_id,
                    launch_receipt=launch_id,
                    exit_code=1,
                )
            except BaseException:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
