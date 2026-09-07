"""Capture an idle, source-bound terminal-SNR confirmation prelaunch state."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - formal execution is POSIX-only
    fcntl = None  # type: ignore[assignment]

import torch

from cofitok.configs import load_config
from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    capture_dataset_provenance,
    validate_dataset_provenance,
)
from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation.exposure_capacity_authorization import (
    checkout_identity,
    identity,
    read_object,
)
from cofitok.generation.terminal_snr_confirmation import (
    validate_terminal_snr_confirmation_preparation_contract,
)
from cofitok.generation.terminal_snr_confirmation_execution import (
    LIVE_SNAPSHOT_ROLE,
    LIVE_SNAPSHOT_SCHEMA,
    terminal_snr_confirmation_execution_lock_path,
    validate_terminal_snr_confirmation_execution_authorization_physical,
    validate_terminal_snr_confirmation_live_snapshot,
)
from cofitok.generation.terminal_snr_screen import ARM_NAMES, ARM_SPECS, STOP_STEP
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.checkpointing import verify_training_checkpoint


DATASET = "imagenet_256"
REFERENCE_ARM = "control_cofitok"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build terminal-SNR confirmation idle prelaunch snapshot."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--expected-reference-config-sha256", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def _stable_load(
    path: Path, *, expected_sha256: str, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = identity(path)
    if before["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    payload = read_object(path, name=name)
    after = identity(path)
    if after != before:
        raise ValueError(f"{name} changed while being read")
    return payload, before


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
            raise ValueError("terminal-SNR confirmation GPU inventory is malformed")
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
            raise ValueError("terminal-SNR confirmation compute inventory is malformed")
        compute.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": int(fields[2]),
            }
        )
    return inventory, compute


def _conflicting_processes(output_root: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    markers = (
        "run_generation_terminal_snr_confirmation.py",
        "generate_samples.py",
        "evaluate_generation_metrics.py",
        "evaluate_generation_class_fidelity.py",
    )
    current = os.getpid()
    root_text = output_root.as_posix()
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


def _lock_free(path: Path) -> bool:
    if fcntl is None:
        raise RuntimeError("terminal-SNR confirmation lock requires POSIX fcntl")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (not path.is_file() or path.is_symlink()):
        return False
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return True


def _verify_frozen_checkpoints(prepared: dict[str, Any]) -> dict[str, Any]:
    frozen = prepared["frozen_arms"]
    result: dict[str, Any] = {}
    for arm in ARM_NAMES:
        checkpoint = frozen[arm]["checkpoint"]
        path = reject_symlink_chain(
            checkpoint["path"], name=f"{arm} frozen checkpoint"
        )
        sidecar = reject_symlink_chain(
            checkpoint["integrity_manifest"]["path"],
            name=f"{arm} frozen checkpoint integrity",
        )
        if not path.is_file() or not sidecar.is_file():
            raise FileNotFoundError(f"{arm} frozen checkpoint evidence is missing")
        verified = verify_training_checkpoint(path)
        if (
            path.stat().st_size != int(checkpoint["bytes"])
            or verified.get("checkpoint_sha256") != checkpoint["sha256"]
            or int(verified.get("step", -1)) != STOP_STEP
            or identity(sidecar) != checkpoint["integrity_manifest"]
        ):
            raise ValueError(f"{arm} frozen checkpoint integrity differs")
        result[arm] = {
            "checkpoint": identity(path),
            "integrity_manifest": identity(sidecar),
            "step": STOP_STEP,
        }
    return result


def build_live_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    project_root = reject_symlink_chain(
        args.project_root, name="terminal-SNR confirmation execution checkout"
    ).resolve()
    execution_git = checkout_identity(project_root)
    if execution_git["tracked_dirty"] is not False:
        raise ValueError("terminal-SNR confirmation checkout has tracked changes")
    output_root = reject_symlink_chain(
        args.output_root, name="terminal-SNR confirmation output root"
    )
    root_text = output_root.as_posix()
    preparation, preparation_id = _stable_load(
        args.preparation,
        expected_sha256=args.expected_preparation_sha256,
        name="terminal-SNR confirmation preparation",
    )
    prepared = validate_terminal_snr_confirmation_preparation_contract(
        preparation, expected_output_root=root_text
    )
    authorization, authorization_id = _stable_load(
        args.authorization,
        expected_sha256=args.expected_authorization_sha256,
        name="terminal-SNR confirmation authorization",
    )
    validate_terminal_snr_confirmation_execution_authorization_physical(
        authorization,
        preparation=prepared,
        preparation_identity=preparation_id,
        expected_execution_checkout=execution_git,
        expected_output_root=root_text,
    )
    _, config_id = _stable_load(
        args.reference_config,
        expected_sha256=args.expected_reference_config_sha256,
        name="terminal-SNR confirmation reference config",
    )
    if config_id != prepared["frozen_arms"][REFERENCE_ARM]["config"]:
        raise ValueError("terminal-SNR confirmation reference config differs")
    config = load_config(args.reference_config)
    if (
        config.data.dataset != DATASET
        or config.model.base_channels != ARM_SPECS[REFERENCE_ARM]["base_channels"]
        or config.diffusion.cosine_endpoint_fraction
        != ARM_SPECS[REFERENCE_ARM]["endpoint_fraction"]
        or config.runtime.device != "cuda"
    ):
        raise ValueError("terminal-SNR confirmation reference config differs")
    if not torch.cuda.is_available():
        raise RuntimeError("terminal-SNR confirmation live snapshot requires CUDA")
    device = torch.device(config.runtime.device)
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.benchmark = config.runtime.cudnn_benchmark
    torch.set_float32_matmul_precision("high")
    runtime_environment = capture_runtime_environment(device, project_root=project_root)
    runtime_sha = runtime_environment_sha256(runtime_environment)
    dataset_spec = FORMAL_GENERATION_DATASETS[DATASET]
    dataset_provenance = capture_dataset_provenance(
        config.data,
        train_images=dataset_spec.train_images,
        val_images=dataset_spec.val_images,
    )
    validated_dataset = validate_dataset_provenance(
        dataset_provenance, expected_dataset=DATASET
    )
    dataset_sha = str(validated_dataset["identity_sha256"])
    frozen_verification = _verify_frozen_checkpoints(prepared)
    exact_lock = terminal_snr_confirmation_execution_lock_path(root_text)
    execution_lock = reject_symlink_chain(
        args.execution_lock, name="terminal-SNR confirmation execution lock"
    )
    if execution_lock.as_posix() != exact_lock:
        raise ValueError("terminal-SNR confirmation execution lock differs")
    storage_path = reject_symlink_chain(
        args.storage_path, name="terminal-SNR confirmation storage path"
    )
    if not storage_path.is_dir():
        raise FileNotFoundError(
            f"terminal-SNR confirmation storage path is missing: {storage_path}"
        )
    gpu_inventory, gpu_compute_processes = _gpu_inventory()
    report = {
        "schema_version": LIVE_SNAPSHOT_SCHEMA,
        "role": LIVE_SNAPSHOT_ROLE,
        "status": "pass",
        "execution_checkout": execution_git,
        "source_evidence": {
            "preparation": preparation_id,
            "execution_authorization": authorization_id,
            "reference_config": config_id,
            "frozen_checkpoint_verification": frozen_verification,
        },
        "output_root": root_text,
        "execution_lock": exact_lock,
        "gpu_inventory": gpu_inventory,
        "gpu_compute_processes": gpu_compute_processes,
        "conflicting_processes": _conflicting_processes(output_root),
        "output_root_absent": not output_root.exists(),
        "execution_lock_free": _lock_free(execution_lock),
        "free_bytes": int(shutil.disk_usage(storage_path).free),
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_sha,
        "dataset_provenance": dataset_provenance,
        "dataset_identity_sha256": dataset_sha,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    return validate_terminal_snr_confirmation_live_snapshot(
        report,
        expected_output_root=root_text,
        expected_execution_checkout=execution_git,
        expected_execution_lock=exact_lock,
    )


def main() -> None:
    args = parse_args()
    output = reject_symlink_chain(
        args.output, name="terminal-SNR confirmation live snapshot output"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite live snapshot: {output}")
    report = build_live_snapshot(args)
    write_json_report(output, report)
    persisted = read_object(output, name="terminal-SNR confirmation live snapshot")
    validate_terminal_snr_confirmation_live_snapshot(
        persisted,
        expected_output_root=report["output_root"],
        expected_execution_checkout=report["execution_checkout"],
        expected_execution_lock=report["execution_lock"],
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "live_snapshot_sha256": file_sha256(output),
                "output_root_absent": True,
                "execution_lock_free": True,
                "full_300k_launch_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
