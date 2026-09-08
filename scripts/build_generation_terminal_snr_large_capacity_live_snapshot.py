"""Capture the idle launch boundary for terminal-SNR large-capacity training."""

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
except ModuleNotFoundError:  # pragma: no cover - formal builder is POSIX-only
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
from cofitok.generation.terminal_snr_large_capacity import (
    METHODS,
    validate_terminal_snr_large_capacity_config_pair,
)
from cofitok.generation.terminal_snr_large_capacity_execution import (
    LIVE_SNAPSHOT_BOUNDARY,
    LIVE_SNAPSHOT_ROLE,
    LIVE_SNAPSHOT_SCHEMA,
    terminal_snr_large_capacity_execution_lock_path,
    validate_terminal_snr_large_capacity_live_snapshot,
    validate_terminal_snr_large_capacity_runtime_selection,
    validate_terminal_snr_large_capacity_storage_capacity,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report


DATASET = "imagenet_256"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-bound idle-GPU and empty-output snapshot for "
            "fresh terminal-SNR endpoint-0.975 large-capacity training."
        )
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--expected-runtime-selection-sha256", required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--expected-storage-capacity-sha256", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--expected-cofitok-config-sha256", required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--expected-dense-config-sha256", required=True)
    parser.add_argument("--cofitok-run-dir", required=True)
    parser.add_argument("--dense-run-dir", required=True)
    parser.add_argument("--benchmark-root", required=True)
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
    result = subprocess.run(
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
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 5)]
        if len(fields) != 6:
            raise ValueError("large-capacity GPU inventory is malformed")
        inventory.append({
            "index": int(fields[0]),
            "uuid": fields[1],
            "name": fields[2],
            "memory_used_mib": int(fields[3]),
            "memory_total_mib": int(fields[4]),
            "utilization_percent": int(fields[5]),
        })
    result = subprocess.run(
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
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise ValueError("large-capacity compute inventory is malformed")
        compute.append({
            "pid": int(fields[0]),
            "process_name": fields[1],
            "used_gpu_memory_mib": int(fields[2]),
        })
    return inventory, compute


def _conflicting_processes(output_root: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    markers = (
        "run_generation_terminal_snr_large_capacity.py",
        "train_generation.py",
        "generate_samples.py",
        "evaluate_generation_metrics.py",
        "evaluate_generation_class_fidelity.py",
        "evaluate_generation_checkpoint.py",
        "evaluate_generation_rollout_stability.py",
    )
    current = os.getpid()
    root_text = output_root.as_posix()
    conflicts: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(maxsplit=2)
        if len(fields) < 3:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid != current and root_text in fields[2] and any(
            marker in fields[2] for marker in markers
        ):
            conflicts.append({"pid": pid, "command": line.strip()})
    return conflicts


def _lock_free(path: Path) -> bool:
    if fcntl is None:
        raise RuntimeError("large-capacity lock probing requires POSIX fcntl")
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


def build_live_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    project_root = reject_symlink_chain(
        args.project_root, name="large-capacity execution checkout"
    ).resolve()
    execution_git = checkout_identity(project_root)
    if execution_git["tracked_dirty"] is not False:
        raise ValueError("large-capacity execution checkout has tracked changes")

    runtime, runtime_id = _stable_load(
        args.runtime_selection,
        expected_sha256=args.expected_runtime_selection_sha256,
        name="large-capacity runtime selection",
    )
    storage, storage_id = _stable_load(
        args.storage_capacity,
        expected_sha256=args.expected_storage_capacity_sha256,
        name="large-capacity storage capacity",
    )
    cofitok_config, cofitok_id = _stable_load(
        args.cofitok_config,
        expected_sha256=args.expected_cofitok_config_sha256,
        name="large-capacity CoFiTok config",
    )
    dense_config, dense_id = _stable_load(
        args.dense_config,
        expected_sha256=args.expected_dense_config_sha256,
        name="large-capacity dense config",
    )
    configs = {"cofitok": cofitok_id, "dense_identity": dense_id}
    validate_terminal_snr_large_capacity_config_pair(
        cofitok_config=cofitok_config,
        dense_config=dense_config,
    )
    run_dirs = {
        "cofitok": str(args.cofitok_run_dir),
        "dense_identity": str(args.dense_run_dir),
    }
    validated_runtime = validate_terminal_snr_large_capacity_runtime_selection(
        runtime,
        execution_checkout=execution_git,
        config_identities=configs,
        run_dirs=run_dirs,
        benchmark_root=str(args.benchmark_root),
    )
    validated_storage = validate_terminal_snr_large_capacity_storage_capacity(
        storage,
        execution_checkout=execution_git,
        output_root=str(args.output_root),
    )

    config = load_config(args.cofitok_config)
    device = torch.device(config.runtime.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("large-capacity live snapshot requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
    torch.backends.cudnn.benchmark = config.runtime.cudnn_benchmark
    torch.set_float32_matmul_precision("high")
    environment = capture_runtime_environment(device, project_root=project_root)
    environment_sha = runtime_environment_sha256(environment)
    if environment_sha != validated_runtime["runtime_environment_sha256"]:
        raise ValueError("large-capacity current runtime differs from selection")
    dataset_spec = FORMAL_GENERATION_DATASETS[DATASET]
    provenance = capture_dataset_provenance(
        config.data,
        train_images=dataset_spec.train_images,
        val_images=dataset_spec.val_images,
    )
    dataset_sha = validate_dataset_provenance(
        provenance, expected_dataset=DATASET
    )["identity_sha256"]
    if dataset_sha != validated_runtime["dataset_identity_sha256"]:
        raise ValueError("large-capacity current dataset differs from selection")

    output_root = reject_symlink_chain(
        args.output_root, name="large-capacity output root"
    )
    execution_lock = reject_symlink_chain(
        args.execution_lock, name="large-capacity execution lock"
    )
    expected_lock = terminal_snr_large_capacity_execution_lock_path(
        output_root.as_posix()
    )
    if execution_lock.as_posix() != expected_lock:
        raise ValueError("large-capacity execution lock is not the exact sibling lock")
    storage_path = reject_symlink_chain(
        args.storage_path, name="large-capacity storage path"
    )
    if not storage_path.is_dir():
        raise FileNotFoundError(
            f"large-capacity storage path is missing: {storage_path}"
        )
    if storage_path.as_posix() != output_root.parent.as_posix():
        raise ValueError("large-capacity storage path differs from output parent")
    normalized_run_dirs = {
        method: reject_symlink_chain(
            run_dirs[method], name=f"large-capacity {method} run directory"
        )
        for method in METHODS
    }
    output_absent = not output_root.exists()
    training_state_absent = output_absent and all(
        not path.exists() for path in normalized_run_dirs.values()
    )
    inventory, compute = _gpu_inventory()
    free_bytes = int(shutil.disk_usage(storage_path).free)
    required_free_bytes = int(
        validated_storage["plan"]["required_free_bytes"]
    )
    report = {
        "schema_version": LIVE_SNAPSHOT_SCHEMA,
        "role": LIVE_SNAPSHOT_ROLE,
        "status": "pass",
        "execution_checkout": execution_git,
        "source_evidence": {
            "runtime_selection": runtime_id,
            "storage_capacity": storage_id,
            "configs": configs,
        },
        "output_root": output_root.as_posix(),
        "execution_lock": execution_lock.as_posix(),
        "training_run_dirs": {
            method: path.as_posix()
            for method, path in normalized_run_dirs.items()
        },
        "gpu_inventory": inventory,
        "gpu_compute_processes": compute,
        "conflicting_processes": _conflicting_processes(output_root),
        "output_root_absent": output_absent,
        "training_state_absent": training_state_absent,
        "execution_lock_free": _lock_free(execution_lock),
        "filesystem": {
            "path": storage_path.as_posix(),
            "free_bytes": free_bytes,
            "required_free_bytes": required_free_bytes,
            "headroom_bytes": free_bytes - required_free_bytes,
        },
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_sha,
        "dataset_provenance": provenance,
        "dataset_identity_sha256": dataset_sha,
        "hostname": os.uname().nodename,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "authorization_boundary": LIVE_SNAPSHOT_BOUNDARY,
    }
    return validate_terminal_snr_large_capacity_live_snapshot(
        report,
        expected_output_root=output_root.as_posix(),
        expected_execution_checkout=execution_git,
        expected_execution_lock=execution_lock.as_posix(),
        expected_runtime_selection_identity=runtime_id,
        expected_storage_capacity_identity=storage_id,
        expected_config_identities=configs,
    )


def main() -> None:
    args = parse_args()
    output = reject_symlink_chain(
        args.output, name="large-capacity live snapshot output"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite live snapshot: {output}")
    report = build_live_snapshot(args)
    write_json_report(output, report)
    persisted = read_object(output, name="large-capacity live snapshot")
    validate_terminal_snr_large_capacity_live_snapshot(
        persisted,
        expected_output_root=report["output_root"],
        expected_execution_checkout=report["execution_checkout"],
        expected_execution_lock=report["execution_lock"],
        expected_runtime_selection_identity=report["source_evidence"][
            "runtime_selection"
        ],
        expected_storage_capacity_identity=report["source_evidence"][
            "storage_capacity"
        ],
        expected_config_identities=report["source_evidence"]["configs"],
    )
    print(json.dumps({
        "status": "pass",
        "live_snapshot_sha256": file_sha256(output),
        "output_root_absent": True,
        "training_state_absent": True,
        "execution_lock_free": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
