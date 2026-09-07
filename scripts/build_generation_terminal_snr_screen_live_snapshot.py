"""Capture the exact idle runtime for the terminal-SNR screen launch."""

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
from cofitok.generation.terminal_snr_screen import DATASET
from cofitok.generation.terminal_snr_screen_execution import (
    LIVE_SNAPSHOT_ROLE,
    LIVE_SNAPSHOT_SCHEMA,
    terminal_snr_screen_execution_lock_path,
    validate_terminal_snr_screen_live_snapshot,
    validate_terminal_snr_screen_runtime_selection,
)
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build terminal-SNR screen live prelaunch snapshot."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--expected-runtime-selection-sha256", required=True)
    parser.add_argument("--endpoint-cofitok-config", type=Path, required=True)
    parser.add_argument("--expected-endpoint-cofitok-config-sha256", required=True)
    parser.add_argument("--endpoint-dense-config", type=Path, required=True)
    parser.add_argument("--expected-endpoint-dense-config-sha256", required=True)
    parser.add_argument("--endpoint-cofitok-run-dir", required=True)
    parser.add_argument("--endpoint-dense-run-dir", required=True)
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
            raise ValueError("terminal-SNR GPU inventory is malformed")
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
            raise ValueError("terminal-SNR compute inventory is malformed")
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
        "run_generation_terminal_snr_screen.py",
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
        raise RuntimeError("terminal-SNR lock probing requires POSIX fcntl")
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
        args.project_root, name="terminal-SNR execution checkout"
    ).resolve()
    execution_git = checkout_identity(project_root)
    if execution_git["tracked_dirty"] is not False:
        raise ValueError("terminal-SNR execution checkout has tracked changes")
    runtime, runtime_id = _stable_load(
        args.runtime_selection,
        expected_sha256=args.expected_runtime_selection_sha256,
        name="terminal-SNR runtime selection",
    )
    _, cofitok_id = _stable_load(
        args.endpoint_cofitok_config,
        expected_sha256=args.expected_endpoint_cofitok_config_sha256,
        name="endpoint CoFiTok config",
    )
    _, dense_id = _stable_load(
        args.endpoint_dense_config,
        expected_sha256=args.expected_endpoint_dense_config_sha256,
        name="endpoint dense config",
    )
    validated_runtime = validate_terminal_snr_screen_runtime_selection(
        runtime,
        execution_git=execution_git,
        endpoint_config_identities={
            "cofitok": cofitok_id,
            "dense_identity": dense_id,
        },
        endpoint_run_dirs=[
            str(args.endpoint_cofitok_run_dir),
            str(args.endpoint_dense_run_dir),
        ],
        benchmark_root=str(args.benchmark_root),
    )
    cofitok = load_config(args.endpoint_cofitok_config)
    dense = load_config(args.endpoint_dense_config)
    if (
        cofitok.data != dense.data
        or cofitok.data.dataset != DATASET
        or cofitok.model.base_channels != 128
        or dense.model.base_channels != 128
        or cofitok.diffusion.cosine_endpoint_fraction != 0.975
        or dense.diffusion.cosine_endpoint_fraction != 0.975
    ):
        raise ValueError("terminal-SNR live configs are not the endpoint matched pair")
    device = torch.device(cofitok.runtime.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("terminal-SNR live snapshot requires CUDA")
    torch.backends.cuda.matmul.allow_tf32 = cofitok.runtime.allow_tf32
    torch.backends.cudnn.allow_tf32 = cofitok.runtime.allow_tf32
    torch.backends.cudnn.benchmark = cofitok.runtime.cudnn_benchmark
    torch.set_float32_matmul_precision("high")
    environment = capture_runtime_environment(device, project_root=project_root)
    environment_sha = runtime_environment_sha256(environment)
    if environment_sha != validated_runtime["runtime_environment_sha256"]:
        raise ValueError("terminal-SNR current runtime differs from selection")
    dataset_spec = FORMAL_GENERATION_DATASETS[DATASET]
    provenance = capture_dataset_provenance(
        cofitok.data,
        train_images=dataset_spec.train_images,
        val_images=dataset_spec.val_images,
    )
    dataset_sha = validate_dataset_provenance(
        provenance, expected_dataset=DATASET
    )["identity_sha256"]
    if dataset_sha != validated_runtime["dataset_identity_sha256"]:
        raise ValueError("terminal-SNR current dataset differs from selection")
    output_root = reject_symlink_chain(
        args.output_root, name="terminal-SNR output root"
    )
    expected_lock = terminal_snr_screen_execution_lock_path(output_root.as_posix())
    execution_lock = reject_symlink_chain(
        args.execution_lock, name="terminal-SNR execution lock"
    )
    if execution_lock.as_posix() != expected_lock:
        raise ValueError("terminal-SNR execution lock is not the exact sibling lock")
    storage_path = reject_symlink_chain(
        args.storage_path, name="terminal-SNR storage path"
    )
    if not storage_path.is_dir():
        raise FileNotFoundError(f"terminal-SNR storage path is missing: {storage_path}")
    inventory, compute = _gpu_inventory()
    report = {
        "schema_version": LIVE_SNAPSHOT_SCHEMA,
        "role": LIVE_SNAPSHOT_ROLE,
        "status": "pass",
        "execution_checkout": execution_git,
        "source_evidence": {
            "runtime_selection": runtime_id,
            "endpoint_configs": {
                "cofitok": cofitok_id,
                "dense_identity": dense_id,
            },
        },
        "output_root": output_root.as_posix(),
        "execution_lock": execution_lock.as_posix(),
        "gpu_inventory": inventory,
        "gpu_compute_processes": compute,
        "conflicting_processes": _conflicting_processes(output_root),
        "output_root_absent": not output_root.exists(),
        "execution_lock_free": _lock_free(execution_lock),
        "free_bytes": int(shutil.disk_usage(storage_path).free),
        "runtime_environment": environment,
        "runtime_environment_sha256": environment_sha,
        "dataset_provenance": provenance,
        "dataset_identity_sha256": dataset_sha,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }
    return validate_terminal_snr_screen_live_snapshot(
        report,
        expected_output_root=output_root.as_posix(),
        expected_execution_checkout=execution_git,
        expected_execution_lock=execution_lock.as_posix(),
    )


def main() -> None:
    args = parse_args()
    output = reject_symlink_chain(
        args.output, name="terminal-SNR live snapshot output"
    )
    if output.exists():
        raise FileExistsError(f"refusing to overwrite live snapshot: {output}")
    report = build_live_snapshot(args)
    write_json_report(output, report)
    persisted = read_object(output, name="terminal-SNR live snapshot")
    validate_terminal_snr_screen_live_snapshot(
        persisted,
        expected_output_root=report["output_root"],
        expected_execution_checkout=report["execution_checkout"],
        expected_execution_lock=report["execution_lock"],
    )
    print(json.dumps({
        "status": "pass",
        "live_snapshot_sha256": file_sha256(output),
        "output_root_absent": True,
        "execution_lock_free": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
