from __future__ import annotations

import argparse
import fcntl
import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from cofitok.data import capture_dataset_provenance
from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation.min_snr_pilot import build_execution_gate
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFLICTING_BASENAMES = (
    "train_generation.py",
    "generate_samples.py",
    "evaluate_generation_checkpoint.py",
    "evaluate_generation_metrics.py",
    "evaluate_generation_class_fidelity.py",
    "generation_min_snr_matched_50k_pilot_v1.sh",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the source-bound matched Min-SNR pilot execution gate."
    )
    parser.add_argument("--preparation", required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--storage-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--execution-lock", required=True)
    parser.add_argument("--user-instruction", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return value


def _identity(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _builder_git() -> dict[str, Any]:
    return {**git_provenance(PROJECT_ROOT), "tree": _git("rev-parse", "HEAD^{tree}")}


def _gpu_inventory() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    inventory = []
    for line in gpu.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 5)]
        if len(fields) != 6:
            raise ValueError("nvidia-smi GPU inventory is malformed")
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
    compute = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    processes = []
    for line in compute.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        processes.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": int(fields[2]),
            }
        )
    return inventory, processes


def _conflicting_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,lstart=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    conflicts = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or not any(name in stripped for name in CONFLICTING_BASENAMES):
            continue
        fields = stripped.split(maxsplit=7)
        if not fields or int(fields[0]) == os.getpid():
            continue
        conflicts.append({"pid": int(fields[0]), "command": stripped})
    return conflicts


def _lock_is_free(path: str | Path) -> bool:
    lock_path = Path(path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return True


def main() -> None:
    args = parse_args()
    preparation_identity = _identity(args.preparation)
    if preparation_identity["sha256"] != args.expected_preparation_sha256:
        raise ValueError("Min-SNR pilot preparation SHA256 differs")
    preparation = _read(args.preparation)
    if str(preparation.get("output_root", "")) != str(Path(args.output_root)):
        raise ValueError("execution output root differs from preparation")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Min-SNR pilot execution gate requires the target CUDA GPU")
    environment = capture_runtime_environment(device, project_root=PROJECT_ROOT)
    dataset = capture_dataset_provenance(
        SimpleNamespace(dataset="imagenet_256", root=args.dataset_root),
        train_images=1_281_167,
        val_images=50_000,
    )
    if dataset.get("status") != "pass":
        raise ValueError("formal ImageNet-256 provenance failed")
    inventory, compute = _gpu_inventory()
    output_root = Path(args.output_root)
    report = build_execution_gate(
        preparation=preparation,
        preparation_identity=preparation_identity,
        gate_builder_git=_builder_git(),
        runtime_environment_sha256=runtime_environment_sha256(environment),
        dataset_identity_sha256=str(dataset["identity_sha256"]),
        gpu_inventory=inventory,
        gpu_compute_processes=compute,
        conflicting_processes=_conflicting_processes(),
        output_root_absent=not output_root.exists(),
        execution_lock_free=_lock_is_free(args.execution_lock),
        free_bytes=shutil.disk_usage(args.storage_path).free,
        authorization_record={
            "scope": "matched_min_snr_50k_pilot_only",
            "approved_by": "user",
            "instruction": args.user_instruction,
            "direct_execution_without_repeated_prompt": True,
            "full_300k_launch_allowed": False,
        },
    )
    write_json_report(args.output, report)
    print(file_sha256(args.output))


if __name__ == "__main__":
    main()
