"""Build the separately authorized exposure-continuation execution record."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch

from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation.exposure_capacity_authorization import (
    _physical_source_bindings,
    _preparation_sources,
    build_authorization,
    checkout_identity,
    identity,
    read_object,
    validate_source_checkout,
)
from cofitok.generation.exposure_capacity_gate import validate_execution_gate
from cofitok.inference_replay import reject_symlink_chain
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training.checkpointing import verify_training_checkpoint


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MIN_FREE_BYTES = 120 * 1024**3


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a source-bound exposure authorization.")
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--standing-authorization", type=Path, required=True)
    parser.add_argument("--expected-standing-authorization-sha256", required=True)
    parser.add_argument(
        "--stage-authorization",
        type=Path,
        required=True,
        help="User-created approval for this exact 100K-to-110K stage.",
    )
    parser.add_argument("--expected-stage-authorization-sha256", required=True)
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--execution-project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--cofitok-source-run", type=Path, required=True)
    parser.add_argument("--dense-source-run", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path, name: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{name} is not a JSON object")
    return value


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
    inventory = []
    for line in inventory_result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 5)]
        if len(fields) != 6:
            raise ValueError("GPU inventory is malformed")
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
    compute = []
    for line in compute_result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 2)]
        if len(fields) != 3:
            raise ValueError("GPU compute inventory is malformed")
        compute.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": int(fields[2]),
            }
        )
    return inventory, compute


def _conflicting_processes() -> list[dict[str, Any]]:
    names = {
        "train_generation.py",
        "generation_exposure_capacity_continuation_110k_execute.sh",
        "run_generation_exposure_capacity_continuation.py",
    }
    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,args="], capture_output=True, text=True, check=True
    )
    conflicts = []
    current = os.getpid()
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
        if any(name in command for name in names):
            conflicts.append({"pid": pid, "command": stripped})
    return conflicts


def _lock_free(path: Path) -> bool:
    import fcntl

    if not path.parent.exists():
        return True
    if not path.parent.is_dir():
        return False
    if not path.exists():
        return True
    if not path.is_file() or path.is_symlink():
        return False
    with path.open("r+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return True


def main() -> None:
    args = _args()
    if args.output.exists():
        raise FileExistsError(f"authorization output already exists: {args.output}")
    if file_sha256(args.preparation) != args.expected_preparation_sha256:
        raise ValueError("preparation SHA256 differs")
    if file_sha256(args.gate) != args.expected_gate_sha256:
        raise ValueError("candidate gate SHA256 differs")
    if file_sha256(args.standing_authorization) != args.expected_standing_authorization_sha256:
        raise ValueError("standing authorization SHA256 differs")
    if file_sha256(args.stage_authorization) != args.expected_stage_authorization_sha256:
        raise ValueError("stage authorization SHA256 differs")
    preparation = _read(args.preparation, "preparation")
    gate = _read(args.gate, "candidate gate")
    standing = _read(args.standing_authorization, "standing authorization")
    stage_authorization = _read(args.stage_authorization, "stage authorization")
    if standing.get("status") != "active":
        raise ValueError("standing authorization is not active")
    prep_id = identity(args.preparation)
    gate_id = identity(args.gate)
    standing_id = identity(args.standing_authorization)
    stage_authorization_id = identity(args.stage_authorization)
    validate_execution_gate(gate, preparation=preparation, preparation_identity=prep_id)
    execution_checkout = checkout_identity(args.execution_project_root)
    if execution_checkout["tracked_dirty"] is not False:
        raise ValueError("execution checkout has tracked changes")
    config_identities = {
        "cofitok": identity(args.cofitok_config),
        "dense_identity": identity(args.dense_config),
    }
    reports = _preparation_sources(preparation)
    del reports
    source_bindings = _physical_source_bindings(preparation)
    source_checkout = validate_source_checkout(
        checkout_identity(args.source_project_root)
    )
    if source_bindings["cofitok"] != gate["source_checkpoint"]:
        raise ValueError(
            "CoFiTok candidate gate is not bound to the complete physical source binding"
        )
    if source_bindings["dense_identity"]["run_dir"] != args.dense_source_run.resolve().as_posix():
        raise ValueError("dense physical source run directory differs from the requested source")
    if source_bindings["cofitok"]["run_dir"] != args.cofitok_source_run.resolve().as_posix():
        raise ValueError("CoFiTok physical source run directory differs from the requested source")
    inventory, compute = _gpu_inventory()
    if len(inventory) != 1 or compute:
        raise ValueError("target GPU is not uniquely idle")
    if any(
        row["memory_used_mib"] > 16 or row["utilization_percent"] > 5 for row in inventory
    ):
        raise ValueError("target GPU is not idle")
    output_root = reject_symlink_chain(args.output_root, name="exposure output root")
    lock = reject_symlink_chain(args.execution_lock, name="exposure execution lock")
    live_environment = capture_runtime_environment(
        torch.device("cuda"), project_root=args.execution_project_root
    )
    live = {
        "runtime_environment_sha256": runtime_environment_sha256(live_environment),
        "dataset_identity_sha256": str(
            _read(args.cofitok_source_run / "latest.json", "CoFiTok latest")[
                "dataset_identity_sha256"
            ]
        ),
        "gpu_inventory": inventory,
        "gpu_compute_processes": [],
        "conflicting_processes": _conflicting_processes(),
        "output_root": output_root.resolve().as_posix(),
        "output_root_absent": not output_root.exists(),
        "execution_lock_free": _lock_free(lock),
        "free_bytes": int(shutil.disk_usage(args.output_root.parent).free),
    }
    if live["conflicting_processes"] or not live["output_root_absent"] or not live["execution_lock_free"]:
        raise ValueError("live prelaunch state is not clean for a new exposure run")
    if live["free_bytes"] < MIN_FREE_BYTES:
        raise ValueError("insufficient free storage for exposure continuation")
    authorization = build_authorization(
        gate=gate,
        preparation=preparation,
        preparation_identity=prep_id,
        gate_identity=gate_id,
        standing_identity=standing_id,
        execution_checkout=execution_checkout,
        config_identities=config_identities,
        source_checkpoints=source_bindings,
        live_prelaunch=live,
        source_checkout=source_checkout,
        stage_authorization=stage_authorization,
        stage_authorization_identity=stage_authorization_id,
    )
    write_json_report(args.output, authorization)
    print(file_sha256(args.output))


if __name__ == "__main__":
    main()
