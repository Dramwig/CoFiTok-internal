"""Build a non-authorizing, source-bound exposure/capacity gate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - exercised by Windows callers
    fcntl = None  # type: ignore[assignment]

from cofitok.generation.exposure_capacity import (
    validate_preparation,
    validate_source_checkpoint_bindings,
)
from cofitok.generation.exposure_capacity_gate import (
    ARM_IDS,
    SOURCE_BRANCH,
    SOURCE_REVISION,
    SOURCE_STEP,
    SOURCE_TREE,
    build_execution_gate,
)
from cofitok.path_security import reject_symlink_chain
from cofitok.training.checkpointing import resolve_latest_checkpoint, verify_training_checkpoint
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = reject_symlink_chain(
    Path(__file__),
    name="exposure/capacity gate script",
).parents[1]
CONFLICTING_BASENAMES = (
    "train_generation.py",
    "generate_samples.py",
    "evaluate_generation_checkpoint.py",
    "evaluate_generation_metrics.py",
    "evaluate_generation_class_fidelity.py",
    "generation_exposure_capacity",
    "generation_stability_full_data_quality_bridge_100k_execute.sh",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound exposure/capacity qualification gate. The "
            "result is deliberately non-authorizing and cannot launch work."
        )
    )
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--expected-preparation-sha256", required=True)
    parser.add_argument("--arm", choices=ARM_IDS, required=True)
    parser.add_argument("--source-project-root", type=Path, required=True)
    parser.add_argument("--runtime-environment-sha256", required=True)
    parser.add_argument("--dataset-identity-sha256", required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--execution-lock", type=Path, required=True)
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        help="100K run directory for the exposure-continuation resume source",
    )
    parser.add_argument(
        "--source-checkpoint",
        type=Path,
        help="100K checkpoint payload for the exposure-continuation resume source",
    )
    parser.add_argument(
        "--source-integrity-manifest",
        type=Path,
        help="100K checkpoint integrity sidecar for the exposure-continuation resume source",
    )
    parser.add_argument(
        "--source-latest",
        type=Path,
        help="100K latest.json pointer for the exposure-continuation resume source",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="gate source")
    if not source.is_file():
        raise FileNotFoundError(f"gate source is missing: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {source}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="gate source")
    if not source.is_file():
        raise FileNotFoundError(f"source is missing: {source}")
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _source_checkpoint_binding(
    run_dir: Path,
    checkpoint: Path,
    integrity_manifest: Path,
    latest: Path,
) -> dict[str, Any]:
    run_dir = reject_symlink_chain(run_dir, name="source checkpoint run directory")
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError("source checkpoint run directory must be a real directory")
    checkpoint = reject_symlink_chain(checkpoint, name="source checkpoint")
    integrity_manifest = reject_symlink_chain(
        integrity_manifest,
        name="source checkpoint integrity manifest",
    )
    latest = reject_symlink_chain(latest, name="source latest pointer")
    resolved_checkpoint = resolve_latest_checkpoint(run_dir)
    if checkpoint != resolved_checkpoint:
        raise ValueError("source checkpoint does not match latest.json")
    expected_integrity = checkpoint.with_name(f"{checkpoint.name}.integrity.json")
    if integrity_manifest != expected_integrity:
        raise ValueError("source checkpoint sidecar does not match checkpoint")
    expected_latest = run_dir / "latest.json"
    if latest != expected_latest:
        raise ValueError("source latest pointer does not match run directory")
    integrity = verify_training_checkpoint(resolved_checkpoint)
    if int(integrity.get("step", -1)) != SOURCE_STEP:
        raise ValueError("source checkpoint must be the completed 100K step")
    return {
        "run_dir": str(run_dir),
        "step": SOURCE_STEP,
        "checkpoint": _identity(resolved_checkpoint),
        "integrity_manifest": _identity(expected_integrity),
        "latest": _identity(expected_latest),
    }


def _verify_preparation_sources(preparation: dict[str, Any]) -> None:
    sources = preparation.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("preparation source descriptors are missing")
    for name, descriptor in sources.items():
        expected = dict(descriptor)
        actual = _identity(Path(str(expected.get("path", ""))))
        if actual != expected:
            raise ValueError(f"preparation source {name} changed")


def _verify_preparation_training_sources(preparation: dict[str, Any]) -> None:
    sources = preparation.get("sources")
    if not isinstance(sources, dict):
        raise ValueError("preparation source descriptors are missing")
    reports: dict[str, dict[str, Any]] = {}
    for method, source_name in (
        ("cofitok", "cofitok_training_report"),
        ("dense_identity", "dense_training_report"),
    ):
        descriptor = sources.get(source_name)
        if not isinstance(descriptor, dict):
            raise ValueError(f"preparation source {source_name} is malformed")
        reports[method] = _read(Path(str(descriptor.get("path", ""))))
    validate_source_checkpoint_bindings(
        preparation,
        cofitok_training_report=reports["cofitok"],
        dense_training_report=reports["dense_identity"],
    )


def _git(project_root: Path) -> dict[str, Any]:
    provenance = git_provenance(project_root)
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {**provenance, "tree": tree}


def _source_checkout(project_root: Path) -> dict[str, Any]:
    identity = _git(project_root)
    expected = {
        "revision": SOURCE_REVISION,
        "tree": SOURCE_TREE,
        "branch": SOURCE_BRANCH,
        "tracked_dirty": False,
    }
    if identity != expected:
        raise ValueError("source project checkout does not match locked bridge")
    return expected


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
            raise ValueError("nvidia-smi compute inventory is malformed")
        compute.append(
            {
                "pid": int(fields[0]),
                "process_name": fields[1],
                "used_gpu_memory_mib": int(fields[2]),
            }
        )
    return inventory, compute


def _conflicting_processes() -> list[dict[str, Any]]:
    def matches_conflicting_command(command: str) -> bool:
        try:
            tokens = shlex.split(command, posix=True)
        except ValueError:
            # A process can expose an incomplete shell command while it is
            # starting; conservative tokenization still avoids substring
            # matches against unrelated parent commands.
            tokens = command.split()
        for raw_token in tokens:
            token = raw_token.split("=", 1)[-1].strip("'\"")
            if not token:
                continue
            basename = PurePosixPath(token).name
            stem = PurePosixPath(basename).stem
            module_leaf = token.rsplit(".", 1)[-1]
            if (
                basename in CONFLICTING_BASENAMES
                or stem in CONFLICTING_BASENAMES
                or module_leaf in CONFLICTING_BASENAMES
            ):
                return True
        return False

    result = subprocess.run(
        ["ps", "-eo", "pid=,ppid=,lstart=,args="],
        check=True,
        capture_output=True,
        text=True,
    )
    conflicts: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped or not matches_conflicting_command(stripped):
            continue
        fields = stripped.split(maxsplit=7)
        if not fields:
            continue
        try:
            pid = int(fields[0])
        except ValueError:
            continue
        if pid == os.getpid():
            continue
        conflicts.append({"pid": pid, "command": stripped})
    return conflicts


def _lock_is_free(path: Path) -> bool:
    """Probe an optional lock without mutating the non-authorizing gate state."""
    # A preparation/gate snapshot must not create a lock or its parent.  The
    # eventual authorized launcher re-probes and acquires its own lock.
    path = reject_symlink_chain(path, name="execution lock")
    if path.is_symlink():
        return False
    if not path.exists():
        return True
    if not path.is_file():
        return False
    if fcntl is None:
        raise RuntimeError("execution lock probing requires POSIX fcntl")
    with path.open("r+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return True


def main() -> None:
    args = _parse_args()
    output = reject_symlink_chain(args.output, name="gate output")
    if output.exists():
        raise FileExistsError(f"gate output already exists: {output}")
    preparation_path = reject_symlink_chain(args.preparation, name="preparation report")
    if not preparation_path.is_file():
        raise FileNotFoundError(f"preparation report is missing: {preparation_path}")
    if file_sha256(preparation_path) != args.expected_preparation_sha256:
        raise ValueError("preparation SHA256 differs")
    preparation = _read(preparation_path)
    validate_preparation(preparation)
    _verify_preparation_sources(preparation)
    _verify_preparation_training_sources(preparation)
    source_checkout = _source_checkout(
        reject_symlink_chain(args.source_project_root, name="source project checkout")
    )
    builder_git = _git(PROJECT_ROOT)
    gpu_inventory, gpu_compute_processes = _gpu_inventory()
    selected_root = preparation["candidate_arms"][args.arm]["output_root"]
    output_root = reject_symlink_chain(selected_root, name="candidate output root")
    output_root_absent = not output_root.exists()
    source_paths = {
        "run_dir": args.source_run_dir,
        "checkpoint": args.source_checkpoint,
        "integrity_manifest": args.source_integrity_manifest,
        "latest": args.source_latest,
    }
    supplied_source_paths = [path for path in source_paths.values() if path is not None]
    if args.arm == "exposure_continuation":
        if len(supplied_source_paths) != len(source_paths):
            raise ValueError(
                "exposure continuation requires source run dir, checkpoint, sidecar, and latest"
            )
        source_checkpoint_binding = _source_checkpoint_binding(
            args.source_run_dir,
            args.source_checkpoint,
            args.source_integrity_manifest,
            args.source_latest,
        )
    elif supplied_source_paths:
        raise ValueError("capacity qualification must not provide a resume source")
    else:
        source_checkpoint_binding = None
    gate = build_execution_gate(
        preparation=preparation,
        preparation_identity=_identity(preparation_path),
        arm_id=args.arm,
        source_checkout=source_checkout,
        gate_builder_git=builder_git,
        runtime_environment_sha256=args.runtime_environment_sha256,
        dataset_identity_sha256=args.dataset_identity_sha256,
        gpu_inventory=gpu_inventory,
        gpu_compute_processes=gpu_compute_processes,
        conflicting_processes=_conflicting_processes(),
        output_root_absent=output_root_absent,
        execution_lock_free=_lock_is_free(args.execution_lock),
        free_bytes=shutil.disk_usage(
            reject_symlink_chain(args.storage_path, name="storage path")
        ).free,
        source_checkpoint_binding=source_checkpoint_binding,
    )
    write_json_report(output, gate)
    print(file_sha256(output))


if __name__ == "__main__":
    main()
