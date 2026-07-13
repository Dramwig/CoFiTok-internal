from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch


RUNTIME_ENVIRONMENT_SCHEMA_VERSION = 1
TRACKED_DISTRIBUTIONS = ("numpy", "pillow", "torch", "torchvision", "tqdm")
TRACKED_ENVIRONMENT_VARIABLES = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_VISIBLE_DEVICES",
    "PYTHONHASHSEED",
    "PYTORCH_ALLOC_CONF",
)


def _distribution_versions() -> dict[str, str | None]:
    versions = {}
    for name in TRACKED_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _file_identity(path: Path, *, root: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _nvidia_driver_version() -> str | None:
    if not torch.cuda.is_available():
        return None
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(versions) or None


def capture_runtime_environment(
    device: torch.device,
    *,
    project_root: str | Path,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    device_payload: dict[str, Any] = {"type": device.type}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_payload.update(
            {
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "total_memory_bytes": properties.total_memory,
                "multi_processor_count": properties.multi_processor_count,
            }
        )
    return {
        "schema_version": RUNTIME_ENVIRONMENT_SCHEMA_VERSION,
        "python": {
            "implementation": platform.python_implementation(),
            "version": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "packages": _distribution_versions(),
        "torch": {
            "version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "nvidia_driver_version": _nvidia_driver_version(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "deterministic_algorithms": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "num_threads": torch.get_num_threads(),
            "num_interop_threads": torch.get_num_interop_threads(),
        },
        "device": device_payload,
        "environment_variables": {
            name: os.environ.get(name) for name in TRACKED_ENVIRONMENT_VARIABLES
        },
        "project_files": {
            name: _file_identity(root / name, root=root)
            for name in ("pyproject.toml", "uv.lock")
        },
    }


def runtime_environment_sha256(environment: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        environment,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(serialized).hexdigest()


def runtime_environment_mismatch_paths(
    expected: Any,
    actual: Any,
    *,
    path: str = "runtime_environment",
) -> list[str]:
    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        mismatches = []
        for key in sorted(set(expected) | set(actual), key=str):
            child = f"{path}.{key}"
            if key not in expected or key not in actual:
                mismatches.append(child)
                continue
            mismatches.extend(
                runtime_environment_mismatch_paths(
                    expected[key], actual[key], path=child
                )
            )
        return mismatches
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [path]
        mismatches = []
        for index, (expected_item, actual_item) in enumerate(
            zip(expected, actual, strict=True)
        ):
            mismatches.extend(
                runtime_environment_mismatch_paths(
                    expected_item, actual_item, path=f"{path}[{index}]"
                )
            )
        return mismatches
    return [] if expected == actual else [path]
