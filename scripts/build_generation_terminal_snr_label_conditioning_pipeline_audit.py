"""Build or validate the terminal-SNR label-conditioning pipeline audit.

This standard-library-only tool physically replays the immutable four-arm
terminal-SNR screen, its label metadata, the generated-sample class protocol,
and the exact execution-revision code that transports class indices.  It is a
diagnostic audit only: it never imports the training package, loads a model,
starts a process, uses a GPU, changes controller state, or grants a later-stage
authorization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import stat
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any


SCHEMA = "cofitok_generation_terminal_snr_label_conditioning_pipeline_audit_v1"
ROLE = "source_bound_terminal_snr_label_conditioning_pipeline_audit"
VALIDATION_SCHEMA = (
    "cofitok_generation_terminal_snr_label_conditioning_pipeline_audit_validation_v1"
)
VALIDATION_ROLE = (
    "content_addressed_terminal_snr_label_conditioning_pipeline_audit_validation"
)

EXECUTION_REVISION = "89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430"
EXECUTION_TREE = "46efd20cff489bccd799bb13c4155a0cc79e7649"
EXECUTION_BRANCH = "terminal-snr-execution-89bcd9a"
NUM_CLASSES = 1000
TRAIN_ROWS = 1_281_167
VALIDATION_ROWS = 50_000
VALIDATION_ROWS_PER_CLASS = 50
SAMPLES_PER_ARM = 1000
EFFECTIVE_BATCH_SIZE = 64
COMPLETED_STEPS = 10_000
EVALUATOR_CATEGORIES_SHA256 = (
    "62fff941ecff3f19de9128c6ca9c2097807c6b7bafc5552d7589a219431221ed"
)

ARM_SPECS = {
    "control_cofitok": {
        "condition": "control",
        "method": "cofitok",
        "endpoint_fraction": 1.0,
        "prefix_budget": 8,
        "config": (
            "configs/generation/"
            "imagenet256_capacity_reference_rgbtail3_rollout_x0_u2_ema_teacher_"
            "k8_100k.json"
        ),
    },
    "control_dense_identity": {
        "condition": "control",
        "method": "dense_identity",
        "endpoint_fraction": 1.0,
        "prefix_budget": 1,
        "config": (
            "configs/generation/"
            "imagenet256_capacity_reference_rollout_x0_u2_ema_teacher_dense_100k.json"
        ),
    },
    "endpoint0975_cofitok": {
        "condition": "endpoint0975",
        "method": "cofitok",
        "endpoint_fraction": 0.975,
        "prefix_budget": 8,
        "config": (
            "configs/generation/"
            "imagenet256_terminal_snr_endpoint0975_rgbtail3_rollout_x0_u2_ema_"
            "teacher_k8_100k.json"
        ),
    },
    "endpoint0975_dense_identity": {
        "condition": "endpoint0975",
        "method": "dense_identity",
        "endpoint_fraction": 0.975,
        "prefix_budget": 1,
        "config": (
            "configs/generation/"
            "imagenet256_terminal_snr_endpoint0975_rollout_x0_u2_ema_teacher_"
            "dense_100k.json"
        ),
    },
}

CODE_FILES = {
    "code_data_registry": "src/cofitok/data/registry.py",
    "code_train_generation": "scripts/train_generation.py",
    "code_generate_samples": "scripts/generate_samples.py",
    "code_generation_session": "src/cofitok/generation/session.py",
    "code_diffusion_sampling": "src/cofitok/diffusion/sampling.py",
    "code_scalable_unet": "src/cofitok/models/scalable_unet.py",
    "code_cofitok_model": "src/cofitok/models/cofitok.py",
    "code_class_fidelity_evaluator": "scripts/evaluate_generation_class_fidelity.py",
    "code_class_fidelity_contract": "src/cofitok/generation_class_fidelity.py",
}

LOCKED_SHA256 = {
    "preparation": "af417ae383ff289a0ad6c91a347046567fb80e7c211b27db754fc3f3fdb75144",
    "execution_authorization": "6f989930ba8eb6150917717a3d0e6fb342a91ebf0a50c82360843aba4e8a4e36",
    "launch_receipt": "24ff4acc496f8ea24698da9c5dfe5fa5c56b6e681b9b8d4db8d84b32ec28c26d",
    "runtime_selection": "cee1bf37b5e8972ec77e498a0b0b4756bced401200d2e2511dea2e2f4d8941cd",
    "live_snapshot": "c3598bcbc87cb0c8091eed0f6c3e010869009d4819ab709f5e773766efd38f8f",
    "storage_capacity": "371f1171dbe16600e30ad5485a6505968447cb2e1d9a1b6075179e17a85794e2",
    "controller_status": "dfba8a85c6aa0b530e27874c5a0187f18ace3ac2624ab0cfedd92b8f4dd45e14",
    "controller_log": "2167f99c0c899c7e79896dc2924849e9a5bd8638e9c985777f13f97edaca084d",
    "terminal_result": "b740c8b21aabf640c156aea076058c73350d26c110e2cc967576a26492a52dea",
    "terminal_result_validation": "3be043d852bd61d6e79005dc9d14f77113f1250bef967ca4e2a8ac74bc715f95",
    "causal_discriminator": "6b0887dec10d6908550bee8e90c3a6e242dbddb3cb62c97935eac841dd262bc0",
    "causal_discriminator_validation": "0043680c6109de3adffae743517ef87b4efd81f3e87f6d540a144aa804354292",
    "execution_label_mapping": "3e8c34f680433258998fdd30cf4c15442e2ce3cbd35c27507becc0f690b19b91",
    "dataset_label_mapping": "3e8c34f680433258998fdd30cf4c15442e2ce3cbd35c27507becc0f690b19b91",
    "dataset_source_mapping": "156a88df0f13a82c66354db1417625baafdecdf84d87d417873acaaf023a4fa9",
    "dataset_manifest": "9a2eec642f0d56162bffaafed84a41267f22abfc9feff4cf41fed9f6881173f0",
    "torchvision_categories": "5eed2c03c74fab0fe073e06993a2fb5f539900cd74b65d41b33d6c955cf44f9e",
    "timm_synsets": "70002b0ff5de60a3a17a82dbfcff291931f96225ddf941ad2e182fc39e183d15",
    "real_calibration": "a4c68b9b1c4dffda89f622887455f46d554fdb0c306d467fecb51d8f5abe6132",
}

AUTHORIZATION_BOUNDARY = {
    "audit_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "gpu_execution_allowed": False,
    "process_signals_allowed": False,
    "frozen_confirmation_preparation_allowed": False,
    "frozen_confirmation_launch_allowed": False,
    "large_capacity_readiness_preparation_allowed": False,
    "full_training_preparation_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "paper_integration_allowed": False,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return list(value)


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _hex(value: Any, *, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not a lowercase {length}-character hex digest")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _assert_finite_tree(value: Any, name: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, f"{name}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, f"{name}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _json_loads(data: bytes, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains non-finite JSON constant {value}")

    try:
        value = json.loads(data, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    result = _object(value, name)
    _assert_finite_tree(result, name)
    return result


def _reject_symlink_chain(path: Path, *, allow_missing_leaf: bool = False) -> None:
    absolute = path if path.is_absolute() else path.absolute()
    parts = absolute.parts
    if not parts:
        raise ValueError(f"invalid path: {path}")
    current = Path(parts[0])
    for index, part in enumerate(parts[1:], start=1):
        current = current / part
        leaf = index == len(parts) - 1
        try:
            mode = os.lstat(current).st_mode
        except FileNotFoundError:
            if allow_missing_leaf and leaf:
                return
            raise
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink path component is forbidden: {current}")


def _before_read(path: Path) -> os.stat_result:
    if not path.is_absolute():
        raise ValueError(f"source path must be absolute: {path}")
    _reject_symlink_chain(path)
    observed = path.stat()
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"source is not a regular file: {path}")
    return observed


def _unchanged(path: Path, before: os.stat_result, bytes_read: int) -> None:
    after = path.stat()
    if (
        bytes_read != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
    ):
        raise RuntimeError(f"source changed while being read: {path}")


def _stable_hash(path: Path) -> dict[str, Any]:
    before = _before_read(path)
    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
            bytes_read += len(block)
    _unchanged(path, before, bytes_read)
    return {"path": str(path), "bytes": bytes_read, "sha256": digest.hexdigest()}


def _stable_read(path: Path) -> tuple[bytes, dict[str, Any]]:
    before = _before_read(path)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    bytes_read = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            chunks.append(block)
            digest.update(block)
            bytes_read += len(block)
    _unchanged(path, before, bytes_read)
    return b"".join(chunks), {
        "path": str(path),
        "bytes": bytes_read,
        "sha256": digest.hexdigest(),
    }


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    if not isinstance(row["path"], str) or not Path(row["path"]).is_absolute():
        raise ValueError(f"{name} path must be absolute")
    if _integer(row["bytes"], f"{name} bytes") < 1:
        raise ValueError(f"{name} bytes must be positive")
    _hex(row["sha256"], length=64, name=f"{name} SHA256")
    return row


def _git_identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"branch", "revision", "tracked_dirty", "tree"}:
        raise ValueError(f"{name} fields differ")
    if not isinstance(row["branch"], str) or not row["branch"]:
        raise ValueError(f"{name} branch is invalid")
    _hex(row["revision"], length=40, name=f"{name} revision")
    _hex(row["tree"], length=40, name=f"{name} tree")
    if row["tracked_dirty"] is not False:
        raise ValueError(f"{name} must be tracked-clean")
    return row


def _checkout_identity(project_root: Path) -> dict[str, Any]:
    if not project_root.is_absolute():
        raise ValueError("project root must be absolute")
    _reject_symlink_chain(project_root)
    if not project_root.is_dir():
        raise ValueError(f"project root is not a directory: {project_root}")

    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError(f"project root must be completely clean: {project_root}")
    branch = git("branch", "--show-current")
    if not branch:
        raise ValueError("project root must be on a named branch")
    return {
        "branch": branch,
        "revision": _hex(
            git("rev-parse", "HEAD^{commit}"), length=40, name="revision"
        ),
        "tracked_dirty": False,
        "tree": _hex(git("rev-parse", "HEAD^{tree}"), length=40, name="tree"),
    }


def _checkout_and_script_identity(
    project_root: Path, script: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    git = _checkout_identity(project_root)
    if not script.is_absolute():
        raise ValueError("script path must be absolute")
    _reject_symlink_chain(script)
    try:
        relative = script.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError("script must be inside the project root") from error
    subprocess.run(
        ["git", "-C", str(project_root), "ls-files", "--error-unmatch", relative],
        check=True,
        capture_output=True,
        text=True,
    )
    _, identity = _stable_read(script)
    return git, identity


def _execution_git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if (
        row.get("branch") != EXECUTION_BRANCH
        or row.get("revision") != EXECUTION_REVISION
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} execution Git differs")
    if "tree" in row and row["tree"] != EXECUTION_TREE:
        raise ValueError(f"{name} execution tree differs")
    return row


def _expect_identity(actual: Any, expected: Mapping[str, Any], name: str) -> None:
    if _identity(actual, name) != dict(expected):
        raise ValueError(f"{name} physical identity differs")


def normalize_label_mapping(
    payload: Mapping[str, Any], *, expected_classes: int = NUM_CLASSES
) -> list[str]:
    row = _object(payload, "label mapping")
    mapping = _object(row.get("label_to_wnid"), "label_to_wnid")
    expected_keys = {str(index) for index in range(expected_classes)}
    if set(mapping) != expected_keys:
        raise ValueError("label mapping indices are not contiguous")
    wnids = [mapping[str(index)] for index in range(expected_classes)]
    if any(
        not isinstance(wnid, str) or re.fullmatch(r"n\d{8}", wnid) is None
        for wnid in wnids
    ):
        raise ValueError("label mapping contains an invalid WNID")
    if len(set(wnids)) != expected_classes:
        raise ValueError("label mapping WNIDs are not unique")
    if wnids != sorted(wnids):
        raise ValueError("label mapping is not lexicographic WNID order")
    return wnids


def audit_categories(
    data: bytes, mapping: Sequence[str], *, name: str = "torchvision categories"
) -> dict[str, Any]:
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not UTF-8") from error
    parsed: list[tuple[str, str]] = []
    for number, line in enumerate(lines, start=1):
        if not line or "," not in line:
            raise ValueError(f"{name} line {number} is malformed")
        category, wnid = line.rsplit(",", 1)
        if not category or re.fullmatch(r"n\d{8}", wnid) is None:
            raise ValueError(f"{name} line {number} is malformed")
        parsed.append((category, wnid))
    if len(parsed) != len(mapping):
        raise ValueError(f"{name} class count differs")
    categories = [category for category, _ in parsed]
    wnids = [wnid for _, wnid in parsed]
    if len(set(categories)) != len(categories):
        raise ValueError(f"{name} names are not unique")
    if wnids != list(mapping):
        raise ValueError(f"{name} WNID order differs from the loader mapping")
    return {
        "class_count": len(parsed),
        "first_category": categories[0],
        "last_category": categories[-1],
        "first_wnid": wnids[0],
        "last_wnid": wnids[-1],
        "categories_sha256": hashlib.sha256(
            json.dumps(categories, ensure_ascii=False, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest(),
    }


def audit_synsets(data: bytes, mapping: Sequence[str]) -> dict[str, Any]:
    try:
        wnids = [line for line in data.decode("utf-8").splitlines() if line]
    except UnicodeDecodeError as error:
        raise ValueError("timm synsets are not UTF-8") from error
    if wnids != list(mapping):
        raise ValueError("timm synset order differs from the loader mapping")
    return {
        "class_count": len(wnids),
        "first_wnid": wnids[0],
        "last_wnid": wnids[-1],
        "class_order_sha256": _canonical_sha256(wnids),
    }


def audit_dataset_manifest(
    path: Path,
    mapping: Sequence[str],
    *,
    expected_split_counts: Mapping[str, int] | None = None,
    expected_validation_per_class: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _before_read(path)
    digest = hashlib.sha256()
    bytes_read = 0
    row_count = 0
    split_counts: Counter[str] = Counter()
    per_split_labels: dict[str, Counter[int]] = {
        "train": Counter(),
        "val": Counter(),
    }
    seen_wnids: set[str] = set()
    for raw_line in path.open("rb"):
        digest.update(raw_line)
        bytes_read += len(raw_line)
        if not raw_line.strip():
            raise ValueError("dataset manifest contains a blank row")
        row_count += 1
        try:
            record = _object(
                json.loads(raw_line, parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite constant {value}")
                )),
                f"dataset manifest row {row_count}",
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"dataset manifest row {row_count} is invalid JSON") from error
        split = record.get("split")
        if split == "validation":
            split = "val"
        if split not in {"train", "val"}:
            raise ValueError(f"dataset manifest row {row_count} split differs")
        label = _integer(record.get("label"), f"dataset manifest row {row_count} label")
        if not 0 <= label < len(mapping):
            raise ValueError(f"dataset manifest row {row_count} label is outside range")
        wnid = record.get("wnid")
        if wnid != mapping[label]:
            raise ValueError(
                f"dataset manifest row {row_count} label/WNID mapping differs"
            )
        relative = record.get("relative_path") or record.get("path")
        if not isinstance(relative, str):
            raise ValueError(f"dataset manifest row {row_count} path is missing")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or len(pure.parts) < 4:
            raise ValueError(f"dataset manifest row {row_count} path is unsafe")
        if pure.parts[:3] != ("extracted", split, wnid):
            raise ValueError(
                f"dataset manifest row {row_count} path/WNID consistency differs"
            )
        if _integer(record.get("width"), "manifest width") != 256 or _integer(
            record.get("height"), "manifest height"
        ) != 256:
            raise ValueError(f"dataset manifest row {row_count} resolution differs")
        split_counts[split] += 1
        per_split_labels[split][label] += 1
        seen_wnids.add(wnid)
    _unchanged(path, before, bytes_read)
    identity = {
        "path": str(path),
        "bytes": bytes_read,
        "sha256": digest.hexdigest(),
    }
    if sorted(seen_wnids) != list(mapping):
        raise ValueError("dataset manifest loader class order differs from mapping")
    if set(per_split_labels["train"]) != set(range(len(mapping))):
        raise ValueError("dataset manifest train split lacks complete class coverage")
    if set(per_split_labels["val"]) != set(range(len(mapping))):
        raise ValueError("dataset manifest validation split lacks complete class coverage")
    if expected_split_counts is not None and dict(split_counts) != dict(
        expected_split_counts
    ):
        raise ValueError("dataset manifest split counts differ")
    if expected_validation_per_class is not None and any(
        count != expected_validation_per_class
        for count in per_split_labels["val"].values()
    ):
        raise ValueError("dataset manifest validation class balance differs")
    return {
        "row_count": row_count,
        "split_counts": dict(sorted(split_counts.items())),
        "class_count": len(mapping),
        "train_class_count_min": min(per_split_labels["train"].values()),
        "train_class_count_max": max(per_split_labels["train"].values()),
        "validation_class_count_min": min(per_split_labels["val"].values()),
        "validation_class_count_max": max(per_split_labels["val"].values()),
        "loader_class_order": "sorted_unique_manifest_wnid",
        "class_order_sha256": _canonical_sha256(list(mapping)),
        "label_matches_wnid_for_every_row": True,
        "path_wnid_matches_for_every_row": True,
        "resolution_matches_for_every_row": True,
    }, identity


def audit_metrics_jsonl(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    before = _before_read(path)
    digest = hashlib.sha256()
    bytes_read = 0
    row_count = 0
    first_step: int | None = None
    previous_step = 0
    validation_events = 0
    final_row: dict[str, Any] | None = None
    with path.open("rb") as handle:
        for raw_line in handle:
            digest.update(raw_line)
            bytes_read += len(raw_line)
            if not raw_line.strip():
                raise ValueError("training metrics contain a blank row")
            row_count += 1
            row = _json_loads(raw_line, f"training metric row {row_count}")
            step = _integer(row.get("step"), f"training metric row {row_count} step")
            samples = _integer(
                row.get("samples_seen"),
                f"training metric row {row_count} samples_seen",
            )
            if step <= previous_step:
                raise ValueError("training metric steps are not strictly increasing")
            if first_step is None:
                first_step = step
            if samples != step * EFFECTIVE_BATCH_SIZE:
                raise ValueError("training metric sample accounting differs")
            if "validation_epsilon_mse" in row:
                _finite(row["validation_epsilon_mse"], "validation epsilon MSE")
                validation_events += 1
            previous_step = step
            final_row = row
    _unchanged(path, before, bytes_read)
    if final_row is None or first_step != 1 or previous_step != COMPLETED_STEPS:
        raise ValueError("training metrics do not end at the qualified checkpoint")
    if validation_events != 10:
        raise ValueError("training metric validation-event count differs")
    return {
        "row_count": row_count,
        "first_step": first_step,
        "final_step": previous_step,
        "final_samples_seen": _integer(final_row["samples_seen"], "final samples"),
        "validation_event_count": validation_events,
        "strictly_increasing": True,
        "all_numeric_values_finite": True,
        "samples_seen_equals_step_times_64": True,
        "final_validation_epsilon_mse": _finite(
            final_row["validation_epsilon_mse"], "final validation epsilon MSE"
        ),
    }, {
        "path": str(path),
        "bytes": bytes_read,
        "sha256": digest.hexdigest(),
    }


CODE_PATTERNS = {
    "code_data_registry": (
        r"classes\s*=\s*sorted\(class_names\)",
        r"class_name\s*=\s*_record_class_name\(record\)",
        r"for\s+key\s+in\s+\(\"class_name\",\s*\"label_name\",\s*\"wnid\"\)",
        r"samples\s*=\s*\[\(path,\s*class_to_idx\[class_name\]\)",
    ),
    "code_train_generation": (
        r"labels\.to\(device=device,\s*dtype=torch\.long",
        r"output\s*=\s*model\(noisy,\s*timesteps,\s*class_labels=labels\)",
        r"class_conditional\s*!=\s*\(config\.model\.num_classes\s*>\s*0\)",
    ),
    "code_generate_samples": (
        r"torch\.arange\(start,\s*start\s*\+\s*count,.*?\)\s*%\s*num_classes",
        r"f\"\{start\s*\+\s*offset:06d\}\.png\"",
        r"\"class_schedule\":\s*\"balanced_modulo\"",
        r"class_labels=class_labels",
    ),
    "code_generation_session": (
        r"label\s*<\s*0\s*or\s*label\s*>=\s*self\.num_classes",
        r"torch\.tensor\(labels,\s*device=self\.device,\s*dtype=torch\.long\)",
        r"class_labels=labels",
    ),
    "code_diffusion_sampling": (
        r"torch\.cat\(.*?class_labels.*?torch\.full_like\(class_labels,\s*null_class\)",
        r"guided\s*=\s*unconditional\s*\+\s*guidance_scale\s*\*\s*\(conditional\s*-\s*unconditional\)",
        r"class_labels=class_labels",
    ),
    "code_scalable_unet": (
        r"nn\.Embedding\(num_classes\s*\+\s*1,\s*embedding_channels\)",
        r"labels\s*=\s*class_labels\.to\(device=timesteps\.device,\s*dtype=torch\.long\)",
        r"return\s+embedding\s*\+\s*self\.class_embed\(labels\)",
    ),
    "code_cofitok_model": (
        r"tokens\s*=\s*self\.predictor\(.*?class_labels=class_labels",
        r"components\s*=\s*self\.synthesis\(tokens\)",
    ),
    "code_class_fidelity_evaluator": (
        r"requested_class\s*=\s*int\(path\.stem\)\s*%\s*self\.num_classes",
        r"target_from_filename\"\s*:\s*\"int\(zero_based_png_stem\) mod 1000\"",
        r"sampling\.get\(\"class_schedule\"\)\s*!=\s*\"balanced_modulo\"",
    ),
    "code_class_fidelity_contract": (
        r"requested_class_count\s*!=\s*num_classes",
        r"target_from_filename.*?int\(zero_based_png_stem\) mod 1000",
        r"requested classes are not balanced",
    ),
}


def audit_code_contract(raw: Mapping[str, bytes]) -> dict[str, Any]:
    checks: dict[str, list[str]] = {}
    for key, patterns in CODE_PATTERNS.items():
        try:
            text = raw[key].decode("utf-8")
        except (KeyError, UnicodeDecodeError) as error:
            raise ValueError(f"execution source is missing or invalid: {key}") from error
        matched: list[str] = []
        for pattern in patterns:
            if re.search(pattern, text, flags=re.DOTALL) is None:
                raise ValueError(f"execution code contract differs: {key}: {pattern}")
            matched.append(pattern)
        checks[key] = matched
    return {
        "execution_revision": EXECUTION_REVISION,
        "execution_tree": EXECUTION_TREE,
        "code_file_count": len(CODE_FILES),
        "all_required_semantics_present": True,
        "loader_uses_sorted_wnid_classes": True,
        "training_passes_integer_labels_to_predictor": True,
        "sampling_uses_global_index_modulo_num_classes": True,
        "cfg_preserves_conditional_class_and_uses_null_class_only_for_unconditional": True,
        "class_embedding_receives_the_same_integer_index": True,
        "synthesis_operator_receives_tokens_not_class_labels": True,
        "class_fidelity_uses_the_same_filename_index_modulo_num_classes": True,
        "pattern_count_by_file": {key: len(value) for key, value in checks.items()},
    }


def _sample_set_sha256(directory: Path, *, count: int) -> dict[str, Any]:
    if not directory.is_absolute():
        raise ValueError("sample directory must be absolute")
    _reject_symlink_chain(directory)
    if not directory.is_dir():
        raise ValueError(f"sample directory is missing: {directory}")
    expected_names = [f"{index:06d}.png" for index in range(count)]
    entries = sorted(directory.iterdir(), key=lambda path: path.name)
    if [entry.name for entry in entries] != expected_names:
        raise ValueError(f"sample directory numbered-file set differs: {directory}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in entries:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"sample set contains a non-file or symlink: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        before = path.stat()
        bytes_read = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                bytes_read += len(block)
        _unchanged(path, before, bytes_read)
        total_bytes += bytes_read
        digest.update(b"\0")
    return {
        "root": str(directory),
        "count": count,
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
        "framing": "filename_utf8_nul_file_bytes_nul",
    }


def _source_paths(
    *,
    execution_root: Path,
    screen_root: Path,
    controller_root: Path,
    prelaunch_root: Path,
    causal_root: Path,
    dataset_root: Path,
    categories_file: Path,
    timm_synsets_file: Path,
    calibration: Path,
) -> dict[str, Path]:
    paths = {
        "preparation": prelaunch_root / "preparation.json",
        "execution_authorization": prelaunch_root / "execution_authorization.json",
        "launch_receipt": prelaunch_root / "launch_receipt.json",
        "runtime_selection": prelaunch_root / "runtime_selection.json",
        "live_snapshot": prelaunch_root / "live_snapshot.json",
        "storage_capacity": prelaunch_root / "storage_capacity.json",
        "controller_status": controller_root / "controller_status.json",
        "controller_log": controller_root / "controller.log",
        "terminal_result": screen_root / "terminal_snr_screen_result.json",
        "terminal_result_validation": (
            screen_root / "terminal_snr_screen_result.validation.json"
        ),
        "causal_discriminator": (
            causal_root / "support_collapse_causal_discriminator.json"
        ),
        "causal_discriminator_validation": (
            causal_root / "support_collapse_causal_discriminator.validation.json"
        ),
        "execution_label_mapping": (
            execution_root
            / "docs/experiment_conditions/generated/"
            "imagenet_label_to_wnid_train_order.json"
        ),
        "dataset_label_mapping": dataset_root / "metadata/label_to_wnid.json",
        "dataset_source_mapping": (
            dataset_root / "metadata/imagenet_label_to_wnid_train_order.json"
        ),
        "dataset_manifest": dataset_root / "metadata/image_manifest.jsonl",
        "torchvision_categories": categories_file,
        "timm_synsets": timm_synsets_file,
        "real_calibration": calibration,
    }
    for key, relative in CODE_FILES.items():
        paths[key] = execution_root / relative
    for arm, spec in ARM_SPECS.items():
        training = screen_root / "training" / arm
        evaluation = screen_root / "evaluations" / arm
        paths.update(
            {
                f"{arm}_config": execution_root / str(spec["config"]),
                f"{arm}_arm_validation": screen_root
                / "arm_validations"
                / f"{arm}.json",
                f"{arm}_run_manifest": training / "run_manifest.json",
                f"{arm}_train_metrics": training / "train_metrics.jsonl",
                f"{arm}_latest": training / "latest.json",
                f"{arm}_training_report": training / "training_report.json",
                f"{arm}_checkpoint_5000": training
                / "checkpoint_step_00005000.pt",
                f"{arm}_checkpoint_5000_sidecar": training
                / "checkpoint_step_00005000.pt.integrity.json",
                f"{arm}_checkpoint_10000": training
                / "checkpoint_step_00010000.pt",
                f"{arm}_checkpoint_10000_sidecar": training
                / "checkpoint_step_00010000.pt.integrity.json",
                f"{arm}_sampling_manifest": evaluation
                / "sampling/sampling_manifest.json",
                f"{arm}_sampling_progress": evaluation
                / "sampling/sampling_progress.json",
                f"{arm}_sampling_report": evaluation
                / "sampling/sampling_report.json",
                f"{arm}_metrics_report": evaluation
                / "metrics/generation_metrics_report.json",
                f"{arm}_class_fidelity_report": evaluation
                / "class_fidelity/class_fidelity_report.json",
                f"{arm}_checkpoint_evaluation_manifest": evaluation
                / "checkpoint_eval/checkpoint_evaluation_manifest.json",
                f"{arm}_checkpoint_evaluation_report": evaluation
                / "checkpoint_eval/checkpoint_evaluation_report.json",
                f"{arm}_rollout_report": evaluation
                / "rollout/rollout_stability_report.json",
            }
        )
    return paths


def _load_sources(
    paths: Mapping[str, Path],
) -> tuple[dict[str, dict[str, Any]], dict[str, bytes], dict[str, dict[str, Any]]]:
    payloads: dict[str, dict[str, Any]] = {}
    raw: dict[str, bytes] = {}
    identities: dict[str, dict[str, Any]] = {}
    hash_only = {
        key
        for key in paths
        if "_checkpoint_5000" in key or "_checkpoint_10000" in key
    } - {
        key for key in paths if key.endswith("_sidecar")
    }
    deferred = {"dataset_manifest"} | {
        key for key in paths if key.endswith("_train_metrics")
    }
    for key in sorted(paths):
        if key in deferred:
            continue
        path = paths[key]
        if key in hash_only:
            identity = _stable_hash(path)
        else:
            data, identity = _stable_read(path)
            raw[key] = data
            if path.suffix == ".json":
                payloads[key] = _json_loads(data, key)
        identities[key] = identity
        expected_sha256 = LOCKED_SHA256.get(key)
        if expected_sha256 is not None and identity["sha256"] != expected_sha256:
            raise ValueError(
                f"locked source SHA256 differs for {key}: {identity['sha256']}"
            )
    return payloads, raw, identities


def _checkpoint_summary(
    *,
    arm: str,
    step: int,
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    checkpoint_key = f"{arm}_checkpoint_{step}"
    sidecar_key = f"{checkpoint_key}_sidecar"
    checkpoint = _identity(identities[checkpoint_key], checkpoint_key)
    sidecar_identity = _identity(identities[sidecar_key], sidecar_key)
    sidecar = _object(payloads[sidecar_key], sidecar_key)
    expected_filename = f"checkpoint_step_{step:08d}.pt"
    if (
        sidecar.get("schema_version") != 1
        or sidecar.get("checkpoint_format_version") != 1
        or sidecar.get("checkpoint") != expected_filename
        or _integer(sidecar.get("step"), f"{sidecar_key} step") != step
        or _integer(sidecar.get("checkpoint_bytes"), "checkpoint bytes")
        != checkpoint["bytes"]
        or sidecar.get("checkpoint_sha256") != checkpoint["sha256"]
        or sidecar.get("git_branch") != EXECUTION_BRANCH
        or sidecar.get("git_revision") != EXECUTION_REVISION
        or sidecar.get("git_dirty") is not False
    ):
        raise ValueError(f"checkpoint sidecar contract differs: {arm} step {step}")
    _hex(
        sidecar.get("dataset_identity_sha256"),
        length=64,
        name="dataset identity SHA256",
    )
    _hex(
        sidecar.get("runtime_environment_sha256"),
        length=64,
        name="runtime environment SHA256",
    )
    return {
        "step": step,
        "checkpoint": checkpoint,
        "integrity_manifest": sidecar_identity,
        "integrity_replayed": True,
    }


def _training_git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if (
        row.get("branch") != EXECUTION_BRANCH
        or row.get("revision") != EXECUTION_REVISION
        or row.get("dirty") is not False
    ):
        raise ValueError(f"{name} differs")
    return row


def _audit_sampling(
    *,
    arm: str,
    spec: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_key = f"{arm}_sampling_manifest"
    progress_key = f"{arm}_sampling_progress"
    report_key = f"{arm}_sampling_report"
    manifest = _object(payloads[manifest_key], manifest_key)
    progress = _object(payloads[progress_key], progress_key)
    report = _object(payloads[report_key], report_key)
    manifest_identity = _identity(identities[manifest_key], manifest_key)
    progress_identity = _identity(identities[progress_key], progress_key)
    report_identity = _identity(identities[report_key], report_key)
    sampling = _object(report.get("sampling"), f"{arm} sampling protocol")
    manifest_sampling = _object(
        manifest.get("sampling"), f"{arm} sampling manifest protocol"
    )
    if sampling != manifest_sampling:
        raise ValueError(f"{arm} sampling manifest/report protocol differs")
    prefix = _integer(spec["prefix_budget"], f"{arm} prefix budget")
    expected_protocol = {
        "num_samples": SAMPLES_PER_ARM,
        "start_index": 0,
        "batch_size": 4,
        "sample_steps": 100,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 2027,
        "precision": "bf16",
        "class_schedule": "balanced_modulo",
        "prefix_budgets": [prefix],
    }
    if any(sampling.get(key) != value for key, value in expected_protocol.items()):
        raise ValueError(f"{arm} sampling protocol differs")
    if (
        sampling.get("sampler") != "ddim"
        or sampling.get("protocol_schema") != "cofitok_ddim_sampling_v1"
        or sampling.get("sample_set_digest")
        != {"algorithm": "sha256", "framing": "filename_utf8_nul_file_bytes_nul"}
        or report.get("schema_version") != 6
        or report.get("status") != "completed"
        or report.get("weights") != "ema"
        or manifest.get("schema_version") != 3
        or manifest.get("weights") != "ema"
    ):
        raise ValueError(f"{arm} sampling contract differs")
    for source in (manifest, report):
        _execution_git(source.get("git"), f"{arm} sampling Git")
        if (
            source.get("checkpoint") != checkpoint["path"]
            or source.get("checkpoint_sha256") != checkpoint["sha256"]
            or source.get("checkpoint_step") != COMPLETED_STEPS
        ):
            raise ValueError(f"{arm} sampling checkpoint binding differs")
    if report.get("sampling_manifest_sha256") != manifest_identity["sha256"]:
        raise ValueError(f"{arm} sampling report manifest binding differs")
    if (
        progress.get("schema_version") != 1
        or progress.get("status") != "completed"
        or progress.get("completed_samples") != SAMPLES_PER_ARM
        or progress.get("total_samples") != SAMPLES_PER_ARM
        or progress.get("last_completed_index") != SAMPLES_PER_ARM - 1
        or progress.get("sampling_manifest_sha256") != manifest_identity["sha256"]
        or progress.get("prefix_budgets") != [prefix]
        or _finite(progress.get("completed_fraction"), "completed fraction") != 1.0
        or _finite(
            progress.get("cumulative_elapsed_seconds"), "sampling elapsed seconds"
        )
        <= 0.0
    ):
        raise ValueError(f"{arm} sampling progress differs")
    output_dirs = _object(report.get("output_dirs"), f"{arm} output directories")
    if output_dirs != manifest.get("output_dirs") or set(output_dirs) != {str(prefix)}:
        raise ValueError(f"{arm} sampling output directories differ")
    sample_set = _sample_set_sha256(
        Path(output_dirs[str(prefix)]), count=SAMPLES_PER_ARM
    )
    declared_sets = _object(report.get("sample_sets"), f"{arm} sample sets")
    progress_sets = _object(progress.get("sample_sets"), f"{arm} progress sample sets")
    expected_set = {"count": SAMPLES_PER_ARM, "sha256": sample_set["sha256"]}
    if declared_sets != {str(prefix): expected_set} or progress_sets != declared_sets:
        raise ValueError(f"{arm} physical sample-set digest differs")
    return {
        "manifest": manifest_identity,
        "progress": progress_identity,
        "report": report_identity,
        "protocol": expected_protocol,
        "sample_set": sample_set,
        "balanced_requested_class_histogram": {
            "class_count": NUM_CLASSES,
            "minimum_count": 1,
            "maximum_count": 1,
            "derivation": "global_sample_index_modulo_1000",
        },
    }, report


def _audit_arm(
    *,
    arm: str,
    spec: Mapping[str, Any],
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    metrics_summary: Mapping[str, Any],
) -> dict[str, Any]:
    config_key = f"{arm}_config"
    validation_key = f"{arm}_arm_validation"
    training_key = f"{arm}_training_report"
    latest_key = f"{arm}_latest"
    config = _object(payloads[config_key], config_key)
    validation = _object(payloads[validation_key], validation_key)
    training = _object(payloads[training_key], training_key)
    latest = _object(payloads[latest_key], latest_key)
    if (
        validation.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_arm_validation_v1"
        or validation.get("role")
        != "physical_terminal_snr_screen_arm_validation"
        or validation.get("status") != "pass"
        or validation.get("arm") != arm
        or validation.get("condition") != spec["condition"]
        or validation.get("method") != spec["method"]
        or _finite(validation.get("endpoint_fraction"), "endpoint fraction")
        != spec["endpoint_fraction"]
    ):
        raise ValueError(f"{arm} validation contract differs")
    _execution_git(validation.get("execution_git"), f"{arm} validation Git")
    if any(_object(validation.get("authorization_boundary"), "arm boundary").values()):
        raise ValueError(f"{arm} validation grants an authorization")

    validation_sources = _object(validation.get("sources"), f"{arm} sources")
    source_bindings = {
        "config": config_key,
        "launch_receipt": "launch_receipt",
        "training_report": training_key,
        "sampling_report": f"{arm}_sampling_report",
        "metrics_report": f"{arm}_metrics_report",
        "class_fidelity_report": f"{arm}_class_fidelity_report",
        "checkpoint_evaluation_report": f"{arm}_checkpoint_evaluation_report",
        "rollout_report": f"{arm}_rollout_report",
    }
    if set(validation_sources) != set(source_bindings):
        raise ValueError(f"{arm} validation source set differs")
    for declared, source_key in source_bindings.items():
        _expect_identity(
            validation_sources[declared], identities[source_key], f"{arm} {declared}"
        )

    data = _object(config.get("data"), f"{arm} config data")
    model = _object(config.get("model"), f"{arm} config model")
    diffusion = _object(config.get("diffusion"), f"{arm} config diffusion")
    optimization = _object(config.get("optimization"), f"{arm} config optimization")
    if (
        data.get("dataset") != "imagenet_256"
        or data.get("class_conditional") is not True
        or model.get("num_classes") != NUM_CLASSES
        or model.get("class_dropout_prob") != 0.1
        or diffusion.get("cosine_endpoint_fraction", 1.0)
        != spec["endpoint_fraction"]
        or _integer(data.get("batch_size"), "configured micro batch")
        * _integer(
            optimization.get("gradient_accumulation_steps"),
            "configured gradient accumulation",
        )
        != EFFECTIVE_BATCH_SIZE
    ):
        raise ValueError(f"{arm} class-conditional config differs")
    expected_tokens = 8 if spec["method"] == "cofitok" else 1
    if model.get("token_count") != expected_tokens:
        raise ValueError(f"{arm} token-count contract differs")

    resolved_config = _object(training.get("config"), f"{arm} resolved config")
    resolved_data = _object(resolved_config.get("data"), f"{arm} resolved data")
    resolved_model = _object(resolved_config.get("model"), f"{arm} resolved model")
    resolved_diffusion = _object(
        resolved_config.get("diffusion"), f"{arm} resolved diffusion"
    )
    resolved_optimization = _object(
        resolved_config.get("optimization"), f"{arm} resolved optimization"
    )
    if (
        training.get("completed_steps") != COMPLETED_STEPS
        or training.get("training_complete") is not False
        or training.get("target_steps") != 100_000
        or training.get("parameter_count") != training.get("trainable_parameter_count")
        or resolved_data.get("dataset") != "imagenet_256"
        or resolved_data.get("class_conditional") is not True
        or resolved_model.get("num_classes") != NUM_CLASSES
        or resolved_model.get("class_dropout_prob") != 0.1
        or resolved_diffusion.get("cosine_endpoint_fraction")
        != spec["endpoint_fraction"]
        or _integer(resolved_data.get("batch_size"), "resolved micro batch")
        * _integer(
            resolved_optimization.get("gradient_accumulation_steps"),
            "resolved gradient accumulation",
        )
        != EFFECTIVE_BATCH_SIZE
    ):
        raise ValueError(f"{arm} training report contract differs")
    _training_git(training.get("git"), f"{arm} training Git")
    final_metrics = _object(training.get("final_metrics"), f"{arm} final metrics")
    if (
        final_metrics.get("step") != COMPLETED_STEPS
        or final_metrics.get("samples_seen")
        != COMPLETED_STEPS * EFFECTIVE_BATCH_SIZE
        or final_metrics.get("validation_epsilon_mse")
        != metrics_summary["final_validation_epsilon_mse"]
    ):
        raise ValueError(f"{arm} training final metrics differ")

    checkpoints = {
        str(step): _checkpoint_summary(
            arm=arm, step=step, payloads=payloads, identities=identities
        )
        for step in (5000, 10000)
    }
    checkpoint = checkpoints["10000"]["checkpoint"]
    sidecar = _object(
        payloads[f"{arm}_checkpoint_10000_sidecar"], "final checkpoint sidecar"
    )
    expected_latest = dict(sidecar)
    expected_latest["integrity_manifest"] = f"checkpoint_step_{COMPLETED_STEPS:08d}.pt.integrity.json"
    if latest != expected_latest or training.get("latest_checkpoint") != latest:
        raise ValueError(f"{arm} latest/report checkpoint binding differs")
    arm_training = _object(validation.get("training"), f"{arm} validation training")
    _expect_identity(
        {
            key: _object(arm_training.get("checkpoint"), "arm checkpoint")[key]
            for key in ("path", "bytes", "sha256")
        },
        checkpoint,
        f"{arm} validated checkpoint",
    )
    _expect_identity(
        _object(arm_training.get("checkpoint"), "arm checkpoint").get(
            "integrity_manifest"
        ),
        identities[f"{arm}_checkpoint_10000_sidecar"],
        f"{arm} validated checkpoint sidecar",
    )

    sampling_summary, sampling_report = _audit_sampling(
        arm=arm,
        spec=spec,
        payloads=payloads,
        identities=identities,
        checkpoint=checkpoint,
    )
    sample_set = sampling_summary["sample_set"]

    metrics_report = _object(
        payloads[f"{arm}_metrics_report"], f"{arm} metrics report"
    )
    if metrics_report.get("status") != "completed":
        raise ValueError(f"{arm} metrics report is incomplete")
    metrics = _object(metrics_report.get("metrics"), f"{arm} distribution metrics")
    metrics_provenance = _object(
        metrics_report.get("sample_provenance"), f"{arm} metrics provenance"
    )
    if (
        metrics_provenance.get("sample_set_sha256") != sample_set["sha256"]
        or metrics_provenance.get("checkpoint_sha256") != checkpoint["sha256"]
    ):
        raise ValueError(f"{arm} metrics sample provenance differs")
    _expect_identity(
        metrics_provenance.get("report_identity"),
        identities[f"{arm}_sampling_report"],
        f"{arm} metrics sampling report",
    )

    fidelity = _object(
        payloads[f"{arm}_class_fidelity_report"], f"{arm} class fidelity"
    )
    fidelity_metrics = _object(
        fidelity.get("metrics"), f"{arm} class fidelity metrics"
    )
    parameters = _object(fidelity.get("parameters"), f"{arm} fidelity parameters")
    fidelity_provenance = _object(
        fidelity.get("sample_provenance"), f"{arm} fidelity provenance"
    )
    if (
        fidelity.get("schema_version") != 2
        or fidelity.get("status") != "completed"
        or fidelity.get("protocol") != "torchvision_imagenet_class_fidelity"
        or parameters.get("target_from_filename")
        != "int(zero_based_png_stem) mod 1000"
        or parameters.get("num_classes") != NUM_CLASSES
        or fidelity_metrics.get("sample_count") != SAMPLES_PER_ARM
        or fidelity_metrics.get("requested_class_count") != NUM_CLASSES
        or fidelity_metrics.get("requested_count_min") != 1
        or fidelity_metrics.get("requested_count_max") != 1
        or fidelity_provenance.get("sample_set_sha256") != sample_set["sha256"]
        or fidelity_provenance.get("checkpoint_sha256") != checkpoint["sha256"]
    ):
        raise ValueError(f"{arm} class-fidelity protocol differs")
    _execution_git(fidelity.get("git"), f"{arm} class-fidelity Git")
    _expect_identity(
        fidelity_provenance.get("report_identity"),
        identities[f"{arm}_sampling_report"],
        f"{arm} class-fidelity sampling report",
    )

    checkpoint_evaluation = _object(
        payloads[f"{arm}_checkpoint_evaluation_report"],
        f"{arm} checkpoint evaluation",
    )
    rollout = _object(payloads[f"{arm}_rollout_report"], f"{arm} rollout")
    for report_name, report in (
        ("checkpoint evaluation", checkpoint_evaluation),
        ("rollout", rollout),
    ):
        if (
            report.get("status") != "completed"
            or report.get("checkpoint_sha256") != checkpoint["sha256"]
            or report.get("checkpoint_step") != COMPLETED_STEPS
        ):
            raise ValueError(f"{arm} {report_name} checkpoint binding differs")
        _execution_git(report.get("git"), f"{arm} {report_name} Git")
    _expect_identity(
        checkpoint_evaluation.get("manifest"),
        identities[f"{arm}_checkpoint_evaluation_manifest"],
        f"{arm} checkpoint evaluation manifest",
    )

    classifier = _object(fidelity.get("classifier"), f"{arm} classifier")
    if (
        classifier.get("num_classes") != NUM_CLASSES
        or classifier.get("categories_sha256") != EVALUATOR_CATEGORIES_SHA256
        or classifier.get("weights_sha256")
        != "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
    ):
        raise ValueError(f"{arm} classifier identity differs")

    return {
        "condition": spec["condition"],
        "method": spec["method"],
        "endpoint_fraction": spec["endpoint_fraction"],
        "config": _identity(identities[config_key], config_key),
        "arm_validation": _identity(identities[validation_key], validation_key),
        "training_report": _identity(identities[training_key], training_key),
        "training_metrics": copy.deepcopy(metrics_summary),
        "checkpoints": checkpoints,
        "sampling": sampling_summary,
        "distribution": {
            "report": _identity(
                identities[f"{arm}_metrics_report"], f"{arm} metrics report"
            ),
            "fid": _finite(metrics.get("frechet_inception_distance"), "FID"),
            "precision": _finite(metrics.get("precision"), "precision"),
            "recall": _finite(metrics.get("recall"), "recall"),
        },
        "class_fidelity": {
            "report": _identity(
                identities[f"{arm}_class_fidelity_report"],
                f"{arm} class-fidelity report",
            ),
            "top1_accuracy": _finite(
                fidelity_metrics.get("top1_accuracy"), "top-1 accuracy"
            ),
            "top5_accuracy": _finite(
                fidelity_metrics.get("top5_accuracy"), "top-5 accuracy"
            ),
            "predicted_class_fraction": _finite(
                fidelity_metrics.get("predicted_class_fraction"),
                "predicted class fraction",
            ),
            "normalized_predicted_class_entropy": _finite(
                fidelity_metrics.get("normalized_predicted_class_entropy"),
                "normalized predicted-class entropy",
            ),
        },
        "checkpoint_diagnostics": _identity(
            identities[f"{arm}_checkpoint_evaluation_report"],
            f"{arm} checkpoint evaluation",
        ),
        "rollout": _identity(
            identities[f"{arm}_rollout_report"], f"{arm} rollout"
        ),
        "class_index_transport_replayed": True,
    }


def _audit_calibration(
    calibration: Mapping[str, Any],
    identities: Mapping[str, Mapping[str, Any]],
    *,
    categories_sha256: str,
    class_order_sha256: str,
) -> dict[str, Any]:
    row = _object(calibration, "real classifier calibration")
    metrics = _object(row.get("metrics"), "real calibration metrics")
    selection = _object(row.get("selection"), "real calibration selection")
    sources = _object(row.get("sources"), "real calibration sources")
    if (
        row.get("schema_version") != 1
        or row.get("role") != "generation_class_fidelity_classifier_real_calibration"
        or row.get("status") != "completed"
        or row.get("calibration_status") != "pass"
        or metrics.get("sample_count") != NUM_CLASSES
        or metrics.get("requested_class_count") != NUM_CLASSES
        or metrics.get("requested_count_min") != 1
        or metrics.get("requested_count_max") != 1
        or _finite(metrics.get("top1_accuracy"), "real top-1") != 0.777
        or _finite(metrics.get("top5_accuracy"), "real top-5") != 0.941
        or selection.get("scheme") != "lexicographic_first_n_per_wnid_v1"
        or selection.get("class_order_sha256") != class_order_sha256
        or selection.get("first_class") != "n01440764"
        or selection.get("last_class") != "n15075141"
    ):
        raise ValueError("real classifier calibration contract differs")
    expected_sources = {
        "label_to_wnid": "dataset_label_mapping",
        "torchvision_categories": "torchvision_categories",
        "timm_synsets": "timm_synsets",
    }
    if set(sources) != set(expected_sources):
        raise ValueError("real classifier calibration sources differ")
    for name, key in expected_sources.items():
        _expect_identity(sources[name], identities[key], f"calibration {name}")
    classifier = _object(row.get("classifier"), "real calibration classifier")
    if (
        classifier.get("num_classes") != NUM_CLASSES
        or classifier.get("categories_sha256") != categories_sha256
        or classifier.get("weights_sha256")
        != "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
    ):
        raise ValueError("real calibration classifier identity differs")
    if any(_object(row.get("claim_boundary"), "calibration boundary").get(key) is True for key in (
        "training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_or_release_allowed",
    )):
        raise ValueError("real calibration grants an authorization")
    return {
        "report": _identity(identities["real_calibration"], "real calibration"),
        "sample_count": NUM_CLASSES,
        "samples_per_class": 1,
        "top1_accuracy": 0.777,
        "top5_accuracy": 0.941,
        "calibration_status": "pass",
        "demonstrates_classifier_index_alignment_on_real_validation": True,
    }


def _build_audit(
    *,
    builder_git: Mapping[str, Any],
    builder_script: Mapping[str, Any],
    execution_root: Path,
    screen_root: Path,
    controller_root: Path,
    prelaunch_root: Path,
    causal_root: Path,
    dataset_root: Path,
    categories_file: Path,
    timm_synsets_file: Path,
    calibration: Path,
) -> dict[str, Any]:
    execution_git = _checkout_identity(execution_root)
    if execution_git != {
        "branch": EXECUTION_BRANCH,
        "revision": EXECUTION_REVISION,
        "tracked_dirty": False,
        "tree": EXECUTION_TREE,
    }:
        raise ValueError("execution checkout identity differs")
    paths = _source_paths(
        execution_root=execution_root,
        screen_root=screen_root,
        controller_root=controller_root,
        prelaunch_root=prelaunch_root,
        causal_root=causal_root,
        dataset_root=dataset_root,
        categories_file=categories_file,
        timm_synsets_file=timm_synsets_file,
        calibration=calibration,
    )
    payloads, raw, identities = _load_sources(paths)

    mapping_sources = (
        "execution_label_mapping",
        "dataset_label_mapping",
        "dataset_source_mapping",
    )
    mappings = {
        key: normalize_label_mapping(payloads[key]) for key in mapping_sources
    }
    mapping = mappings["execution_label_mapping"]
    if any(value != mapping for value in mappings.values()):
        raise ValueError("execution and dataset label mappings differ semantically")
    categories = audit_categories(raw["torchvision_categories"], mapping)
    synsets = audit_synsets(raw["timm_synsets"], mapping)
    manifest_summary, manifest_identity = audit_dataset_manifest(
        paths["dataset_manifest"],
        mapping,
        expected_split_counts={"train": TRAIN_ROWS, "val": VALIDATION_ROWS},
        expected_validation_per_class=VALIDATION_ROWS_PER_CLASS,
    )
    identities["dataset_manifest"] = manifest_identity
    if manifest_identity["sha256"] != LOCKED_SHA256["dataset_manifest"]:
        raise ValueError("locked dataset manifest SHA256 differs")

    for arm in ARM_SPECS:
        summary, identity = audit_metrics_jsonl(paths[f"{arm}_train_metrics"])
        identities[f"{arm}_train_metrics"] = identity
        payloads[f"{arm}_metrics_summary"] = summary

    code_contract = audit_code_contract(raw)
    calibration_summary = _audit_calibration(
        payloads["real_calibration"],
        identities,
        categories_sha256=EVALUATOR_CATEGORIES_SHA256,
        class_order_sha256=manifest_summary["class_order_sha256"],
    )

    controller = _object(payloads["controller_status"], "controller status")
    if (
        controller.get("role") != "generation_terminal_snr_screen_controller"
        or controller.get("status") != "completed"
        or controller.get("stage") != "complete"
        or controller.get("terminal_status") != "hold"
        or controller.get("generation_advantage_proven") is not False
    ):
        raise ValueError("active controller terminal state differs")
    for field in (
        "frozen_confirmation_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_allowed",
        "export_allowed",
        "release_allowed",
        "process_signals_allowed",
    ):
        if controller.get(field) is not False:
            raise ValueError(f"active controller permission differs: {field}")
    _expect_identity(
        controller.get("authorization"),
        identities["execution_authorization"],
        "controller authorization",
    )
    _expect_identity(
        controller.get("launch_receipt"),
        identities["launch_receipt"],
        "controller launch receipt",
    )

    terminal = _object(payloads["terminal_result"], "terminal result")
    terminal_validation = _object(
        payloads["terminal_result_validation"], "terminal result validation"
    )
    if (
        terminal.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_result_v1"
        or terminal.get("status") != "completed"
        or terminal.get("operational_status") != "pass"
        or terminal.get("scientific_status") != "hold"
        or terminal.get("terminal_status") != "hold"
        or terminal.get("screen_pass") is not False
        or terminal.get("generation_advantage_proven") is not False
        or terminal.get("failed_checks")
        != [
            "cofitok.relative_fid_improvement",
            "dense_identity.relative_fid_improvement",
        ]
    ):
        raise ValueError("terminal result hold contract differs")
    _execution_git(terminal.get("result_git"), "terminal result Git")
    if (
        terminal_validation.get("status") != "pass"
        or terminal_validation.get("scientific_status") != "hold"
        or terminal_validation.get("screen_pass") is not False
        or terminal_validation.get("generation_advantage_proven") is not False
    ):
        raise ValueError("terminal result validation contract differs")
    _expect_identity(
        terminal_validation.get("result"),
        identities["terminal_result"],
        "validated terminal result",
    )
    if any(_object(terminal_validation.get("authorization_boundary"), "terminal validation boundary").values()):
        raise ValueError("terminal result validation grants an authorization")

    arm_replays = {
        arm: _audit_arm(
            arm=arm,
            spec=spec,
            payloads=payloads,
            identities=identities,
            metrics_summary=payloads[f"{arm}_metrics_summary"],
        )
        for arm, spec in ARM_SPECS.items()
    }
    terminal_arm_sources = _object(
        _object(terminal.get("source_evidence"), "terminal sources").get(
            "arm_validations"
        ),
        "terminal arm validations",
    )
    terminal_summaries = _object(terminal.get("arm_summaries"), "terminal arms")
    if set(terminal_arm_sources) != set(ARM_SPECS) or set(terminal_summaries) != set(
        ARM_SPECS
    ):
        raise ValueError("terminal arm set differs")
    for arm, replay in arm_replays.items():
        _expect_identity(
            terminal_arm_sources[arm],
            identities[f"{arm}_arm_validation"],
            f"terminal {arm} validation",
        )
        summary = _object(terminal_summaries[arm], f"terminal {arm} summary")
        if (
            summary.get("sample_count") != SAMPLES_PER_ARM
            or summary.get("sample_set_sha256")
            != replay["sampling"]["sample_set"]["sha256"]
            or summary.get("fid") != replay["distribution"]["fid"]
            or summary.get("precision") != replay["distribution"]["precision"]
            or summary.get("recall") != replay["distribution"]["recall"]
            or summary.get("class_top1")
            != replay["class_fidelity"]["top1_accuracy"]
            or summary.get("class_top5")
            != replay["class_fidelity"]["top5_accuracy"]
        ):
            raise ValueError(f"terminal {arm} summary differs from physical sources")

    causal = _object(payloads["causal_discriminator"], "causal discriminator")
    causal_validation = _object(
        payloads["causal_discriminator_validation"], "causal validation"
    )
    if (
        causal.get("schema_version")
        != "cofitok_generation_support_collapse_causal_discriminator_v1"
        or causal.get("status") != "completed"
        or causal.get("operational_status") != "pass"
        or causal.get("scientific_status") != "hold"
        or causal.get("decision") != "no_defensible_shared_intervention_selected"
        or causal.get("selected_candidate") is not None
        or _object(causal.get("support_quality"), "support quality").get(
            "shared_support_collapse_consistent"
        )
        is not True
        or _object(causal.get("support_quality"), "support quality").get(
            "common_cause_proven"
        )
        is not False
    ):
        raise ValueError("causal discriminator contract differs")
    if any(_object(causal.get("authorization_boundary"), "causal boundary").values()):
        raise ValueError("causal discriminator grants an authorization")
    if causal_validation.get("status") != "pass":
        raise ValueError("causal discriminator validation did not pass")
    _expect_identity(
        causal_validation.get("decision"),
        identities["causal_discriminator"],
        "validated causal discriminator",
    )
    causal_sources = _object(
        _object(causal.get("source_evidence"), "causal sources").get(
            "all_bound_sources"
        ),
        "causal bound sources",
    )
    for causal_key, source_key in (
        ("controller_status", "controller_status"),
        ("controller_log", "controller_log"),
        ("terminal_result", "terminal_result"),
        ("terminal_result_validation", "terminal_result_validation"),
        ("arm_control_cofitok", "control_cofitok_arm_validation"),
        ("arm_control_dense_identity", "control_dense_identity_arm_validation"),
        ("arm_endpoint0975_cofitok", "endpoint0975_cofitok_arm_validation"),
        (
            "arm_endpoint0975_dense_identity",
            "endpoint0975_dense_identity_arm_validation",
        ),
    ):
        _expect_identity(
            causal_sources[causal_key], identities[source_key], f"causal {causal_key}"
        )

    recalls = {arm: replay["distribution"]["recall"] for arm, replay in arm_replays.items()}
    if any(value != 0.0 for value in recalls.values()):
        raise ValueError("expected shared zero-recall support collapse is absent")
    comparisons = _object(terminal.get("comparisons"), "terminal comparisons")
    fid_improvements = {
        method: _finite(
            _object(comparisons[method], f"{method} comparison").get(
                "relative_fid_improvement"
            ),
            f"{method} relative FID improvement",
        )
        for method in ("cofitok", "dense_identity")
    }
    if fid_improvements != {
        "cofitok": -0.11363547384116164,
        "dense_identity": -0.09408838680587096,
    }:
        raise ValueError("terminal FID judgment differs")

    source_evidence = {key: copy.deepcopy(value) for key, value in sorted(identities.items())}
    report = {
        "schema_version": SCHEMA,
        "role": ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "decision": "class_index_mapping_mismatch_ruled_out_support_collapse_unexplained",
        "generation_advantage_proven": False,
        "builder_git": _git_identity(builder_git, "builder Git"),
        "builder_script": _identity(builder_script, "builder script"),
        "execution_git": execution_git,
        "source_evidence": source_evidence,
        "dataset_label_pipeline": {
            "execution_mapping": source_evidence["execution_label_mapping"],
            "dataset_mapping": source_evidence["dataset_label_mapping"],
            "dataset_source_mapping": source_evidence["dataset_source_mapping"],
            "all_three_mappings_semantically_identical": True,
            "class_count": NUM_CLASSES,
            "first_label": {"index": 0, "wnid": mapping[0]},
            "last_label": {"index": NUM_CLASSES - 1, "wnid": mapping[-1]},
            "class_order": "official_lexicographic_wnid_order",
            "class_order_sha256": manifest_summary["class_order_sha256"],
            "manifest": manifest_summary | {"identity": manifest_identity},
            "torchvision_categories": categories
            | {"identity": source_evidence["torchvision_categories"]},
            "timm_synsets": synsets | {"identity": source_evidence["timm_synsets"]},
        },
        "execution_code_pipeline": code_contract,
        "real_classifier_calibration": calibration_summary,
        "arm_replays": arm_replays,
        "screen_judgment": {
            "failed_checks": copy.deepcopy(terminal["failed_checks"]),
            "relative_fid_improvement": fid_improvements,
            "recall_by_arm": recalls,
            "screen_pass": False,
            "generation_advantage_proven": False,
        },
        "causal_context": {
            "report": source_evidence["causal_discriminator"],
            "validation": source_evidence["causal_discriminator_validation"],
            "decision": "no_defensible_shared_intervention_selected",
            "shared_support_collapse_consistent": True,
            "common_cause_proven": False,
            "selected_candidate": None,
        },
        "scientific_judgment": {
            "class_index_mapping_mismatch": False,
            "mapping_can_explain_support_collapse": False,
            "shared_support_collapse_consistent": True,
            "common_cause_proven": False,
            "interpretation": (
                "The training loader derives lexicographic WNID indices that match the "
                "dataset mappings, torchvision category order, and timm synsets. The "
                "trainer, sampler, CFG path, model class embedding, and class-fidelity "
                "evaluator transport the same integer index; real validation calibration "
                "is strong. A class-index mapping mismatch therefore cannot explain the "
                "shared generated-support collapse, whose common cause remains unproven."
            ),
        },
        "physical_replay": {
            "performed": True,
            "source_file_count": len(source_evidence),
            "checkpoint_count": 2 * len(ARM_SPECS),
            "generated_sample_count": SAMPLES_PER_ARM * len(ARM_SPECS),
            "manifest_rows_streamed": manifest_summary["row_count"],
            "large_files_loaded_into_memory": False,
            "gpu_used": False,
        },
        "next_stage": {
            "route": "hold",
            "frozen_confirmation_preparation_allowed": False,
            "frozen_confirmation_launch_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "export_allowed": False,
            "release_allowed": False,
            "paper_integration_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }
    _assert_finite_tree(report, "label-conditioning pipeline audit")
    return report


def validate_audit(value: Mapping[str, Any]) -> dict[str, Any]:
    row = _object(value, "label-conditioning pipeline audit")
    if (
        row.get("schema_version") != SCHEMA
        or row.get("role") != ROLE
        or row.get("status") != "completed"
        or row.get("operational_status") != "pass"
        or row.get("scientific_status") != "hold"
        or row.get("terminal_status") != "hold"
        or row.get("decision")
        != "class_index_mapping_mismatch_ruled_out_support_collapse_unexplained"
        or row.get("generation_advantage_proven") is not False
        or row.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("label-conditioning pipeline audit contract differs")
    _git_identity(row.get("builder_git"), "builder Git")
    _identity(row.get("builder_script"), "builder script")
    execution_git = _git_identity(row.get("execution_git"), "execution Git")
    if execution_git != {
        "branch": EXECUTION_BRANCH,
        "revision": EXECUTION_REVISION,
        "tracked_dirty": False,
        "tree": EXECUTION_TREE,
    }:
        raise ValueError("audited execution Git differs")
    judgment = _object(row.get("scientific_judgment"), "scientific judgment")
    expected = {
        "class_index_mapping_mismatch": False,
        "mapping_can_explain_support_collapse": False,
        "shared_support_collapse_consistent": True,
        "common_cause_proven": False,
    }
    if any(judgment.get(key) is not value for key, value in expected.items()):
        raise ValueError("scientific judgment differs")
    arms = _object(row.get("arm_replays"), "arm replays")
    if set(arms) != set(ARM_SPECS):
        raise ValueError("audited arm set differs")
    recalls = _object(
        _object(row.get("screen_judgment"), "screen judgment").get("recall_by_arm"),
        "recall by arm",
    )
    if set(recalls) != set(ARM_SPECS) or any(value != 0.0 for value in recalls.values()):
        raise ValueError("audited shared recall collapse differs")
    evidence = _object(row.get("source_evidence"), "source evidence")
    for key, identity in evidence.items():
        _identity(identity, f"source evidence {key}")
    if _object(row.get("next_stage"), "next stage").get("route") != "hold":
        raise ValueError("audit next-stage route differs")
    if any(
        value is True
        for key, value in _object(row.get("next_stage"), "next stage").items()
        if key != "route"
    ):
        raise ValueError("audit next stage grants an authorization")
    _assert_finite_tree(row, "label-conditioning pipeline audit")
    return copy.deepcopy(row)


def _validation_basis(
    audit: Mapping[str, Any], audit_identity: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "audit": copy.deepcopy(audit_identity),
        "builder_git": copy.deepcopy(audit["builder_git"]),
        "builder_script": copy.deepcopy(audit["builder_script"]),
        "execution_git": copy.deepcopy(audit["execution_git"]),
        "source_evidence": copy.deepcopy(audit["source_evidence"]),
        "dataset_label_pipeline": copy.deepcopy(audit["dataset_label_pipeline"]),
        "execution_code_pipeline": copy.deepcopy(audit["execution_code_pipeline"]),
        "arm_replays": copy.deepcopy(audit["arm_replays"]),
        "scientific_judgment": copy.deepcopy(audit["scientific_judgment"]),
    }


def build_validation(
    *,
    audit: Mapping[str, Any],
    audit_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
    validator_script: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_audit(audit)
    audit_id = _identity(audit_identity, "audit")
    basis = _validation_basis(validated, audit_id)
    return {
        "schema_version": VALIDATION_SCHEMA,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "scientific_status": "hold",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "audit": audit_id,
        "builder_git": copy.deepcopy(validated["builder_git"]),
        "builder_script": copy.deepcopy(validated["builder_script"]),
        "validator_git": _git_identity(validator_git, "validator Git"),
        "validator_script": _identity(validator_script, "validator script"),
        "scientific_judgment": copy.deepcopy(validated["scientific_judgment"]),
        "physical_source_replay": {
            "performed": True,
            "source_file_count": len(validated["source_evidence"]),
            "recomputed_audit_canonical_sha256": _canonical_sha256(validated),
        },
        "validation_basis_sha256": _canonical_sha256(basis),
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def validate_validation(
    value: Mapping[str, Any],
    *,
    audit: Mapping[str, Any],
    audit_identity: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(value, "label-conditioning pipeline audit validation")
    expected = build_validation(
        audit=audit,
        audit_identity=audit_identity,
        validator_git=_git_identity(row.get("validator_git"), "validator Git"),
        validator_script=_identity(row.get("validator_script"), "validator script"),
    )
    if row != expected:
        raise ValueError("label-conditioning audit validation is not reproducible")
    return expected


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("output path must be absolute")
    _reject_symlink_chain(path, allow_missing_leaf=True)
    encoded = (
        json.dumps(
            payload,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise
    _, identity = _stable_read(path)
    return identity


def _persist_and_replay(
    path: Path, payload: Mapping[str, Any], *, name: str
) -> dict[str, Any]:
    identity = _write_exclusive(path, payload)
    encoded, observed_identity = _stable_read(path)
    if observed_identity != identity or _json_loads(encoded, name) != payload:
        raise RuntimeError(f"{name} did not replay after exclusive write")
    return identity


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--execution-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--controller-root", type=Path, required=True)
    parser.add_argument("--prelaunch-root", type=Path, required=True)
    parser.add_argument("--causal-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--categories-file", type=Path, required=True)
    parser.add_argument("--timm-synsets-file", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="build the immutable diagnostic audit")
    _add_common_arguments(build)
    validate = commands.add_parser(
        "validate", help="independently replay and validate the diagnostic audit"
    )
    _add_common_arguments(validate)
    validate.add_argument("--audit", type=Path, required=True)
    validate.add_argument("--audit-sha256", required=True)
    return parser


def _build_from_arguments(
    arguments: argparse.Namespace,
    *,
    builder_git: Mapping[str, Any],
    builder_script: Mapping[str, Any],
) -> dict[str, Any]:
    return _build_audit(
        builder_git=builder_git,
        builder_script=builder_script,
        execution_root=arguments.execution_root,
        screen_root=arguments.screen_root,
        controller_root=arguments.controller_root,
        prelaunch_root=arguments.prelaunch_root,
        causal_root=arguments.causal_root,
        dataset_root=arguments.dataset_root,
        categories_file=arguments.categories_file,
        timm_synsets_file=arguments.timm_synsets_file,
        calibration=arguments.calibration,
    )


def _build_command(arguments: argparse.Namespace) -> dict[str, Any]:
    builder_git, builder_script = _checkout_and_script_identity(
        arguments.project_root, arguments.script
    )
    report = _build_from_arguments(
        arguments, builder_git=builder_git, builder_script=builder_script
    )
    validate_audit(report)
    identity = _persist_and_replay(
        arguments.output, report, name="label-conditioning pipeline audit"
    )
    return {
        "command": "build",
        "status": "pass",
        "scientific_status": "hold",
        "decision": report["decision"],
        "output": identity,
        "source_file_count": len(report["source_evidence"]),
        "builder_git": builder_git,
    }


def _validate_command(arguments: argparse.Namespace) -> dict[str, Any]:
    validator_git, validator_script = _checkout_and_script_identity(
        arguments.project_root, arguments.script
    )
    expected_sha256 = _hex(
        arguments.audit_sha256, length=64, name="expected audit SHA256"
    )
    audit_bytes, audit_identity = _stable_read(arguments.audit)
    if audit_identity["sha256"] != expected_sha256:
        raise ValueError("audit SHA256 differs")
    audit = validate_audit(_json_loads(audit_bytes, "label-conditioning audit"))
    replayed = _build_from_arguments(
        arguments,
        builder_git=audit["builder_git"],
        builder_script=audit["builder_script"],
    )
    if replayed != audit:
        raise ValueError("independent physical replay does not reproduce the audit")
    receipt = build_validation(
        audit=audit,
        audit_identity=audit_identity,
        validator_git=validator_git,
        validator_script=validator_script,
    )
    validate_validation(receipt, audit=audit, audit_identity=audit_identity)
    identity = _persist_and_replay(
        arguments.output,
        receipt,
        name="label-conditioning pipeline audit validation",
    )
    persisted, _ = _stable_read(arguments.output)
    validate_validation(
        _json_loads(persisted, "label-conditioning pipeline audit validation"),
        audit=audit,
        audit_identity=audit_identity,
    )
    return {
        "command": "validate",
        "status": "pass",
        "scientific_status": "hold",
        "decision": audit["decision"],
        "audit": audit_identity,
        "output": identity,
        "source_file_count": len(audit["source_evidence"]),
        "validator_git": validator_git,
        "physical_source_replay": True,
    }


def main() -> int:
    arguments = _parser().parse_args()
    summary = (
        _build_command(arguments)
        if arguments.command == "build"
        else _validate_command(arguments)
    )
    print(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
