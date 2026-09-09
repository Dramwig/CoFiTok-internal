"""Build or validate a non-authorizing support-collapse causal discriminator.

This module intentionally uses only the Python standard library.  It replays
content-addressed diagnostic reports and emits an immutable, fail-closed
scientific discriminator.  It never loads checkpoints, launches a process,
uses a GPU, changes a controller, or authorizes a later stage.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import stat
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SCHEMA = "cofitok_generation_support_collapse_causal_discriminator_v1"
ROLE = "source_bound_support_collapse_causal_discriminator"
VALIDATION_SCHEMA = (
    "cofitok_generation_support_collapse_causal_discriminator_validation_v1"
)
VALIDATION_ROLE = (
    "content_addressed_support_collapse_causal_discriminator_validation"
)

PARENT_REVISION = "7ec178f3f1bf1d3e23ab899750754b47bbc4626c"
EXECUTION_REVISION = "89bcd9adb2a1e4625a9cd48dc2be82cbee8c6430"
EXECUTION_TREE = "46efd20cff489bccd799bb13c4155a0cc79e7649"
EXECUTION_BRANCH = "terminal-snr-execution-89bcd9a"

SOURCE_KEYS = (
    "arm_control_cofitok",
    "arm_control_dense_identity",
    "arm_endpoint0975_cofitok",
    "arm_endpoint0975_dense_identity",
    "capacity_hold_decision",
    "capacity_hold_validation",
    "capacity_result",
    "capacity_result_validation",
    "conditioning_control_cofitok_manifest",
    "conditioning_control_cofitok_report",
    "conditioning_control_dense_manifest",
    "conditioning_control_dense_report",
    "conditioning_ranked_cofitok_manifest",
    "conditioning_ranked_cofitok_report",
    "conditioning_ranked_dense_manifest",
    "conditioning_ranked_dense_report",
    "conditioning_ranking_postevaluation",
    "controller_log",
    "controller_status",
    "execution_authorization",
    "exposure_cofitok_1250_manifest",
    "exposure_cofitok_1250_report",
    "exposure_cofitok_2500_manifest",
    "exposure_cofitok_2500_report",
    "exposure_cofitok_5000_manifest",
    "exposure_cofitok_5000_report",
    "exposure_controller_log",
    "exposure_controller_status",
    "exposure_decision",
    "exposure_decision_validation",
    "exposure_dense_1250_manifest",
    "exposure_dense_1250_report",
    "exposure_dense_2500_manifest",
    "exposure_dense_2500_report",
    "exposure_dense_5000_manifest",
    "exposure_dense_5000_report",
    "exposure_result",
    "exposure_result_validation",
    "exposure_semantic_trajectory",
    "exposure_semantic_trajectory_replay_audit",
    "launch_receipt",
    "live_snapshot",
    "min_snr_physical_guard",
    "min_snr_result",
    "preparation",
    "prior_reassessment",
    "prior_reassessment_validation",
    "prior_terminal_reassessment",
    "prior_terminal_reassessment_validation",
    "quality_bridge_comparison",
    "quality_bridge_result",
    "quality_class_fidelity_qualification",
    "quality_cofitok_class_fidelity",
    "quality_cofitok_metrics",
    "quality_dense_class_fidelity",
    "quality_dense_metrics",
    "runtime_selection",
    "sampling_recovery_result",
    "semantic_control_cofitok_manifest",
    "semantic_control_cofitok_report",
    "semantic_control_dense_manifest",
    "semantic_control_dense_report",
    "semantic_residual_cofitok_manifest",
    "semantic_residual_cofitok_report",
    "semantic_residual_dense_manifest",
    "semantic_residual_dense_report",
    "semantic_residual_postevaluation",
    "storage_capacity",
    "terminal_result",
    "terminal_result_validation",
)

LOG_KEYS = {"controller_log", "exposure_controller_log"}
JSON_KEYS = set(SOURCE_KEYS) - LOG_KEYS

CONDITIONING = {
    "cofitok": {
        "control_report": "conditioning_control_cofitok_report",
        "control_manifest": "conditioning_control_cofitok_manifest",
        "intervention_report": "conditioning_ranked_cofitok_report",
        "intervention_manifest": "conditioning_ranked_cofitok_manifest",
    },
    "dense_identity": {
        "control_report": "conditioning_control_dense_report",
        "control_manifest": "conditioning_control_dense_manifest",
        "intervention_report": "conditioning_ranked_dense_report",
        "intervention_manifest": "conditioning_ranked_dense_manifest",
    },
}

SEMANTIC_RESIDUAL = {
    "cofitok": {
        "control_report": "semantic_control_cofitok_report",
        "control_manifest": "semantic_control_cofitok_manifest",
        "intervention_report": "semantic_residual_cofitok_report",
        "intervention_manifest": "semantic_residual_cofitok_manifest",
    },
    "dense_identity": {
        "control_report": "semantic_control_dense_report",
        "control_manifest": "semantic_control_dense_manifest",
        "intervention_report": "semantic_residual_dense_report",
        "intervention_manifest": "semantic_residual_dense_manifest",
    },
}

EXPOSURE = {
    "cofitok": {
        1250: ("exposure_cofitok_1250_report", "exposure_cofitok_1250_manifest"),
        2500: ("exposure_cofitok_2500_report", "exposure_cofitok_2500_manifest"),
        5000: ("exposure_cofitok_5000_report", "exposure_cofitok_5000_manifest"),
    },
    "dense_identity": {
        1250: ("exposure_dense_1250_report", "exposure_dense_1250_manifest"),
        2500: ("exposure_dense_2500_report", "exposure_dense_2500_manifest"),
        5000: ("exposure_dense_5000_report", "exposure_dense_5000_manifest"),
    },
}

CONDITIONING_TIMESTEPS = (100, 500, 900)
SEMANTIC_TIMESTEPS = (100, 500, 700, 900)
EXPOSURE_TIMESTEPS = (100, 500, 700, 900)
EXPOSURE_STEPS = (1250, 2500, 5000)

SIGNIFICANCE_LEVEL = 0.05
MIN_POSITIVE_SAMPLE_FRACTION = 0.75
MAX_CORRECT_MSE_RATIO = 1.02

# These are the existing conservative distribution-support floors used by the
# stability qualification family.  They are diagnostic safety floors here;
# passing them never grants a training or sampling permission.
SUPPORT_THRESHOLDS = {
    "min_precision": 0.10,
    "min_recall": 0.10,
    "max_precision_regression": 0.05,
    "max_recall_regression": 0.05,
    "min_predicted_class_fraction": 0.25,
    "min_normalized_predicted_class_entropy": 0.50,
}

AUTHORIZATION_BOUNDARY = {
    "decision_is_execution_authorization": False,
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

DECISION = "no_defensible_shared_intervention_selected"

CLAIM_BOUNDARY = {
    "diagnostic_only": True,
    "generates_new_samples": False,
    "authorizes_training": False,
    "authorizes_sampling": False,
    "replaces_formal_quality_gate": False,
    "shared_support_collapse_consistent": True,
    "common_cause_proven": False,
}

CANDIDATE_IDS = ("conditioning_ranking", "semantic_residual_alignment")

SENSITIVITY_SPECS = {
    "conditioning_ranking": {
        "control": {
            "cofitok": ("conditioning_control_cofitok_report", "conditioning_control_cofitok_manifest"),
            "dense_identity": ("conditioning_control_dense_report", "conditioning_control_dense_manifest"),
        },
        "intervention": {
            "cofitok": ("conditioning_ranked_cofitok_report", "conditioning_ranked_cofitok_manifest"),
            "dense_identity": ("conditioning_ranked_dense_report", "conditioning_ranked_dense_manifest"),
        },
        "all_timesteps": CONDITIONING_TIMESTEPS,
        "eligible_timesteps": (500, 900),
        "expected_samples": 8,
        "intervention_label": "ranked",
        "source_intervention_label": "ranked",
    },
    "semantic_residual_alignment": {
        "control": {
            "cofitok": ("semantic_control_cofitok_report", "semantic_control_cofitok_manifest"),
            "dense_identity": ("semantic_control_dense_report", "semantic_control_dense_manifest"),
        },
        "intervention": {
            "cofitok": ("semantic_residual_cofitok_report", "semantic_residual_cofitok_manifest"),
            "dense_identity": ("semantic_residual_dense_report", "semantic_residual_dense_manifest"),
        },
        "all_timesteps": SEMANTIC_TIMESTEPS,
        "eligible_timesteps": (500, 700, 900),
        "expected_samples": 8,
        "intervention_label": "residual",
        "source_intervention_label": "ranked",
    },
}

QUALITY_METRIC_FIELDS = (
    "frechet_inception_distance",
    "precision",
    "recall",
)
QUALITY_FIDELITY_FIELDS = (
    "predicted_class_fraction",
    "normalized_predicted_class_entropy",
    "top1_accuracy",
    "top5_accuracy",
)

LOCKED_ACTIVE_SHA256 = {
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
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return dict(value)


def _sequence(value: Any, name: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return list(value)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _positive(value: Any, name: str) -> float:
    result = _finite(value, name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _hex(value: Any, *, length: int, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} is not a lowercase {length}-character hex digest")
    return value


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


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


def _stable_read(path: Path) -> tuple[bytes, dict[str, Any]]:
    if not path.is_absolute():
        raise ValueError(f"source path must be absolute: {path}")
    _reject_symlink_chain(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"source is not a regular file: {path}")
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    bytes_read = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            chunks.append(block)
            digest.update(block)
            bytes_read += len(block)
    after = path.stat()
    if (
        bytes_read != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
        or after.st_ino != before.st_ino
        or after.st_dev != before.st_dev
    ):
        raise RuntimeError(f"source changed while being read: {path}")
    return b"".join(chunks), {
        "path": str(path),
        "bytes": bytes_read,
        "sha256": digest.hexdigest(),
    }


def _read_json(data: bytes, name: str) -> dict[str, Any]:
    try:
        payload = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not valid JSON") from exc
    return _object(payload, name)


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    if not isinstance(row["path"], str) or not row["path"].startswith("/"):
        raise ValueError(f"{name} path must be absolute")
    if isinstance(row["bytes"], bool) or not isinstance(row["bytes"], int):
        raise ValueError(f"{name} bytes must be an integer")
    if row["bytes"] < 1:
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
        raise ValueError(f"project root must be absolute: {project_root}")
    _reject_symlink_chain(project_root)
    if not project_root.is_dir():
        raise ValueError(f"project root is not a directory: {project_root}")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(project_root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    status = git("status", "--porcelain=v1", "--untracked-files=all")
    if status:
        raise ValueError(f"project root must be completely clean: {project_root}")
    branch = git("branch", "--show-current")
    if not branch:
        raise ValueError(f"project root must be on a named branch: {project_root}")
    return {
        "branch": branch,
        "revision": _hex(
            git("rev-parse", "HEAD^{commit}"), length=40, name="revision"
        ),
        "tracked_dirty": False,
        "tree": _hex(git("rev-parse", "HEAD^{tree}"), length=40, name="tree"),
    }


def _checkout_and_script_identity(
    project_root: Path, script_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind an exact tracked script to a completely clean named checkout."""
    checkout = _checkout_identity(project_root)
    if not script_path.is_absolute():
        raise ValueError(f"script path must be absolute: {script_path}")
    _reject_symlink_chain(script_path)
    if not script_path.is_file():
        raise ValueError(f"script path is not a regular file: {script_path}")

    root = project_root.resolve(strict=True)
    script = script_path.resolve(strict=True)
    try:
        relative = script.relative_to(root)
    except ValueError as exc:
        raise ValueError("script must be inside the project checkout") from exc
    git_path = relative.as_posix()
    subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", git_path],
        check=True,
        capture_output=True,
    )
    data, identity = _stable_read(script)
    committed = subprocess.run(
        ["git", "-C", str(root), "show", f"HEAD:{git_path}"],
        check=True,
        capture_output=True,
    ).stdout
    if committed != data:
        raise ValueError("tracked script bytes differ from the exact HEAD blob")
    return checkout, identity


def _load_sources(
    rows: Sequence[Sequence[str]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], dict[str, bytes]]:
    paths: dict[str, Path] = {}
    expected: dict[str, str] = {}
    for row in rows:
        if len(row) != 3:
            raise ValueError("each --source requires KEY PATH EXPECTED_SHA256")
        key, raw_path, digest = row
        if key in paths:
            raise ValueError(f"duplicate source key: {key}")
        paths[key] = Path(raw_path)
        expected[key] = _hex(digest, length=64, name=f"{key} expected SHA256")
    if set(paths) != set(SOURCE_KEYS):
        missing = sorted(set(SOURCE_KEYS) - set(paths))
        extra = sorted(set(paths) - set(SOURCE_KEYS))
        raise ValueError(f"source key set differs; missing={missing}, extra={extra}")

    payloads: dict[str, dict[str, Any]] = {}
    identities: dict[str, dict[str, Any]] = {}
    raw: dict[str, bytes] = {}
    for key in sorted(paths):
        data, descriptor = _stable_read(paths[key])
        if descriptor["sha256"] != expected[key]:
            raise ValueError(
                f"{key} SHA256 differs: expected {expected[key]}, "
                f"observed {descriptor['sha256']}"
            )
        raw[key] = data
        identities[key] = descriptor
        if key in JSON_KEYS:
            payloads[key] = _read_json(data, key)
    return payloads, identities, raw


def _require_false_fields(row: Mapping[str, Any], fields: Sequence[str], name: str) -> None:
    for field in fields:
        if row.get(field) is not False:
            raise ValueError(f"{name}.{field} must be false")


def _row_key(row: Mapping[str, Any]) -> tuple[int, int, int, int, int]:
    return (
        int(row["sample_index"]),
        int(row["timestep"]),
        int(row["correct_label"]),
        int(row["wrong_label"]),
        int(row["noise_seed"]),
    )


def _sample_key(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row["sample_index"]),
        int(row["correct_label"]),
        int(row["wrong_label"]),
        int(row["noise_seed"]),
    )


def _condition_mse(row: Mapping[str, Any], condition: str) -> float:
    try:
        value = float(row["conditions"][condition]["epsilon_mse_to_noise"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row lacks {condition} epsilon MSE") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"row {condition} epsilon MSE is invalid")
    return value


def _margin(row: Mapping[str, Any], comparison: str) -> float:
    reference = {"versus_null": "null", "versus_wrong": "wrong"}[comparison]
    correct = _condition_mse(row, "correct")
    reference_mse = _condition_mse(row, reference)
    return (reference_mse - correct) / reference_mse


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("cannot average non-finite values")
    return sum(float(value) for value in values) / len(values)


def _median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot take an empty median")
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _population_std(values: Sequence[float], mean: float | None = None) -> float:
    centre = _mean(values) if mean is None else float(mean)
    return math.sqrt(_mean([(float(value) - centre) ** 2 for value in values]))


def _one_sided_binomial_pvalue(successes: int, trials: int) -> float:
    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("invalid sign-test counts")
    return sum(math.comb(trials, value) for value in range(successes, trials + 1)) / (
        2**trials
    )


def _signed_summary(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot summarize empty values")
    mean = _mean(numbers)
    positive = sum(value > 0.0 for value in numbers)
    negative = sum(value < 0.0 for value in numbers)
    zero = len(numbers) - positive - negative
    direction = "positive" if mean > 0.0 else "negative" if mean < 0.0 else "zero"
    return {
        "count": len(numbers),
        "mean": mean,
        "median": _median(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "population_std": _population_std(numbers, mean),
        "positive_count": positive,
        "negative_count": negative,
        "zero_count": zero,
        "positive_fraction": positive / len(numbers),
        "negative_fraction": negative / len(numbers),
        "direction": direction,
        "one_sided_sign_test_pvalue": _one_sided_binomial_pvalue(positive, len(numbers)),
    }


def _mse_summary(values: Sequence[float]) -> dict[str, Any]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("cannot summarize empty MSE values")
    mean = _mean(numbers)
    return {
        "count": len(numbers),
        "mean": mean,
        "median": _median(numbers),
        "min": min(numbers),
        "max": max(numbers),
        "population_std": _population_std(numbers, mean),
    }


def _slope(points: Sequence[tuple[float, float]]) -> float:
    if len(points) < 2:
        raise ValueError("at least two points are required for a slope")
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    xbar = _mean(xs)
    ybar = _mean(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    if denominator <= 0.0:
        raise ValueError("exposure slope has degenerate x values")
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator


def _manifest_report_binding(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    *,
    report_identity: Mapping[str, Any],
    name: str,
) -> None:
    if (
        manifest.get("schema_version") != 1
        or manifest.get("role") != "generation_conditioning_sensitivity_manifest"
        or manifest.get("report") != report_identity["path"]
    ):
        raise ValueError(f"{name} manifest/report path binding differs")
    declared_manifest = report.get("manifest")
    if _identity(declared_manifest, f"{name} report manifest") != manifest_identity:
        raise ValueError(f"{name} report manifest identity differs")
    for field in ("checkpoint", "dataset", "git", "request"):
        if manifest.get(field) != report.get(field):
            raise ValueError(f"{name} manifest/report {field} binding differs")


def _validate_sensitivity_report(
    report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_identity: Mapping[str, Any],
    report_identity: Mapping[str, Any],
    *,
    name: str,
    expected_timesteps: Sequence[int],
    expected_samples: int,
) -> dict[str, Any]:
    if (
        report.get("schema_version") != 1
        or report.get("role") != "generation_conditioning_sensitivity_report"
        or report.get("status") != "completed"
        or report.get("weights") != "ema"
    ):
        raise ValueError(f"{name} sensitivity report state differs")
    boundary = report.get("claim_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("diagnostic_only") is not True
        or any(
            boundary.get(field) is not False
            for field in (
                "generates_new_samples",
                "authorizes_training",
                "authorizes_sampling",
                "replaces_formal_quality_gate",
            )
        )
    ):
        raise ValueError(f"{name} sensitivity report is authorizing")
    request = _object(report.get("request"), f"{name} request")
    if tuple(int(value) for value in request.get("timesteps", [])) != tuple(expected_timesteps):
        raise ValueError(f"{name} timesteps differ")
    if int(request.get("num_samples", -1)) != expected_samples:
        raise ValueError(f"{name} sample count differs")
    if manifest.get("request") != request:
        raise ValueError(f"{name} manifest request differs")
    if manifest.get("report") != report_identity["path"]:
        raise ValueError(f"{name} manifest does not bind report path")
    _manifest_report_binding(
        report,
        manifest,
        manifest_identity,
        report_identity=report_identity,
        name=name,
    )
    rows = _sequence(report.get("sample_rows"), f"{name} sample rows")
    expected_count = expected_samples * len(expected_timesteps)
    if len(rows) != expected_count:
        raise ValueError(f"{name} row count differs")
    keys = [_row_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} has duplicate row identities")
    expected_keys = {
        (
         sample,
         int(timestep),
         (int(request["start_label"]) + sample) % 1000,
         (
             int(request["start_label"])
             + sample
             + int(request["wrong_label_offset"])
         )
         % 1000,
         int(request["noise_seed"]) + sample,
        )
        for sample in range(expected_samples)
        for timestep in expected_timesteps
    }
    if set(keys) != expected_keys:
        raise ValueError(f"{name} row identities differ")
    for row in rows:
        for condition in ("correct", "null", "wrong"):
            _condition_mse(row, condition)
        for comparison in ("versus_null", "versus_wrong"):
            recomputed = _margin(row, comparison)
            reported = float(row["correct_relative_mse_improvement"][comparison])
            if not math.isfinite(reported) or not math.isclose(
                recomputed, reported, rel_tol=0.0, abs_tol=1e-12
            ):
                raise ValueError(f"{name} reported margin differs from raw MSE")
        if row.get("correct_better", {}).get("than_null") is not (
            _condition_mse(row, "correct") < _condition_mse(row, "null")
        ):
            raise ValueError(f"{name} correct-vs-null flag differs")
        if row.get("correct_better", {}).get("than_wrong") is not (
            _condition_mse(row, "correct") < _condition_mse(row, "wrong")
        ):
            raise ValueError(f"{name} correct-vs-wrong flag differs")
    return {
        "request": copy.deepcopy(request),
        "timesteps": list(expected_timesteps),
        "sample_count": expected_samples,
        "row_count": len(rows),
        "rows": rows,
    }


def _load_report_with_manifest(
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    report_key: str,
    manifest_key: str,
    *,
    name: str,
    expected_timesteps: Sequence[int],
    expected_samples: int,
) -> dict[str, Any]:
    report = payloads[report_key]
    manifest = payloads[manifest_key]
    return _validate_sensitivity_report(
        report,
        manifest,
        identities[manifest_key],
        identities[report_key],
        name=name,
        expected_timesteps=expected_timesteps,
        expected_samples=expected_samples,
    )


def _check_paired_row_keys(reports: Mapping[str, Mapping[str, Any]], name: str) -> None:
    keys: set[tuple[int, int, int, int, int]] | None = None
    for method, report in reports.items():
        current = {_row_key(row) for row in report["rows"]}
        if keys is None:
            keys = current
        elif current != keys:
            raise ValueError(f"{name} row identities differ between methods")


def _validate_active_and_prior_sources(
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    raw: Mapping[str, bytes],
) -> dict[str, Any]:
    for key, expected_sha in LOCKED_ACTIVE_SHA256.items():
        if identities[key]["sha256"] != expected_sha:
            raise ValueError(f"locked active source hash differs for {key}")
    terminal = payloads["terminal_result"]
    if (
        terminal.get("schema_version") != "cofitok_generation_terminal_snr_screen_result_v1"
        or terminal.get("role") != "source_bound_terminal_snr_screen_scientific_result"
        or terminal.get("status") != "completed"
        or terminal.get("operational_status") != "pass"
        or terminal.get("scientific_status") != "hold"
        or terminal.get("terminal_status") != "hold"
        or terminal.get("screen_pass") is not False
        or terminal.get("generation_advantage_proven") is not False
    ):
        raise ValueError("active terminal-SNR result state differs")
    terminal_validation = payloads["terminal_result_validation"]
    if (
        terminal_validation.get("status") != "pass"
        or terminal_validation.get("scientific_status") != "hold"
        or terminal_validation.get("screen_pass") is not False
        or _identity(terminal_validation.get("result"), "terminal result validation")
        != identities["terminal_result"]
    ):
        raise ValueError("active terminal-SNR result validation differs")
    controller = payloads["controller_status"]
    if (
        controller.get("schema_version") != 1
        or controller.get("role") != "generation_terminal_snr_screen_controller"
        or controller.get("status") != "completed"
        or controller.get("stage") != "complete"
        or controller.get("terminal_status") != "hold"
        or controller.get("generation_advantage_proven") is not False
    ):
        raise ValueError("active terminal controller state differs")
    _require_false_fields(
        controller,
        [
            "frozen_confirmation_allowed",
            "full_training_launch_allowed",
            "full_300k_launch_allowed",
            "promotion_allowed",
            "export_allowed",
            "release_allowed",
            "process_signals_allowed",
        ],
        "terminal controller",
    )
    controller_log = raw["controller_log"].decode("utf-8", errors="strict")
    if controller_log.count("generated 1000/1000") != 4:
        raise ValueError("active controller log lacks four completed samplings")
    if controller_log.count("scripts/train_generation.py") != 4:
        raise ValueError("active controller log lacks four training arms")
    preparation = payloads["preparation"]
    if (
        preparation.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_preparation_v1"
        or preparation.get("status") != "prepared"
        or preparation.get("generation_advantage_proven") is not False
    ):
        raise ValueError("active terminal preparation state differs")
    authorization = payloads["execution_authorization"]
    if (
        authorization.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_execution_authorization_v1"
        or authorization.get("status") != "authorized"
    ):
        raise ValueError("active terminal authorization state differs")
    launch = payloads["launch_receipt"]
    if (
        launch.get("schema_version")
        != "cofitok_generation_terminal_snr_screen_launch_receipt_v1"
        or launch.get("status") != "pass"
    ):
        raise ValueError("active terminal launch receipt state differs")
    launch_sources = _object(launch.get("source_evidence"), "terminal launch sources")
    for key in (
        "preparation",
        "execution_authorization",
        "runtime_selection",
        "live_snapshot",
        "storage_capacity",
    ):
        if _identity(launch_sources.get(key), f"launch {key}") != identities[key]:
            raise ValueError(f"launch receipt does not bind {key}")
    if payloads["runtime_selection"].get("status") != "selected":
        raise ValueError("active runtime selection state differs")
    if payloads["live_snapshot"].get("status") != "pass":
        raise ValueError("active live snapshot state differs")
    if payloads["storage_capacity"].get("status") != "pass":
        raise ValueError("active storage-capacity state differs")
    for key, arm in (
        ("arm_control_cofitok", "control_cofitok"),
        ("arm_control_dense_identity", "control_dense_identity"),
        ("arm_endpoint0975_cofitok", "endpoint0975_cofitok"),
        ("arm_endpoint0975_dense_identity", "endpoint0975_dense_identity"),
    ):
        validation = payloads[key]
        if (
            validation.get("schema_version")
            != "cofitok_generation_terminal_snr_screen_arm_validation_v1"
            or validation.get("role") != "physical_terminal_snr_screen_arm_validation"
            or validation.get("status") != "pass"
            or validation.get("arm") != arm
            or validation.get("execution_git")
            != {
                "branch": EXECUTION_BRANCH,
                "revision": EXECUTION_REVISION,
                "tracked_dirty": False,
                "tree": EXECUTION_TREE,
            }
        ):
            raise ValueError(f"{arm} arm validation state differs")
        result_arms = _object(
            _object(terminal.get("source_evidence"), "terminal source evidence").get(
                "arm_validations"
            ),
            "terminal result arm validations",
        )
        if _identity(result_arms.get(arm), f"terminal {arm} identity") != identities[key]:
            raise ValueError(f"terminal result does not bind {arm}")

    historical_controller = payloads["exposure_controller_status"]
    if (
        historical_controller.get("role")
        != "generation_exposure_capacity_continuation_controller"
        or historical_controller.get("status") != "failed"
        or historical_controller.get("stage") != "failed"
        or historical_controller.get("terminal_status") != "hold"
        or historical_controller.get("generation_advantage_proven") is not False
    ):
        raise ValueError("historical exposure controller state differs")
    historical_log = raw["exposure_controller_log"].decode("utf-8", errors="strict")
    if "ValueError: cofitok runtime environment differs from authorization" not in historical_log:
        raise ValueError("historical exposure failure evidence differs")

    prior = payloads["prior_reassessment"]
    if (
        prior.get("schema_version") != "cofitok_generation_terminal_snr_intervention_reassessment_v1"
        or prior.get("decision") != "no_defensible_shared_intervention_selected"
        or prior.get("scientific_status") != "hold"
        or prior.get("generation_advantage_proven") is not False
    ):
        raise ValueError("prior terminal reassessment state differs")
    prior_validation = payloads["prior_reassessment_validation"]
    if (
        prior_validation.get("status") != "pass"
        or _identity(prior_validation.get("decision"), "prior reassessment validation")
        != identities["prior_reassessment"]
    ):
        raise ValueError("prior terminal reassessment validation differs")

    prior_terminal = payloads["prior_terminal_reassessment"]
    if (
        prior_terminal.get("scientific_status") not in {"hold", "bounded_screen_selected_not_executed"}
        or prior_terminal.get("generation_advantage_proven") is not False
    ):
        raise ValueError("prior objective reassessment state differs")
    for key in (
        "capacity_result",
        "capacity_result_validation",
        "capacity_hold_decision",
        "capacity_hold_validation",
        "sampling_recovery_result",
        "min_snr_result",
        "min_snr_physical_guard",
        "exposure_result",
        "exposure_result_validation",
        "exposure_decision",
        "exposure_decision_validation",
    ):
        if payloads[key].get("generation_advantage_proven") is not False:
            raise ValueError(f"{key} unexpectedly authorizes an advantage")
    return {
        "terminal_result": copy.deepcopy(identities["terminal_result"]),
        "terminal_result_validation": copy.deepcopy(identities["terminal_result_validation"]),
        "controller_status": copy.deepcopy(identities["controller_status"]),
        "controller_log": copy.deepcopy(identities["controller_log"]),
        "prelaunch_evidence": {
            key: copy.deepcopy(identities[key])
            for key in (
                "preparation",
                "execution_authorization",
                "launch_receipt",
                "runtime_selection",
                "live_snapshot",
                "storage_capacity",
            )
        },
        "arm_validations": {
            arm: copy.deepcopy(identities[key])
            for key, arm in (
                ("arm_control_cofitok", "control_cofitok"),
                ("arm_control_dense_identity", "control_dense_identity"),
                ("arm_endpoint0975_cofitok", "endpoint0975_cofitok"),
                ("arm_endpoint0975_dense_identity", "endpoint0975_dense_identity"),
            )
        },
        "historical_exposure_controller": {
            "status": "failed_historical_evidence",
            "controller_status": copy.deepcopy(identities["exposure_controller_status"]),
            "controller_log": copy.deepcopy(identities["exposure_controller_log"]),
        },
    }


def _assert_finite_tree(value: Any, name: str) -> None:
    """Reject non-finite numeric values anywhere in a derived report."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite_tree(item, f"{name}.{key}")
        return
    if isinstance(value, Sequence):
        for index, item in enumerate(value):
            _assert_finite_tree(item, f"{name}[{index}]")
        return
    raise ValueError(f"{name} contains an unsupported value type")


def _diagnostic_boundary(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    boundary = payload.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        boundary = payload.get("authorization_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError(f"{name} has no claim/authorization boundary")
    for field in (
        "authorizes_training",
        "authorizes_sampling",
        "authorizes_full_training",
        "authorizes_300k_training",
        "authorizes_checkpoint_promotion",
        "authorizes_export",
        "authorizes_release",
        "training_launch_allowed",
        "sampling_launch_allowed",
        "full_training_launch_allowed",
        "full_300k_launch_allowed",
        "promotion_allowed",
        "export_allowed",
        "release_allowed",
        "process_signals_allowed",
        "gpu_execution_allowed",
        "remote_mutation_allowed",
    ):
        if field in boundary and boundary[field] is not False:
            raise ValueError(f"{name}.{field} is authorizing")
    return boundary


def _validate_support_diagnostic_sources(
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the non-authorizing diagnostic families before recomputation."""
    exact_roles = {
        "conditioning_ranking_postevaluation": "generation_conditioning_ranking_four_arm_postevaluation",
        "semantic_residual_postevaluation": "generation_semantic_residual_alignment_four_arm_postevaluation",
        "exposure_semantic_trajectory": "generation_exposure_semantic_trajectory_report",
        "exposure_semantic_trajectory_replay_audit": "generation_exposure_semantic_trajectory_replay_audit",
        "quality_bridge_result": "stability_full_data_quality_bridge_result",
        "quality_bridge_comparison": "stability_full_data_quality_bridge_comparison",
        "quality_class_fidelity_qualification": "generation_class_fidelity_qualification",
    }
    for key, role in exact_roles.items():
        payload = _object(payloads[key], key)
        if payload.get("role") != role:
            raise ValueError(f"{key} role differs")
        _diagnostic_boundary(payload, key)
        if (
            "generation_advantage_proven" in payload
            and payload.get("generation_advantage_proven") is not False
        ):
            raise ValueError(f"{key} unexpectedly proves generation advantage")

    if payloads["conditioning_ranking_postevaluation"].get("status") != "completed":
        raise ValueError("conditioning-ranking post-evaluation is not complete")
    if payloads["semantic_residual_postevaluation"].get("status") != "completed":
        raise ValueError("semantic-residual post-evaluation is not complete")
    if payloads["exposure_semantic_trajectory"].get("status") != "completed":
        raise ValueError("exposure trajectory is not complete")
    trajectory_audit = payloads["exposure_semantic_trajectory_replay_audit"]
    if (
        trajectory_audit.get("status") != "pass"
        or trajectory_audit.get("canonical_byte_exact_replay") is not True
        or trajectory_audit.get("scientific_decision_preserved")
        != payloads["exposure_semantic_trajectory"].get("decision")
    ):
        raise ValueError("exposure trajectory replay audit is not a preserved replay")
    if payloads["quality_bridge_result"].get("status") != "completed":
        raise ValueError("quality bridge result is not complete")
    if payloads["quality_bridge_comparison"].get("status") != "hold":
        raise ValueError("quality bridge comparison is not held")
    if payloads["quality_bridge_comparison"].get("decision") != (
        "terminal_system_evidence_complete_without_qualified_matched_advantage"
    ):
        raise ValueError("quality bridge comparison decision differs")
    if payloads["quality_class_fidelity_qualification"].get("status") != "hold":
        raise ValueError("class-fidelity qualification is not held")

    return {
        key: copy.deepcopy(identities[key])
        for key in exact_roles
    }


def _aggregate_sensitivity(
    validated: Mapping[str, Any],
    *,
    eligible_timesteps: Sequence[int],
    name: str,
) -> dict[str, Any]:
    """Recompute raw margins and aggregate timestep rows by held-out sample."""
    rows = [_object(row, f"{name} row") for row in validated["rows"]]
    eligible = tuple(int(value) for value in eligible_timesteps)
    all_timesteps = tuple(int(value) for value in validated["timesteps"])
    if not set(eligible).issubset(all_timesteps):
        raise ValueError(f"{name} eligible timestep set is invalid")

    timestep_summaries: dict[str, Any] = {}
    for timestep in all_timesteps:
        selected = [row for row in rows if int(row["timestep"]) == timestep]
        if not selected:
            raise ValueError(f"{name} has no rows at timestep {timestep}")
        timestep_summaries[str(timestep)] = {
            "eligible": timestep in eligible,
            "row_count": len(selected),
            "correct_mse": _mse_summary(
                [_condition_mse(row, "correct") for row in selected]
            ),
            "margins": {
                comparison: _signed_summary(
                    [_margin(row, comparison) for row in selected]
                )
                for comparison in ("versus_null", "versus_wrong")
            },
        }

    grouped: dict[tuple[int, int, int, int], dict[str, list[float]]] = {}
    timestep_sets: dict[tuple[int, int, int, int], set[int]] = {}
    for row in rows:
        timestep = int(row["timestep"])
        if timestep not in eligible:
            continue
        key = _sample_key(row)
        grouped.setdefault(
            key,
            {
                "correct_mse": [],
                "null_mse": [],
                "wrong_mse": [],
                "versus_null": [],
                "versus_wrong": [],
            },
        )
        timestep_sets.setdefault(key, set()).add(timestep)
        grouped[key]["correct_mse"].append(_condition_mse(row, "correct"))
        grouped[key]["null_mse"].append(_condition_mse(row, "null"))
        grouped[key]["wrong_mse"].append(_condition_mse(row, "wrong"))
        grouped[key]["versus_null"].append(_margin(row, "versus_null"))
        grouped[key]["versus_wrong"].append(_margin(row, "versus_wrong"))

    if len(grouped) != int(validated["sample_count"]):
        raise ValueError(f"{name} independent sample count differs")
    sample_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        observed_timesteps = sorted(timestep_sets[key])
        if tuple(observed_timesteps) != eligible:
            raise ValueError(f"{name} sample has incomplete eligible timesteps")
        values = grouped[key]
        condition_mse = {
            condition: _mean(values[f"{condition}_mse"])
            for condition in ("correct", "null", "wrong")
        }
        aggregated_margins = {
            "versus_null": (
                condition_mse["null"] - condition_mse["correct"]
            )
            / condition_mse["null"],
            "versus_wrong": (
                condition_mse["wrong"] - condition_mse["correct"]
            )
            / condition_mse["wrong"],
        }
        sample_rows.append(
            {
                "sample_index": key[0],
                "correct_label": key[1],
                "wrong_label": key[2],
                "noise_seed": key[3],
                "eligible_timesteps": observed_timesteps,
                "correct_mse": condition_mse["correct"],
                "condition_mse": condition_mse,
                "margins": {
                    comparison: _mean(values[comparison])
                    for comparison in ("versus_null", "versus_wrong")
                },
                "aggregated_condition_mse_margins": aggregated_margins,
            }
        )

    return {
        "name": name,
        "independent_unit": "held_out_validation_image",
        "all_timesteps": list(all_timesteps),
        "eligible_timesteps": list(eligible),
        "descriptive_only_timesteps": [
            timestep for timestep in all_timesteps if timestep not in eligible
        ],
        "row_count": len(rows),
        "sample_count": len(sample_rows),
        "timestep_rows_aggregated_before_sign_test": True,
        "held_out_sample_margin_aggregation": "mean_raw_timestep_margin",
        "trajectory_compatible_margin_aggregation": (
            "mean_condition_mse_across_eligible_timesteps_then_compute_margin"
        ),
        "raw_margin_formula": {
            "versus_null": "(null_epsilon_mse - correct_epsilon_mse) / null_epsilon_mse",
            "versus_wrong": "(wrong_epsilon_mse - correct_epsilon_mse) / wrong_epsilon_mse",
        },
        "timestep_summaries": timestep_summaries,
        "per_sample": sample_rows,
        "aggregate": {
            "correct_mse": _mse_summary(
                [row["correct_mse"] for row in sample_rows]
            ),
            "margins": {
                comparison: _signed_summary(
                    [row["margins"][comparison] for row in sample_rows]
                )
                for comparison in ("versus_null", "versus_wrong")
            },
            "aggregated_condition_mse_margins": {
                comparison: _signed_summary(
                    [
                        row["aggregated_condition_mse_margins"][comparison]
                        for row in sample_rows
                    ]
                )
                for comparison in ("versus_null", "versus_wrong")
            },
        },
    }


def _paired_intervention_effect(
    control: Mapping[str, Any],
    intervention: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    control_rows = {
        _sample_key(row): row for row in control["per_sample"]
    }
    intervention_rows = {
        _sample_key(row): row for row in intervention["per_sample"]
    }
    if set(control_rows) != set(intervention_rows):
        raise ValueError(f"{name} paired sample identities differ")
    paired_rows: list[dict[str, Any]] = []
    for key in sorted(control_rows):
        before = control_rows[key]
        after = intervention_rows[key]
        ratios = _positive(after["correct_mse"], f"{name} intervention correct MSE") / _positive(
            before["correct_mse"], f"{name} control correct MSE"
        )
        paired_rows.append(
            {
                "sample_index": key[0],
                "correct_label": key[1],
                "wrong_label": key[2],
                "noise_seed": key[3],
                "correct_mse_ratio": ratios,
                "margin_improvement": {
                    comparison: float(after["margins"][comparison])
                    - float(before["margins"][comparison])
                    for comparison in ("versus_null", "versus_wrong")
                },
            }
        )
    return {
        "paired_unit": "held_out_validation_image",
        "sample_count": len(paired_rows),
        "per_sample": paired_rows,
        "margin_improvement": {
            comparison: _signed_summary(
                [row["margin_improvement"][comparison] for row in paired_rows]
            )
            for comparison in ("versus_null", "versus_wrong")
        },
        "correct_mse_ratio": _mse_summary(
            [row["correct_mse_ratio"] for row in paired_rows]
        ),
        "mean_correct_mse_ratio": _mean(
            [row["correct_mse_ratio"] for row in paired_rows]
        ),
        "max_correct_mse_ratio": max(
            row["correct_mse_ratio"] for row in paired_rows
        ),
        "aggregate_correct_mse_ratio": _positive(
            intervention["aggregate"]["correct_mse"]["mean"],
            f"{name} intervention aggregate correct MSE",
        )
        / _positive(
            control["aggregate"]["correct_mse"]["mean"],
            f"{name} control aggregate correct MSE",
        ),
    }


def _validate_quality_sources(
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method, metric_key, fidelity_key in (
        ("cofitok", "quality_cofitok_metrics", "quality_cofitok_class_fidelity"),
        ("dense_identity", "quality_dense_metrics", "quality_dense_class_fidelity"),
    ):
        metrics_report = _object(payloads[metric_key], metric_key)
        fidelity_report = _object(payloads[fidelity_key], fidelity_key)
        if (
            metrics_report.get("schema_version") != 3
            or metrics_report.get("role") != "generation_directory_metrics_report"
            or metrics_report.get("status") != "completed"
        ):
            raise ValueError(f"{method} metrics source state differs")
        if (
            fidelity_report.get("schema_version") != 2
            or fidelity_report.get("role") != "generation_class_fidelity_report"
            or fidelity_report.get("status") != "completed"
        ):
            raise ValueError(f"{method} class-fidelity source state differs")
        counts = _object(metrics_report.get("counts"), f"{method} metric counts")
        if (
            int(counts.get("generated_image_count", -1)) != 10000
            or int(counts.get("real_image_count", -1)) != 50000
        ):
            raise ValueError(f"{method} quality sample counts differ")
        metric_values = _object(metrics_report.get("metrics"), f"{method} metrics")
        fidelity_values = _object(
            fidelity_report.get("metrics"), f"{method} class-fidelity metrics"
        )
        values: dict[str, float] = {}
        for field in QUALITY_METRIC_FIELDS:
            values[field] = _finite(metric_values.get(field), f"{method}.{field}")
        for field in QUALITY_FIDELITY_FIELDS:
            values[field] = _finite(
                fidelity_values.get(field), f"{method}.{field}"
            )
        if int(fidelity_values.get("sample_count", -1)) != 10000:
            raise ValueError(f"{method} class-fidelity sample count differs")
        if int(fidelity_values.get("num_classes", -1)) != 1000:
            raise ValueError(f"{method} class-fidelity class count differs")
        if not 0.0 <= values["precision"] <= 1.0 or not 0.0 <= values["recall"] <= 1.0:
            raise ValueError(f"{method} precision/recall is outside [0,1]")
        if not 0.0 <= values["predicted_class_fraction"] <= 1.0:
            raise ValueError(f"{method} predicted-class fraction is outside [0,1]")
        if not 0.0 <= values["normalized_predicted_class_entropy"] <= 1.0:
            raise ValueError(f"{method} normalized entropy is outside [0,1]")
        if not 0.0 <= values["top1_accuracy"] <= 1.0 or not 0.0 <= values["top5_accuracy"] <= 1.0:
            raise ValueError(f"{method} class accuracy is outside [0,1]")
        methods[method] = {
            "metrics_report": copy.deepcopy(identities[metric_key]),
            "class_fidelity_report": copy.deepcopy(identities[fidelity_key]),
            "generated_image_count": 10000,
            "real_image_count": 50000,
            "sample_count": 10000,
            "values": values,
        }

    cofitok = methods["cofitok"]["values"]
    dense = methods["dense_identity"]["values"]
    regression: dict[str, float] = {}
    for field in ("precision", "recall"):
        denominator = dense[field]
        if denominator <= 0.0:
            raise ValueError(f"dense {field} cannot be zero for regression")
        regression[field] = max(0.0, (dense[field] - cofitok[field]) / denominator)
    floor_checks: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        values = methods[method]["values"]
        floor_checks[f"{method}_precision_floor"] = {
            "observed": values["precision"],
            "threshold": SUPPORT_THRESHOLDS["min_precision"],
            "pass": values["precision"] >= SUPPORT_THRESHOLDS["min_precision"],
        }
        floor_checks[f"{method}_recall_floor"] = {
            "observed": values["recall"],
            "threshold": SUPPORT_THRESHOLDS["min_recall"],
            "pass": values["recall"] >= SUPPORT_THRESHOLDS["min_recall"],
        }
        floor_checks[f"{method}_predicted_class_fraction_floor"] = {
            "observed": values["predicted_class_fraction"],
            "threshold": SUPPORT_THRESHOLDS["min_predicted_class_fraction"],
            "pass": values["predicted_class_fraction"]
            >= SUPPORT_THRESHOLDS["min_predicted_class_fraction"],
        }
        floor_checks[f"{method}_normalized_entropy_floor"] = {
            "observed": values["normalized_predicted_class_entropy"],
            "threshold": SUPPORT_THRESHOLDS["min_normalized_predicted_class_entropy"],
            "pass": values["normalized_predicted_class_entropy"]
            >= SUPPORT_THRESHOLDS["min_normalized_predicted_class_entropy"],
        }
    for field in ("precision", "recall"):
        floor_checks[f"matched_{field}_regression"] = {
            "observed": regression[field],
            "threshold": SUPPORT_THRESHOLDS[f"max_{field}_regression"],
            "pass": regression[field] <= SUPPORT_THRESHOLDS[f"max_{field}_regression"],
        }
    complete = all(
        isinstance(methods[method].get("values"), Mapping)
        and set(methods[method]["values"]) == set(
            QUALITY_METRIC_FIELDS + QUALITY_FIDELITY_FIELDS
        )
        for method in methods
    )
    non_regressive = complete and all(item["pass"] for item in floor_checks.values())
    shared_support_collapse_consistent = all(
        methods[method]["values"]["recall"] < SUPPORT_THRESHOLDS["min_recall"]
        and methods[method]["values"]["top1_accuracy"] < 0.01
        and methods[method]["values"]["top5_accuracy"] < 0.05
        for method in ("cofitok", "dense_identity")
    )
    return {
        "methods": methods,
        "matched_regression": regression,
        "thresholds": copy.deepcopy(SUPPORT_THRESHOLDS),
        "checks": floor_checks,
        "complete": complete,
        "non_regressive": non_regressive,
        "shared_support_collapse_consistent": shared_support_collapse_consistent,
        "common_cause_proven": False,
        "interpretation": (
            "Both methods show the same low-recall/class-fidelity support-collapse pattern; "
            "this is consistent with a shared support problem but does not identify or prove a common cause."
        ),
    }


def _validate_postevaluation_binding(
    payload: Mapping[str, Any],
    *,
    candidate_id: str,
    calculated: Mapping[str, Any],
) -> None:
    methods = _object(payload.get("methods"), f"{candidate_id} post-evaluation methods")
    for method in ("cofitok", "dense_identity"):
        source = _object(methods.get(method), f"{candidate_id} {method} post-evaluation")
        expected = calculated[method]
        existing_mse = _object(source.get("correct_mse"), f"{candidate_id} {method} MSE")
        if not math.isclose(
            _finite(existing_mse.get("control_mean"), "post-evaluation control MSE"),
            expected["control"]["aggregate"]["correct_mse"]["mean"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            _finite(existing_mse.get(f"{SENSITIVITY_SPECS[candidate_id]['source_intervention_label']}_mean"), "post-evaluation intervention MSE"),
            expected["intervention"]["aggregate"]["correct_mse"]["mean"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{candidate_id} {method} MSE replay differs")
        existing_ratio = _finite(
            existing_mse.get(f"{SENSITIVITY_SPECS[candidate_id]['source_intervention_label']}_to_control_ratio"),
            "post-evaluation MSE ratio",
        )
        if not math.isclose(
            existing_ratio,
            expected["paired"]["aggregate_correct_mse_ratio"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"{candidate_id} {method} MSE ratio replay differs")
        absolute = _object(
            source.get(f"{SENSITIVITY_SPECS[candidate_id]['source_intervention_label']}_absolute"),
            f"{candidate_id} {method} absolute margins",
        )
        paired = _object(
            source.get(f"{SENSITIVITY_SPECS[candidate_id]['source_intervention_label']}_minus_control"),
            f"{candidate_id} {method} paired margins",
        )
        for comparison in ("versus_null", "versus_wrong"):
            existing_abs = _finite(
                _object(absolute.get(comparison), "absolute margin").get("mean"),
                "post-evaluation absolute margin",
            )
            existing_pair = _finite(
                _object(paired.get(comparison), "paired margin").get("mean"),
                "post-evaluation paired margin",
            )
            if not math.isclose(
                existing_abs,
                expected["intervention"]["aggregate"]["margins"][comparison]["mean"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ) or not math.isclose(
                existing_pair,
                expected["paired"]["margin_improvement"][comparison]["mean"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"{candidate_id} {method} margin replay differs")


def _build_candidate_evaluation(
    candidate_id: str,
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    support_quality: Mapping[str, Any],
) -> dict[str, Any]:
    spec = SENSITIVITY_SPECS[candidate_id]
    methods: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        control_report, control_manifest = spec["control"][method]
        intervention_report, intervention_manifest = spec["intervention"][method]
        control = _load_report_with_manifest(
            payloads,
            identities,
            control_report,
            control_manifest,
            name=f"{candidate_id} {method} control",
            expected_timesteps=spec["all_timesteps"],
            expected_samples=spec["expected_samples"],
        )
        intervention = _load_report_with_manifest(
            payloads,
            identities,
            intervention_report,
            intervention_manifest,
            name=f"{candidate_id} {method} intervention",
            expected_timesteps=spec["all_timesteps"],
            expected_samples=spec["expected_samples"],
        )
        _check_paired_row_keys(
            {"control": control, "intervention": intervention},
            f"{candidate_id} {method}",
        )
        control_aggregate = _aggregate_sensitivity(
            control,
            eligible_timesteps=spec["eligible_timesteps"],
            name=f"{candidate_id} {method} control",
        )
        intervention_aggregate = _aggregate_sensitivity(
            intervention,
            eligible_timesteps=spec["eligible_timesteps"],
            name=f"{candidate_id} {method} intervention",
        )
        paired = _paired_intervention_effect(
            control_aggregate,
            intervention_aggregate,
            name=f"{candidate_id} {method}",
        )
        methods[method] = {
            "sources": {
                "control_report": copy.deepcopy(identities[control_report]),
                "control_manifest": copy.deepcopy(identities[control_manifest]),
                "intervention_report": copy.deepcopy(identities[intervention_report]),
                "intervention_manifest": copy.deepcopy(identities[intervention_manifest]),
            },
            "control": control_aggregate,
            "intervention": intervention_aggregate,
            "paired": paired,
        }

    posteval_key = (
        "conditioning_ranking_postevaluation"
        if candidate_id == "conditioning_ranking"
        else "semantic_residual_postevaluation"
    )
    _validate_postevaluation_binding(
        payloads[posteval_key], candidate_id=candidate_id, calculated=methods
    )

    method_gates: dict[str, Any] = {}
    for method, values in methods.items():
        absolute_margins = values["intervention"]["aggregate"]["margins"]
        paired_margins = values["paired"]["margin_improvement"]
        ratio = values["paired"]["aggregate_correct_mse_ratio"]
        method_gates[method] = {
            "absolute_intervention_margins_positive": {
                comparison: absolute_margins[comparison]["mean"] > 0.0
                for comparison in ("versus_null", "versus_wrong")
            },
            "positive_independent_sample_fraction_at_least": {
                comparison: absolute_margins[comparison]["positive_fraction"]
                >= MIN_POSITIVE_SAMPLE_FRACTION
                for comparison in ("versus_null", "versus_wrong")
            },
            "paired_margin_improvement_positive": {
                comparison: paired_margins[comparison]["mean"] > 0.0
                for comparison in ("versus_null", "versus_wrong")
            },
            "correct_mse_ratio_within_limit": ratio <= MAX_CORRECT_MSE_RATIO,
            "correct_mse_ratio": ratio,
        }
        method_gates[method]["pass"] = all(
            list(method_gates[method]["absolute_intervention_margins_positive"].values())
            + list(method_gates[method]["positive_independent_sample_fraction_at_least"].values())
            + list(method_gates[method]["paired_margin_improvement_positive"].values())
            + [method_gates[method]["correct_mse_ratio_within_limit"]]
        )

    cross_method: dict[str, Any] = {}
    for comparison in ("versus_null", "versus_wrong"):
        absolute = {
            method: methods[method]["intervention"]["aggregate"]["margins"][comparison]["mean"]
            for method in methods
        }
        paired = {
            method: methods[method]["paired"]["margin_improvement"][comparison]["mean"]
            for method in methods
        }
        cross_method[comparison] = {
            "absolute_intervention_mean": absolute,
            "paired_intervention_minus_control_mean": paired,
            "common_absolute_direction": (
                (absolute["cofitok"] > 0.0 and absolute["dense_identity"] > 0.0)
                or (absolute["cofitok"] < 0.0 and absolute["dense_identity"] < 0.0)
            ),
            "common_paired_direction": (
                (paired["cofitok"] > 0.0 and paired["dense_identity"] > 0.0)
                or (paired["cofitok"] < 0.0 and paired["dense_identity"] < 0.0)
            ),
            "difference_in_differences_cofitok_minus_dense": paired["cofitok"]
            - paired["dense_identity"],
        }
    gate_values = {
        "absolute_intervention_margins_positive_both_methods_and_comparisons": all(
            method_gates[method]["absolute_intervention_margins_positive"][comparison]
            for method in methods
            for comparison in ("versus_null", "versus_wrong")
        ),
        "positive_independent_sample_fraction_at_least_0p75_both_methods_and_comparisons": all(
            method_gates[method]["positive_independent_sample_fraction_at_least"][comparison]
            for method in methods
            for comparison in ("versus_null", "versus_wrong")
        ),
        "paired_margin_improvement_positive_both_methods_and_comparisons": all(
            method_gates[method]["paired_margin_improvement_positive"][comparison]
            for method in methods
            for comparison in ("versus_null", "versus_wrong")
        ),
        "correct_mse_ratio_no_worse_than_1p02_both_methods": all(
            method_gates[method]["correct_mse_ratio_within_limit"] for method in methods
        ),
        "support_quality_evidence_complete": bool(support_quality["complete"]),
        "support_quality_evidence_non_regressive": bool(
            support_quality["non_regressive"]
        ),
    }
    gate_values["candidate_pass"] = all(gate_values.values())
    return {
        "id": candidate_id,
        "intervention_label": spec["intervention_label"],
        "all_timesteps": list(spec["all_timesteps"]),
        "eligible_timesteps": list(spec["eligible_timesteps"]),
        "expected_samples": spec["expected_samples"],
        "methods": methods,
        "method_gates": method_gates,
        "cross_method": cross_method,
        "gates": gate_values,
    }


def _validate_exposure_sources(
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    for method in ("cofitok", "dense_identity"):
        per_step: dict[str, Any] = {}
        validated_reports: dict[int, dict[str, Any]] = {}
        for step in EXPOSURE_STEPS:
            report_key, manifest_key = EXPOSURE[method][step]
            validated_reports[step] = _load_report_with_manifest(
                payloads,
                identities,
                report_key,
                manifest_key,
                name=f"exposure {method} step {step}",
                expected_timesteps=EXPOSURE_TIMESTEPS,
                expected_samples=16,
            )
            aggregate = _aggregate_sensitivity(
                validated_reports[step],
                eligible_timesteps=(500, 700, 900),
                name=f"exposure {method} step {step}",
            )
            per_step[str(step)] = {
                "sources": {
                    "report": copy.deepcopy(identities[report_key]),
                    "manifest": copy.deepcopy(identities[manifest_key]),
                },
                "aggregate": aggregate["aggregate"],
                "sample_count": aggregate["sample_count"],
                "per_sample": aggregate["per_sample"],
                "timestep_summaries": aggregate["timestep_summaries"],
            }

        trajectory_method = _object(
            _object(payloads["exposure_semantic_trajectory"].get("methods"), "trajectory methods").get(method),
            f"exposure trajectory {method}",
        )
        trajectory = _object(trajectory_method.get("trajectory"), f"exposure {method} trajectory")
        if [int(x) for x in trajectory.get("steps", [])] != list(EXPOSURE_STEPS):
            raise ValueError(f"exposure {method} step sequence differs")
        if int(trajectory.get("sample_count", -1)) != 16:
            raise ValueError(f"exposure {method} sample count differs")

        metric_points = {
            "correct_mse": [
                (float(step), per_step[str(step)]["aggregate"]["correct_mse"]["mean"])
                for step in EXPOSURE_STEPS
            ],
            "versus_null": [
                (
                    float(step),
                    per_step[str(step)]["aggregate"][
                        "aggregated_condition_mse_margins"
                    ]["versus_null"]["mean"],
                )
                for step in EXPOSURE_STEPS
            ],
            "versus_wrong": [
                (
                    float(step),
                    per_step[str(step)]["aggregate"][
                        "aggregated_condition_mse_margins"
                    ]["versus_wrong"]["mean"],
                )
                for step in EXPOSURE_STEPS
            ],
        }
        slopes = {metric: _slope(points) for metric, points in metric_points.items()}
        first = per_step[str(EXPOSURE_STEPS[0])]["aggregate"]
        last = per_step[str(EXPOSURE_STEPS[-1])]["aggregate"]
        rows_by_step = {
            step: {
                _sample_key(row): row
                for row in per_step[str(step)]["per_sample"]
            }
            for step in EXPOSURE_STEPS
        }
        sample_keys = set(rows_by_step[EXPOSURE_STEPS[0]])
        if any(set(rows_by_step[step]) != sample_keys for step in EXPOSURE_STEPS):
            raise ValueError(f"exposure {method} paired sample identities differ")
        per_sample_trajectory: list[dict[str, Any]] = []
        for key in sorted(sample_keys):
            correct_by_step = {
                str(step): float(rows_by_step[step][key]["correct_mse"])
                for step in EXPOSURE_STEPS
            }
            semantic_by_step = {
                comparison: {
                    str(step): float(
                        rows_by_step[step][key][
                            "aggregated_condition_mse_margins"
                        ][comparison]
                    )
                    for step in EXPOSURE_STEPS
                }
                for comparison in ("versus_null", "versus_wrong")
            }
            initial_mse = _positive(
                correct_by_step[str(EXPOSURE_STEPS[0])],
                f"exposure {method} initial sample MSE",
            )
            final_mse = _positive(
                correct_by_step[str(EXPOSURE_STEPS[-1])],
                f"exposure {method} final sample MSE",
            )
            per_sample_trajectory.append(
                {
                    "sample_index": key[0],
                    "correct_label": key[1],
                    "wrong_label": key[2],
                    "noise_seed": key[3],
                    "correct_mse_by_step": correct_by_step,
                    "correct_mse_reduction_1250_to_5000": (
                        initial_mse - final_mse
                    )
                    / initial_mse,
                    "correct_mse_strictly_improves_each_interval": all(
                        correct_by_step[str(left)] > correct_by_step[str(right)]
                        for left, right in zip(EXPOSURE_STEPS, EXPOSURE_STEPS[1:])
                    ),
                    "semantic_advantage_by_step": semantic_by_step,
                    "semantic_advantage_change_1250_to_5000": {
                        comparison: semantic_by_step[comparison][
                            str(EXPOSURE_STEPS[-1])
                        ]
                        - semantic_by_step[comparison][str(EXPOSURE_STEPS[0])]
                        for comparison in ("versus_null", "versus_wrong")
                    },
                    "semantic_advantage_strictly_improves_each_interval": {
                        comparison: all(
                            semantic_by_step[comparison][str(left)]
                            < semantic_by_step[comparison][str(right)]
                            for left, right in zip(
                                EXPOSURE_STEPS, EXPOSURE_STEPS[1:]
                            )
                        )
                        for comparison in ("versus_null", "versus_wrong")
                    },
                }
            )

        source_sample_rows = {
            _sample_key(row): _object(row, f"exposure {method} trajectory row")
            for row in _sequence(
                trajectory.get("per_sample"), f"exposure {method} trajectory rows"
            )
        }
        if set(source_sample_rows) != sample_keys:
            raise ValueError(f"exposure {method} source trajectory identities differ")
        for derived in per_sample_trajectory:
            key = _sample_key(derived)
            source = source_sample_rows[key]
            for step in EXPOSURE_STEPS:
                if not math.isclose(
                    _finite(
                        _object(
                            source.get("correct_mse_by_step"),
                            "source exposure correct MSE",
                        ).get(str(step)),
                        "source exposure correct MSE",
                    ),
                    derived["correct_mse_by_step"][str(step)],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"exposure {method} source sample MSE replay differs"
                    )
                for comparison in ("versus_null", "versus_wrong"):
                    if not math.isclose(
                        _finite(
                            _object(
                                _object(
                                    source.get("semantic_advantage_by_step"),
                                    "source exposure semantic trajectory",
                                ).get(comparison),
                                "source exposure semantic comparison",
                            ).get(str(step)),
                            "source exposure semantic margin",
                        ),
                        derived["semantic_advantage_by_step"][comparison][str(step)],
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    ):
                        raise ValueError(
                            f"exposure {method} source semantic replay differs"
                        )
            if not math.isclose(
                _finite(
                    source.get("correct_mse_reduction_1250_to_5000"),
                    "source exposure sample MSE reduction",
                ),
                derived["correct_mse_reduction_1250_to_5000"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    f"exposure {method} source sample MSE reduction differs"
                )
            if source.get("correct_mse_strictly_improves_each_interval") is not derived[
                "correct_mse_strictly_improves_each_interval"
            ]:
                raise ValueError(f"exposure {method} source MSE ordering differs")
            source_change = _object(
                source.get("semantic_advantage_change_1250_to_5000"),
                "source exposure semantic change",
            )
            source_strict = _object(
                source.get("semantic_advantage_strictly_improves_each_interval"),
                "source exposure semantic ordering",
            )
            for comparison in ("versus_null", "versus_wrong"):
                if not math.isclose(
                    _finite(
                        source_change.get(comparison),
                        "source exposure semantic change",
                    ),
                    derived["semantic_advantage_change_1250_to_5000"][comparison],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        f"exposure {method} source semantic change differs"
                    )
                if source_strict.get(comparison) is not derived[
                    "semantic_advantage_strictly_improves_each_interval"
                ][comparison]:
                    raise ValueError(
                        f"exposure {method} source semantic ordering differs"
                    )

        relative_reduction = _signed_summary(
            [
                row["correct_mse_reduction_1250_to_5000"]
                for row in per_sample_trajectory
            ]
        )
        semantic_change = {
            comparison: _signed_summary(
                [
                    row["semantic_advantage_change_1250_to_5000"][comparison]
                    for row in per_sample_trajectory
                ]
            )
            for comparison in ("versus_null", "versus_wrong")
        }
        endpoint_change = {
            "correct_mse": last["correct_mse"]["mean"] - first["correct_mse"]["mean"],
            "correct_mse_absolute_reduction": first["correct_mse"]["mean"]
            - last["correct_mse"]["mean"],
            "correct_mse_mean_relative_reduction": relative_reduction["mean"],
            "versus_null": last["aggregated_condition_mse_margins"][
                "versus_null"
            ]["mean"]
            - first["aggregated_condition_mse_margins"]["versus_null"]["mean"],
            "versus_wrong": last["aggregated_condition_mse_margins"][
                "versus_wrong"
            ]["mean"]
            - first["aggregated_condition_mse_margins"]["versus_wrong"]["mean"],
        }
        ratio = last["correct_mse"]["mean"] / first["correct_mse"]["mean"]

        existing_reduction = _object(
            trajectory.get("denoising_correct_mse_reduction"),
            f"exposure {method} denoising reduction",
        )
        existing_change = _object(
            trajectory.get("semantic_advantage_change"),
            f"exposure {method} semantic change",
        )
        if not math.isclose(
            _finite(existing_reduction.get("mean"), "exposure reduction"),
            relative_reduction["mean"],
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            _finite(trajectory.get("correct_mse_final_to_initial_ratio"), "exposure MSE ratio"),
            ratio,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"exposure {method} correct-MSE replay differs")
        for comparison in ("versus_null", "versus_wrong"):
            if not math.isclose(
                _finite(existing_change[comparison].get("mean"), "exposure semantic change"),
                semantic_change[comparison]["mean"],
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ValueError(f"exposure {method} semantic-change replay differs")

        for source_summary, derived_summary, name in (
            (existing_reduction, relative_reduction, "MSE reduction"),
            *(
                (
                    _object(existing_change.get(comparison), "source semantic summary"),
                    semantic_change[comparison],
                    f"{comparison} semantic change",
                )
                for comparison in ("versus_null", "versus_wrong")
            ),
        ):
            for field in (
                "mean",
                "median",
                "positive_count",
                "positive_fraction",
                "one_sided_sign_test_pvalue",
            ):
                source_value = source_summary.get(field)
                derived_value = derived_summary[field]
                if isinstance(derived_value, int):
                    matches = source_value == derived_value
                else:
                    matches = math.isclose(
                        _finite(source_value, f"source exposure {name} {field}"),
                        float(derived_value),
                        rel_tol=0.0,
                        abs_tol=1e-12,
                    )
                if not matches:
                    raise ValueError(
                        f"exposure {method} source {name} summary differs"
                    )

        strict_counts = {
            "correct_mse": sum(
                row["correct_mse_strictly_improves_each_interval"]
                for row in per_sample_trajectory
            ),
            "semantic_advantage": {
                comparison: sum(
                    row["semantic_advantage_strictly_improves_each_interval"][
                        comparison
                    ]
                    for row in per_sample_trajectory
                )
                for comparison in ("versus_null", "versus_wrong")
            },
        }
        if trajectory.get("strict_interval_improvement_counts") != strict_counts:
            raise ValueError(f"exposure {method} strict improvement counts differ")

        methods[method] = {
            "steps": list(EXPOSURE_STEPS),
            "eligible_timesteps": [500, 700, 900],
            "independent_unit": "held_out_validation_image",
            "per_step": per_step,
            "per_sample_trajectory": per_sample_trajectory,
            "slopes": slopes,
            "endpoint_change_1250_to_5000": endpoint_change,
            "correct_mse_relative_reduction": relative_reduction,
            "semantic_advantage_change": semantic_change,
            "correct_mse_final_to_initial_ratio": ratio,
            "strict_interval_improvement_counts": strict_counts,
            "trajectory_gates_replayed": copy.deepcopy(trajectory.get("gates")),
            "source_trajectory": copy.deepcopy(identities["exposure_semantic_trajectory"]),
            "source_replay_audit": copy.deepcopy(
                identities["exposure_semantic_trajectory_replay_audit"]
            ),
        }

    cross_method: dict[str, Any] = {}
    for metric in ("correct_mse", "versus_null", "versus_wrong"):
        values = {
            method: methods[method]["slopes"][metric] for method in methods
        }
        changes = {
            method: methods[method]["endpoint_change_1250_to_5000"][metric]
            for method in methods
        }
        cross_method[metric] = {
            "slopes": values,
            "endpoint_changes": changes,
            "same_slope_direction": (
                (values["cofitok"] > 0.0 and values["dense_identity"] > 0.0)
                or (values["cofitok"] < 0.0 and values["dense_identity"] < 0.0)
            ),
            "same_endpoint_direction": (
                (changes["cofitok"] > 0.0 and changes["dense_identity"] > 0.0)
                or (changes["cofitok"] < 0.0 and changes["dense_identity"] < 0.0)
            ),
            "cofitok_minus_dense_endpoint_change": changes["cofitok"]
            - changes["dense_identity"],
        }
    return {"methods": methods, "cross_method": cross_method}


def _source_identity_groups(
    identities: Mapping[str, Mapping[str, Any]],
    active_prior: Mapping[str, Any],
    diagnostic_sources: Mapping[str, Any],
) -> dict[str, Any]:
    groups = {
        "active_terminal_and_prelaunch": copy.deepcopy(
            active_prior["terminal_result"]
        ),
        "active_terminal_result_validation": copy.deepcopy(
            active_prior["terminal_result_validation"]
        ),
        "active_controller": {
            "status": copy.deepcopy(active_prior["controller_status"]),
            "log": copy.deepcopy(active_prior["controller_log"]),
            "arm_validations": copy.deepcopy(active_prior["arm_validations"]),
            "prelaunch_evidence": copy.deepcopy(active_prior["prelaunch_evidence"]),
        },
        "historical_and_prior": copy.deepcopy(
            active_prior["historical_exposure_controller"]
        ),
        "diagnostic_reports": copy.deepcopy(diagnostic_sources),
        "all_bound_sources": {
            key: copy.deepcopy(identities[key]) for key in sorted(identities)
        },
    }
    return groups


def _build_discriminator(
    *,
    payloads: Mapping[str, Mapping[str, Any]],
    identities: Mapping[str, Mapping[str, Any]],
    raw: Mapping[str, bytes],
    builder_git: Mapping[str, Any],
    builder_script: Mapping[str, Any],
) -> dict[str, Any]:
    active_prior = _validate_active_and_prior_sources(payloads, identities, raw)
    diagnostic_sources = _validate_support_diagnostic_sources(payloads, identities)
    support_quality = _validate_quality_sources(payloads, identities)
    candidates = [
        _build_candidate_evaluation(
            candidate_id,
            payloads=payloads,
            identities=identities,
            support_quality=support_quality,
        )
        for candidate_id in CANDIDATE_IDS
    ]
    exposure = _validate_exposure_sources(payloads, identities)
    selected = [candidate for candidate in candidates if candidate["gates"]["candidate_pass"]]
    selected_candidate = selected[0]["id"] if len(selected) == 1 else None
    scientific_status = "pass" if len(selected) == 1 else "hold"
    decision = (
        "one_source_compatible_shared_intervention_qualified"
        if len(selected) == 1
        else DECISION
    )

    cross_method_candidate = {
        candidate["id"]: copy.deepcopy(candidate["cross_method"])
        for candidate in candidates
    }
    claim_boundary = copy.deepcopy(CLAIM_BOUNDARY)
    claim_boundary["shared_support_collapse_consistent"] = bool(
        support_quality["shared_support_collapse_consistent"]
    )
    claim_boundary["common_cause_proven"] = False
    report = {
        "schema_version": SCHEMA,
        "role": ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": scientific_status,
        "terminal_status": "hold" if selected_candidate is None else "qualified_candidate_hold",
        "generation_advantage_proven": False,
        "decision": decision,
        "selected_candidate": selected_candidate,
        "candidate_count": len(candidates),
        "builder_git": _git_identity(builder_git, "builder Git"),
        "builder_script": _identity(builder_script, "builder script"),
        "protocol": {
            "independent_unit": "held_out_validation_image",
            "raw_margin_formula": {
                "versus_null": "(null_epsilon_mse - correct_epsilon_mse) / null_epsilon_mse",
                "versus_wrong": "(wrong_epsilon_mse - correct_epsilon_mse) / wrong_epsilon_mse",
            },
            "descriptive_only_timesteps": {"conditioning_ranking": [100], "semantic_residual_alignment": [100]},
            "eligible_timesteps": {"conditioning_ranking": [500, 900], "semantic_residual_alignment": [500, 700, 900]},
            "timestep_rows_are_not_independent_units": True,
            "positive_independent_sample_fraction_threshold": MIN_POSITIVE_SAMPLE_FRACTION,
            "maximum_correct_mse_ratio": MAX_CORRECT_MSE_RATIO,
            "support_quality_thresholds": copy.deepcopy(SUPPORT_THRESHOLDS),
        },
        "source_evidence": _source_identity_groups(
            identities, active_prior, diagnostic_sources
        ),
        "support_quality": support_quality,
        "candidate_evaluations": candidates,
        "exposure_trajectory": exposure,
        "cross_method_evidence": {
            "candidate_intervention_effects": cross_method_candidate,
            "exposure": copy.deepcopy(exposure["cross_method"]),
            "shared_support_collapse_consistent": bool(
                support_quality["shared_support_collapse_consistent"]
            ),
            "common_cause_proven": False,
            "difference_in_differences_are_descriptive_only": True,
        },
        "scientific_judgment": {
            "absolute_intervention_margins_passed_for_selected_candidate": (
                len(selected) == 1
            ),
            "paired_intervention_effects_passed_for_selected_candidate": (
                len(selected) == 1
            ),
            "support_quality_is_non_regressive": bool(support_quality["non_regressive"]),
            "shared_support_collapse_consistent": bool(
                support_quality["shared_support_collapse_consistent"]
            ),
            "common_cause_proven": False,
            "interpretation": (
                "The held-out sensitivity replays recompute raw conditioning margins and paired intervention effects. "
                "Neither predeclared candidate satisfies every margin, paired-effect, MSE-ratio, and support-quality gate; "
                "the shared low-recall pattern is consistent with support collapse, but a common cause is not proven."
            ),
        },
        "next_stage": {
            "route": "hold" if selected_candidate is None else "candidate_qualification_requires_separate_authorization",
            "selected_candidate": selected_candidate,
            "confirmation_preparation_allowed": False,
            "confirmation_launch_allowed": False,
            "large_capacity_readiness_preparation_allowed": False,
            "full_training_preparation_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "export_allowed": False,
            "release_allowed": False,
        },
        "claim_boundary": claim_boundary,
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }
    validate_discriminator(report)
    return report


def validate_discriminator(report: Mapping[str, Any]) -> dict[str, Any]:
    row = _object(report, "support-collapse causal discriminator")
    if (
        row.get("schema_version") != SCHEMA
        or row.get("role") != ROLE
        or row.get("status") != "completed"
        or row.get("operational_status") != "pass"
        or row.get("generation_advantage_proven") is not False
        or row.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("support-collapse discriminator contract differs")
    if row.get("scientific_status") not in {"hold", "pass"}:
        raise ValueError("support-collapse discriminator scientific status differs")
    _git_identity(row.get("builder_git"), "builder Git")
    _identity(row.get("builder_script"), "builder script")
    _object(row.get("protocol"), "discriminator protocol")
    support = _object(row.get("support_quality"), "support quality")
    if support.get("common_cause_proven") is not False:
        raise ValueError("support quality common-cause flag differs")
    candidates = _sequence(row.get("candidate_evaluations"), "candidate evaluations")
    if [
        _object(candidate, "candidate").get("id") for candidate in candidates
    ] != list(CANDIDATE_IDS):
        raise ValueError("candidate ordering differs")
    selected = [
        _object(candidate, "candidate")
        for candidate in candidates
        if _object(candidate, "candidate").get("gates", {}).get("candidate_pass") is True
    ]
    if (row.get("selected_candidate") is not None) != (len(selected) == 1):
        raise ValueError("selected-candidate cardinality differs")
    if len(selected) == 1 and row.get("selected_candidate") != selected[0].get("id"):
        raise ValueError("selected-candidate identity differs")
    if len(selected) != 1:
        if row.get("scientific_status") != "hold" or row.get("selected_candidate") is not None:
            raise ValueError("held discriminator must have no selected candidate")
        if row.get("decision") != DECISION:
            raise ValueError("held discriminator decision differs")
        if row.get("terminal_status") != "hold":
            raise ValueError("held discriminator terminal status differs")
    else:
        if row.get("scientific_status") != "pass":
            raise ValueError("qualified discriminator status differs")
    claim = _object(row.get("claim_boundary"), "claim boundary")
    if claim.get("diagnostic_only") is not True or claim.get("common_cause_proven") is not False:
        raise ValueError("claim boundary differs")
    for field in (
        "generates_new_samples",
        "authorizes_training",
        "authorizes_sampling",
        "replaces_formal_quality_gate",
        "shared_support_collapse_consistent",
    ):
        if field not in claim or not isinstance(claim[field], bool):
            raise ValueError(f"claim boundary field missing: {field}")
    _assert_finite_tree(row, "discriminator")
    return copy.deepcopy(row)


def _validation_basis(report: Mapping[str, Any], decision_identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": copy.deepcopy(decision_identity),
        "builder_git": copy.deepcopy(report["builder_git"]),
        "builder_script": copy.deepcopy(report["builder_script"]),
        "source_evidence": copy.deepcopy(report["source_evidence"]),
        "support_quality": copy.deepcopy(report["support_quality"]),
        "candidate_evaluations": copy.deepcopy(report["candidate_evaluations"]),
        "exposure_trajectory": copy.deepcopy(report["exposure_trajectory"]),
        "scientific_judgment": copy.deepcopy(report["scientific_judgment"]),
    }


def build_validation(
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
    validator_git: Mapping[str, Any],
    validator_script: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_discriminator(decision)
    decision_id = _identity(decision_identity, "discriminator")
    validator = _git_identity(validator_git, "validator Git")
    script = _identity(validator_script, "validator script")
    basis = _validation_basis(validated, decision_id)
    return {
        "schema_version": VALIDATION_SCHEMA,
        "role": VALIDATION_ROLE,
        "status": "pass",
        "scientific_status": validated["scientific_status"],
        "terminal_status": validated["terminal_status"],
        "generation_advantage_proven": False,
        "decision": decision_id,
        "selected_candidate": validated["selected_candidate"],
        "builder_git": copy.deepcopy(validated["builder_git"]),
        "builder_script": copy.deepcopy(validated["builder_script"]),
        "validator_git": validator,
        "validator_script": script,
        "source_evidence": copy.deepcopy(validated["source_evidence"]),
        "physical_source_replay": {
            "performed": True,
            "source_count": len(SOURCE_KEYS),
            "recomputed_decision_canonical_sha256": _canonical_sha256(validated),
        },
        "validation_basis_sha256": _canonical_sha256(basis),
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }


def validate_validation(
    receipt: Mapping[str, Any],
    *,
    decision: Mapping[str, Any],
    decision_identity: Mapping[str, Any],
) -> dict[str, Any]:
    row = _object(receipt, "support-collapse discriminator validation")
    expected = build_validation(
        decision=decision,
        decision_identity=decision_identity,
        validator_git=_git_identity(row.get("validator_git"), "validator Git"),
        validator_script=_identity(row.get("validator_script"), "validator script"),
    )
    if row != expected:
        raise ValueError("support-collapse discriminator validation is not reproducible")
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
    path: Path,
    payload: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    identity = _write_exclusive(path, payload)
    encoded, observed_identity = _stable_read(path)
    if observed_identity != identity:
        raise RuntimeError(f"{name} identity changed after exclusive write")
    observed = _read_json(encoded, name)
    if observed != payload:
        raise RuntimeError(f"{name} bytes do not replay to the built payload")
    return identity


def _add_source_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source",
        action="append",
        nargs=3,
        required=True,
        metavar=("KEY", "ABSOLUTE_PATH", "EXPECTED_SHA256"),
        help="repeat once for every required content-addressed source",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser(
        "build", help="replay every source and write the immutable discriminator"
    )
    build.add_argument("--project-root", type=Path, required=True)
    build.add_argument("--script", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    _add_source_arguments(build)

    validate = subparsers.add_parser(
        "validate",
        help="independently replay every source and write an immutable validation receipt",
    )
    validate.add_argument("--project-root", type=Path, required=True)
    validate.add_argument("--script", type=Path, required=True)
    validate.add_argument("--decision", type=Path, required=True)
    validate.add_argument("--decision-sha256", required=True)
    validate.add_argument("--output", type=Path, required=True)
    _add_source_arguments(validate)
    return parser


def _build_command(arguments: argparse.Namespace) -> dict[str, Any]:
    builder_git, builder_script = _checkout_and_script_identity(
        arguments.project_root, arguments.script
    )
    payloads, identities, raw = _load_sources(arguments.source)
    report = _build_discriminator(
        payloads=payloads,
        identities=identities,
        raw=raw,
        builder_git=builder_git,
        builder_script=builder_script,
    )
    validate_discriminator(report)
    identity = _persist_and_replay(
        arguments.output, report, name="support-collapse causal discriminator"
    )
    return {
        "command": "build",
        "status": "pass",
        "scientific_status": report["scientific_status"],
        "selected_candidate": report["selected_candidate"],
        "output": identity,
        "source_count": len(identities),
        "builder_git": builder_git,
    }


def _validate_command(arguments: argparse.Namespace) -> dict[str, Any]:
    validator_git, validator_script = _checkout_and_script_identity(
        arguments.project_root, arguments.script
    )
    expected_decision_sha256 = _hex(
        arguments.decision_sha256,
        length=64,
        name="expected discriminator SHA256",
    )
    decision_bytes, decision_identity = _stable_read(arguments.decision)
    if decision_identity["sha256"] != expected_decision_sha256:
        raise ValueError(
            "discriminator SHA256 differs: "
            f"expected {expected_decision_sha256}, "
            f"observed {decision_identity['sha256']}"
        )
    decision = _read_json(decision_bytes, "support-collapse causal discriminator")
    validated = validate_discriminator(decision)

    payloads, identities, raw = _load_sources(arguments.source)
    replayed = _build_discriminator(
        payloads=payloads,
        identities=identities,
        raw=raw,
        builder_git=validated["builder_git"],
        builder_script=validated["builder_script"],
    )
    if replayed != validated:
        raise ValueError(
            "independent physical source replay does not reproduce the discriminator"
        )

    receipt = build_validation(
        decision=validated,
        decision_identity=decision_identity,
        validator_git=validator_git,
        validator_script=validator_script,
    )
    validate_validation(
        receipt,
        decision=validated,
        decision_identity=decision_identity,
    )
    receipt_identity = _persist_and_replay(
        arguments.output,
        receipt,
        name="support-collapse causal discriminator validation",
    )
    persisted_bytes, _ = _stable_read(arguments.output)
    persisted_receipt = _read_json(
        persisted_bytes, "support-collapse causal discriminator validation"
    )
    validate_validation(
        persisted_receipt,
        decision=validated,
        decision_identity=decision_identity,
    )
    return {
        "command": "validate",
        "status": "pass",
        "scientific_status": validated["scientific_status"],
        "selected_candidate": validated["selected_candidate"],
        "decision": decision_identity,
        "output": receipt_identity,
        "source_count": len(identities),
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
