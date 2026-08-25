from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from cofitok.configs import load_config
from cofitok.generation import SAMPLING_REPORT_SCHEMA_VERSION
from cofitok.generation import min_snr_pilot
from cofitok.generation_class_fidelity import (
    CLASS_FIDELITY_CATEGORIES_SHA256,
    CLASS_FIDELITY_CLASSIFIER_BYTES,
    CLASS_FIDELITY_CLASSIFIER_NAME,
    CLASS_FIDELITY_CLASSIFIER_SHA256,
    CLASS_FIDELITY_PREPROCESSING,
    CLASS_FIDELITY_REPORT_SCHEMA_VERSION,
    validate_class_fidelity_report,
)
from cofitok.image_integrity import (
    IMAGE_TREE_DIGEST_SCHEMA,
    image_tree_sha256,
    sample_set_sha256,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)
from scripts.evaluate_generation_checkpoint import (
    MANIFEST_FILENAME as CHECKPOINT_EVALUATION_MANIFEST_FILENAME,
    _validate_completed_report as validate_checkpoint_evaluation_report,
)
from scripts.evaluate_generation_metrics import (
    GENERATION_METRICS_REPORT_ROLE,
    GENERATION_METRICS_REPORT_SCHEMA_VERSION,
    find_images,
    validate_sampling_provenance,
)


TERMINAL_GUARD_SCHEMA = "cofitok_matched_min_snr_terminal_physical_guard_v1"
TERMINAL_GUARD_ROLE = "generation_matched_min_snr_terminal_physical_guard"
TERMINAL_GUARD_DIRECTORY = "terminal_physical_guard_v1"
TERMINAL_GUARD_FILENAME = "terminal_physical_guard.json"
REAL_IMAGE_COUNT = 50_000
CANONICAL_REAL_SET = Path(
    "/root/autodl-tmp/CoFiTok/datasets/imagenet_256/extracted/val"
)
PREPARATION_SOURCE_NAMES = {
    "post_diagnostic_decision",
    "pair_monitor",
    "legacy_cofitok_config",
    "legacy_dense_config",
    "pilot_cofitok_config",
    "pilot_dense_config",
    "legacy_cofitok_training_report",
    "legacy_dense_training_report",
    "legacy_cofitok_checkpoint_audit_50k",
    "legacy_dense_checkpoint_audit_50k",
}
TRAINING_REPLAY_CODE_PATHS = (
    "src/cofitok/configs.py",
    "src/cofitok/generation/min_snr_pilot.py",
    "src/cofitok/generation_class_fidelity.py",
    "src/cofitok/image_integrity.py",
    "src/cofitok/sampling_progress.py",
    "src/cofitok/training/checkpointing.py",
    "scripts/audit_generation_min_snr_pilot_training.py",
    "scripts/build_generation_min_snr_pilot_result.py",
    "scripts/evaluate_generation_checkpoint.py",
    "scripts/evaluate_generation_class_fidelity.py",
    "scripts/evaluate_generation_metrics.py",
    "scripts/generate_samples.py",
    "scripts/preflight_generation_sampling.py",
)
SOURCE_CODE_PATHS = (
    "src/cofitok/generation/min_snr_terminal_guard.py",
    "scripts/build_generation_min_snr_terminal_physical_guard.py",
    "scripts/verify_generation_min_snr_terminal_physical_guard.py",
    *TRAINING_REPLAY_CODE_PATHS,
)
ARM_SPECS = {
    "legacy_gamma0_cofitok": {
        "method": "cofitok",
        "pilot": False,
        "prefix_budget": 8,
        "random_orders": 4,
    },
    "legacy_gamma0_dense_identity": {
        "method": "dense_identity",
        "pilot": False,
        "prefix_budget": 1,
        "random_orders": 0,
    },
    "pilot_gamma5_cofitok": {
        "method": "cofitok",
        "pilot": True,
        "prefix_budget": 8,
        "random_orders": 4,
    },
    "pilot_gamma5_dense_identity": {
        "method": "dense_identity",
        "pilot": True,
        "prefix_budget": 1,
        "random_orders": 0,
    },
}
SOURCE_KINDS = (
    "generation",
    "class_fidelity",
    "checkpoint_eval",
    "sampling_preflight",
)
AUTHORIZATION_BOUNDARY = {
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "continuation_beyond_50000_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "inference_export_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "gpu_execution_allowed": False,
}


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} is missing or malformed")
    return value


def _canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _finite(value: Any, *, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is malformed") from error
    if not math.isfinite(numeric):
        raise ValueError(f"{label} is not finite")
    return numeric


def _assert_all_numbers_finite(value: Any, *, label: str) -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{label} contains a non-finite numeric value")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_all_numbers_finite(item, label=f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _assert_all_numbers_finite(item, label=f"{label}[{index}]")


def _read_file_with_identity(
    path: str | Path,
    *,
    label: str,
) -> tuple[dict[str, Any], bytes]:
    source = reject_symlink_chain(path, name=label)
    if not source.is_file():
        raise FileNotFoundError(f"{label} is missing: {source}")
    resolved = source.resolve()
    before = resolved.stat()
    payload = resolved.read_bytes()
    after = resolved.stat()
    if (
        len(payload) != before.st_size
        or after.st_size != before.st_size
        or after.st_mtime_ns != before.st_mtime_ns
    ):
        raise RuntimeError(f"{label} changed while it was read")
    return (
        {
            "path": resolved.as_posix(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        payload,
    )


def _decode_json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value


def _read_json_with_identity(
    path: str | Path,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity, payload = _read_file_with_identity(path, label=label)
    return identity, _decode_json_object(payload, label=label)


def _read_bound_json(
    claimed_value: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    claimed = min_snr_pilot.normalize_identity(claimed_value, label=label)
    observed, payload = _read_json_with_identity(claimed["path"], label=label)
    if observed != claimed:
        raise ValueError(f"{label} changed from its bound identity")
    return observed, payload


def _verify_source_code_files(
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    files = _mapping(binding.get("files"), label="terminal guard source files")
    observed = _actual_source_identities(files)
    aggregate = _canonical_digest(observed)
    if aggregate != binding.get("aggregate_sha256"):
        raise ValueError("Min-SNR terminal source-code aggregate changed")
    return {
        "files": observed,
        "aggregate_sha256": aggregate,
        "physical_identities_verified": True,
    }


def _clean_git(
    value: Any,
    *,
    label: str,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    git = _mapping(value, label=label)
    if (
        git.get("revision") != expected_revision
        or git.get("branch") != expected_branch
        or git.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} differs")
    return copy.deepcopy(dict(git))


def _guard_git(value: Any) -> dict[str, Any]:
    git = _mapping(value, label="terminal guard Git")
    revision = str(git.get("revision", ""))
    tree = str(git.get("tree", ""))
    branch = str(git.get("branch", ""))
    if (
        len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or len(tree) != 40
        or any(character not in "0123456789abcdef" for character in tree)
        or not branch
        or git.get("tracked_dirty") is not False
    ):
        raise ValueError("terminal guard Git identity differs")
    return copy.deepcopy(dict(git))


def _source_code_binding(value: Any) -> dict[str, Any]:
    binding = _mapping(value, label="terminal guard source-code binding")
    files = _mapping(binding.get("files"), label="terminal guard source files")
    training_source = _mapping(
        binding.get("training_source"), label="terminal guard training source"
    )
    guard_source = _mapping(
        binding.get("guard_source"), label="terminal guard implementation source"
    )
    replay_blobs = _mapping(
        binding.get("training_replay_blobs"),
        label="terminal guard training replay blobs",
    )
    if set(files) != set(SOURCE_CODE_PATHS):
        raise ValueError("terminal guard source-code set differs")
    if set(replay_blobs) != set(TRAINING_REPLAY_CODE_PATHS):
        raise ValueError("terminal guard training replay blob set differs")
    normalized = {
        name: min_snr_pilot.normalize_identity(identity, label=name)
        for name, identity in files.items()
    }
    normalized_blobs = {str(name): str(digest) for name, digest in replay_blobs.items()}
    if any(
        len(digest) != 40
        or any(character not in "0123456789abcdef" for character in digest)
        for digest in normalized_blobs.values()
    ):
        raise ValueError("terminal guard training replay blob identity is malformed")
    for name, source in (
        ("training", training_source),
        ("guard", guard_source),
    ):
        if (
            len(str(source.get("revision", ""))) != 40
            or len(str(source.get("tree", ""))) != 40
            or not str(source.get("branch", ""))
            or source.get("tracked_dirty") is not False
        ):
            raise ValueError(f"terminal guard {name} source identity differs")
    if binding.get("training_source_is_ancestor") is not True:
        raise ValueError("terminal guard training source ancestry differs")
    expected_digest = _canonical_digest(normalized)
    if binding.get("aggregate_sha256") != expected_digest:
        raise ValueError("terminal guard source-code aggregate differs")
    return {
        "training_source": copy.deepcopy(dict(training_source)),
        "guard_source": copy.deepcopy(dict(guard_source)),
        "training_source_is_ancestor": True,
        "training_replay_blobs": normalized_blobs,
        "files": normalized,
        "aggregate_sha256": expected_digest,
    }


def _actual_source_identities(
    expected: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    actual: dict[str, dict[str, Any]] = {}
    for name, claimed_value in sorted(expected.items()):
        claimed = min_snr_pilot.normalize_identity(claimed_value, label=name)
        observed = file_identity(claimed["path"])
        if observed != claimed:
            raise ValueError(f"Min-SNR terminal source changed: {name}")
        actual[name] = observed
    return actual


def _logical_result_replay(
    result_path: Path,
    *,
    expected_result_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    result_identity, result_payload = _read_json_with_identity(
        result_path,
        label="Min-SNR pilot result",
    )
    if result_identity["sha256"] != expected_result_sha256:
        raise ValueError("Min-SNR pilot result SHA256 differs")
    result = min_snr_pilot.validate_result(result_payload)
    _assert_all_numbers_finite(result, label="Min-SNR pilot result")
    expected_sources = _mapping(
        result.get("source_identities"), label="Min-SNR pilot result sources"
    )
    actual_sources: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, claimed in expected_sources.items():
        identity, payload = _read_bound_json(
            claimed,
            label=name.replace("_", " "),
        )
        actual_sources[str(name)] = identity
        payloads[str(name)] = payload
    audits = {
        "cofitok": payloads["pilot_cofitok_training_audit"],
        "dense_identity": payloads["pilot_dense_training_audit"],
    }
    arms = {
        arm: {
            kind: payloads[f"{arm}_{kind}"]
            for kind in SOURCE_KINDS
        }
        for arm in ARM_SPECS
    }
    rebuilt = min_snr_pilot.build_result(
        preparation=payloads["preparation"],
        preparation_identity=actual_sources["preparation"],
        execution_gate=payloads["execution_gate"],
        execution_gate_identity=actual_sources["execution_gate"],
        pilot_training_audits=audits,
        arms=arms,
        source_identities=actual_sources,
        builder_git=result["builder_git"],
    )
    if rebuilt != result:
        raise ValueError("Min-SNR pilot result does not replay exactly")
    if file_identity(result_path) != result_identity:
        raise ValueError("Min-SNR pilot result changed during logical replay")
    if _actual_source_identities(actual_sources) != actual_sources:
        raise ValueError("Min-SNR pilot sources changed during logical replay")
    return result, result_identity, actual_sources, payloads


def _verify_checkpoint(
    checkpoint_value: str | Path,
    *,
    label: str,
    expected_step: int,
) -> dict[str, Any]:
    checkpoint = reject_symlink_chain(checkpoint_value, name=f"{label} checkpoint")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"{label} checkpoint is missing: {checkpoint}")
    integrity_path = reject_symlink_chain(
        checkpoint_integrity_path(checkpoint),
        name=f"{label} checkpoint integrity manifest",
    )
    integrity = verify_training_checkpoint(checkpoint)
    if int(integrity.get("step", -1)) != expected_step:
        raise ValueError(f"{label} checkpoint step differs")
    payload = {
        "path": checkpoint.resolve().as_posix(),
        "bytes": int(integrity.get("checkpoint_bytes", -1)),
        "sha256": str(integrity.get("checkpoint_sha256", "")),
    }
    if (
        payload["bytes"] != checkpoint.stat().st_size
        or len(payload["sha256"]) != 64
    ):
        raise ValueError(f"{label} checkpoint physical identity is malformed")
    return {
        "payload": payload,
        "integrity_manifest": file_identity(integrity_path),
        "step": expected_step,
        "format_version": int(integrity.get("checkpoint_format_version", -1)),
        "git": {
            "revision": integrity.get("git_revision"),
            "branch": integrity.get("git_branch"),
            "tracked_dirty": integrity.get("git_dirty"),
        },
        "dataset_identity_sha256": integrity.get("dataset_identity_sha256"),
        "runtime_environment_sha256": integrity.get(
            "runtime_environment_sha256"
        ),
        "physical_sha256_verified": True,
    }


def _read_metrics(
    path: Path,
    *,
    label: str,
    expected_identity: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    identity, payload = _read_file_with_identity(path, label=f"{label} metrics")
    if expected_identity is not None:
        expected = min_snr_pilot.normalize_identity(
            expected_identity,
            label=f"{label} metrics identity",
        )
        if identity != expected:
            raise ValueError(f"{label} metrics changed from their bound identity")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} metrics are unreadable") from error
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{label} metrics line {line_number} is malformed"
            ) from error
        if not isinstance(row, dict):
            raise ValueError(f"{label} metrics line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{label} metrics are empty")
    return rows


def _verify_min_snr_metrics(
    rows: list[dict[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    steps = [int(row.get("step", -1)) for row in rows]
    if (
        steps[0] != 1
        or steps[-1] != min_snr_pilot.PILOT_STEP
        or any(current <= previous for previous, current in zip(steps, steps[1:]))
    ):
        raise ValueError(f"pilot {method} metrics steps differ")
    weights: list[float] = []
    epsilon_ratios: list[float] = []
    downweighted_count = 0
    for index, row in enumerate(rows):
        _assert_all_numbers_finite(
            row,
            label=f"pilot {method} metrics row {index}",
        )
        step = int(row.get("step", -1))
        if int(row.get("samples_seen", -1)) != step * min_snr_pilot.EFFECTIVE_BATCH_SIZE:
            raise ValueError(f"pilot {method} metrics row {index} exposure differs")
        values = {
            field: _finite(row.get(field), label=f"pilot {method} row {index} {field}")
            for field in (
                "total",
                "epsilon",
                "epsilon_unweighted",
                "min_snr_weight_mean",
                "learning_rate",
                "grad_norm",
            )
        }
        weight = values["min_snr_weight_mean"]
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"pilot {method} metrics row {index} Min-SNR weight differs")
        if values["epsilon"] > values["epsilon_unweighted"] + 1e-12:
            raise ValueError(
                f"pilot {method} metrics row {index} weighted epsilon exceeds unweighted"
            )
        if values["epsilon_unweighted"] <= 0.0:
            raise ValueError(f"pilot {method} metrics row {index} unweighted epsilon is invalid")
        weights.append(weight)
        epsilon_ratios.append(values["epsilon"] / values["epsilon_unweighted"])
        downweighted_count += values["epsilon"] < values["epsilon_unweighted"]
    if not any(weight < 0.999999 for weight in weights):
        raise ValueError(f"pilot {method} metrics never apply Min-SNR")
    return {
        "row_count": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "samples_seen": int(rows[-1]["samples_seen"]),
        "strictly_increasing": True,
        "samples_seen_binding_verified": True,
        "all_numeric_fields_finite": True,
        "epsilon_le_unweighted_all_rows": True,
        "downweighted_row_count": downweighted_count,
        "min_snr_weight_min": min(weights),
        "min_snr_weight_mean": sum(weights) / len(weights),
        "min_snr_weight_max": max(weights),
        "epsilon_ratio_min": min(epsilon_ratios),
        "epsilon_ratio_mean": sum(epsilon_ratios) / len(epsilon_ratios),
        "epsilon_ratio_max": max(epsilon_ratios),
    }


def _verify_pilot_training(
    *,
    method: str,
    audit_path: Path,
    audit: Mapping[str, Any],
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    logical = min_snr_pilot._validate_training_audit(audit, method=method)
    audit_git = _mapping(audit.get("git"), label=f"pilot {method} audit Git")
    if (
        audit_git.get("revision") != expected_revision
        or audit_git.get("tree") != expected_tree
        or audit_git.get("branch") != expected_branch
        or audit_git.get("tracked_dirty") is not False
    ):
        raise ValueError(f"pilot {method} audit Git identity differs")
    checkpoint_claim = _mapping(
        audit.get("checkpoint"), label=f"pilot {method} audit checkpoint"
    )
    physical = _verify_checkpoint(
        str(checkpoint_claim.get("path", "")),
        label=f"pilot {method}",
        expected_step=min_snr_pilot.PILOT_STEP,
    )
    if (
        physical["payload"]
        != {
            "path": Path(str(checkpoint_claim.get("path", ""))).resolve().as_posix(),
            "bytes": int(checkpoint_claim.get("bytes", -1)),
            "sha256": checkpoint_claim.get("sha256"),
        }
        or physical["integrity_manifest"]
        != min_snr_pilot.normalize_identity(
            checkpoint_claim.get("integrity_manifest"),
            label=f"pilot {method} integrity manifest",
        )
        or physical["git"]
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or physical["dataset_identity_sha256"]
        != min_snr_pilot.DATASET_IDENTITY_SHA256
        or physical["runtime_environment_sha256"]
        != min_snr_pilot.LEGACY_RUNTIME_ENVIRONMENT_SHA256
    ):
        raise ValueError(f"pilot {method} physical checkpoint differs from audit")
    if logical["checkpoint"] != physical["payload"]:
        raise ValueError(f"pilot {method} result checkpoint differs from audit")

    sources = _mapping(audit.get("sources"), label=f"pilot {method} audit sources")
    expected_source_names = {"training_report", "latest", "metrics"}
    if set(sources) != expected_source_names:
        raise ValueError(f"pilot {method} audit source set differs")
    source_identities = _actual_source_identities(sources)
    run_dir = Path(physical["payload"]["path"]).parent.resolve()
    expected_paths = {
        "training_report": run_dir / "training_report.json",
        "latest": run_dir / "latest.json",
        "metrics": run_dir / "train_metrics.jsonl",
    }
    for name, expected_path in expected_paths.items():
        if Path(source_identities[name]["path"]) != expected_path:
            raise ValueError(f"pilot {method} {name} path is not canonical")
    _, report = _read_bound_json(
        source_identities["training_report"],
        label=f"pilot {method} training report",
    )
    _, latest = _read_bound_json(
        source_identities["latest"],
        label=f"pilot {method} latest checkpoint pointer",
    )
    config_identity, _ = _read_bound_json(
        audit.get("config"),
        label=f"pilot {method} resolved config source",
    )
    configured = load_config(config_identity["path"])
    resolved_config = asdict(
        replace(
            configured,
            data=replace(
                configured.data,
                batch_size=min_snr_pilot.EFFECTIVE_BATCH_SIZE,
            ),
            optimization=replace(
                configured.optimization,
                gradient_accumulation_steps=1,
            ),
        )
    )
    _assert_all_numbers_finite(report, label=f"pilot {method} training report")
    _assert_all_numbers_finite(latest, label=f"pilot {method} latest pointer")
    report_git = _mapping(report.get("git"), label=f"pilot {method} training Git")
    provenance = _mapping(
        report.get("dataset_provenance"), label=f"pilot {method} dataset provenance"
    )
    config = _mapping(report.get("config"), label=f"pilot {method} training config")
    loss = _mapping(config.get("loss"), label=f"pilot {method} loss config")
    runtime = _mapping(config.get("runtime"), label=f"pilot {method} runtime config")
    if (
        int(report.get("completed_steps", -1)) != min_snr_pilot.PILOT_STEP
        or int(report.get("target_steps", -1)) != min_snr_pilot.SCHEDULER_HORIZON
        or report.get("training_complete") is not False
        or report.get("stop_requested") is not False
        or report.get("stop_signal") is not None
        or report_git.get("revision") != expected_revision
        or report_git.get("branch") != expected_branch
        or report_git.get("dirty") is not False
        or provenance.get("identity_sha256")
        != min_snr_pilot.DATASET_IDENTITY_SHA256
        or report.get("runtime_environment_sha256")
        != min_snr_pilot.LEGACY_RUNTIME_ENVIRONMENT_SHA256
        or report.get("config") != resolved_config
        or float(loss.get("min_snr_gamma", -1.0)) != min_snr_pilot.GAMMA
        or int(runtime.get("steps", -1)) != min_snr_pilot.SCHEDULER_HORIZON
        or int(runtime.get("seed", -1)) != 2027
        or int(latest.get("step", -1)) != min_snr_pilot.PILOT_STEP
        or latest.get("checkpoint") != Path(physical["payload"]["path"]).name
        or latest.get("checkpoint_sha256") != physical["payload"]["sha256"]
        or latest.get("integrity_manifest")
        != Path(physical["integrity_manifest"]["path"]).name
        or latest.get("git_revision") != expected_revision
        or latest.get("git_branch") != expected_branch
        or latest.get("git_dirty") is not False
    ):
        raise ValueError(f"pilot {method} training terminal state differs")

    rows = _read_metrics(
        expected_paths["metrics"],
        label=f"pilot {method}",
        expected_identity=source_identities["metrics"],
    )
    metrics_evidence = _verify_min_snr_metrics(rows, method=method)
    training_claim = _mapping(
        audit.get("training"),
        label=f"pilot {method} audit training",
    )
    if (
        int(training_claim.get("metric_row_count", -1)) != len(rows)
        or int(training_claim.get("samples_seen", -1))
        != min_snr_pilot.IMAGES_PER_METHOD
        or file_identity(audit_path).get("path") != audit_path.resolve().as_posix()
        or file_identity(config_identity["path"]) != config_identity
        or _actual_source_identities(source_identities) != source_identities
    ):
        raise ValueError(f"pilot {method} source replay differs")
    return {
        "method": method,
        "audit": file_identity(audit_path),
        "checkpoint": physical,
        "sources": source_identities,
        "config": config_identity,
        "metrics": metrics_evidence,
    }


def _verify_preparation_sources(
    preparation: Mapping[str, Any],
) -> dict[str, Any]:
    prepared = min_snr_pilot.validate_preparation(preparation)
    claimed_sources = _mapping(
        prepared.get("source_identities"),
        label="Min-SNR preparation sources",
    )
    if set(claimed_sources) != PREPARATION_SOURCE_NAMES:
        raise ValueError("Min-SNR preparation source set differs")
    source_identities: dict[str, dict[str, Any]] = {}
    payloads: dict[str, dict[str, Any]] = {}
    for name, claimed in claimed_sources.items():
        identity, payload = _read_bound_json(
            claimed,
            label=f"Min-SNR preparation {name.replace('_', ' ')}",
        )
        source_identities[str(name)] = identity
        payloads[str(name)] = payload
    legacy_configs = {
        "cofitok": payloads["legacy_cofitok_config"],
        "dense_identity": payloads["legacy_dense_config"],
    }
    pilot_configs = {
        "cofitok": payloads["pilot_cofitok_config"],
        "dense_identity": payloads["pilot_dense_config"],
    }
    legacy_training_reports = {
        "cofitok": payloads["legacy_cofitok_training_report"],
        "dense_identity": payloads["legacy_dense_training_report"],
    }
    legacy_checkpoint_audits = {
        "cofitok": payloads["legacy_cofitok_checkpoint_audit_50k"],
        "dense_identity": payloads["legacy_dense_checkpoint_audit_50k"],
    }
    rebuilt = min_snr_pilot.build_preparation(
        post_diagnostic_decision=payloads["post_diagnostic_decision"],
        pair_monitor=payloads["pair_monitor"],
        legacy_configs=legacy_configs,
        pilot_configs=pilot_configs,
        legacy_training_reports=legacy_training_reports,
        legacy_checkpoint_audits=legacy_checkpoint_audits,
        source_identities=source_identities,
        builder_git=prepared["builder_git"],
        training_source_delta=prepared["training_source_delta"],
        output_root=str(prepared["output_root"]),
    )
    if rebuilt != prepared:
        raise ValueError("Min-SNR preparation does not replay exactly")
    training_delta = _mapping(
        prepared.get("training_source_delta"),
        label="Min-SNR preparation training-source delta",
    )
    semantic_claims = _mapping(
        training_delta.get("semantic_file_identities"),
        label="Min-SNR semantic file identities",
    )
    if set(semantic_claims) != set(min_snr_pilot.TRAINING_SEMANTIC_FILES):
        raise ValueError("Min-SNR semantic source set differs")
    semantic_sources = _actual_source_identities(semantic_claims)
    legacy_controls: dict[str, dict[str, Any]] = {}
    prepared_controls = _mapping(
        _mapping(
            prepared.get("legacy_control_policy"),
            label="Min-SNR legacy control policy",
        ).get("controls"),
        label="Min-SNR legacy controls",
    )
    for method in min_snr_pilot.METHODS:
        audit = legacy_checkpoint_audits[method]
        logical = min_snr_pilot._validate_legacy_audit(audit, method=method)
        prepared_control = _mapping(
            prepared_controls.get(method),
            label=f"legacy {method} prepared control",
        )
        if (
            min_snr_pilot.normalize_identity(
                prepared_control.get("checkpoint"),
                label=f"legacy {method} prepared checkpoint",
            )
            != logical["checkpoint"]
            or int(prepared_control.get("step", -1))
            != min_snr_pilot.PILOT_STEP
            or int(prepared_control.get("samples_seen", -1))
            != min_snr_pilot.IMAGES_PER_METHOD
            or prepared_control.get("physical_sha256_verified") is not True
        ):
            raise ValueError(f"legacy {method} preparation control differs")
        physical = _verify_checkpoint(
            logical["checkpoint"]["path"],
            label=f"legacy {method}",
            expected_step=min_snr_pilot.PILOT_STEP,
        )
        audit_checkpoint = _mapping(
            audit.get("checkpoint"),
            label=f"legacy {method} audit checkpoint",
        )
        audit_integrity = _mapping(
            audit_checkpoint.get("integrity_manifest"),
            label=f"legacy {method} audit integrity manifest",
        )
        latest_pointer = _mapping(
            audit.get("latest_pointer"),
            label=f"legacy {method} latest pointer",
        )
        metrics = _mapping(
            audit.get("metrics"),
            label=f"legacy {method} metrics audit",
        )
        _, integrity_payload = _read_bound_json(
            audit_integrity,
            label=f"legacy {method} checkpoint integrity manifest",
        )
        latest_identity = min_snr_pilot.normalize_identity(
            latest_pointer.get("identity"),
            label=f"legacy {method} historical latest identity",
        )
        latest_payload = _mapping(
            latest_pointer.get("content"),
            label=f"legacy {method} historical latest content",
        )
        latest_bytes = (
            json.dumps(latest_payload, indent=2, sort_keys=True) + "\n"
        ).encode("utf-8")
        metrics_identity = min_snr_pilot.normalize_identity(
            metrics.get("identity"),
            label=f"legacy {method} historical metrics identity",
        )
        target_row = _mapping(
            metrics.get("target_row"),
            label=f"legacy {method} historical target row",
        )
        _assert_all_numbers_finite(
            target_row,
            label=f"legacy {method} historical target row",
        )
        expected_physical = {
            "payload": logical["checkpoint"],
            "integrity_manifest": min_snr_pilot.normalize_identity(
                audit_integrity,
                label=f"legacy {method} audit integrity manifest",
            ),
            "step": min_snr_pilot.PILOT_STEP,
            "format_version": 1,
            "git": {
                "revision": min_snr_pilot.LEGACY_REVISION,
                "branch": min_snr_pilot.LEGACY_BRANCH,
                "tracked_dirty": False,
            },
            "dataset_identity_sha256": min_snr_pilot.DATASET_IDENTITY_SHA256,
            "runtime_environment_sha256": (
                min_snr_pilot.LEGACY_RUNTIME_ENVIRONMENT_SHA256
            ),
            "physical_sha256_verified": True,
        }
        if (
            physical != expected_physical
            or int(logical.get("samples_seen", -1)) != min_snr_pilot.IMAGES_PER_METHOD
            or integrity_payload != audit_checkpoint.get("integrity")
            or latest_pointer.get("exact_target_binding") is not True
            or latest_identity["bytes"] != len(latest_bytes)
            or latest_identity["sha256"] != hashlib.sha256(latest_bytes).hexdigest()
            or Path(latest_identity["path"]).name != "latest.json"
            or int(latest_payload.get("step", -1)) != min_snr_pilot.PILOT_STEP
            or latest_payload.get("checkpoint")
            != Path(physical["payload"]["path"]).name
            or latest_payload.get("checkpoint_sha256")
            != physical["payload"]["sha256"]
            or latest_payload.get("integrity_manifest")
            != Path(physical["integrity_manifest"]["path"]).name
            or metrics.get("strictly_increasing") is not True
            or metrics.get("samples_seen_binding_verified") is not True
            or int(metrics.get("first_step", -1)) != 1
            or int(metrics.get("last_step", -1)) != min_snr_pilot.PILOT_STEP
            or int(metrics.get("row_count", -1)) < 1
            or Path(metrics_identity["path"]).name != "train_metrics.jsonl"
            or int(target_row.get("step", -1)) != min_snr_pilot.PILOT_STEP
            or int(target_row.get("samples_seen", -1))
            != min_snr_pilot.IMAGES_PER_METHOD
        ):
            raise ValueError(f"legacy {method} physical control differs")
        legacy_controls[method] = {
            "audit": source_identities[
                f"legacy_{'cofitok' if method == 'cofitok' else 'dense'}_checkpoint_audit_50k"
            ],
            "checkpoint": physical,
            "historical_latest": latest_identity,
            "historical_metrics": metrics_identity,
            "historical_latest_bytes_reconstructed": True,
            "historical_metrics_target_row_verified": True,
            "preparation_control_match": True,
        }
    if _actual_source_identities(source_identities) != source_identities:
        raise ValueError("Min-SNR preparation sources changed during replay")
    if _actual_source_identities(semantic_sources) != semantic_sources:
        raise ValueError("Min-SNR semantic sources changed during replay")
    return {
        "source_reports": source_identities,
        "semantic_sources": semantic_sources,
        "logical_replay_exact": True,
        "legacy_controls": legacy_controls,
    }


def _verify_classifier(
    report: Mapping[str, Any],
    *,
    label: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    classifier = _mapping(report.get("classifier"), label=f"{label} classifier")
    if (
        classifier.get("name") != CLASS_FIDELITY_CLASSIFIER_NAME
        or classifier.get("weights_enum")
        != "ResNet50_Weights.IMAGENET1K_V2"
        or int(classifier.get("weights_bytes", -1))
        != CLASS_FIDELITY_CLASSIFIER_BYTES
        or classifier.get("weights_sha256")
        != CLASS_FIDELITY_CLASSIFIER_SHA256
        or int(classifier.get("num_classes", -1)) != 1000
        or classifier.get("categories_sha256")
        != CLASS_FIDELITY_CATEGORIES_SHA256
        or classifier.get("preprocessing") != CLASS_FIDELITY_PREPROCESSING
    ):
        raise ValueError(f"{label} classifier contract differs")
    weights_path = classifier.get("weights_path")
    if not isinstance(weights_path, str) or not Path(weights_path).is_absolute():
        raise ValueError(f"{label} classifier weight path is malformed")
    resolved = reject_symlink_chain(weights_path, name=f"{label} classifier weight").resolve()
    cache_key = resolved.as_posix()
    if cache_key not in cache:
        cache[cache_key] = file_identity(resolved)
    physical = cache[cache_key]
    if (
        physical["path"] != resolved.as_posix()
        or physical["bytes"] != CLASS_FIDELITY_CLASSIFIER_BYTES
        or physical["sha256"] != CLASS_FIDELITY_CLASSIFIER_SHA256
        or classifier.get("weights_path") != resolved.as_posix()
    ):
        raise ValueError(f"{label} classifier physical identity differs")
    return {
        "contract": copy.deepcopy(dict(classifier)),
        "physical_weights": copy.deepcopy(physical),
        "physical_sha256_verified": True,
    }


def _verify_real_set(
    report: Mapping[str, Any],
    *,
    label: str,
    cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    paths = _mapping(report.get("paths"), label=f"{label} generation paths")
    claimed = _mapping(report.get("real_set"), label=f"{label} real set")
    real_dir = reject_symlink_chain(paths.get("real_dir", ""), name=f"{label} real set")
    resolved = real_dir.resolve()
    if resolved != CANONICAL_REAL_SET.resolve():
        raise ValueError(f"{label} real set is not canonical")
    key = resolved.as_posix()
    if key not in cache:
        images = find_images(resolved)
        cache[key] = {
            "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
            "root": key,
            "image_count": len(images),
            "sha256": image_tree_sha256(images, root=resolved),
        }
    physical = cache[key]
    counts = _mapping(report.get("counts"), label=f"{label} generation counts")
    if (
        physical != dict(claimed)
        or physical["image_count"] != REAL_IMAGE_COUNT
        or int(counts.get("real_image_count", -1)) != physical["image_count"]
    ):
        raise ValueError(f"{label} physical real-set evidence differs")
    return copy.deepcopy(physical)


def _verify_checkpoint_evaluation(
    report_path: Path,
    report: Mapping[str, Any],
    *,
    label: str,
    expected_checkpoint: Mapping[str, Any],
    expected_revision: str,
    expected_branch: str,
    random_orders: int,
) -> dict[str, Any]:
    manifest_path = report_path.parent / CHECKPOINT_EVALUATION_MANIFEST_FILENAME
    manifest_identity = file_identity(manifest_path)
    manifest = read_json_object(manifest_path, name=f"{label} checkpoint manifest")
    _assert_all_numbers_finite(
        manifest,
        label=f"{label} checkpoint evaluation manifest",
    )
    _assert_all_numbers_finite(
        report,
        label=f"{label} checkpoint evaluation report",
    )
    validate_checkpoint_evaluation_report(
        dict(report), manifest=manifest, manifest_identity=manifest_identity
    )
    request = _mapping(manifest.get("request"), label=f"{label} checkpoint request")
    checkpoint = _mapping(
        manifest.get("checkpoint"), label=f"{label} checkpoint manifest identity"
    )
    _clean_git(
        manifest.get("git"),
        label=f"{label} checkpoint manifest Git",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    _clean_git(
        report.get("git"),
        label=f"{label} checkpoint evaluator Git",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    if (
        request
        != {
            "num_images": 256,
            "timestep": 500,
            "random_orders": random_orders,
            "seed": 2027,
            "weights": "ema",
            "precision": "bf16",
        }
        or checkpoint.get("path") != expected_checkpoint["payload"]["path"]
        or checkpoint.get("bytes") != expected_checkpoint["payload"]["bytes"]
        or checkpoint.get("sha256") != expected_checkpoint["payload"]["sha256"]
        or checkpoint.get("integrity_manifest")
        != expected_checkpoint["integrity_manifest"]
        or int(checkpoint.get("step", -1)) != min_snr_pilot.PILOT_STEP
        or report.get("checkpoint") != expected_checkpoint["payload"]["path"]
        or report.get("checkpoint_sha256")
        != expected_checkpoint["payload"]["sha256"]
    ):
        raise ValueError(f"{label} checkpoint evaluation binding differs")
    if file_identity(manifest_path) != manifest_identity:
        raise ValueError(f"{label} checkpoint manifest changed during replay")
    return {
        "report": file_identity(report_path),
        "manifest": manifest_identity,
        "request": copy.deepcopy(dict(request)),
        "checkpoint": copy.deepcopy(dict(expected_checkpoint)),
    }


def _verify_generation_report_contract(
    report: Mapping[str, Any],
    *,
    label: str,
    expected_revision: str,
    expected_branch: str,
) -> None:
    _assert_all_numbers_finite(report, label=f"{label} generation report")
    metrics = _mapping(report.get("metrics"), label=f"{label} generation metrics")
    runtime = _mapping(report.get("runtime"), label=f"{label} generation runtime")
    counts = _mapping(report.get("counts"), label=f"{label} generation counts")
    if (
        report.get("schema_version") != GENERATION_METRICS_REPORT_SCHEMA_VERSION
        or report.get("role") != GENERATION_METRICS_REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("protocol") != "torch_fidelity_directory_metrics"
        or _finite(metrics.get("frechet_inception_distance"), label=f"{label} FID")
        < 0.0
        or _finite(
            metrics.get("inception_score_mean"),
            label=f"{label} Inception score",
        )
        <= 0.0
        or _finite(
            metrics.get("inception_score_std"),
            label=f"{label} Inception score std",
        )
        < 0.0
        or not 0.0
        <= _finite(metrics.get("precision"), label=f"{label} precision")
        <= 1.0
        or not 0.0 <= _finite(metrics.get("recall"), label=f"{label} recall") <= 1.0
        or _finite(runtime.get("elapsed_seconds"), label=f"{label} metric elapsed")
        <= 0.0
        or int(counts.get("generated_image_count", -1))
        != min_snr_pilot.SAMPLE_COUNT
        or int(counts.get("real_image_count", -1)) != REAL_IMAGE_COUNT
    ):
        raise ValueError(f"{label} generation report contract differs")
    _clean_git(
        report.get("git"),
        label=f"{label} generation evaluator Git",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )


def _verify_sampling_evidence(
    *,
    arm: str,
    generation: Mapping[str, Any],
    preflight: Mapping[str, Any],
    expected_prefix: int,
    expected_revision: str,
    expected_branch: str,
    expected_checkpoint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    preflight_row = min_snr_pilot.validate_sampling_preflight(
        preflight,
        arm=arm,
        expected_prefix_budget=expected_prefix,
    )
    _clean_git(
        preflight.get("git"),
        label=f"{arm} sampling preflight Git",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    _verify_generation_report_contract(
        generation,
        label=arm,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    paths = _mapping(generation.get("paths"), label=f"{arm} generation paths")
    generated_dir = reject_symlink_chain(
        paths.get("generated_dir", ""), name=f"{arm} generated directory"
    ).resolve()
    sampling_report_path = reject_symlink_chain(
        paths.get("sampling_report", ""), name=f"{arm} sampling report"
    ).resolve()
    generated_images = find_images(generated_dir)
    if any(path.parent != generated_dir for path in generated_images):
        raise ValueError(f"{arm} generated sample tree is not flat")
    for path in generated_images:
        reject_symlink_chain(path, name=f"{arm} generated PNG")
    sampling = validate_sampling_provenance(
        sampling_report_path,
        generated_dir,
        generated_images,
    )
    if sampling != generation.get("sample_provenance"):
        raise ValueError(f"{arm} physical sampling provenance differs")
    protocol = _mapping(sampling.get("sampling"), label=f"{arm} sampling protocol")
    if (
        int(protocol.get("num_samples", -1)) != min_snr_pilot.SAMPLE_COUNT
        or len(generated_images) != min_snr_pilot.SAMPLE_COUNT
        or protocol.get("sampler") != "ddim"
        or int(protocol.get("sample_steps", -1)) != min_snr_pilot.SAMPLE_STEPS
        or int(protocol.get("num_train_timesteps", -1)) != 1000
        or int(protocol.get("seed", -1)) != min_snr_pilot.SAMPLE_SEED
        or int(protocol.get("start_index", -1)) != 0
        or float(protocol.get("guidance_scale", -1.0)) != 1.5
        or float(protocol.get("guidance_rescale", -1.0)) != 0.0
        or protocol.get("cfg_batch_mode") != "batched"
        or float(protocol.get("eta", -1.0)) != 0.0
        or protocol.get("clip_x0") is not True
        or protocol.get("precision") != "bf16"
        or protocol.get("class_schedule") != "balanced_modulo"
        or protocol.get("prefix_budgets") != [expected_prefix]
        or int(sampling.get("selected_prefix_budget", -1)) != expected_prefix
        or preflight_row["checkpoint_sha256"] != sampling["checkpoint_sha256"]
        or preflight_row["runtime_environment_sha256"]
        != sampling["runtime_environment_sha256"]
    ):
        raise ValueError(f"{arm} physical sampling protocol differs")
    _clean_git(
        sampling.get("git"),
        label=f"{arm} sampling Git",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    checkpoint = (
        copy.deepcopy(dict(expected_checkpoint))
        if expected_checkpoint is not None
        else _verify_checkpoint(
            sampling["checkpoint"],
            label=arm,
            expected_step=min_snr_pilot.PILOT_STEP,
        )
    )
    if (
        checkpoint["payload"]["sha256"] != sampling["checkpoint_sha256"]
        or Path(str(sampling.get("checkpoint", ""))).resolve().as_posix()
        != checkpoint["payload"]["path"]
        or checkpoint["integrity_manifest"]["path"]
        != Path(sampling["checkpoint_integrity_manifest"]).resolve().as_posix()
        or Path(str(preflight.get("checkpoint", ""))).resolve().as_posix()
        != checkpoint["payload"]["path"]
        or Path(str(preflight.get("checkpoint_integrity_manifest", ""))).resolve().as_posix()
        != checkpoint["integrity_manifest"]["path"]
    ):
        raise ValueError(f"{arm} physical checkpoint differs from sampling")
    progress = _mapping(
        sampling.get("sampling_progress"),
        label=f"{arm} sampling progress",
    )
    _, progress_payload = _read_bound_json(
        progress.get("identity"),
        label=f"{arm} sampling progress source",
    )
    _assert_all_numbers_finite(
        progress_payload,
        label=f"{arm} sampling progress source",
    )
    if (
        int(progress_payload.get("schema_version", -1))
        != SAMPLING_REPORT_SCHEMA_VERSION
        or progress.get("status") != "completed"
        or int(progress.get("invocation", -1)) < 1
        or int(progress.get("completed_samples", -1))
        != min_snr_pilot.SAMPLE_COUNT
        or _finite(
            progress.get("cumulative_elapsed_seconds"),
            label=f"{arm} sampling elapsed",
        )
        <= 0.0
        or progress_payload.get("status") != progress.get("status")
        or int(progress_payload.get("invocation", -1))
        != int(progress.get("invocation", -2))
        or int(progress_payload.get("completed_samples", -1))
        != int(progress.get("completed_samples", -2))
        or _finite(
            progress_payload.get("cumulative_elapsed_seconds"),
            label=f"{arm} physical sampling elapsed",
        )
        != _finite(
            progress.get("cumulative_elapsed_seconds"),
            label=f"{arm} replayed sampling elapsed",
        )
    ):
        raise ValueError(f"{arm} sampling progress differs")
    nested_identities = (
        sampling["report_identity"],
        sampling["manifest_identity"],
        progress["identity"],
    )
    if any(file_identity(identity["path"]) != identity for identity in nested_identities):
        raise ValueError(f"{arm} sampling sources changed during replay")
    if sample_set_sha256(generated_images) != sampling["sample_set_sha256"]:
        raise ValueError(f"{arm} sample set changed during replay")
    return {
        "preflight": preflight_row,
        "generated_dir": generated_dir,
        "generated_images": generated_images,
        "sampling_report_path": sampling_report_path,
        "sampling": sampling,
        "protocol": copy.deepcopy(dict(protocol)),
        "checkpoint": checkpoint,
    }


def _verify_arm(
    *,
    arm: str,
    spec: Mapping[str, Any],
    source_identities: Mapping[str, Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
    expected_revision: str,
    expected_branch: str,
    training_checkpoint: Mapping[str, Any],
    training_binding_label: str,
    real_cache: dict[str, dict[str, Any]],
    classifier_cache: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    expected_prefix = int(spec["prefix_budget"])
    source = {
        kind: payloads[f"{arm}_{kind}"]
        for kind in SOURCE_KINDS
    }
    for kind, payload in source.items():
        _assert_all_numbers_finite(
            payload,
            label=f"{arm} {kind.replace('_', ' ')} report",
        )
    logical = min_snr_pilot.validate_evaluation_arm(
        source,
        arm=arm,
        expected_prefix_budget=expected_prefix,
    )
    generation = source["generation"]
    sampling_evidence = _verify_sampling_evidence(
        arm=arm,
        generation=generation,
        preflight=source["sampling_preflight"],
        expected_prefix=expected_prefix,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_checkpoint=training_checkpoint,
    )
    generated_dir = sampling_evidence["generated_dir"]
    generated_images = sampling_evidence["generated_images"]
    sampling_report_path = sampling_evidence["sampling_report_path"]
    sampling = sampling_evidence["sampling"]
    protocol = sampling_evidence["protocol"]
    checkpoint = sampling_evidence["checkpoint"]
    generation_report_path = Path(
        source_identities[f"{arm}_generation"]["path"]
    )
    samples_root = generation_report_path.parent.parent
    if (
        generated_dir != (samples_root / f"prefix_{expected_prefix}").resolve()
        or sampling_report_path != (samples_root / "sampling_report.json").resolve()
        or Path(sampling["manifest_identity"]["path"])
        != (samples_root / "sampling_manifest.json").resolve()
        or Path(sampling["sampling_progress"]["identity"]["path"])
        != (samples_root / "sampling_progress.json").resolve()
    ):
        raise ValueError(f"{arm} sampling paths are not canonical")
    if checkpoint != training_checkpoint:
        raise ValueError(f"{arm} checkpoint differs from {training_binding_label}")

    real_set = _verify_real_set(
        generation,
        label=arm,
        cache=real_cache,
    )
    class_report = source["class_fidelity"]
    validate_class_fidelity_report(dict(class_report))
    class_paths = _mapping(
        class_report.get("paths"), label=f"{arm} class-fidelity paths"
    )
    if (
        int(class_report.get("schema_version", -1))
        != CLASS_FIDELITY_REPORT_SCHEMA_VERSION
        or class_report.get("sample_provenance") != sampling
        or Path(str(class_paths.get("generated_dir", ""))).resolve()
        != generated_dir
        or Path(str(class_paths.get("sampling_report", ""))).resolve()
        != sampling_report_path
        or Path(str(class_paths.get("report", ""))).resolve()
        != Path(source_identities[f"{arm}_class_fidelity"]["path"])
    ):
        raise ValueError(f"{arm} class-fidelity sampling binding differs")
    _clean_git(
        class_report.get("git"),
        label=f"{arm} class-fidelity evaluator Git",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    classifier = _verify_classifier(
        class_report,
        label=arm,
        cache=classifier_cache,
    )
    checkpoint_eval_path = Path(
        source_identities[f"{arm}_checkpoint_eval"]["path"]
    )
    checkpoint_evaluation = _verify_checkpoint_evaluation(
        checkpoint_eval_path,
        source["checkpoint_eval"],
        label=arm,
        expected_checkpoint=checkpoint,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        random_orders=int(spec["random_orders"]),
    )
    return {
        "arm": arm,
        "method": spec["method"],
        "recipe": "pilot_gamma5" if spec["pilot"] else "legacy_gamma0",
        "source_reports": {
            kind: copy.deepcopy(dict(source_identities[f"{arm}_{kind}"]))
            for kind in SOURCE_KINDS
        },
        "logical_evaluation": logical,
        "sampling": {
            "report": sampling["report_identity"],
            "manifest": sampling["manifest_identity"],
            "progress": sampling["sampling_progress"]["identity"],
            "sample_count": len(generated_images),
            "sample_set_sha256": sampling["sample_set_sha256"],
            "protocol": copy.deepcopy(dict(protocol)),
        },
        "checkpoint": checkpoint,
        "checkpoint_evaluation": checkpoint_evaluation,
        "class_fidelity": {
            "report": copy.deepcopy(
                dict(source_identities[f"{arm}_class_fidelity"])
            ),
            "classifier": classifier,
            "sample_provenance_matches_generation": True,
        },
        "real_set": real_set,
        "cross_bindings": {
            "preflight_generation_checkpoint_match": True,
            "generation_class_fidelity_sample_set_match": True,
            "generation_checkpoint_evaluation_checkpoint_match": True,
            "training_control_checkpoint_match": True,
            "training_binding_label": training_binding_label,
        },
    }


def physical_replay(
    *,
    output_root: Path,
    result_path: Path,
    expected_result_sha256: str,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
) -> dict[str, Any]:
    result, result_identity, source_identities, payloads = _logical_result_replay(
        result_path,
        expected_result_sha256=expected_result_sha256,
    )
    canonical_result = output_root / "reports" / "min_snr_pilot_result.json"
    if result_path.resolve() != canonical_result.resolve():
        raise ValueError("Min-SNR pilot result path is not canonical")
    preparation_payload = _mapping(
        payloads.get("preparation"),
        label="Min-SNR terminal preparation",
    )
    preparation_git = _clean_git(
        preparation_payload.get("builder_git"),
        label="Min-SNR preparation builder Git",
        expected_revision=expected_revision,
        expected_branch=expected_branch,
    )
    if (
        str(preparation_payload.get("output_root"))
        != output_root.resolve().as_posix()
        or preparation_git.get("tree") != expected_tree
        or result.get("builder_git") != preparation_payload.get("builder_git")
    ):
        raise ValueError("Min-SNR terminal preparation identity differs")
    expected_canonical_paths = {
        "pilot_cofitok_training_audit": output_root
        / "reports/training_audits/cofitok_50k_physical_audit.json",
        "pilot_dense_training_audit": output_root
        / "reports/training_audits/dense_identity_50k_physical_audit.json",
    }
    for arm in ARM_SPECS:
        root = output_root / "evaluations" / arm
        expected_canonical_paths.update(
            {
                f"{arm}_sampling_preflight": root / "sampling_preflight.json",
                f"{arm}_generation": root
                / "samples/metrics/generation_metrics_report.json",
                f"{arm}_class_fidelity": root
                / "samples/class_fidelity/class_fidelity_report.json",
                f"{arm}_checkpoint_eval": root
                / "checkpoint_eval/checkpoint_evaluation_report.json",
            }
        )
    for name, path in expected_canonical_paths.items():
        if Path(source_identities[name]["path"]) != path.resolve():
            raise ValueError(f"Min-SNR terminal source path is not canonical: {name}")

    preparation = _verify_preparation_sources(payloads["preparation"])

    audit_source_names = {
        "cofitok": "pilot_cofitok_training_audit",
        "dense_identity": "pilot_dense_training_audit",
    }
    training = {
        method: _verify_pilot_training(
            method=method,
            audit_path=Path(
                source_identities[audit_source_names[method]]["path"]
            ),
            audit=payloads[audit_source_names[method]],
            expected_revision=expected_revision,
            expected_tree=expected_tree,
            expected_branch=expected_branch,
        )
        for method in min_snr_pilot.METHODS
    }
    real_cache: dict[str, dict[str, Any]] = {}
    classifier_cache: dict[str, dict[str, Any]] = {}
    arms = {
        arm: _verify_arm(
            arm=arm,
            spec=spec,
            source_identities=source_identities,
            payloads=payloads,
            expected_revision=expected_revision,
            expected_branch=expected_branch,
            training_checkpoint=(
                training[str(spec["method"])]["checkpoint"]
                if spec["pilot"]
                else preparation["legacy_controls"][str(spec["method"])][
                    "checkpoint"
                ]
            ),
            training_binding_label=(
                "pilot training audit"
                if spec["pilot"]
                else "legacy preparation control"
            ),
            real_cache=real_cache,
            classifier_cache=classifier_cache,
        )
        for arm, spec in ARM_SPECS.items()
    }
    classifiers = {
        _canonical_digest(row["class_fidelity"]["classifier"])
        for row in arms.values()
    }
    real_sets = {_canonical_digest(row["real_set"]) for row in arms.values()}
    if len(classifiers) != 1:
        raise ValueError("Min-SNR terminal arms use different classifiers")
    if len(real_sets) != 1:
        raise ValueError("Min-SNR terminal arms use different real sets")
    checkpoint_sources = {
        "legacy_cofitok": preparation["legacy_controls"]["cofitok"]["checkpoint"],
        "legacy_dense_identity": preparation["legacy_controls"]["dense_identity"][
            "checkpoint"
        ],
        "pilot_cofitok": training["cofitok"]["checkpoint"],
        "pilot_dense_identity": training["dense_identity"]["checkpoint"],
    }
    for name, expected_checkpoint in checkpoint_sources.items():
        replayed_checkpoint = _verify_checkpoint(
            expected_checkpoint["payload"]["path"],
            label=f"terminal post-replay {name}",
            expected_step=min_snr_pilot.PILOT_STEP,
        )
        if replayed_checkpoint != expected_checkpoint:
            raise ValueError(f"terminal checkpoint changed during replay: {name}")
    shared_classifier = next(iter(arms.values()))["class_fidelity"]["classifier"]
    if (
        file_identity(shared_classifier["physical_weights"]["path"])
        != shared_classifier["physical_weights"]
    ):
        raise ValueError("terminal classifier changed during replay")
    if file_identity(result_path) != result_identity:
        raise ValueError("Min-SNR pilot result changed during physical replay")
    if _actual_source_identities(source_identities) != source_identities:
        raise ValueError("Min-SNR terminal reports changed during physical replay")
    return {
        "result": result_identity,
        "result_logical_replay_exact": True,
        "selection_status": result["selection_status"],
        "source_reports": source_identities,
        "preparation": preparation,
        "pilot_training": training,
        "evaluation_arms": arms,
        "shared_classifier": shared_classifier,
        "shared_real_set": next(iter(arms.values()))["real_set"],
        "all_checkpoint_payloads_physically_hashed": True,
        "all_sample_sets_physically_hashed": True,
        "all_classifier_weights_physically_hashed": True,
        "all_source_reports_physically_hashed": True,
        "all_training_control_checkpoints_physically_bound": True,
        "all_checkpoint_payloads_rehashed_after_replay": True,
        "all_terminal_sources_rehashed_after_replay": True,
    }


def build_terminal_guard(
    *,
    output_root: Path,
    result_path: Path,
    expected_result_sha256: str,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    guard_git: Mapping[str, Any],
    source_code_binding: Mapping[str, Any],
) -> dict[str, Any]:
    validated_guard_git = _guard_git(guard_git)
    validated_source_code = _source_code_binding(source_code_binding)
    if (
        validated_source_code["training_source"]
        != {
            "revision": expected_revision,
            "tree": expected_tree,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or validated_source_code["guard_source"] != validated_guard_git
    ):
        raise ValueError("Min-SNR terminal source-code Git binding differs")
    source_before = _verify_source_code_files(validated_source_code)
    first = physical_replay(
        output_root=output_root,
        result_path=result_path,
        expected_result_sha256=expected_result_sha256,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
    )
    source_between = _verify_source_code_files(validated_source_code)
    second = physical_replay(
        output_root=output_root,
        result_path=result_path,
        expected_result_sha256=expected_result_sha256,
        expected_revision=expected_revision,
        expected_tree=expected_tree,
        expected_branch=expected_branch,
    )
    source_after = _verify_source_code_files(validated_source_code)
    first_digest = _canonical_digest(first)
    second_digest = _canonical_digest(second)
    if (
        first != second
        or first_digest != second_digest
        or source_before != source_between
        or source_before != source_after
    ):
        raise ValueError("Min-SNR terminal evidence changed during double replay")
    report = {
        "schema": TERMINAL_GUARD_SCHEMA,
        "role": TERMINAL_GUARD_ROLE,
        "status": "pass",
        "scientific_status": "physical_evidence_replayed_non_authorizing",
        "terminal_status": "hold",
        "generation_advantage_proven": False,
        "selection_status": first["selection_status"],
        "guard_git": validated_guard_git,
        "source_code_binding": validated_source_code,
        "source_code_replay": {
            "pass_count": 3,
            "aggregate_sha256": source_before["aggregate_sha256"],
            "physical_identities_verified": True,
            "identical": True,
        },
        "execution_policy": {
            "cpu_only": True,
            "cuda_visible_devices_empty_required": True,
            "omp_num_threads": 1,
            "mkl_num_threads": 1,
            "permanently_non_authorizing": True,
            "may_run_only_after_terminal_result_exists": True,
            "persistent_process_allowed": False,
            "waiter_deployment_allowed": False,
        },
        "double_replay": {
            "pass_count": 2,
            "first_sha256": first_digest,
            "second_sha256": second_digest,
            "identical": True,
        },
        "evidence": first,
        "claim_policy": {
            "pilot_screening_result_physically_replayed": True,
            "matched_generation_advantage_claim_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "absolute_generation_usability_claim_allowed": False,
            "continuation_decision_made": False,
            "screening_result_may_be_used_only_by_a_separate_future_gate": True,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }
    validate_terminal_guard(report)
    return report


def validate_terminal_guard(report: Mapping[str, Any]) -> dict[str, Any]:
    replay = _mapping(report.get("double_replay"), label="terminal double replay")
    evidence = _mapping(report.get("evidence"), label="terminal guard evidence")
    claims = _mapping(report.get("claim_policy"), label="terminal claim policy")
    execution = _mapping(report.get("execution_policy"), label="terminal execution policy")
    source_replay = _mapping(
        report.get("source_code_replay"),
        label="terminal source-code replay",
    )
    guard_git = _guard_git(report.get("guard_git"))
    source_code = _source_code_binding(report.get("source_code_binding"))
    digest = _canonical_digest(evidence)
    if (
        report.get("schema") != TERMINAL_GUARD_SCHEMA
        or report.get("role") != TERMINAL_GUARD_ROLE
        or report.get("status") != "pass"
        or report.get("scientific_status")
        != "physical_evidence_replayed_non_authorizing"
        or report.get("terminal_status") != "hold"
        or report.get("generation_advantage_proven") is not False
        or report.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or execution
        != {
            "cpu_only": True,
            "cuda_visible_devices_empty_required": True,
            "omp_num_threads": 1,
            "mkl_num_threads": 1,
            "permanently_non_authorizing": True,
            "may_run_only_after_terminal_result_exists": True,
            "persistent_process_allowed": False,
            "waiter_deployment_allowed": False,
        }
        or source_code["guard_source"] != guard_git
        or source_replay
        != {
            "pass_count": 3,
            "aggregate_sha256": source_code["aggregate_sha256"],
            "physical_identities_verified": True,
            "identical": True,
        }
        or int(replay.get("pass_count", -1)) != 2
        or replay.get("identical") is not True
        or replay.get("first_sha256") != digest
        or replay.get("second_sha256") != digest
        or claims
        != {
            "pilot_screening_result_physically_replayed": True,
            "matched_generation_advantage_claim_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "absolute_generation_usability_claim_allowed": False,
            "continuation_decision_made": False,
            "screening_result_may_be_used_only_by_a_separate_future_gate": True,
        }
        or evidence.get("selection_status") != report.get("selection_status")
        or evidence.get("result_logical_replay_exact") is not True
        or evidence.get("all_checkpoint_payloads_physically_hashed") is not True
        or evidence.get("all_sample_sets_physically_hashed") is not True
        or evidence.get("all_classifier_weights_physically_hashed") is not True
        or evidence.get("all_source_reports_physically_hashed") is not True
        or evidence.get("all_training_control_checkpoints_physically_bound") is not True
        or evidence.get("all_checkpoint_payloads_rehashed_after_replay") is not True
        or evidence.get("all_terminal_sources_rehashed_after_replay") is not True
    ):
        raise ValueError("Min-SNR terminal physical guard contract differs")
    return copy.deepcopy(dict(report))
