"""User-stage authorization boundary for terminal-SNR large-capacity work.

The stage authorization created here is deliberately *not* a training or
execution authorization.  It binds the user's explicit 250M/300K goal to one
passing terminal-SNR preparation and one exact clean execution checkout, then
permits only the prelaunch evidence collection needed to construct a later
source-bound execution authorization and immutable launch receipt.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.data.provenance import validate_dataset_provenance
from cofitok.environment import runtime_environment_sha256
from cofitok.generation.terminal_snr_large_capacity import (
    BASE_CHANNELS,
    CONFIG_FILENAMES,
    EFFECTIVE_BATCH_SIZE,
    ENDPOINT_FRACTION,
    EXPECTED_PARAMETER_COUNTS,
    FORMAL_EVALUATION_CONTRACT,
    FULL_STAGE_DIRNAME,
    METHODS,
    MILESTONE_STEPS,
    TARGET_STEPS,
    validate_terminal_snr_large_capacity_config_pair,
    validate_terminal_snr_large_capacity_preparation_contract,
)


STAGE_AUTHORIZATION_SCHEMA = (
    "cofitok_generation_terminal_snr_large_capacity_stage_authorization_v1"
)
STAGE_AUTHORIZATION_ROLE = (
    "user_goal_bound_terminal_snr_large_capacity_prelaunch_approval"
)
STAGE_AUTHORIZATION_SCOPE = (
    "fresh_endpoint0975_matched_base256_250m_300k_prelaunch_evidence_only"
)
PAIR_VALIDATION_SCHEMA = (
    "cofitok_generation_terminal_snr_large_capacity_pair_validation_v1"
)
PAIR_VALIDATION_ROLE = (
    "source_bound_terminal_snr_large_capacity_matched_pair_validation"
)
LIVE_SNAPSHOT_SCHEMA = (
    "cofitok_generation_terminal_snr_large_capacity_live_snapshot_v1"
)
LIVE_SNAPSHOT_ROLE = (
    "terminal_snr_large_capacity_idle_gpu_and_output_absence_snapshot"
)

RUNTIME_CANDIDATES = ((1, 64), (2, 32), (4, 16))
RUNTIME_BASELINE = (1, 64)
RUNTIME_BENCHMARK_STEPS = 8
RUNTIME_WARMUP_STEPS = 2
RUNTIME_MAX_MEMORY_FRACTION = 0.90

STORAGE_STAGE = "terminal_snr_endpoint0975_large_capacity_300k_training"
# Eight protected 50K/100K/200K/300K checkpoints plus three rolling recovery
# checkpoints per method.  The sample reserve covers both formal 50K sets and
# every 2,048-sample matched milestone trend evaluation.
STORAGE_CHECKPOINT_COUNT = 14
# The live reference is a base-128 checkpoint; base-256 is approximately four
# times larger, with an additional 6.25% reserve for payload/schema overhead.
STORAGE_CHECKPOINT_SIZE_MULTIPLIER = 4.25
STORAGE_SAMPLE_COUNT = 2 * 50_000 + 4 * 2 * 2_048
STORAGE_ESTIMATED_SAMPLE_BYTES = 256 * 1024
STORAGE_ADDITIONAL_BYTES = 32 * 1024**3
STORAGE_SAFETY_MARGIN_BYTES = 128 * 1024**3
LIVE_MAX_IDLE_MEMORY_MIB = 16
LIVE_MAX_IDLE_UTILIZATION_PERCENT = 5
LIVE_MIN_FREE_BYTES = STORAGE_SAFETY_MARGIN_BYTES

GOAL_BINDING = {
    "project": "CoFiTok",
    "training": {
        "initialization": "fresh",
        "dataset": "imagenet_256",
        "condition": "terminal_snr_endpoint0975",
        "methods": list(METHODS),
        "base_channels": BASE_CHANNELS,
        "parameter_counts": copy.deepcopy(EXPECTED_PARAMETER_COUNTS),
        "steps_per_method": TARGET_STEPS,
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
    },
    "formal_evaluation": copy.deepcopy(FORMAL_EVALUATION_CONTRACT),
    "terminal_condition": (
        "release requires a passing final gate, release-authorized EMA "
        "artifacts, locked paper integration, and terminal completion audit"
    ),
}

STAGE_BOUNDARY = {
    "user_goal_bound": True,
    "decision_is_execution_authorization": False,
    "readiness_evidence_collection_allowed": True,
    "pair_validation_allowed": True,
    "gpu_runtime_preflight_allowed": True,
    "storage_preflight_allowed": True,
    "live_snapshot_capture_allowed": True,
    "remote_checkout_mutation_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "checkpoint_mutation_allowed": False,
    "resume_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "separate_execution_authorization_required": True,
    "immutable_launch_receipt_required": True,
}

PAIR_VALIDATION_BOUNDARY = {
    "matched_pair_validated": True,
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "checkpoint_mutation_allowed": False,
    "resume_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "runtime_selection_still_required": True,
    "storage_capacity_still_required": True,
    "live_snapshot_still_required": True,
    "separate_execution_authorization_required": True,
    "immutable_launch_receipt_required": True,
}

LIVE_SNAPSHOT_BOUNDARY = {
    "evidence_only": True,
    "decision_is_execution_authorization": False,
    "remote_mutation_allowed": False,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "checkpoint_mutation_allowed": False,
    "resume_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_allowed": False,
    "export_allowed": False,
    "release_allowed": False,
    "process_signals_allowed": False,
    "separate_execution_authorization_required": True,
    "immutable_launch_receipt_required": True,
}


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _float(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return math.nan
    return float(value)


def _identity(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"path", "bytes", "sha256"}:
        raise ValueError(f"{name} identity fields differ")
    path = str(row.get("path", ""))
    if (
        not path
        or not (Path(path).is_absolute() or PurePosixPath(path).is_absolute())
        or not isinstance(row.get("bytes"), int)
        or isinstance(row.get("bytes"), bool)
        or row["bytes"] < 1
        or not _hex(row.get("sha256"), 64)
    ):
        raise ValueError(f"{name} identity is malformed")
    return {key: row[key] for key in ("path", "bytes", "sha256")}


def _git(value: Any, name: str) -> dict[str, Any]:
    row = _object(value, name)
    if set(row) != {"revision", "tree", "branch", "tracked_dirty"}:
        raise ValueError(f"{name} Git identity fields differ")
    if (
        not _hex(row.get("revision"), 40)
        or not _hex(row.get("tree"), 40)
        or not isinstance(row.get("branch"), str)
        or not row["branch"]
        or row.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{name} must identify one exact clean checkout")
    return copy.deepcopy(row)


def _absolute(value: Any, name: str) -> str:
    text = str(value or "")
    if not text or not (
        Path(text).is_absolute() or PurePosixPath(text).is_absolute()
    ):
        raise ValueError(f"{name} must be absolute")
    return text


def _path_is_within(path: str, parent: str) -> bool:
    try:
        PurePosixPath(path).relative_to(PurePosixPath(parent))
    except ValueError:
        return False
    return True


def terminal_snr_large_capacity_execution_lock_path(output_root: str) -> str:
    root = PurePosixPath(
        _absolute(output_root, "large-capacity output root")
    )
    if root.name != FULL_STAGE_DIRNAME:
        raise ValueError("large-capacity output root basename differs")
    return (
        root.parent
        / f".{root.name}.terminal_snr_large_capacity_execution.lock"
    ).as_posix()


def _expected_training_run_dirs(output_root: str) -> dict[str, str]:
    root = PurePosixPath(
        _absolute(output_root, "large-capacity output root")
    )
    if root.name != FULL_STAGE_DIRNAME:
        raise ValueError("large-capacity output root basename differs")
    return {
        method: (root / "training" / method).as_posix()
        for method in METHODS
    }


def _explicit_goal_instruction(value: Any) -> str:
    text = str(value or "").strip()
    normalized = text.lower()
    required = ("250m", "300k", "50k", "ddim-250", "terminal")
    if not text or any(token not in normalized for token in required):
        raise ValueError(
            "large-capacity source instruction must bind the explicit "
            "250M/300K, 50K DDIM-250, and terminal-audit goal"
        )
    return text


def _selection(
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    prepared = validate_terminal_snr_large_capacity_preparation_contract(
        preparation
    )
    root = _absolute(output_root, "large-capacity output root")
    selected = _object(prepared.get("selection"), "large-capacity selection")
    sources = _object(prepared.get("source_evidence"), "preparation sources")
    configs = _object(sources.get("configs"), "large-capacity configs")
    if (
        selected.get("output_root") != root
        or selected.get("fresh_initialization_required") is not True
        or selected.get("resume_checkpoint_allowed") is not False
        or selected.get("condition") != "endpoint0975"
        or selected.get("methods") != list(METHODS)
        or int(selected.get("configured_training_steps", -1)) != TARGET_STEPS
        or set(configs) != set(METHODS)
    ):
        raise ValueError("large-capacity preparation selection differs")
    config_ids = {
        method: _identity(configs[method], f"{method} prepared config")
        for method in METHODS
    }
    return {
        "preparation": _identity(
            preparation_identity, "large-capacity preparation"
        ),
        "execution_checkout": _git(
            execution_checkout, "large-capacity execution checkout"
        ),
        "output_root": root,
        "configs": config_ids,
        "condition": "terminal_snr_endpoint0975",
        "cosine_endpoint_fraction": ENDPOINT_FRACTION,
        "methods": list(METHODS),
        "base_channels": BASE_CHANNELS,
        "parameter_counts": copy.deepcopy(EXPECTED_PARAMETER_COUNTS),
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "configured_training_steps": TARGET_STEPS,
        "milestone_steps": list(MILESTONE_STEPS),
        "fresh_initialization_required": True,
        "resume_allowed": False,
        "formal_evaluation": copy.deepcopy(FORMAL_EVALUATION_CONTRACT),
    }


def build_terminal_snr_large_capacity_stage_authorization(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
    approved_by: str,
    approved_at: str,
    source_instruction: str,
) -> dict[str, Any]:
    report = {
        "schema_version": STAGE_AUTHORIZATION_SCHEMA,
        "role": STAGE_AUTHORIZATION_ROLE,
        "status": "approved",
        "scope": STAGE_AUTHORIZATION_SCOPE,
        "selection": _selection(
            preparation,
            preparation_identity,
            execution_checkout,
            output_root,
        ),
        "approval_record": {
            "approved_by": str(approved_by).strip(),
            "approved_at": str(approved_at).strip(),
            "source_instruction": _explicit_goal_instruction(source_instruction),
            "goal_binding": copy.deepcopy(GOAL_BINDING),
        },
        "next_stage": {
            "route": "collect_source_bound_large_capacity_prelaunch_evidence",
            "pair_validation_required": True,
            "runtime_selection_required": True,
            "storage_capacity_required": True,
            "live_snapshot_required": True,
            "separate_execution_authorization_required": True,
            "immutable_launch_receipt_required": True,
            "execution_ready": False,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(STAGE_BOUNDARY),
    }
    return validate_terminal_snr_large_capacity_stage_authorization(
        report,
        preparation=preparation,
        preparation_identity=preparation_identity,
        execution_checkout=execution_checkout,
        expected_output_root=output_root,
    )


def validate_terminal_snr_large_capacity_stage_authorization(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "large-capacity stage authorization")
    approval = _object(row.get("approval_record"), "large-capacity approval")
    next_stage = _object(row.get("next_stage"), "large-capacity next stage")
    expected_selection = _selection(
        preparation,
        preparation_identity,
        execution_checkout,
        expected_output_root,
    )
    if (
        set(row)
        != {
            "schema_version",
            "role",
            "status",
            "scope",
            "selection",
            "approval_record",
            "next_stage",
            "authorization_boundary",
        }
        or set(approval)
        != {"approved_by", "approved_at", "source_instruction", "goal_binding"}
        or set(next_stage)
        != {
            "route",
            "pair_validation_required",
            "runtime_selection_required",
            "storage_capacity_required",
            "live_snapshot_required",
            "separate_execution_authorization_required",
            "immutable_launch_receipt_required",
            "execution_ready",
            "training_launch_allowed",
            "full_300k_launch_allowed",
        }
        or row.get("schema_version") != STAGE_AUTHORIZATION_SCHEMA
        or row.get("role") != STAGE_AUTHORIZATION_ROLE
        or row.get("status") != "approved"
        or row.get("scope") != STAGE_AUTHORIZATION_SCOPE
        or row.get("selection") != expected_selection
        or row.get("authorization_boundary") != STAGE_BOUNDARY
        or not str(approval.get("approved_by", "")).strip()
        or not str(approval.get("approved_at", "")).strip()
        or _explicit_goal_instruction(approval.get("source_instruction"))
        != approval.get("source_instruction")
        or approval.get("goal_binding") != GOAL_BINDING
        or next_stage.get("route")
        != "collect_source_bound_large_capacity_prelaunch_evidence"
        or next_stage.get("pair_validation_required") is not True
        or next_stage.get("runtime_selection_required") is not True
        or next_stage.get("storage_capacity_required") is not True
        or next_stage.get("live_snapshot_required") is not True
        or next_stage.get("separate_execution_authorization_required") is not True
        or next_stage.get("immutable_launch_receipt_required") is not True
        or next_stage.get("execution_ready") is not False
        or next_stage.get("training_launch_allowed") is not False
        or next_stage.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("terminal-SNR large-capacity stage authorization differs")
    return copy.deepcopy(row)


def _pair_validation_report(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    cofitok_config: Mapping[str, Any],
    cofitok_config_identity: Mapping[str, Any],
    dense_config: Mapping[str, Any],
    dense_config_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    prepared = validate_terminal_snr_large_capacity_preparation_contract(
        preparation
    )
    prepared_id = _identity(
        preparation_identity, "large-capacity preparation"
    )
    execution = _git(
        execution_checkout, "large-capacity execution checkout"
    )
    root = _absolute(output_root, "large-capacity output root")
    stage = validate_terminal_snr_large_capacity_stage_authorization(
        stage_authorization,
        preparation=prepared,
        preparation_identity=prepared_id,
        execution_checkout=execution,
        expected_output_root=root,
    )
    stage_id = _identity(
        stage_authorization_identity, "large-capacity stage authorization"
    )
    prepared_sources = _object(
        prepared.get("source_evidence"), "large-capacity preparation sources"
    )
    prepared_configs = _object(
        prepared_sources.get("configs"), "large-capacity prepared configs"
    )
    config_ids = {
        "cofitok": _identity(
            cofitok_config_identity, "large-capacity CoFiTok config"
        ),
        "dense_identity": _identity(
            dense_config_identity, "large-capacity dense config"
        ),
    }
    if set(prepared_configs) != set(METHODS) or any(
        config_ids[method]
        != _identity(prepared_configs[method], f"prepared {method} config")
        for method in METHODS
    ):
        raise ValueError("large-capacity pair-validation config identity differs")
    for method in METHODS:
        if PurePosixPath(config_ids[method]["path"]).name != CONFIG_FILENAMES[method]:
            raise ValueError(
                f"large-capacity {method} pair-validation config path differs"
            )
    validation = validate_terminal_snr_large_capacity_config_pair(
        cofitok_config=cofitok_config,
        dense_config=dense_config,
    )
    prepared_selection = _object(
        prepared.get("selection"), "large-capacity prepared selection"
    )
    if validation != prepared_selection.get("config_validation"):
        raise ValueError(
            "large-capacity pair validation differs from preparation"
        )
    return {
        "schema_version": PAIR_VALIDATION_SCHEMA,
        "role": PAIR_VALIDATION_ROLE,
        "status": "pass",
        "execution_checkout": execution,
        "source_evidence": {
            "preparation": prepared_id,
            "stage_authorization": stage_id,
            "configs": config_ids,
        },
        "selection": {
            "output_root": root,
            "condition": "terminal_snr_endpoint0975",
            "methods": list(METHODS),
            "base_channels": BASE_CHANNELS,
            "parameter_counts": copy.deepcopy(EXPECTED_PARAMETER_COUNTS),
            "effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "configured_training_steps": TARGET_STEPS,
            "milestone_steps": list(MILESTONE_STEPS),
            "fresh_initialization_required": True,
            "resume_allowed": False,
        },
        "validation": validation,
        "next_stage": {
            "route": "collect_large_capacity_runtime_storage_and_live_snapshot",
            "runtime_selection_required": True,
            "storage_capacity_required": True,
            "live_snapshot_required": True,
            "separate_execution_authorization_required": True,
            "immutable_launch_receipt_required": True,
            "execution_ready": False,
            "training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(PAIR_VALIDATION_BOUNDARY),
    }


def build_terminal_snr_large_capacity_pair_validation(
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    cofitok_config: Mapping[str, Any],
    cofitok_config_identity: Mapping[str, Any],
    dense_config: Mapping[str, Any],
    dense_config_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    report = _pair_validation_report(
        preparation=preparation,
        preparation_identity=preparation_identity,
        stage_authorization=stage_authorization,
        stage_authorization_identity=stage_authorization_identity,
        cofitok_config=cofitok_config,
        cofitok_config_identity=cofitok_config_identity,
        dense_config=dense_config,
        dense_config_identity=dense_config_identity,
        execution_checkout=execution_checkout,
        output_root=output_root,
    )
    return validate_terminal_snr_large_capacity_pair_validation(
        report,
        preparation=preparation,
        preparation_identity=preparation_identity,
        stage_authorization=stage_authorization,
        stage_authorization_identity=stage_authorization_identity,
        cofitok_config=cofitok_config,
        cofitok_config_identity=cofitok_config_identity,
        dense_config=dense_config,
        dense_config_identity=dense_config_identity,
        execution_checkout=execution_checkout,
        expected_output_root=output_root,
    )


def validate_terminal_snr_large_capacity_pair_validation(
    report: Mapping[str, Any],
    *,
    preparation: Mapping[str, Any],
    preparation_identity: Mapping[str, Any],
    stage_authorization: Mapping[str, Any],
    stage_authorization_identity: Mapping[str, Any],
    cofitok_config: Mapping[str, Any],
    cofitok_config_identity: Mapping[str, Any],
    dense_config: Mapping[str, Any],
    dense_config_identity: Mapping[str, Any],
    execution_checkout: Mapping[str, Any],
    expected_output_root: str,
) -> dict[str, Any]:
    row = _object(report, "large-capacity pair validation")
    expected = _pair_validation_report(
        preparation=preparation,
        preparation_identity=preparation_identity,
        stage_authorization=stage_authorization,
        stage_authorization_identity=stage_authorization_identity,
        cofitok_config=cofitok_config,
        cofitok_config_identity=cofitok_config_identity,
        dense_config=dense_config,
        dense_config_identity=dense_config_identity,
        execution_checkout=execution_checkout,
        output_root=expected_output_root,
    )
    if row != expected:
        raise ValueError("terminal-SNR large-capacity pair validation differs")
    return copy.deepcopy(row)


def _runtime_git(execution_checkout: Mapping[str, Any]) -> dict[str, Any]:
    git = _git(execution_checkout, "large-capacity runtime checkout")
    return {
        "revision": git["revision"],
        "branch": git["branch"],
        "tracked_dirty": False,
    }


def validate_terminal_snr_large_capacity_runtime_selection(
    report: Mapping[str, Any],
    *,
    execution_checkout: Mapping[str, Any],
    config_identities: Mapping[str, Mapping[str, Any]],
    run_dirs: Mapping[str, str],
    benchmark_root: str,
) -> dict[str, Any]:
    row = _object(report, "large-capacity runtime selection")
    if set(config_identities) != set(METHODS) or set(run_dirs) != set(METHODS):
        raise ValueError("large-capacity runtime method set differs")
    execution = _git(
        execution_checkout, "large-capacity runtime execution checkout"
    )
    selected = _object(row.get("selected"), "large-capacity selected runtime")
    baseline = _object(row.get("baseline"), "large-capacity runtime baseline")
    policy = _object(row.get("policy"), "large-capacity runtime policy")
    lock = _object(row.get("selection_lock"), "large-capacity runtime lock")
    configs = {
        method: _identity(
            config_identities[method], f"large-capacity {method} config"
        )
        for method in METHODS
    }
    normalized_runs = {
        method: _absolute(run_dirs[method], f"large-capacity {method} run")
        for method in METHODS
    }
    inferred_root = PurePosixPath(normalized_runs["cofitok"]).parent.parent
    expected_run_layout = {
        method: (inferred_root / "training" / method).as_posix()
        for method in METHODS
    }
    if (
        inferred_root.name != FULL_STAGE_DIRNAME
        or normalized_runs != expected_run_layout
    ):
        raise ValueError("large-capacity runtime run layout differs")
    expected_runs = [normalized_runs[method] for method in METHODS]
    expected_candidates = [
        {
            "micro_batch_size": micro,
            "gradient_accumulation_steps": accumulation,
        }
        for micro, accumulation in RUNTIME_CANDIDATES
    ]
    expected_config_sha = {
        method: configs[method]["sha256"] for method in METHODS
    }
    benchmark = _absolute(benchmark_root, "large-capacity benchmark root")
    if _path_is_within(benchmark, inferred_root.as_posix()):
        raise ValueError("large-capacity runtime benchmarks must be outside output root")
    candidates = row.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(
        RUNTIME_CANDIDATES
    ):
        raise ValueError("large-capacity runtime candidate set differs")
    observed_candidates: list[tuple[int, int]] = []
    eligible_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        value = _object(candidate, "large-capacity runtime candidate")
        micro = int(value.get("micro_batch_size", -1))
        accumulation = int(value.get("gradient_accumulation_steps", -1))
        observed_candidates.append((micro, accumulation))
        methods = _object(
            value.get("methods"), "large-capacity runtime candidate methods"
        )
        if (
            micro * accumulation != EFFECTIVE_BATCH_SIZE
            or int(value.get("effective_batch_size", -1))
            != EFFECTIVE_BATCH_SIZE
            or set(methods) != set(METHODS)
        ):
            raise ValueError("large-capacity runtime candidate differs")
        method_scores: list[float] = []
        method_memory: list[float] = []
        method_runtime_shas: list[str] = []
        method_dataset_shas: list[str] = []
        all_completed = True
        for method in METHODS:
            method_report = _object(
                methods[method], f"large-capacity {method} runtime benchmark"
            )
            if method_report.get("status") != "completed":
                all_completed = False
                continue
            environment = _object(
                method_report.get("runtime_environment"),
                f"large-capacity {method} runtime environment",
            )
            environment_sha = runtime_environment_sha256(environment)
            provenance = _object(
                method_report.get("dataset_provenance"),
                f"large-capacity {method} dataset provenance",
            )
            dataset_sha = validate_dataset_provenance(
                provenance, expected_dataset="imagenet_256"
            )["identity_sha256"]
            seconds = _float(method_report.get("mean_optimizer_step_seconds"))
            throughput = _float(method_report.get("images_per_second"))
            peak = int(method_report.get("peak_vram_bytes", 0))
            total = int(method_report.get("device_total_memory_bytes", 0))
            if (
                int(method_report.get("effective_batch_size", -1))
                != EFFECTIVE_BATCH_SIZE
                or method_report.get("runtime_environment_sha256")
                != environment_sha
                or provenance.get("identity_sha256") != dataset_sha
                or not math.isfinite(seconds)
                or seconds <= 0.0
                or not math.isfinite(throughput)
                or throughput <= 0.0
                or peak <= 0
                or total <= 0
                or peak > total
            ):
                raise ValueError(
                    "large-capacity runtime benchmark provenance differs"
                )
            method_scores.append(seconds)
            method_memory.append(peak / total)
            method_runtime_shas.append(environment_sha)
            method_dataset_shas.append(dataset_sha)
        reasons = value.get("ineligible_reasons")
        if not isinstance(reasons, list) or any(
            not isinstance(reason, str) or not reason for reason in reasons
        ):
            raise ValueError("large-capacity runtime ineligible reasons differ")
        eligible = value.get("eligible") is True
        candidate_score = value.get("selection_score_seconds")
        candidate_memory = value.get("max_memory_fraction")
        candidate_score_value = _float(candidate_score)
        candidate_memory_value = _float(candidate_memory)
        if eligible:
            expected_score = max(method_scores) if len(method_scores) == 2 else None
            expected_memory = max(method_memory) if len(method_memory) == 2 else None
            if (
                reasons != []
                or not all_completed
                or expected_score is None
                or expected_memory is None
                or value.get("runtime_environment_sha256")
                != row.get("runtime_environment_sha256")
                or value.get("dataset_identity_sha256")
                != row.get("dataset_identity_sha256")
                or len(set(method_runtime_shas)) != 1
                or len(set(method_dataset_shas)) != 1
                or not math.isclose(
                    candidate_score_value, expected_score, rel_tol=1e-12
                )
                or not math.isclose(
                    candidate_memory_value, expected_memory, rel_tol=1e-12
                )
                or expected_memory > RUNTIME_MAX_MEMORY_FRACTION
            ):
                raise ValueError("large-capacity eligible runtime differs")
            eligible_candidates.append(
                {
                    "pair": (micro, accumulation),
                    "score": expected_score,
                    "memory": expected_memory,
                }
            )
        elif not reasons or candidate_score is not None or candidate_memory is not None:
            raise ValueError("large-capacity ineligible runtime differs")
    selected_pair = (
        int(selected.get("micro_batch_size", -1)),
        int(selected.get("gradient_accumulation_steps", -1)),
    )
    baseline_pair = (
        int(baseline.get("micro_batch_size", -1)),
        int(baseline.get("gradient_accumulation_steps", -1)),
    )
    selection_score = _float(selected.get("selection_score_seconds"))
    memory_fraction = _float(selected.get("max_memory_fraction"))
    speedup = _float(selected.get("estimated_speedup_over_baseline"))
    baseline_score = _float(baseline.get("selection_score_seconds"))
    baseline_memory = _float(baseline.get("max_memory_fraction"))
    if not eligible_candidates:
        raise ValueError("large-capacity runtime has no eligible candidate")
    recomputed_selected = min(
        eligible_candidates,
        key=lambda candidate: (candidate["score"], candidate["pair"][0]),
    )
    eligible_by_pair = {
        candidate["pair"]: candidate for candidate in eligible_candidates
    }
    recomputed_baseline = eligible_by_pair.get(RUNTIME_BASELINE)
    if (
        set(row)
        != {
            "schema_version",
            "status",
            "policy",
            "selected",
            "runtime_environment_sha256",
            "dataset_identity_sha256",
            "candidates",
            "baseline",
            "git_revision",
            "config_sha256",
            "benchmark_root",
            "selection_lock",
        }
        or int(row.get("schema_version", -1)) != 3
        or row.get("status") != "selected"
        or observed_candidates != list(RUNTIME_CANDIDATES)
        or selected_pair != recomputed_selected["pair"]
        or baseline_pair != RUNTIME_BASELINE
        or recomputed_baseline is None
        or int(selected.get("effective_batch_size", -1))
        != EFFECTIVE_BATCH_SIZE
        or int(baseline.get("effective_batch_size", -1))
        != EFFECTIVE_BATCH_SIZE
        or not math.isfinite(selection_score)
        or selection_score <= 0.0
        or not math.isfinite(memory_fraction)
        or not 0.0 < memory_fraction <= RUNTIME_MAX_MEMORY_FRACTION
        or not math.isfinite(speedup)
        or speedup <= 0.0
        or not math.isfinite(baseline_score)
        or baseline_score <= 0.0
        or not math.isfinite(baseline_memory)
        or not 0.0 < baseline_memory <= RUNTIME_MAX_MEMORY_FRACTION
        or not math.isclose(
            selection_score, recomputed_selected["score"], rel_tol=1e-12
        )
        or not math.isclose(
            memory_fraction, recomputed_selected["memory"], rel_tol=1e-12
        )
        or not math.isclose(
            baseline_score, recomputed_baseline["score"], rel_tol=1e-12
        )
        or not math.isclose(
            baseline_memory, recomputed_baseline["memory"], rel_tol=1e-12
        )
        or not math.isclose(
            speedup, baseline_score / selection_score, rel_tol=1e-12
        )
        or policy
        != {
            "shared_candidate_required": True,
            "score": "minimize_worst_method_mean_optimizer_step_seconds",
            "expected_effective_batch_size": EFFECTIVE_BATCH_SIZE,
            "max_memory_fraction": RUNTIME_MAX_MEMORY_FRACTION,
        }
        or row.get("git_revision") != execution["revision"]
        or row.get("config_sha256") != expected_config_sha
        or row.get("benchmark_root") != benchmark
        or lock.get("schema_version") != 2
        or lock.get("mode") != "freeze_on_training_state"
        or lock.get("training_run_dirs") != expected_runs
        or lock.get("candidates") != expected_candidates
        or lock.get("baseline_candidate")
        != {
            "micro_batch_size": RUNTIME_BASELINE[0],
            "gradient_accumulation_steps": RUNTIME_BASELINE[1],
        }
        or int(lock.get("expected_effective_batch_size", -1))
        != EFFECTIVE_BATCH_SIZE
        or int(lock.get("benchmark_steps", -1))
        != RUNTIME_BENCHMARK_STEPS
        or int(lock.get("warmup_steps", -1)) != RUNTIME_WARMUP_STEPS
        or _float(lock.get("max_memory_fraction"))
        != RUNTIME_MAX_MEMORY_FRACTION
        or int(lock.get("training_target_steps", -1)) != TARGET_STEPS
        or lock.get("git") != _runtime_git(execution)
        or lock.get("config_sha256") != expected_config_sha
        or lock.get("benchmark_root") != benchmark
        or not _hex(row.get("runtime_environment_sha256"), 64)
        or not _hex(row.get("dataset_identity_sha256"), 64)
    ):
        raise ValueError("terminal-SNR large-capacity runtime selection differs")
    return {
        "micro_batch_size": selected_pair[0],
        "gradient_accumulation_steps": selected_pair[1],
        "effective_batch_size": EFFECTIVE_BATCH_SIZE,
        "selection_score_seconds": selection_score,
        "max_memory_fraction": memory_fraction,
        "estimated_speedup_over_baseline": speedup,
        "runtime_environment_sha256": row["runtime_environment_sha256"],
        "dataset_identity_sha256": row["dataset_identity_sha256"],
    }


def validate_terminal_snr_large_capacity_storage_capacity(
    report: Mapping[str, Any],
    *,
    execution_checkout: Mapping[str, Any],
    output_root: str,
) -> dict[str, Any]:
    row = _object(report, "large-capacity storage capacity")
    filesystem = _object(
        row.get("filesystem"), "large-capacity storage filesystem"
    )
    plan = _object(row.get("plan"), "large-capacity storage plan")
    root = _absolute(output_root, "large-capacity output root")
    storage_path = PurePosixPath(root).parent.as_posix()
    reference_bytes = int(plan.get("reference_checkpoint_bytes_each", -1))
    checkpoint_bytes = int(plan.get("checkpoint_bytes_each", -1))
    expected_checkpoint_bytes = (
        math.ceil(reference_bytes * STORAGE_CHECKPOINT_SIZE_MULTIPLIER)
        if reference_bytes > 0
        else -1
    )
    expected_required = (
        STORAGE_CHECKPOINT_COUNT * expected_checkpoint_bytes
        + STORAGE_SAMPLE_COUNT * STORAGE_ESTIMATED_SAMPLE_BYTES
        + STORAGE_ADDITIONAL_BYTES
        + STORAGE_SAFETY_MARGIN_BYTES
    )
    free_bytes = int(filesystem.get("free_bytes", -1))
    required_bytes = int(plan.get("required_free_bytes", -1))
    if (
        int(row.get("schema_version", -1)) != 2
        or row.get("role") != "generation_storage_capacity_preflight"
        or row.get("stage") != STORAGE_STAGE
        or row.get("status") != "pass"
        or row.get("git") != _runtime_git(execution_checkout)
        or filesystem.get("path") != storage_path
        or int(filesystem.get("total_bytes", 0)) < 1
        or int(filesystem.get("used_bytes", -1)) < 0
        or free_bytes < 0
        or int(filesystem.get("used_bytes", -1)) + free_bytes
        > int(filesystem.get("total_bytes", 0))
        or int(plan.get("checkpoint_count", -1))
        != STORAGE_CHECKPOINT_COUNT
        or reference_bytes < 1
        or _float(plan.get("checkpoint_size_multiplier"))
        != STORAGE_CHECKPOINT_SIZE_MULTIPLIER
        or checkpoint_bytes != expected_checkpoint_bytes
        or int(plan.get("checkpoint_reserve_bytes", -1))
        != STORAGE_CHECKPOINT_COUNT * checkpoint_bytes
        or int(plan.get("sample_count", -1)) != STORAGE_SAMPLE_COUNT
        or int(plan.get("estimated_sample_bytes_each", -1))
        != STORAGE_ESTIMATED_SAMPLE_BYTES
        or int(plan.get("sample_reserve_bytes", -1))
        != STORAGE_SAMPLE_COUNT * STORAGE_ESTIMATED_SAMPLE_BYTES
        or int(plan.get("additional_bytes", -1)) != STORAGE_ADDITIONAL_BYTES
        or int(plan.get("safety_margin_bytes", -1))
        != STORAGE_SAFETY_MARGIN_BYTES
        or required_bytes != expected_required
        or int(row.get("headroom_bytes", -1))
        != free_bytes - required_bytes
        or free_bytes < required_bytes
    ):
        raise ValueError("terminal-SNR large-capacity storage evidence differs")
    return {
        "filesystem": copy.deepcopy(filesystem),
        "plan": copy.deepcopy(plan),
        "headroom_bytes": int(row["headroom_bytes"]),
    }


def validate_terminal_snr_large_capacity_live_snapshot(
    report: Mapping[str, Any],
    *,
    expected_output_root: str,
    expected_execution_checkout: Mapping[str, Any],
    expected_execution_lock: str,
    expected_runtime_selection_identity: Mapping[str, Any],
    expected_storage_capacity_identity: Mapping[str, Any],
    expected_config_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate the final idle, clean, empty-output prelaunch observation.

    The snapshot is evidence-only.  A later execution authorization and
    immutable launch receipt must bind and physically replay it together with
    every other large-capacity prelaunch source before training can start.
    """

    row = _object(report, "large-capacity live snapshot")
    root = _absolute(expected_output_root, "large-capacity output root")
    if PurePosixPath(root).name != FULL_STAGE_DIRNAME:
        raise ValueError("large-capacity output root basename differs")
    execution = _git(
        expected_execution_checkout,
        "large-capacity live execution checkout",
    )
    exact_lock = terminal_snr_large_capacity_execution_lock_path(root)
    supplied_lock = _absolute(
        expected_execution_lock, "large-capacity execution lock"
    )
    expected_runs = _expected_training_run_dirs(root)
    sources = _object(
        row.get("source_evidence"), "large-capacity live sources"
    )
    configs = _object(
        sources.get("configs"), "large-capacity live configs"
    )
    if set(expected_config_identities) != set(METHODS):
        raise ValueError("large-capacity live config set differs")
    expected_configs = {
        method: _identity(
            expected_config_identities[method],
            f"large-capacity live {method} config",
        )
        for method in METHODS
    }
    if set(configs) == set(METHODS):
        configs = {
            method: _identity(
                configs[method], f"large-capacity live {method} config"
            )
            for method in METHODS
        }
    runtime_id = _identity(
        expected_runtime_selection_identity,
        "large-capacity runtime selection",
    )
    storage_id = _identity(
        expected_storage_capacity_identity,
        "large-capacity storage capacity",
    )
    gpu = row.get("gpu_inventory")
    if not isinstance(gpu, list) or len(gpu) != 1:
        raise ValueError("large-capacity launch requires exactly one target GPU")
    device = _object(gpu[0], "large-capacity GPU inventory")
    compute = row.get("gpu_compute_processes")
    conflicts = row.get("conflicting_processes")
    filesystem = _object(
        row.get("filesystem"), "large-capacity live filesystem"
    )
    environment = _object(
        row.get("runtime_environment"),
        "large-capacity live runtime environment",
    )
    provenance = _object(
        row.get("dataset_provenance"),
        "large-capacity live dataset provenance",
    )
    environment_sha = runtime_environment_sha256(environment)
    dataset_sha = validate_dataset_provenance(
        provenance, expected_dataset="imagenet_256"
    )["identity_sha256"]
    run_dirs = _object(
        row.get("training_run_dirs"), "large-capacity training run dirs"
    )
    if (
        set(row)
        != {
            "schema_version",
            "role",
            "status",
            "execution_checkout",
            "source_evidence",
            "output_root",
            "execution_lock",
            "training_run_dirs",
            "gpu_inventory",
            "gpu_compute_processes",
            "conflicting_processes",
            "output_root_absent",
            "training_state_absent",
            "execution_lock_free",
            "filesystem",
            "runtime_environment",
            "runtime_environment_sha256",
            "dataset_provenance",
            "dataset_identity_sha256",
            "hostname",
            "captured_at",
            "authorization_boundary",
        }
        or set(sources) != {"runtime_selection", "storage_capacity", "configs"}
        or row.get("schema_version") != LIVE_SNAPSHOT_SCHEMA
        or row.get("role") != LIVE_SNAPSHOT_ROLE
        or row.get("status") != "pass"
        or row.get("execution_checkout") != execution
        or row.get("source_evidence")
        != {
            "runtime_selection": runtime_id,
            "storage_capacity": storage_id,
            "configs": expected_configs,
        }
        or row.get("output_root") != root
        or row.get("execution_lock") != exact_lock
        or supplied_lock != exact_lock
        or run_dirs != expected_runs
        or row.get("output_root_absent") is not True
        or row.get("training_state_absent") is not True
        or row.get("execution_lock_free") is not True
        or compute != []
        or conflicts != []
        or set(device)
        != {
            "index",
            "uuid",
            "name",
            "memory_used_mib",
            "memory_total_mib",
            "utilization_percent",
        }
        or int(device.get("index", -1)) != 0
        or not str(device.get("uuid", "")).strip()
        or not str(device.get("name", "")).strip()
        or int(device.get("memory_used_mib", -1)) < 0
        or int(device.get("memory_used_mib", LIVE_MAX_IDLE_MEMORY_MIB + 1))
        > LIVE_MAX_IDLE_MEMORY_MIB
        or int(device.get("memory_total_mib", 0)) < 1
        or int(
            device.get(
                "utilization_percent",
                LIVE_MAX_IDLE_UTILIZATION_PERCENT + 1,
            )
        )
        > LIVE_MAX_IDLE_UTILIZATION_PERCENT
        or set(filesystem)
        != {"path", "free_bytes", "required_free_bytes", "headroom_bytes"}
        or filesystem.get("path") != PurePosixPath(root).parent.as_posix()
        or int(filesystem.get("free_bytes", 0)) < LIVE_MIN_FREE_BYTES
        or int(filesystem.get("required_free_bytes", 0)) < 1
        or int(filesystem.get("free_bytes", -1))
        < int(filesystem.get("required_free_bytes", 0))
        or int(filesystem.get("headroom_bytes", -1))
        != int(filesystem.get("free_bytes", -1))
        - int(filesystem.get("required_free_bytes", 0))
        or row.get("runtime_environment_sha256") != environment_sha
        or row.get("dataset_identity_sha256") != dataset_sha
        or not str(row.get("hostname", "")).strip()
        or not str(row.get("captured_at", "")).strip()
        or row.get("authorization_boundary") != LIVE_SNAPSHOT_BOUNDARY
    ):
        raise ValueError("terminal-SNR large-capacity live snapshot differs")
    return copy.deepcopy(row)


__all__ = [
    "GOAL_BINDING",
    "LIVE_MAX_IDLE_MEMORY_MIB",
    "LIVE_MAX_IDLE_UTILIZATION_PERCENT",
    "LIVE_MIN_FREE_BYTES",
    "LIVE_SNAPSHOT_BOUNDARY",
    "LIVE_SNAPSHOT_ROLE",
    "LIVE_SNAPSHOT_SCHEMA",
    "PAIR_VALIDATION_BOUNDARY",
    "PAIR_VALIDATION_ROLE",
    "PAIR_VALIDATION_SCHEMA",
    "RUNTIME_BASELINE",
    "RUNTIME_BENCHMARK_STEPS",
    "RUNTIME_CANDIDATES",
    "RUNTIME_MAX_MEMORY_FRACTION",
    "RUNTIME_WARMUP_STEPS",
    "STORAGE_ADDITIONAL_BYTES",
    "STORAGE_CHECKPOINT_COUNT",
    "STORAGE_CHECKPOINT_SIZE_MULTIPLIER",
    "STORAGE_ESTIMATED_SAMPLE_BYTES",
    "STORAGE_SAFETY_MARGIN_BYTES",
    "STORAGE_SAMPLE_COUNT",
    "STORAGE_STAGE",
    "STAGE_AUTHORIZATION_ROLE",
    "STAGE_AUTHORIZATION_SCHEMA",
    "STAGE_AUTHORIZATION_SCOPE",
    "STAGE_BOUNDARY",
    "build_terminal_snr_large_capacity_pair_validation",
    "build_terminal_snr_large_capacity_stage_authorization",
    "terminal_snr_large_capacity_execution_lock_path",
    "validate_terminal_snr_large_capacity_live_snapshot",
    "validate_terminal_snr_large_capacity_pair_validation",
    "validate_terminal_snr_large_capacity_runtime_selection",
    "validate_terminal_snr_large_capacity_stage_authorization",
    "validate_terminal_snr_large_capacity_storage_capacity",
]
