from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import math
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from cofitok.generation import sampling_protocol_contract
from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_DATASET,
    QUALITY_BRIDGE_EFFECTIVE_BATCH,
    QUALITY_BRIDGE_RECIPE_STAGE,
    QUALITY_BRIDGE_RESULT_ROLE,
    QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
    QUALITY_BRIDGE_STEPS,
    QUALITY_BRIDGE_TERMINAL_SAMPLES,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.generation_class_fidelity import (
    validate_class_fidelity_qualification,
    validate_class_fidelity_report,
)
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import write_text_report

REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "stability_full_data_quality_bridge_comparison"
TERMINAL_GUARD_ROLE = "generation_terminal_system_claim_guard"
RUNTIME_GUARD_ROLE = "generation_runtime_compute_claim_guard"
RUNTIME_FAIRNESS_ROLE = "quality_bridge_runtime_compute_fairness_audit"
EXPECTED_DATASET = QUALITY_BRIDGE_DATASET
EXPECTED_RESOLUTION = 256
EXPECTED_STEPS = QUALITY_BRIDGE_STEPS
EXPECTED_EFFECTIVE_BATCH = QUALITY_BRIDGE_EFFECTIVE_BATCH
EXPECTED_TRAINING_IMAGES = EXPECTED_STEPS * EXPECTED_EFFECTIVE_BATCH
EXPECTED_TERMINAL_SAMPLES = QUALITY_BRIDGE_TERMINAL_SAMPLES
EXPECTED_METHODS = ("cofitok", "dense_identity")
EXPECTED_MONITOR = "generation_stability_full_data_quality_bridge_100k"
EXTERNAL_ALIASES = {"d_ar", "mar", "retok"}
EXTERNAL_METHODS = {"d_ar": "D-AR", "mar": "MAR", "retok": "ReTok"}

AUTHORIZATION_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
    "cross_tier_numeric_ranking_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "sota_claim_allowed": False,
    "independent_replication_claim_allowed": False,
    "multiple_independent_terminal_streams_claim_allowed": False,
}

TERMINAL_CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
    "replaces_bound_source_reports": False,
    "visual_audit_is_quantitative_quality_evidence": False,
    "absolute_usability_claim_allowed": False,
    "cross_tier_numeric_ranking_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "sota_claim_allowed": False,
    "independent_replication_claim_allowed": False,
    "multiple_independent_terminal_streams_claim_allowed": False,
}

REPLICATION_INTERPRETATION = (
    "paired_reanalysis_of_one_exact_bound_terminal_sample_stream"
)
UNPAIRED_REPLICATION_INTERPRETATION = (
    "bound_terminal_stream_without_paired_reanalysis"
)

RUNTIME_CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "sampling_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
    "replaces_runtime_fairness_report": False,
    "replaces_pair_monitor": False,
    "quality_claim_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _finite_positive(value: Any, *, label: str) -> float:
    result = _finite(value, label=label)
    if result <= 0.0:
        raise ValueError(f"{label} must be positive")
    return result


def _same_float(left: Any, right: Any) -> bool:
    return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=1e-12)


def _normalized_real_set(
    value: Any,
    *,
    label: str,
    allow_legacy_missing_digest_schema: bool,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} real-set identity is missing")
    required_fields = {"root", "sha256", "image_count"}
    observed_fields = set(value)
    allowed_fields = required_fields | {"digest_schema"}
    if (
        not required_fields.issubset(observed_fields)
        or not observed_fields.issubset(allowed_fields)
        or (
            "digest_schema" not in observed_fields
            and not allow_legacy_missing_digest_schema
        )
    ):
        raise ValueError(f"{label} real-set field set differs")
    root = str(value.get("root", ""))
    digest_schema = value.get("digest_schema", IMAGE_TREE_DIGEST_SCHEMA)
    image_count = int(value.get("image_count", -1))
    if (
        not root
        or not PurePosixPath(root).is_absolute()
        or not _is_sha256(value.get("sha256"))
        or image_count < EXPECTED_TERMINAL_SAMPLES
        or digest_schema != IMAGE_TREE_DIGEST_SCHEMA
    ):
        raise ValueError(f"{label} real-set identity differs")
    return {
        "root": root,
        "digest_schema": IMAGE_TREE_DIGEST_SCHEMA,
        "sha256": value["sha256"],
        "image_count": image_count,
    }


def _normalized_git(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} Git identity is missing")
    revision = value.get("revision")
    branch = value.get("branch")
    dirty_fields = [value[name] for name in ("tracked_dirty", "dirty") if name in value]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
        or not isinstance(branch, str)
        or not branch
        or len(dirty_fields) != 1
        or dirty_fields[0] is not False
    ):
        raise ValueError(f"{label} Git identity is invalid")
    return {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }


def _bound_json(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} expected SHA256 is invalid")
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return identity, read_json_object(source, name=label)


def _replay_identity(
    descriptor: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    actual = _replay_file_identity(descriptor, label=label)
    return actual, read_json_object(actual["path"], name=label)


def _replay_file_identity(
    descriptor: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise TypeError(f"{label} identity is missing")
    expected = dict(descriptor)
    if int(expected.get("bytes", 0)) < 1 or not _is_sha256(expected.get("sha256")):
        raise ValueError(f"{label} identity is malformed")
    source = reject_symlink_chain(Path(str(expected.get("path", ""))), name=label)
    actual = file_identity(source)
    if actual != expected:
        raise ValueError(f"{label} changed after binding")
    return actual


def _critical_false(mapping: Mapping[str, Any], *names: str) -> bool:
    return all(mapping.get(name) is False for name in names)


def _validate_terminal_guard(report: Mapping[str, Any]) -> dict[str, Any]:
    policy = report.get("claim_policy")
    boundary = report.get("claim_boundary")
    sources = report.get("sources")
    evidence = report.get("evidence")
    classifier_integrity = (
        evidence.get("class_fidelity_classifier_integrity")
        if isinstance(evidence, Mapping)
        else None
    )
    statistical = (
        evidence.get("matched_statistical_advantage")
        if isinstance(evidence, Mapping)
        else None
    )
    replication = (
        statistical.get("replication_scope")
        if isinstance(statistical, Mapping)
        else None
    )
    paired_kid_evaluated = (
        policy.get("paired_kid_statistical_evidence_available")
        if isinstance(policy, Mapping)
        else None
    )
    expected_interpretation = (
        REPLICATION_INTERPRETATION
        if paired_kid_evaluated is True
        else UNPAIRED_REPLICATION_INTERPRETATION
    )
    if (
        report.get("schema_version") != 1
        or report.get("role") != TERMINAL_GUARD_ROLE
        or report.get("status") not in {"pass", "hold"}
        or not isinstance(policy, Mapping)
        or policy.get("terminal_system_evidence_complete") is not True
        or policy.get("class_fidelity_classifier_physical_integrity_verified")
        is not True
        or not isinstance(classifier_integrity, Mapping)
        or classifier_integrity.get("status") != "verified"
        or not _critical_false(
            policy,
            "absolute_usability_claim_allowed",
            "fid_statistical_significance_claim_allowed",
            "fid_confidence_interval_claim_allowed",
            "cross_tier_numeric_ranking_allowed",
            "broad_generation_superiority_claim_allowed",
            "sota_claim_allowed",
            "independent_replication_claim_allowed",
            "multiple_independent_terminal_streams_claim_allowed",
            "larger_training_launch_allowed",
            "inference_export_authorization_allowed",
            "release_authorization_allowed",
        )
        or boundary != TERMINAL_CLAIM_BOUNDARY
        or not isinstance(sources, Mapping)
        or set(sources)
        != {
            "quality_bridge_result",
            "statistical_claim_language_guard",
            "requested_class_visual_audit_waiter_status",
            "runtime_compute_claim_guard",
        }
        or not isinstance(replication, Mapping)
        or replication.get("bound_terminal_stream_count") != 1
        or replication.get("independent_replication_count") != 0
        or replication.get("independent_replication_supported") is not False
        or not isinstance(paired_kid_evaluated, bool)
        or replication.get("interpretation") != expected_interpretation
        or int(replication.get("start_index", -1)) != 0
        or int(replication.get("end_index_exclusive", -1))
        != EXPECTED_TERMINAL_SAMPLES
        or int(replication.get("sample_count", -1)) != EXPECTED_TERMINAL_SAMPLES
        or not isinstance(replication.get("bound_stream_id"), str)
        or not replication.get("bound_stream_id")
    ):
        raise ValueError("terminal system claim guard contract differs")
    advantage_allowed = report.get("status") == "pass"
    if (
        (advantage_allowed and paired_kid_evaluated is not True)
        or policy.get("matched_distribution_quality_claim_allowed")
        is not advantage_allowed
        or policy.get("lower_fid_point_estimate_statement_allowed")
        is not advantage_allowed
        or policy.get("paired_kid_statistical_support_statement_allowed")
        is not advantage_allowed
        or policy.get("replication_language_requires_distinct_bound_streams")
        is not True
        or report.get("decision")
        != (
            "matched_quality_advantage_qualified_with_terminal_system_evidence"
            if advantage_allowed
            else "terminal_system_evidence_complete_without_qualified_matched_advantage"
        )
    ):
        raise ValueError("terminal system guard status and policy differ")
    return {
        "status": str(report["status"]),
        "decision": str(report.get("decision", "")),
        "claim_policy": copy.deepcopy(dict(policy)),
        "paired_kid_evaluated": paired_kid_evaluated,
        "replication_scope": copy.deepcopy(dict(replication)),
        "quality_result": sources["quality_bridge_result"],
        "runtime_guard": sources["runtime_compute_claim_guard"],
        "scope": copy.deepcopy(report.get("scope")),
    }


def _validate_training_report(
    report: Mapping[str, Any],
    *,
    method: str,
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    config = report.get("config")
    data = config.get("data") if isinstance(config, Mapping) else None
    model = config.get("model") if isinstance(config, Mapping) else None
    optimization = config.get("optimization") if isinstance(config, Mapping) else None
    final_metrics = report.get("final_metrics")
    dataset_provenance = report.get("dataset_provenance")
    report_git = _normalized_git(
        report.get("git"),
        label=f"{method} quality-bridge training report",
    )
    if (
        report.get("training_complete") is not True
        or int(report.get("completed_steps", -1)) != EXPECTED_STEPS
        or int(report.get("target_steps", -1)) != EXPECTED_STEPS
        or report_git != dict(expected_git)
        or not isinstance(data, Mapping)
        or not isinstance(model, Mapping)
        or not isinstance(optimization, Mapping)
        or data.get("dataset") != EXPECTED_DATASET
        or int(model.get("image_size", -1)) != EXPECTED_RESOLUTION
        or not isinstance(final_metrics, Mapping)
        or not isinstance(dataset_provenance, Mapping)
        or not _is_sha256(dataset_provenance.get("identity_sha256"))
        or int(final_metrics.get("step", -1)) != EXPECTED_STEPS
        or int(final_metrics.get("samples_seen", -1)) != EXPECTED_TRAINING_IMAGES
    ):
        raise ValueError(f"{method} quality-bridge training report differs")
    expected_model = (
        {
            "synthesis_mode": "fixed_basis",
            "token_count": 8,
            "predictor_use_feedback": True,
        }
        if method == "CoFiTok"
        else {
            "synthesis_mode": "dense_identity",
            "token_count": 1,
            "predictor_use_feedback": False,
        }
    )
    if any(model.get(name) != expected for name, expected in expected_model.items()):
        raise ValueError(f"{method} model identity differs")
    micro_batch = int(data.get("batch_size", -1))
    accumulation = int(optimization.get("gradient_accumulation_steps", -1))
    if (
        micro_batch < 1
        or accumulation < 1
        or (micro_batch * accumulation != EXPECTED_EFFECTIVE_BATCH)
    ):
        raise ValueError(f"{method} effective training batch differs")
    parameter_count = int(report.get("parameter_count", -1))
    if parameter_count < 1:
        raise ValueError(f"{method} parameter count is invalid")
    return {
        "parameter_count": parameter_count,
        "training_steps": EXPECTED_STEPS,
        "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
        "training_images_seen": EXPECTED_TRAINING_IMAGES,
        "dataset": EXPECTED_DATASET,
        "resolution": EXPECTED_RESOLUTION,
        "dataset_identity_sha256": dataset_provenance["identity_sha256"],
        "training_runtime_environment_sha256": report.get("runtime_environment_sha256"),
    }


def _validate_cost(
    evidence: Any,
    *,
    method: str,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or evidence.get("status") != "verified":
        raise ValueError(f"{method} runtime fairness evidence is invalid")
    cost = evidence.get("training_cost")
    if not isinstance(cost, Mapping) or cost.get("valid") is not True:
        raise ValueError(f"{method} runtime cost is invalid")
    elapsed = _finite_positive(
        cost.get("elapsed_seconds"),
        label=f"{method} adjusted elapsed seconds",
    )
    throughput = _finite_positive(
        cost.get("images_per_second"),
        label=f"{method} training throughput",
    )
    reported = _finite_positive(
        cost.get("reported_elapsed_seconds"),
        label=f"{method} reported elapsed seconds",
    )
    peak_vram = int(cost.get("peak_vram_bytes", -1))
    if (
        int(cost.get("target_steps", -1)) != EXPECTED_STEPS
        or int(cost.get("effective_batch_size", -1)) != EXPECTED_EFFECTIVE_BATCH
        or int(cost.get("samples_seen", -1)) != EXPECTED_TRAINING_IMAGES
        or int(cost.get("expected_samples_seen", -1)) != EXPECTED_TRAINING_IMAGES
        or peak_vram <= 0
        or not _same_float(throughput, EXPECTED_TRAINING_IMAGES / elapsed)
    ):
        raise ValueError(f"{method} runtime cost does not match the training budget")
    adjustment = cost.get("resume_compute_adjustment")
    if not isinstance(adjustment, Mapping) or adjustment.get("valid") is not True:
        raise ValueError(f"{method} recovery-compute adjustment is invalid")
    applied = adjustment.get("applied") is True
    seconds = _finite(
        adjustment.get("seconds"),
        label=f"{method} recovery adjustment seconds",
    )
    event_count = int(adjustment.get("event_count", -1))
    orphaned_steps = int(adjustment.get("orphaned_optimizer_steps_lower_bound", -1))
    orphaned_images = int(adjustment.get("orphaned_images_lower_bound", -1))
    discovered = int(adjustment.get("discovered_orphan_archive_count", -1))
    covered = int(adjustment.get("covered_orphan_archive_count", -1))
    required_reasons = adjustment.get("required_reasons")
    if (
        adjustment.get("required") is not applied
        or adjustment.get("provided") is not applied
        or seconds < 0.0
        or not _same_float(elapsed, reported + seconds)
        or int(adjustment.get("continuity_end_step", -1)) != EXPECTED_STEPS
        or not isinstance(required_reasons, list)
        or adjustment.get("issues") != []
    ):
        raise ValueError(f"{method} recovery adjustment accounting differs")
    if applied:
        if (
            seconds <= 0.0
            or event_count < 1
            or orphaned_steps < 1
            or orphaned_images != orphaned_steps * EXPECTED_EFFECTIVE_BATCH
            or discovered < 1
            or covered != discovered
            or event_count != discovered
            or not required_reasons
            or cost.get("elapsed_seconds_role")
            != "physical_lower_bound_including_orphaned_recovery_compute"
        ):
            raise ValueError(f"{method} recovery adjustment evidence differs")
    elif (
        seconds != 0.0
        or event_count != 0
        or orphaned_steps != 0
        or orphaned_images != 0
        or discovered != 0
        or covered != 0
        or required_reasons
        or cost.get("elapsed_seconds_role") != "reported_training_elapsed_seconds"
    ):
        raise ValueError(f"{method} unexpected recovery adjustment evidence")
    return copy.deepcopy(dict(cost))


def _validate_runtime_sources(
    runtime_guard: Mapping[str, Any],
    *,
    expected_git: Mapping[str, Any],
    training_identities: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    policy = runtime_guard.get("claim_policy")
    contract = runtime_guard.get("matched_training_contract")
    sources = runtime_guard.get("sources")
    if (
        runtime_guard.get("schema_version") != 1
        or runtime_guard.get("role") != RUNTIME_GUARD_ROLE
        or runtime_guard.get("status") != "pass"
        or runtime_guard.get("decision")
        not in {
            "direct_runtime_outcome_comparison_allowed",
            "runtime_cost_claims_observational_only",
        }
        or not isinstance(policy, Mapping)
        or runtime_guard.get("claim_boundary") != RUNTIME_CLAIM_BOUNDARY
        or not isinstance(contract, Mapping)
        or contract.get("status") != "verified"
        or _normalized_git(
            contract.get("training_git"),
            label="runtime claim guard training contract",
        )
        != dict(expected_git)
        or contract.get("dataset") != EXPECTED_DATASET
        or int(contract.get("target_steps_per_method", -1)) != EXPECTED_STEPS
        or not isinstance(sources, Mapping)
        or set(sources) != {"runtime_compute_fairness", "terminal_pair_monitor"}
        or not _critical_false(
            policy,
            "equal_wall_clock_budget_claim_allowed",
            "equal_gpu_hours_budget_claim_allowed",
            "equal_training_flops_budget_claim_allowed",
            "peak_vram_advantage_claim_allowed",
            "quality_or_generation_advantage_claim_allowed",
        )
    ):
        raise ValueError("runtime claim guard contract differs")
    fairness_identity, fairness = _replay_identity(
        sources["runtime_compute_fairness"],
        label="quality-bridge runtime fairness report",
    )
    pair_identity, pair_monitor = _replay_identity(
        sources["terminal_pair_monitor"],
        label="quality-bridge terminal pair monitor",
    )
    terminal_sources = fairness.get("terminal_sources")
    methods = fairness.get("methods")
    fairness_contract = fairness.get("contract")
    pair_runs = pair_monitor.get("runs")
    if (
        fairness.get("schema_version") != 1
        or fairness.get("role") != RUNTIME_FAIRNESS_ROLE
        or fairness.get("status") != "pass"
        or not isinstance(terminal_sources, Mapping)
        or set(terminal_sources)
        != {
            "cofitok_training",
            "dense_training",
            "cofitok_latest",
            "dense_latest",
        }
        or terminal_sources.get("cofitok_training")
        != dict(training_identities["cofitok"])
        or terminal_sources.get("dense_training")
        != dict(training_identities["dense_identity"])
        or not isinstance(fairness_contract, Mapping)
        or fairness_contract.get("status") != "verified"
        or _normalized_git(
            fairness_contract.get("training_git"),
            label="runtime fairness training contract",
        )
        != dict(expected_git)
        or not isinstance(methods, Mapping)
        or set(methods) != set(EXPECTED_METHODS)
        or pair_monitor.get("status") != "pass"
        or pair_monitor.get("stage") != "complete"
        or int(pair_monitor.get("schema_version", -1)) != 2
        or pair_monitor.get("monitor") != EXPECTED_MONITOR
        or _normalized_git(
            pair_monitor.get("git"),
            label="terminal pair monitor",
        )
        != dict(expected_git)
        or not isinstance(pair_runs, Mapping)
        or set(pair_runs) != set(EXPECTED_METHODS)
        or any(
            not isinstance(pair_runs[method], Mapping)
            or pair_runs[method].get("complete") is not True
            or int(pair_runs[method].get("last_step", -1)) != EXPECTED_STEPS
            for method in EXPECTED_METHODS
        )
    ):
        raise ValueError("runtime fairness sources do not match the terminal pair")
    latest_identities: dict[str, dict[str, Any]] = {}
    latest_records: dict[str, dict[str, Any]] = {}
    for method, source_name in (
        ("cofitok", "cofitok_latest"),
        ("dense_identity", "dense_latest"),
    ):
        latest_identity, latest = _replay_identity(
            terminal_sources[source_name],
            label=f"{method} terminal latest checkpoint pointer",
        )
        latest_identities[method] = latest_identity
        latest_records[method] = latest
        if (
            int(latest.get("step", -1)) != EXPECTED_STEPS
            or _normalized_git(
                {
                    "revision": latest.get("git_revision"),
                    "branch": latest.get("git_branch"),
                    "dirty": latest.get("git_dirty"),
                },
                label=f"{method} terminal latest checkpoint",
            )
            != dict(expected_git)
            or not _is_sha256(latest.get("checkpoint_sha256"))
            or not str(latest.get("checkpoint", "")).endswith(
                "checkpoint_step_00100000.pt"
            )
            or latest.get("integrity_manifest")
            != "checkpoint_step_00100000.pt.integrity.json"
        ):
            raise ValueError(f"{method} terminal latest checkpoint differs")
    direct = (
        runtime_guard.get("decision") == "direct_runtime_outcome_comparison_allowed"
    )
    if (
        policy.get("training_wall_clock_direct_comparison_allowed") is not direct
        or policy.get("training_throughput_direct_comparison_allowed") is not direct
        or policy.get("cost_efficiency_ranking_allowed") is not direct
    ):
        raise ValueError("runtime direct-comparison policy differs")
    return {
        "direct_runtime_comparison_allowed": direct,
        "fairness_identity": fairness_identity,
        "pair_monitor_identity": pair_identity,
        "latest_identities": latest_identities,
        "latest_records": latest_records,
        "costs": {
            method: _validate_cost(methods[method], method=method)
            for method in EXPECTED_METHODS
        },
    }


def _validate_physical_evidence(
    evidence: Any,
    *,
    method: Mapping[str, Any],
    latest: Mapping[str, Any],
    expected_git: Mapping[str, Any],
    expected_dataset_sha256: str,
    expected_runtime_sha256: str,
    label: str,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping):
        raise TypeError(f"{label} terminal physical evidence is missing")
    expected_fields = {
        "checkpoint",
        "checkpoint_integrity_manifest",
        "checkpoint_step",
        "sampling_report",
        "sampling_manifest",
        "sampling_progress",
        "sample_set_sha256",
        "sample_count",
        "real_set",
    }
    if set(evidence) != expected_fields:
        raise ValueError(f"{label} terminal physical evidence set differs")
    checkpoint = _replay_file_identity(
        evidence.get("checkpoint"),
        label=f"{label} physical checkpoint",
    )
    integrity = _replay_file_identity(
        evidence.get("checkpoint_integrity_manifest"),
        label=f"{label} physical checkpoint integrity manifest",
    )
    integrity_report = read_json_object(
        integrity["path"],
        name=f"{label} checkpoint integrity manifest",
    )
    checkpoint_path = Path(checkpoint["path"])
    expected_integrity_path = checkpoint_path.with_name(
        f"{checkpoint_path.name}.integrity.json"
    )
    replayed_sampling = {
        name: _replay_file_identity(
            evidence.get(name),
            label=f"{label} physical {name.replace('_', ' ')}",
        )
        for name in ("sampling_report", "sampling_manifest", "sampling_progress")
    }
    real_set = evidence.get("real_set")
    physical_real_set = _normalized_real_set(
        real_set,
        label=f"{label} physical",
        allow_legacy_missing_digest_schema=True,
    )
    method_real_set = _normalized_real_set(
        method.get("real_set"),
        label=f"{label} terminal",
        allow_legacy_missing_digest_schema=False,
    )
    if (
        checkpoint != evidence.get("checkpoint")
        or checkpoint["path"] != method.get("checkpoint")
        or checkpoint["sha256"] != method.get("checkpoint_sha256")
        or checkpoint["sha256"] != latest.get("checkpoint_sha256")
        or checkpoint_path.name != latest.get("checkpoint")
        or integrity != evidence.get("checkpoint_integrity_manifest")
        or Path(integrity["path"]) != expected_integrity_path
        or expected_integrity_path.name != latest.get("integrity_manifest")
        or int(integrity_report.get("schema_version", -1)) != 1
        or integrity_report.get("checkpoint") != checkpoint_path.name
        or int(integrity_report.get("checkpoint_bytes", -1)) != int(checkpoint["bytes"])
        or integrity_report.get("checkpoint_sha256") != checkpoint["sha256"]
        or int(integrity_report.get("checkpoint_format_version", -1)) < 1
        or int(integrity_report.get("step", -1)) != EXPECTED_STEPS
        or integrity_report.get("git_revision") != expected_git.get("revision")
        or integrity_report.get("git_branch") != expected_git.get("branch")
        or integrity_report.get("git_dirty") is not False
        or integrity_report.get("dataset_identity_sha256") != expected_dataset_sha256
        or integrity_report.get("runtime_environment_sha256") != expected_runtime_sha256
        or int(latest.get("checkpoint_bytes", -1)) != int(checkpoint["bytes"])
        or int(latest.get("checkpoint_format_version", -1))
        != int(integrity_report["checkpoint_format_version"])
        or latest.get("dataset_identity_sha256") != expected_dataset_sha256
        or latest.get("runtime_environment_sha256") != expected_runtime_sha256
        or int(evidence.get("checkpoint_step", -1)) != EXPECTED_STEPS
        or evidence.get("sample_set_sha256") != method.get("sample_set_sha256")
        or int(evidence.get("sample_count", -1)) != EXPECTED_TERMINAL_SAMPLES
        or replayed_sampling["sampling_report"] != method.get("sampling_report")
        or replayed_sampling["sampling_manifest"] != method.get("sampling_manifest")
        or replayed_sampling["sampling_progress"] != method.get("sampling_progress")
        or physical_real_set != method_real_set
    ):
        raise ValueError(f"{label} terminal physical evidence differs")
    return {
        "checkpoint": checkpoint,
        "checkpoint_integrity_manifest": integrity,
        "checkpoint_step": EXPECTED_STEPS,
        **replayed_sampling,
        "sample_set_sha256": method["sample_set_sha256"],
        "sample_count": EXPECTED_TERMINAL_SAMPLES,
        "real_set": physical_real_set,
    }


def _validate_class_fidelity_sources(
    qualification: Mapping[str, Any],
    *,
    methods: Mapping[str, Mapping[str, Any]],
    source_reports: Mapping[str, Any],
    expected_git: Mapping[str, Any],
) -> dict[str, Any]:
    sources = qualification.get("sources")
    contract = qualification.get("sampling_contract")
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(EXPECTED_METHODS)
        or not isinstance(contract, Mapping)
    ):
        raise ValueError("quality-bridge class-fidelity source contract differs")
    expected_sources = {
        "cofitok": source_reports.get("cofitok_class_fidelity"),
        "dense_identity": source_reports.get("dense_class_fidelity"),
    }
    if any(not isinstance(value, Mapping) for value in expected_sources.values()):
        raise ValueError("quality-bridge class-fidelity source identities are missing")
    if any(sources[method] != expected_sources[method] for method in EXPECTED_METHODS):
        raise ValueError("quality-bridge class-fidelity source identities differ")
    evaluator_git = _normalized_git(
        contract.get("evaluator_git"),
        label="class-fidelity qualification evaluator",
    )
    sampling_git = _normalized_git(
        contract.get("sampling_git"),
        label="class-fidelity qualification sampler",
    )
    evaluator_runtime = contract.get("evaluator_runtime_environment_sha256")
    if (
        evaluator_git != dict(expected_git)
        or sampling_git != dict(expected_git)
        or not _is_sha256(evaluator_runtime)
    ):
        raise ValueError("quality-bridge class-fidelity evaluator identity differs")
    identities: dict[str, dict[str, Any]] = {}
    for method in EXPECTED_METHODS:
        identity, report = _replay_identity(
            sources[method],
            label=f"quality-bridge {method} class-fidelity report",
        )
        validate_class_fidelity_report(dict(report))
        identities[method] = identity
        terminal = methods[method]
        sample = report.get("sample_provenance")
        if not isinstance(sample, Mapping):
            raise TypeError(f"{method} class-fidelity sample provenance is missing")
        expected_budget = 8 if method == "cofitok" else 1
        if (
            _normalized_git(
                report.get("git"),
                label=f"{method} class-fidelity evaluator",
            )
            != evaluator_git
            or _normalized_git(
                sample.get("git"),
                label=f"{method} class-fidelity sampler",
            )
            != sampling_git
            or report.get("runtime_environment_sha256") != evaluator_runtime
            or _normalized_git(
                terminal.get("metrics_evaluator_git"),
                label=f"{method} terminal evaluator",
            )
            != evaluator_git
            or terminal.get("metrics_runtime_environment_sha256") != evaluator_runtime
            or sample.get("checkpoint_sha256") != terminal.get("checkpoint_sha256")
            or sample.get("sample_set_sha256") != terminal.get("sample_set_sha256")
            or int(sample.get("selected_prefix_budget", -1)) != expected_budget
            or sample.get("weights") != "ema"
            or sample.get("sampling") != terminal.get("sampling")
            or report.get("metrics") != qualification.get("metrics", {}).get(method)
        ):
            raise ValueError(f"{method} class-fidelity terminal identity differs")
    if (
        contract.get("cofitok_checkpoint_sha256")
        != methods["cofitok"].get("checkpoint_sha256")
        or contract.get("dense_checkpoint_sha256")
        != methods["dense_identity"].get("checkpoint_sha256")
        or contract.get("cofitok_sample_set_sha256")
        != methods["cofitok"].get("sample_set_sha256")
        or contract.get("dense_sample_set_sha256")
        != methods["dense_identity"].get("sample_set_sha256")
    ):
        raise ValueError("quality-bridge class fidelity uses another terminal pair")
    return {
        "sources": identities,
        "evaluator_git": evaluator_git,
        "sampling_git": sampling_git,
        "evaluator_runtime_environment_sha256": evaluator_runtime,
        "classifier": copy.deepcopy(qualification.get("classifier")),
    }


def _sampling_progress(method: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    identity, progress = _replay_identity(
        method.get("sampling_progress"),
        label=f"{label} terminal sampling progress",
    )
    elapsed = _finite_positive(
        progress.get("cumulative_elapsed_seconds"),
        label=f"{label} terminal sampling elapsed seconds",
    )
    if (
        int(progress.get("schema_version", -1)) != 1
        or progress.get("status") != "completed"
        or int(progress.get("completed_samples", -1)) != EXPECTED_TERMINAL_SAMPLES
        or int(progress.get("total_samples", -1)) != EXPECTED_TERMINAL_SAMPLES
        or int(progress.get("start_index", -1))
        != int(method.get("sampling", {}).get("start_index", -2))
        or progress.get("prefix_budgets")
        != method.get("sampling", {}).get("prefix_budgets")
        or progress.get("sampling_manifest_sha256")
        != method.get("sampling_manifest", {}).get("sha256")
        or progress.get("sample_sets", {}).get(
            str(method.get("selected_prefix_budget"))
        )
        != {
            "count": EXPECTED_TERMINAL_SAMPLES,
            "sha256": method.get("sample_set_sha256"),
        }
        or progress.get("error_type") is not None
        or progress.get("error") is not None
    ):
        raise ValueError(f"{label} terminal sampling progress is incomplete")
    invocation = int(progress.get("invocation", -1))
    if invocation < 1:
        raise ValueError(f"{label} terminal sampling invocation is invalid")
    return {
        "identity": identity,
        "elapsed_seconds": elapsed,
        "invocation": invocation,
    }


def _class_metrics(
    qualification: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    metrics = qualification.get("metrics")
    row = metrics.get(method) if isinstance(metrics, Mapping) else None
    if not isinstance(row, Mapping):
        raise TypeError(f"{method} class-fidelity metrics are missing")
    sample_count = int(row.get("sample_count", -1))
    values = {
        "class_fidelity_sample_count": sample_count,
        "class_top1_accuracy": _finite(
            row.get("top1_accuracy"), label=f"{method} class top-1"
        ),
        "class_top5_accuracy": _finite(
            row.get("top5_accuracy"), label=f"{method} class top-5"
        ),
        "class_mean_target_probability": _finite(
            row.get("mean_target_probability"),
            label=f"{method} mean target probability",
        ),
        "class_target_negative_log_likelihood": _finite(
            row.get("target_negative_log_likelihood"),
            label=f"{method} target negative log likelihood",
        ),
        "class_predicted_class_fraction": _finite(
            row.get("predicted_class_fraction"),
            label=f"{method} predicted class fraction",
        ),
        "class_normalized_predicted_entropy": _finite(
            row.get("normalized_predicted_class_entropy"),
            label=f"{method} normalized predicted entropy",
        ),
    }
    if sample_count != EXPECTED_TERMINAL_SAMPLES or any(
        not 0.0 <= values[name] <= 1.0
        for name in (
            "class_top1_accuracy",
            "class_top5_accuracy",
            "class_mean_target_probability",
            "class_predicted_class_fraction",
            "class_normalized_predicted_entropy",
        )
    ):
        raise ValueError(f"{method} class-fidelity metrics are out of range")
    return values


def _matched_row(
    *,
    method: str,
    method_key: str,
    training: Mapping[str, Any],
    terminal: Mapping[str, Any],
    physical_evidence: Mapping[str, Any],
    runtime_cost: Mapping[str, Any],
    class_fidelity: Mapping[str, Any],
    class_fidelity_source: Mapping[str, Any],
    direct_runtime_comparison_allowed: bool,
) -> dict[str, Any]:
    expected_budget = 8 if method_key == "cofitok" else 1
    sampling = terminal.get("sampling")
    real_set = terminal.get("real_set")
    if not isinstance(sampling, Mapping) or not isinstance(real_set, Mapping):
        raise TypeError(f"{method} terminal protocol is incomplete")
    contract = sampling_protocol_contract(
        dict(sampling),
        stage="scaling",
        expected_num_train_timesteps=1_000,
    )
    if (
        contract["valid"] is not True
        or int(terminal.get("checkpoint_step", -1)) != EXPECTED_STEPS
        or int(terminal.get("sample_count", -1)) != EXPECTED_TERMINAL_SAMPLES
        or int(terminal.get("selected_prefix_budget", -1)) != expected_budget
        or terminal.get("weights") != "ema"
        or sampling.get("prefix_budgets") != [expected_budget]
        or not _is_sha256(terminal.get("checkpoint_sha256"))
        or not _is_sha256(terminal.get("sample_set_sha256"))
        or not _is_sha256(real_set.get("sha256"))
        or int(real_set.get("image_count", -1)) < EXPECTED_TERMINAL_SAMPLES
    ):
        raise ValueError(f"{method} terminal identity differs")
    progress = _sampling_progress(terminal, label=method)
    fid = _finite(terminal.get("fid"), label=f"{method} FID")
    inception = _finite(
        terminal.get("inception_score"), label=f"{method} inception score"
    )
    precision = _finite(terminal.get("precision"), label=f"{method} precision")
    recall = _finite(terminal.get("recall"), label=f"{method} recall")
    if (
        fid < 0.0
        or inception <= 0.0
        or not 0.0 <= precision <= 1.0
        or not 0.0 <= recall <= 1.0
    ):
        raise ValueError(f"{method} distribution metrics are out of range")
    return {
        "method": method,
        "comparison_tier": "matched_training_direct",
        "directly_comparable_to_cofitok": True,
        **dict(training),
        "training_elapsed_seconds": runtime_cost["elapsed_seconds"],
        "training_images_per_second": runtime_cost["images_per_second"],
        "training_elapsed_seconds_role": runtime_cost["elapsed_seconds_role"],
        "training_wall_clock_directly_comparable": (direct_runtime_comparison_allowed),
        "training_throughput_directly_comparable": (direct_runtime_comparison_allowed),
        "peak_vram_bytes": runtime_cost["peak_vram_bytes"],
        "training_budget_basis": "matched_steps_and_training_images",
        "compute_matched_claim_allowed": False,
        "sample_count": EXPECTED_TERMINAL_SAMPLES,
        "sample_batch_size": int(sampling["batch_size"]),
        "sampling_elapsed_seconds": progress["elapsed_seconds"],
        "sampling_images_per_second": (
            EXPECTED_TERMINAL_SAMPLES / progress["elapsed_seconds"]
        ),
        "sampling_invocations": progress["invocation"],
        "weights": "ema",
        "sampler": sampling["sampler"],
        "sample_steps": int(sampling["sample_steps"]),
        "guidance_scale": float(sampling["guidance_scale"]),
        "guidance_rescale": float(sampling["guidance_rescale"]),
        "cfg_batch_mode": sampling["cfg_batch_mode"],
        "sampling_precision": sampling["precision"],
        "sampling_seed": int(sampling["seed"]),
        "sampling_start_index": int(sampling["start_index"]),
        "class_schedule": sampling["class_schedule"],
        "sampling_random_stream": copy.deepcopy(sampling["random_stream"]),
        "prefix_budgets": copy.deepcopy(sampling["prefix_budgets"]),
        "fid": fid,
        "inception_score": inception,
        "precision": precision,
        "recall": recall,
        "checkpoint_sha256": terminal["checkpoint_sha256"],
        "checkpoint_integrity_manifest": terminal["checkpoint_integrity_manifest"],
        "checkpoint_identity": copy.deepcopy(physical_evidence["checkpoint"]),
        "checkpoint_integrity_manifest_identity": copy.deepcopy(
            physical_evidence["checkpoint_integrity_manifest"]
        ),
        "sample_set_sha256": terminal["sample_set_sha256"],
        "real_set_root": real_set["root"],
        "real_set_digest_schema": real_set.get("digest_schema"),
        "real_set_sha256": real_set["sha256"],
        "real_image_count": int(real_set["image_count"]),
        "evaluator_git": copy.deepcopy(terminal["metrics_evaluator_git"]),
        "evaluator_runtime_environment_sha256": terminal[
            "metrics_runtime_environment_sha256"
        ],
        "class_fidelity_report": copy.deepcopy(dict(class_fidelity_source)),
        **_class_metrics(class_fidelity, method=method_key),
        "protocol_note": (
            "Matched full-data ImageNet-256, shared backbone/training contract, "
            "100K optimizer steps, 6.4M training images, EMA, and 10K DDIM-100 "
            "terminal samples. Runtime values are recovery-adjusted measured "
            "outcomes, not equal wall-clock/GPU-hour/FLOP budgets."
        ),
    }


def _validate_quality_result(
    report: Mapping[str, Any],
    *,
    guard: Mapping[str, Any],
) -> dict[str, Any]:
    git = _normalized_git(
        report.get("git"),
        label="quality-bridge terminal result",
    )
    terminal = report.get("terminal")
    methods = terminal.get("methods") if isinstance(terminal, Mapping) else None
    physical_evidence = (
        terminal.get("physical_evidence") if isinstance(terminal, Mapping) else None
    )
    class_fidelity = (
        terminal.get("class_fidelity") if isinstance(terminal, Mapping) else None
    )
    sources = report.get("source_reports")
    boundary = report.get("authorization_boundary")
    quality_screen = report.get("quality_screen")
    if (
        report.get("schema_version") != QUALITY_BRIDGE_RESULT_SCHEMA_VERSION
        or report.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or not isinstance(methods, Mapping)
        or set(methods) != set(EXPECTED_METHODS)
        or not isinstance(physical_evidence, Mapping)
        or set(physical_evidence) != set(EXPECTED_METHODS)
        or not isinstance(class_fidelity, Mapping)
        or not isinstance(sources, Mapping)
        or boundary != RESULT_AUTHORIZATION_BOUNDARY
        or not isinstance(quality_screen, Mapping)
        or quality_screen.get("status") not in {"pass", "hold"}
    ):
        raise ValueError("quality-bridge terminal result contract differs")
    guard_screen = guard.get("evidence", {}).get("quality_screen", {})
    if (
        guard_screen.get("status") != quality_screen.get("status")
        or guard_screen.get("failed_checks") != quality_screen.get("failed_checks")
        or int(guard_screen.get("check_count", -1))
        != len(quality_screen.get("checks", []))
    ):
        raise ValueError("terminal guard quality screen does not match its source")
    return {
        "git": git,
        "methods": methods,
        "physical_evidence": physical_evidence,
        "class_fidelity": class_fidelity,
        "source_reports": sources,
        "quality_screen": quality_screen,
    }


def _external_rows(official: Mapping[str, Any]) -> list[dict[str, Any]]:
    if official.get("schema_version") != 1:
        raise ValueError("unsupported official related-method table schema")
    rows = official.get("rows")
    if not isinstance(rows, list):
        raise TypeError("official related-method rows are missing")
    aliases = {str(row.get("alias")) for row in rows if isinstance(row, Mapping)}
    if len(rows) != len(aliases) or aliases != EXTERNAL_ALIASES:
        raise ValueError(
            f"official related-method aliases do not match: {sorted(aliases)}"
        )
    output = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("official related-method row is malformed")
        alias = str(row.get("alias"))
        if row.get("method") != EXTERNAL_METHODS[alias]:
            raise ValueError(f"official baseline method identity mismatch: {alias}")
        if row.get("status") != "completed_eval_only_50k":
            raise ValueError(f"official baseline is incomplete: {alias}")
        if row.get("paper_table_role") != "secondary related-method only":
            raise ValueError(f"official baseline has an unsafe table role: {alias}")
        if (
            row.get("dataset") != EXPECTED_DATASET
            or int(row.get("resolution", 0)) != EXPECTED_RESOLUTION
            or int(row.get("sample_count", 0)) != 50_000
        ):
            raise ValueError(f"official baseline protocol differs: {alias}")
        metrics = {
            name: _finite(row.get(name), label=f"{alias} {name}")
            for name in ("fid", "inception_score", "precision", "recall")
        }
        if (
            metrics["fid"] < 0.0
            or metrics["inception_score"] <= 0.0
            or not 0.0 <= metrics["precision"] <= 1.0
            or not 0.0 <= metrics["recall"] <= 1.0
        ):
            raise ValueError(f"official baseline metrics are out of range: {alias}")
        metrics_txt = str(row.get("metrics_txt", ""))
        if not PurePosixPath(metrics_txt).is_absolute() or not metrics_txt.endswith(
            ".txt"
        ):
            raise ValueError(f"official baseline metrics source is invalid: {alias}")
        output.append(
            {
                "alias": alias,
                "method": row["method"],
                "comparison_tier": "official_pretrained_contextual",
                "directly_comparable_to_cofitok": False,
                "dataset": row["dataset"],
                "resolution": int(row["resolution"]),
                "training_steps": None,
                "parameter_count": None,
                "sample_count": int(row["sample_count"]),
                **metrics,
                "evaluator": {
                    "package": "ADM TensorFlow evaluation graph",
                    "version": "pinned baseline protocol",
                },
                "sample_steps": None,
                "guidance_scale": None,
                "checkpoint_sha256": None,
                "sample_set_sha256": None,
                "protocol_note": row["protocol"],
                "source_metrics": metrics_txt,
                "source_npz": row.get("npz"),
                "source_kind": row.get("source_kind"),
                "source_status": row["status"],
                "paper_table_role": row["paper_table_role"],
            }
        )
    return output


def build_report(
    *,
    terminal_guard_identity: Mapping[str, Any],
    terminal_guard: Mapping[str, Any],
    official_identity: Mapping[str, Any],
    official_related: Mapping[str, Any],
) -> dict[str, Any]:
    guard = _validate_terminal_guard(terminal_guard)
    quality_identity, quality_report = _replay_identity(
        guard["quality_result"],
        label="terminal quality-bridge result",
    )
    quality = _validate_quality_result(quality_report, guard=terminal_guard)
    runtime_identity, runtime_guard = _replay_identity(
        guard["runtime_guard"],
        label="terminal runtime claim guard",
    )
    training_identities = {
        "cofitok": quality["source_reports"].get("cofitok_training"),
        "dense_identity": quality["source_reports"].get("dense_training"),
    }
    if not all(isinstance(value, Mapping) for value in training_identities.values()):
        raise ValueError("quality-bridge training source identities are missing")
    cofitok_training_identity, cofitok_training_report = _replay_identity(
        training_identities["cofitok"],
        label="quality-bridge CoFiTok training report",
    )
    dense_training_identity, dense_training_report = _replay_identity(
        training_identities["dense_identity"],
        label="quality-bridge dense training report",
    )
    training = {
        "cofitok": _validate_training_report(
            cofitok_training_report,
            method="CoFiTok",
            expected_git=quality["git"],
        ),
        "dense_identity": _validate_training_report(
            dense_training_report,
            method="dense_identity",
            expected_git=quality["git"],
        ),
    }
    class_fidelity_identity, class_fidelity_report = _replay_identity(
        quality["source_reports"].get("class_fidelity_qualification"),
        label="quality-bridge class-fidelity qualification",
    )
    class_fidelity = validate_class_fidelity_qualification(
        class_fidelity_report,
        expected_stage="scaling",
        expected_revision=quality["git"]["revision"],
        expected_branch=quality["git"]["branch"],
        require_pass=False,
    )
    if class_fidelity != dict(quality["class_fidelity"]):
        raise ValueError("quality-bridge class-fidelity qualification differs")
    if (
        not _is_sha256(training["cofitok"]["dataset_identity_sha256"])
        or training["cofitok"]["dataset_identity_sha256"]
        != training["dense_identity"]["dataset_identity_sha256"]
        or not _is_sha256(training["cofitok"]["training_runtime_environment_sha256"])
        or training["cofitok"]["training_runtime_environment_sha256"]
        != training["dense_identity"]["training_runtime_environment_sha256"]
    ):
        raise ValueError("quality-bridge training data/runtime identities differ")
    parameter_gap = (
        training["cofitok"]["parameter_count"]
        - training["dense_identity"]["parameter_count"]
    ) / training["dense_identity"]["parameter_count"]
    if abs(parameter_gap) > 0.02:
        raise ValueError("quality-bridge parameter gap exceeds two percent")
    runtime = _validate_runtime_sources(
        runtime_guard,
        expected_git=quality["git"],
        training_identities={
            "cofitok": cofitok_training_identity,
            "dense_identity": dense_training_identity,
        },
    )
    physical_evidence = {
        method: _validate_physical_evidence(
            quality["physical_evidence"][method],
            method=quality["methods"][method],
            latest=runtime["latest_records"][method],
            expected_git=quality["git"],
            expected_dataset_sha256=training[method]["dataset_identity_sha256"],
            expected_runtime_sha256=training[method][
                "training_runtime_environment_sha256"
            ],
            label=method,
        )
        for method in EXPECTED_METHODS
    }
    normalized_protocols = {
        method: {
            key: copy.deepcopy(value)
            for key, value in quality["methods"][method]["sampling"].items()
            if key != "prefix_budgets"
        }
        for method in EXPECTED_METHODS
    }
    if normalized_protocols["cofitok"] != normalized_protocols["dense_identity"]:
        raise ValueError("quality-bridge methods use different terminal protocols")
    if (
        quality["methods"]["cofitok"]["real_set"]
        != quality["methods"]["dense_identity"]["real_set"]
        or quality["methods"]["cofitok"]["metrics_runtime_environment_sha256"]
        != quality["methods"]["dense_identity"]["metrics_runtime_environment_sha256"]
    ):
        raise ValueError("quality-bridge methods use different evaluator evidence")
    evaluator_sha = quality["methods"]["cofitok"].get(
        "metrics_runtime_environment_sha256"
    )
    if not _is_sha256(evaluator_sha) or any(
        _normalized_git(
            quality["methods"][method].get("metrics_evaluator_git"),
            label=f"{method} terminal evaluator",
        )
        != quality["git"]
        for method in EXPECTED_METHODS
    ):
        raise ValueError("quality-bridge evaluator provenance differs")
    class_fidelity_evidence = _validate_class_fidelity_sources(
        class_fidelity,
        methods=quality["methods"],
        source_reports=quality["source_reports"],
        expected_git=quality["git"],
    )
    matched = [
        _matched_row(
            method="CoFiTok K=8",
            method_key="cofitok",
            training=training["cofitok"],
            terminal=quality["methods"]["cofitok"],
            physical_evidence=physical_evidence["cofitok"],
            runtime_cost=runtime["costs"]["cofitok"],
            class_fidelity=class_fidelity,
            class_fidelity_source=class_fidelity_evidence["sources"]["cofitok"],
            direct_runtime_comparison_allowed=runtime[
                "direct_runtime_comparison_allowed"
            ],
        ),
        _matched_row(
            method="Dense identity",
            method_key="dense_identity",
            training=training["dense_identity"],
            terminal=quality["methods"]["dense_identity"],
            physical_evidence=physical_evidence["dense_identity"],
            runtime_cost=runtime["costs"]["dense_identity"],
            class_fidelity=class_fidelity,
            class_fidelity_source=class_fidelity_evidence["sources"]["dense_identity"],
            direct_runtime_comparison_allowed=runtime[
                "direct_runtime_comparison_allowed"
            ],
        ),
    ]
    external = _external_rows(dict(official_related))
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": guard["status"],
        "decision": guard["decision"],
        "source_profile": "quality_bridge_100k_terminal",
        "scope": {
            "dataset": EXPECTED_DATASET,
            "resolution": EXPECTED_RESOLUTION,
            "training_steps_per_method": EXPECTED_STEPS,
            "training_images_per_method": EXPECTED_TRAINING_IMAGES,
            "terminal_samples_per_method": EXPECTED_TERMINAL_SAMPLES,
            "training_git": quality["git"],
        },
        "comparison_policy": {
            "primary_direct_tier": "matched_training_direct",
            "external_context_tier": "official_pretrained_contextual",
            "cross_tier_numeric_ranking_allowed": False,
            "direct_quality_comparison_allowed": True,
            "training_budget_basis": "matched_steps_and_training_images",
            "equal_wall_clock_budget": False,
            "equal_gpu_hours_budget": False,
            "equal_training_flops_budget": False,
            "compute_matched_claim_allowed": False,
            "runtime_direct_comparison_allowed": runtime[
                "direct_runtime_comparison_allowed"
            ],
            "absolute_usability_claim_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "independent_replication_claim_allowed": False,
            "multiple_independent_terminal_streams_claim_allowed": False,
            "replication_language_requires_distinct_bound_streams": True,
            "reason": (
                "CoFiTok and dense_identity are the only matched-training direct "
                "rows. D-AR, MAR, and ReTok use official pretrained checkpoints, "
                "50K samples, and another evaluator/protocol, so they remain a "
                "separate contextual panel without cross-tier ranking."
            ),
        },
        "quality_screen": copy.deepcopy(dict(quality["quality_screen"])),
        "replication_scope": guard["replication_scope"],
        "terminal_claim_policy": guard["claim_policy"],
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
        "source_reports": {
            "terminal_system_claim_guard": dict(terminal_guard_identity),
            "quality_bridge_result": quality_identity,
            "runtime_compute_claim_guard": runtime_identity,
            "runtime_compute_fairness": runtime["fairness_identity"],
            "terminal_pair_monitor": runtime["pair_monitor_identity"],
            "cofitok_latest": runtime["latest_identities"]["cofitok"],
            "dense_latest": runtime["latest_identities"]["dense_identity"],
            "cofitok_training": cofitok_training_identity,
            "dense_training": dense_training_identity,
            "class_fidelity_qualification": class_fidelity_identity,
            "cofitok_class_fidelity": class_fidelity_evidence["sources"]["cofitok"],
            "dense_class_fidelity": class_fidelity_evidence["sources"][
                "dense_identity"
            ],
            "official_related_methods": dict(official_identity),
        },
        "matched_training_rows": matched,
        "official_context_rows": external,
        "matched_summary": {
            "cofitok_minus_dense_fid": matched[0]["fid"] - matched[1]["fid"],
            "cofitok_relative_fid": matched[0]["fid"] / matched[1]["fid"] - 1.0,
            "cofitok_minus_dense_precision": (
                matched[0]["precision"] - matched[1]["precision"]
            ),
            "cofitok_minus_dense_recall": (matched[0]["recall"] - matched[1]["recall"]),
            "cofitok_minus_dense_class_top1": (
                matched[0]["class_top1_accuracy"] - matched[1]["class_top1_accuracy"]
            ),
            "cofitok_minus_dense_class_top5": (
                matched[0]["class_top5_accuracy"] - matched[1]["class_top5_accuracy"]
            ),
            "relative_parameter_gap": parameter_gap,
        },
        "limitations": [
            (
                "The direct panel is a non-authorizing 100K/10K quality bridge, "
                "not the formal 300K/50K release evaluation."
            ),
            (
                "Official pretrained context cannot be numerically ranked against "
                "the matched-training direct panel."
            ),
            (
                (
                    "FID uses the exact bound terminal sample stream; paired "
                    "block-KID was not evaluated. This report records zero "
                    "independent replications."
                )
                if not guard["paired_kid_evaluated"]
                else (
                    "FID and paired block-KID reuse one exact bound terminal sample "
                    "stream. This report records zero independent replications and "
                    "cannot support a multiple-stream replication claim."
                )
            ),
            (
                "This report does not authorize larger training, sampling, export, "
                "promotion, release, broad superiority, or SOTA claims."
            ),
        ],
    }


def verify_source_reports(report: Mapping[str, Any]) -> dict[str, Any]:
    sources = report.get("source_reports")
    if not isinstance(sources, Mapping) or set(sources) != {
        "terminal_system_claim_guard",
        "quality_bridge_result",
        "runtime_compute_claim_guard",
        "runtime_compute_fairness",
        "terminal_pair_monitor",
        "cofitok_latest",
        "dense_latest",
        "cofitok_training",
        "dense_training",
        "class_fidelity_qualification",
        "cofitok_class_fidelity",
        "dense_class_fidelity",
        "official_related_methods",
    }:
        raise ValueError("quality-bridge comparison source set differs")
    verified = {}
    for name, expected in sources.items():
        if not isinstance(expected, Mapping):
            raise TypeError(f"comparison source identity is malformed: {name}")
        actual = file_identity(
            reject_symlink_chain(Path(str(expected.get("path", ""))), name=name)
        )
        if actual != dict(expected):
            raise ValueError(f"comparison source changed after binding: {name}")
        verified[name] = actual
    physical_verified = {}
    rows = report.get("matched_training_rows")
    if not isinstance(rows, list) or len(rows) != len(EXPECTED_METHODS):
        raise ValueError("comparison matched rows are missing")
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("comparison matched row is malformed")
        method = str(row.get("method", ""))
        checkpoint = _replay_file_identity(
            row.get("checkpoint_identity"),
            label=f"{method} comparison checkpoint",
        )
        integrity = _replay_file_identity(
            row.get("checkpoint_integrity_manifest_identity"),
            label=f"{method} comparison checkpoint integrity manifest",
        )
        if checkpoint.get("sha256") != row.get("checkpoint_sha256") or integrity.get(
            "path"
        ) != row.get("checkpoint_integrity_manifest"):
            raise ValueError(f"comparison physical source differs: {method}")
        physical_verified[method] = {
            "checkpoint": checkpoint,
            "checkpoint_integrity_manifest": integrity,
        }
    return {
        "status": "verified",
        "source_reports": verified,
        "physical_sources": physical_verified,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    paired_kid_evaluated = report["terminal_claim_policy"].get(
        "paired_kid_statistical_evidence_available"
    )
    lines = [
        "# Quality-Bridge Strong-Baseline Comparison",
        "",
        f"Terminal system status: `{report['status']}`; decision: `{report['decision']}`.",
        "",
        "## Matched training direct (100K steps / 10K terminal samples)",
        "",
        "| method | params | steps | train images | train h | VRAM GiB | sample h | FID | IS | precision | recall | class top-1 | class top-5 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["matched_training_rows"]:
        lines.append(
            "| {method} | {params} | {steps} | {images} | {train_h:.3f} | "
            "{vram:.3f} | {sample_h:.3f} | {fid:.4f} | {iscore:.4f} | "
            "{precision:.4f} | {recall:.4f} | {top1:.4f} | {top5:.4f} |".format(
                method=row["method"],
                params=row["parameter_count"],
                steps=row["training_steps"],
                images=row["training_images_seen"],
                train_h=row["training_elapsed_seconds"] / 3600.0,
                vram=row["peak_vram_bytes"] / (1024**3),
                sample_h=row["sampling_elapsed_seconds"] / 3600.0,
                fid=row["fid"],
                iscore=row["inception_score"],
                precision=row["precision"],
                recall=row["recall"],
                top1=row["class_top1_accuracy"],
                top5=row["class_top5_accuracy"],
            )
        )
    lines.extend(
        [
            "",
            (
                "Training is matched by dataset, architecture contract, effective "
                "batch, optimizer steps, and training images. Runtime and VRAM are "
                "measured outcomes, not equal wall-clock/GPU-hour/FLOP budgets."
            ),
            "",
            "## Official pretrained context (not a direct ranking)",
            "",
            "| method | samples | FID | IS | precision | recall | protocol |",
            "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in report["official_context_rows"]:
        lines.append(
            "| {method} | {samples} | {fid:.4f} | {iscore:.4f} | "
            "{precision:.4f} | {recall:.4f} | {protocol} |".format(
                method=row["method"],
                samples=row["sample_count"],
                fid=row["fid"],
                iscore=row["inception_score"],
                precision=row["precision"],
                recall=row["recall"],
                protocol=row["protocol_note"],
            )
        )
    lines.extend(
        [
            "",
            "Do not rank across the two panels: checkpoint source, training budget, sample count, and evaluator differ.",
            "",
            "This report is permanently non-authorizing and does not establish release readiness or broad generation superiority.",
            "",
            (
                "Replication boundary: FID uses one exact bound terminal sample "
                "stream; paired block-KID was not evaluated; independent "
                "replications: 0."
                if paired_kid_evaluated is False
                else (
                    "Replication boundary: FID and paired block-KID re-analyze one "
                    "exact bound terminal sample stream; independent replications: 0."
                )
            ),
            "",
        ]
    )
    return "\n".join(lines)


def render_csv(report: Mapping[str, Any]) -> str:
    fields = [
        "comparison_tier",
        "directly_comparable_to_cofitok",
        "method",
        "dataset",
        "resolution",
        "parameter_count",
        "training_steps",
        "effective_batch_size",
        "training_images_seen",
        "training_elapsed_seconds",
        "training_images_per_second",
        "peak_vram_bytes",
        "sample_count",
        "sample_batch_size",
        "sampling_elapsed_seconds",
        "sampling_images_per_second",
        "sample_steps",
        "guidance_scale",
        "fid",
        "inception_score",
        "precision",
        "recall",
        "class_top1_accuracy",
        "class_top5_accuracy",
        "class_predicted_class_fraction",
        "class_normalized_predicted_entropy",
        "protocol_note",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(report["matched_training_rows"])
    writer.writerows(report["official_context_rows"])
    return output.getvalue()


def _expected_payload(path: Path, payload: str) -> None:
    encoded = payload.encode("utf-8")
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError(f"existing comparison output differs: {path}")
        return
    write_text_report(path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound, non-authorizing two-tier comparison for the "
            "full-data 100K quality bridge."
        )
    )
    parser.add_argument("--terminal-system-guard", type=Path, required=True)
    parser.add_argument("--expected-terminal-system-guard-sha256", required=True)
    parser.add_argument("--official-related", type=Path, required=True)
    parser.add_argument("--expected-official-related-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    guard_identity, guard = _bound_json(
        args.terminal_system_guard,
        expected_sha256=args.expected_terminal_system_guard_sha256,
        label="terminal system claim guard",
    )
    official_identity, official = _bound_json(
        args.official_related,
        expected_sha256=args.expected_official_related_sha256,
        label="official related-method table",
    )
    report = build_report(
        terminal_guard_identity=guard_identity,
        terminal_guard=guard,
        official_identity=official_identity,
        official_related=official,
    )
    verify_source_reports(report)
    output_dir = reject_symlink_chain(
        args.output_dir,
        name="quality-bridge comparison output directory",
    ).resolve()
    json_path = output_dir / "quality_bridge_comparison.json"
    markdown_path = output_dir / "quality_bridge_comparison.md"
    csv_path = output_dir / "quality_bridge_comparison.csv"
    json_payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    _expected_payload(json_path, json_payload)
    _expected_payload(markdown_path, render_markdown(report))
    _expected_payload(csv_path, render_csv(report))
    print(json_path)


if __name__ == "__main__":
    main()
