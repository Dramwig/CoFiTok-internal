from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation_metrics_integrity import (
    validate_generation_reports_against_metrics_trust,
    verify_metrics_trust_receipt,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report
from scripts import build_generation_quality_bridge_comparison as comparison_builder
from scripts import (
    verify_generation_checkpoint_integrity_audit_replay as checkpoint_replay,
)
from scripts.build_generation_quality_bridge_result import build_from_args

SCHEMA_VERSION = 1
ROLE = "generation_quality_bridge_terminal_completion_audit"
TERMINAL_GUARD_ROLE = "generation_terminal_system_claim_guard"
TERMINAL_WAITER_ROLE = "generation_terminal_system_claim_guard_waiter"
COMPARISON_ROLE = "stability_full_data_quality_bridge_comparison"
COMPARISON_WAITER_ROLE = "generation_quality_bridge_comparison_waiter"
REPLAY_WAITER_ROLE = "generation_checkpoint_physical_integrity_audit_replay_waiter"
REPLAY_ROLE = "generation_checkpoint_physical_integrity_audit_replay"
EXPECTED_METHODS = ("cofitok", "dense_identity")
RUN_NAMES = {
    "cofitok": "cofitok_rgbtail3_rollout_x0_u2_ema_teacher",
    "dense_identity": "dense_rollout_x0_u2_ema_teacher",
}
AUDIT_ALIASES = {"cofitok": "cofitok", "dense_identity": "dense"}
CHECKPOINT_STEPS = (90_000, 95_000, 100_000)
EXPECTED_TRAINING_STEPS = 100_000
EXPECTED_EFFECTIVE_BATCH = 64
EXPECTED_TERMINAL_SAMPLES = 10_000
REPLICATION_INTERPRETATION = (
    "paired_reanalysis_of_one_exact_bound_terminal_sample_stream"
)
UNPAIRED_REPLICATION_INTERPRETATION = (
    "bound_terminal_stream_without_paired_reanalysis"
)
TERMINAL_DECISIONS = {
    "pass": "matched_quality_advantage_qualified_with_terminal_system_evidence",
    "hold": "terminal_system_evidence_complete_without_qualified_matched_advantage",
}

AUTHORIZATION_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "cpu_only_evidence_replay_allowed": True,
    "gpu_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "inference_export_authorization_allowed": False,
    "process_signals_allowed": False,
    "upstream_decisions_modified": False,
    "cross_tier_numeric_ranking_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "sota_claim_allowed": False,
    "independent_replication_claim_allowed": False,
    "multiple_independent_terminal_streams_claim_allowed": False,
}


def validated_report_dir_name(value: str, *, prefix: str, label: str) -> str:
    if (
        not value.startswith(prefix)
        or re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", value) is None
        or Path(value).name != value
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def terminal_claim_outcome(*, status: str, decision: str) -> dict[str, Any]:
    expected_decision = TERMINAL_DECISIONS.get(status)
    if expected_decision is None or decision != expected_decision:
        raise ValueError("terminal system guard status and decision are inconsistent")
    return {
        "terminal_status": status,
        "terminal_decision": decision,
        "generation_advantage_proven": status == "pass",
    }


def validate_replication_boundary(
    terminal_guard: Mapping[str, Any],
) -> dict[str, Any]:
    policy = terminal_guard.get("claim_policy")
    evidence = terminal_guard.get("evidence")
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
        not isinstance(policy, Mapping)
        or policy.get("independent_replication_claim_allowed") is not False
        or policy.get("multiple_independent_terminal_streams_claim_allowed")
        is not False
        or policy.get("replication_language_requires_distinct_bound_streams")
        is not True
        or not isinstance(replication, Mapping)
        or replication.get("bound_terminal_stream_count") != 1
        or replication.get("independent_replication_count") != 0
        or replication.get("independent_replication_supported") is not False
        or not isinstance(paired_kid_evaluated, bool)
        or (
            terminal_guard.get("status") == "pass"
            and paired_kid_evaluated is not True
        )
        or replication.get("interpretation") != expected_interpretation
        or int(replication.get("start_index", -1)) != 0
        or int(replication.get("end_index_exclusive", -1))
        != EXPECTED_TERMINAL_SAMPLES
        or int(replication.get("sample_count", -1)) != EXPECTED_TERMINAL_SAMPLES
        or not isinstance(replication.get("bound_stream_id"), str)
        or not replication.get("bound_stream_id")
    ):
        raise ValueError("terminal replication boundary differs")
    return copy.deepcopy(dict(replication))


def _tree(project: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(project), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()


def require_git_identity(
    project: Path,
    *,
    revision: str,
    tree: str,
    branch: str,
    label: str,
) -> dict[str, Any]:
    root = reject_symlink_chain(project, name=label).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"{label} is missing: {root}")
    observed = {**git_provenance(root), "tree": _tree(root)}
    expected = {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }
    if observed != expected:
        raise ValueError(f"{label} Git identity differs: {observed}")
    return observed


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
    if not isinstance(descriptor, Mapping):
        raise TypeError(f"{label} identity is missing")
    expected = dict(descriptor)
    if int(expected.get("bytes", 0)) < 1 or not _is_sha256(expected.get("sha256")):
        raise ValueError(f"{label} identity is malformed")
    source = reject_symlink_chain(
        Path(str(expected.get("path", ""))),
        name=label,
    ).resolve()
    actual = file_identity(source)
    if actual != expected:
        raise ValueError(f"{label} changed after binding")
    return actual, read_json_object(source, name=label)


def _quality_result_args(report: Mapping[str, Any]) -> argparse.Namespace:
    sources = report.get("source_reports")
    git = report.get("git")
    expected_names = {
        "preparation",
        "launch_receipt",
        "cofitok_training",
        "dense_training",
        "training_pair_validation",
        "cofitok_training_audit",
        "dense_training_audit",
        "milestone_50000",
        "milestone_100000",
        "cofitok_sampling_preflight",
        "dense_sampling_preflight",
        "cofitok_generation",
        "dense_generation",
        "cofitok_checkpoint_eval",
        "dense_checkpoint_eval",
        "class_fidelity_qualification",
        "cofitok_class_fidelity",
        "dense_class_fidelity",
    }
    if not isinstance(sources, Mapping) or set(sources) != expected_names:
        raise ValueError("terminal quality result source set differs")
    if not isinstance(git, Mapping):
        raise TypeError("terminal quality result Git identity is missing")
    normalized: dict[str, dict[str, Any]] = {}
    for name, descriptor in sources.items():
        if not isinstance(descriptor, Mapping):
            raise TypeError(f"quality result source is malformed: {name}")
        identity = dict(descriptor)
        if int(identity.get("bytes", 0)) < 1 or not _is_sha256(identity.get("sha256")):
            raise ValueError(f"quality result source identity is invalid: {name}")
        normalized[name] = identity
    return argparse.Namespace(
        preparation=Path(normalized["preparation"]["path"]),
        expected_preparation_sha256=normalized["preparation"]["sha256"],
        launch_receipt=Path(normalized["launch_receipt"]["path"]),
        expected_launch_receipt_sha256=normalized["launch_receipt"]["sha256"],
        cofitok_training=Path(normalized["cofitok_training"]["path"]),
        dense_training=Path(normalized["dense_training"]["path"]),
        training_pair_validation=Path(normalized["training_pair_validation"]["path"]),
        cofitok_training_audit=Path(normalized["cofitok_training_audit"]["path"]),
        dense_training_audit=Path(normalized["dense_training_audit"]["path"]),
        milestone_50000=Path(normalized["milestone_50000"]["path"]),
        milestone_100000=Path(normalized["milestone_100000"]["path"]),
        cofitok_sampling_preflight=Path(
            normalized["cofitok_sampling_preflight"]["path"]
        ),
        dense_sampling_preflight=Path(normalized["dense_sampling_preflight"]["path"]),
        cofitok_generation=Path(normalized["cofitok_generation"]["path"]),
        dense_generation=Path(normalized["dense_generation"]["path"]),
        cofitok_checkpoint_eval=Path(normalized["cofitok_checkpoint_eval"]["path"]),
        dense_checkpoint_eval=Path(normalized["dense_checkpoint_eval"]["path"]),
        class_fidelity_qualification=Path(
            normalized["class_fidelity_qualification"]["path"]
        ),
        cofitok_class_fidelity=Path(normalized["cofitok_class_fidelity"]["path"]),
        dense_class_fidelity=Path(normalized["dense_class_fidelity"]["path"]),
        expected_revision=str(git.get("revision", "")),
        expected_branch=str(git.get("branch", "")),
    )


def replay_quality_result(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    replayed = build_from_args(_quality_result_args(report))
    if replayed != dict(report):
        raise ValueError("terminal quality result does not replay exactly")
    sources = report["source_reports"]
    verified_sources: dict[str, dict[str, Any]] = {}
    generation_metric_reports: dict[str, dict[str, Any]] = {}
    for name, descriptor in sources.items():
        actual, source_report = _replay_identity(
            descriptor,
            label=f"quality result source {name}",
        )
        verified_sources[name] = actual
        if name in {"cofitok_generation", "dense_generation"}:
            generation_metric_reports[
                "cofitok" if name == "cofitok_generation" else "dense_identity"
            ] = {
                field: copy.deepcopy(source_report.get(field))
                for field in (
                    "schema_version",
                    "role",
                    "status",
                    "protocol",
                    "implementation",
                    "runtime_environment_sha256",
                    "real_set",
                    "parameters",
                )
            }
    terminal = report.get("terminal")
    if not isinstance(terminal, Mapping):
        raise TypeError("terminal quality result lacks terminal evidence")
    physical = terminal.get("physical_evidence")
    methods = terminal.get("methods")
    class_fidelity = terminal.get("class_fidelity")
    if (
        not isinstance(physical, Mapping)
        or set(physical) != set(EXPECTED_METHODS)
        or not isinstance(methods, Mapping)
        or set(methods) != set(EXPECTED_METHODS)
        or not isinstance(class_fidelity, Mapping)
    ):
        raise ValueError("terminal quality result physical evidence differs")
    return {
        "identity": dict(identity),
        "exact_replay": True,
        "source_reports": verified_sources,
        "generation_metric_reports": generation_metric_reports,
        "git": copy.deepcopy(dict(report["git"])),
        "quality_screen": copy.deepcopy(dict(report["quality_screen"])),
        "physical_evidence": copy.deepcopy(dict(physical)),
        "methods": copy.deepcopy(dict(methods)),
        "class_fidelity": copy.deepcopy(dict(class_fidelity)),
    }


def verify_class_fidelity_classifier_sources(
    source_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    names = (
        "class_fidelity_qualification",
        "cofitok_class_fidelity",
        "dense_class_fidelity",
    )
    identities: dict[str, dict[str, Any]] = {}
    classifiers: dict[str, dict[str, Any]] = {}
    for name in names:
        descriptor = source_reports.get(name)
        identity, report = _replay_identity(
            descriptor,
            label=f"terminal {name.replace('_', ' ')}",
        )
        classifier = report.get("classifier")
        if not isinstance(classifier, Mapping):
            raise TypeError(f"terminal {name} classifier identity is missing")
        identities[name] = identity
        classifiers[name] = copy.deepcopy(dict(classifier))

    reference = classifiers["class_fidelity_qualification"]
    if any(classifier != reference for classifier in classifiers.values()):
        raise ValueError("terminal class-fidelity classifier identities differ")
    weights_path = reference.get("weights_path")
    if (
        not isinstance(weights_path, str)
        or not weights_path
        or not Path(weights_path).is_absolute()
    ):
        raise ValueError("terminal class-fidelity classifier path is malformed")
    weights = file_identity(
        reject_symlink_chain(
            Path(weights_path),
            name="terminal class-fidelity classifier weight",
        ).resolve()
    )
    if (
        weights["path"] != Path(weights_path).resolve().as_posix()
        or int(reference.get("weights_bytes", -1)) != int(weights["bytes"])
        or reference.get("weights_sha256") != weights["sha256"]
    ):
        raise ValueError(
            "terminal class-fidelity classifier physical identity differs"
        )
    return {
        "status": "verified",
        "classifier": reference,
        "physical_weights": weights,
        "source_reports": identities,
        "matched_report_classifier_identity": True,
        "physical_weight_sha256_replayed": True,
    }


def replay_comparison(
    *,
    terminal_guard_identity: Mapping[str, Any],
    terminal_guard: Mapping[str, Any],
    comparison_identity: Mapping[str, Any],
    comparison_report: Mapping[str, Any],
) -> dict[str, Any]:
    replication_scope = validate_replication_boundary(terminal_guard)
    sources = comparison_report.get("source_reports")
    if not isinstance(sources, Mapping):
        raise TypeError("terminal comparison sources are missing")
    official_identity, official = _replay_identity(
        sources.get("official_related_methods"),
        label="official related-method table",
    )
    replayed = comparison_builder.build_report(
        terminal_guard_identity=terminal_guard_identity,
        terminal_guard=terminal_guard,
        official_identity=official_identity,
        official_related=official,
    )
    if replayed != dict(comparison_report):
        raise ValueError("terminal comparison does not replay exactly")
    verified = comparison_builder.verify_source_reports(comparison_report)
    if sources.get("terminal_system_claim_guard") != dict(terminal_guard_identity):
        raise ValueError("terminal comparison uses another system guard")
    comparison_policy = comparison_report.get("comparison_policy")
    if (
        not isinstance(comparison_policy, Mapping)
        or comparison_policy.get("independent_replication_claim_allowed") is not False
        or comparison_policy.get(
            "multiple_independent_terminal_streams_claim_allowed"
        )
        is not False
        or comparison_policy.get(
            "replication_language_requires_distinct_bound_streams"
        )
        is not True
        or comparison_report.get("replication_scope") != replication_scope
    ):
        raise ValueError("terminal comparison replication boundary differs")
    return {
        "identity": dict(comparison_identity),
        "exact_replay": True,
        "source_verification": verified,
        "official_related_methods": official_identity,
        "status": str(comparison_report.get("status", "")),
        "decision": str(comparison_report.get("decision", "")),
        "replication_scope": replication_scope,
    }


def validate_status_bindings(
    *,
    terminal_status_identity: Mapping[str, Any],
    terminal_status: Mapping[str, Any],
    terminal_guard_identity: Mapping[str, Any],
    terminal_guard: Mapping[str, Any],
    comparison_status_identity: Mapping[str, Any],
    comparison_status: Mapping[str, Any],
    comparison_identity: Mapping[str, Any],
) -> dict[str, Any]:
    replication_scope = validate_replication_boundary(terminal_guard)
    terminal_claim_status = str(terminal_guard.get("status", ""))
    if (
        terminal_status.get("schema_version") != 1
        or terminal_status.get("role") != TERMINAL_WAITER_ROLE
        or terminal_status.get("status") != "completed"
        or terminal_status.get("detail")
        != "terminal_system_claim_guard_source_revalidated"
        or terminal_status.get("guard") != dict(terminal_guard_identity)
        or terminal_status.get("guard_status") != terminal_claim_status
        or terminal_claim_status not in {"pass", "hold"}
    ):
        raise ValueError("terminal system guard waiter status differs")
    comparison_outputs = comparison_status.get("comparison")
    comparison_terminal = comparison_status.get("terminal")
    if (
        comparison_status.get("schema_version") != 1
        or comparison_status.get("role") != COMPARISON_WAITER_ROLE
        or comparison_status.get("status") != "pass"
        or comparison_status.get("detail")
        != "terminal_quality_bridge_comparison_revalidated"
        or comparison_status.get("terminal_status") != terminal_claim_status
        or not isinstance(comparison_terminal, Mapping)
        or comparison_terminal.get("replication_scope") != replication_scope
        or not isinstance(comparison_outputs, Mapping)
        or set(comparison_outputs) != {"json", "markdown", "csv"}
        or comparison_outputs.get("json") != dict(comparison_identity)
    ):
        raise ValueError("terminal comparison waiter status differs")
    output_identities: dict[str, dict[str, Any]] = {}
    for name in ("json", "markdown", "csv"):
        descriptor = comparison_outputs.get(name)
        actual, _ = (
            _replay_identity(descriptor, label=f"terminal comparison {name}")
            if name == "json"
            else (_replay_file(descriptor, label=f"terminal comparison {name}"), None)
        )
        output_identities[name] = actual
    return {
        "terminal_waiter_status": dict(terminal_status_identity),
        "comparison_waiter_status": dict(comparison_status_identity),
        "comparison_outputs": output_identities,
        "terminal_status": terminal_claim_status,
        "terminal_decision": str(terminal_guard.get("decision", "")),
        "replication_scope": replication_scope,
    }


def validate_comparison_renderings(
    report: Mapping[str, Any],
    outputs: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    json_descriptor = outputs.get("json")
    if not isinstance(json_descriptor, Mapping):
        raise TypeError("terminal comparison JSON identity is missing")
    json_path = reject_symlink_chain(
        Path(str(json_descriptor.get("path", ""))),
        name="terminal comparison JSON",
    ).resolve()
    expected = {
        "markdown": (
            json_path.with_name("quality_bridge_comparison.md"),
            comparison_builder.render_markdown(report),
        ),
        "csv": (
            json_path.with_name("quality_bridge_comparison.csv"),
            comparison_builder.render_csv(report),
        ),
    }
    verified: dict[str, dict[str, Any]] = {}
    for name, (expected_path, payload) in expected.items():
        descriptor = outputs.get(name)
        if not isinstance(descriptor, Mapping):
            raise TypeError(f"terminal comparison {name} identity is missing")
        source = reject_symlink_chain(
            Path(str(descriptor.get("path", ""))),
            name=f"terminal comparison {name}",
        ).resolve()
        if source != expected_path:
            raise ValueError(f"terminal comparison {name} path is not canonical")
        if source.read_bytes() != payload.encode("utf-8"):
            raise ValueError(f"terminal comparison {name} does not replay exactly")
        actual = file_identity(source)
        if actual != dict(descriptor):
            raise ValueError(f"terminal comparison {name} changed during replay")
        verified[name] = actual
    return verified


def revalidate_checkpoint_chain_sources(
    checkpoint_chain: Mapping[str, Any],
) -> dict[str, Any]:
    methods = checkpoint_chain.get("methods")
    replay_waiter = checkpoint_chain.get("dense_replay_waiter")
    physical_auditor = checkpoint_chain.get("physical_auditor")
    checkpoint_replay_info = checkpoint_chain.get("checkpoint_replay")
    if not all(
        isinstance(value, Mapping)
        for value in (methods, replay_waiter, physical_auditor, checkpoint_replay_info)
    ):
        raise TypeError("checkpoint-chain completion evidence is malformed")

    def stable(expected: Any, *, label: str) -> dict[str, Any]:
        if not isinstance(expected, Mapping):
            raise TypeError(f"{label} identity is missing")
        identity = {key: expected.get(key) for key in ("path", "bytes", "sha256")}
        actual = checkpoint_replay.stable_file_identity(
            Path(str(identity.get("path", ""))),
            name=label,
        )
        if actual != identity:
            raise ValueError(f"{label} changed after physical replay")
        return actual

    revalidated_methods: dict[str, Any] = {}
    for method in EXPECTED_METHODS:
        method_report = methods.get(method)
        if not isinstance(method_report, Mapping):
            raise TypeError(f"{method} checkpoint completion evidence is missing")
        steps = method_report.get("steps")
        if not isinstance(steps, Mapping) or set(steps) != {
            str(step) for step in CHECKPOINT_STEPS
        }:
            raise ValueError(f"{method} checkpoint completion steps differ")
        revalidated_steps: dict[str, Any] = {}
        for step in CHECKPOINT_STEPS:
            row = steps[str(step)]
            checkpoint = row.get("checkpoint") if isinstance(row, Mapping) else None
            if not isinstance(checkpoint, Mapping):
                raise TypeError(f"{method} checkpoint {step} evidence is malformed")
            sources = {
                "audit_report": stable(
                    row.get("audit_report"),
                    label=f"{method} checkpoint {step} audit report",
                ),
                "waiter_status": stable(
                    row.get("waiter_status"),
                    label=f"{method} checkpoint {step} waiter status",
                ),
                "checkpoint_payload": stable(
                    checkpoint.get("payload"),
                    label=f"{method} checkpoint {step} payload",
                ),
                "checkpoint_integrity_manifest": stable(
                    checkpoint.get("integrity_manifest"),
                    label=f"{method} checkpoint {step} integrity manifest",
                ),
            }
            if method == "dense_identity":
                sources["independent_replay_receipt"] = stable(
                    row.get("independent_replay_receipt"),
                    label=f"dense checkpoint {step} independent replay receipt",
                )
            revalidated_steps[str(step)] = sources
        revalidated_methods[method] = {
            "steps": revalidated_steps,
            "canonical_latest": stable(
                method_report.get("canonical_latest"),
                label=f"{method} canonical latest pointer",
            ),
            "canonical_metrics": stable(
                method_report.get("canonical_metrics"),
                label=f"{method} canonical metrics",
            ),
        }
    return {
        "methods": revalidated_methods,
        "dense_replay_waiter": stable(
            replay_waiter.get("identity"),
            label="dense checkpoint replay waiter status",
        ),
        "physical_auditor_source": stable(
            physical_auditor.get("source"),
            label="physical checkpoint auditor source",
        ),
        "checkpoint_replay_verifier_source": stable(
            checkpoint_replay_info.get("verifier_source"),
            label="checkpoint replay verifier source",
        ),
    }


def _replay_file(descriptor: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(descriptor, Mapping):
        raise TypeError(f"{label} identity is missing")
    expected = dict(descriptor)
    if int(expected.get("bytes", 0)) < 1 or not _is_sha256(expected.get("sha256")):
        raise ValueError(f"{label} identity is malformed")
    path = reject_symlink_chain(Path(str(expected.get("path", ""))), name=label)
    actual = file_identity(path)
    if actual != expected:
        raise ValueError(f"{label} changed after binding")
    return actual


def _audit_paths(audit_dir: Path, *, alias: str, step: int) -> dict[str, Path]:
    stem = f"{alias}_checkpoint_step_{step:08d}"
    return {
        "status": audit_dir / f"{stem}_waiter_status.json",
        "audit": audit_dir / f"{stem}_physical_integrity_audit.json",
        "replay": audit_dir / f"{stem}_physical_integrity_audit_replay.json",
    }


def _receipt_view(receipt: Mapping[str, Any]) -> dict[str, Any]:
    latest = receipt.get("latest_pointer_replay")
    metrics = receipt.get("metrics_replay")
    if not isinstance(latest, Mapping) or not isinstance(metrics, Mapping):
        raise TypeError("checkpoint replay receipt historical evidence is malformed")
    return {
        "audit_report": copy.deepcopy(receipt.get("audit_report")),
        "waiter_status": copy.deepcopy(receipt.get("waiter_status")),
        "auditor_source": copy.deepcopy(receipt.get("auditor_source")),
        "auditor_checkout": copy.deepcopy(receipt.get("auditor_checkout")),
        "training_checkout": copy.deepcopy(receipt.get("training_checkout")),
        "checkpoint": copy.deepcopy(receipt.get("checkpoint")),
        "historical_latest": {
            "historical_identity": copy.deepcopy(latest.get("historical_identity")),
            "historical_content": copy.deepcopy(latest.get("historical_content")),
            "historical_byte_identity_reconstructed": latest.get(
                "historical_byte_identity_reconstructed"
            ),
        },
        "historical_metrics": {
            "audit_time_prefix": copy.deepcopy(metrics.get("audit_time_prefix")),
            "audit_time_prefix_byte_exact": metrics.get("audit_time_prefix_byte_exact"),
            "target_row": copy.deepcopy(metrics.get("target_row")),
        },
        "checks": copy.deepcopy(receipt.get("checks")),
        "scope": copy.deepcopy(receipt.get("scope")),
    }


def _validate_checkout_git_receipt(
    identity: Any,
    *,
    expected_git: Mapping[str, Any],
    checkout: Path,
    label: str,
) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise TypeError(f"{label} is missing")
    expected = {
        **dict(expected_git),
        "path": checkout.resolve().as_posix(),
    }
    if dict(identity) != expected:
        raise ValueError(f"{label} differs")
    return expected


def _validate_dense_replay_status(
    *,
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    expected_receipts: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    milestones = report.get("milestones")
    if (
        report.get("schema_version") != 1
        or report.get("role") != REPLAY_WAITER_ROLE
        or report.get("status") != "pass"
        or report.get("detail") != "all_checkpoint_integrity_replays_completed"
        or int(report.get("completed_count", -1)) != len(CHECKPOINT_STEPS)
        or int(report.get("expected_count", -1)) != len(CHECKPOINT_STEPS)
        or not isinstance(milestones, Mapping)
        or set(milestones) != {str(step) for step in CHECKPOINT_STEPS}
    ):
        raise ValueError("dense checkpoint replay waiter status differs")
    for step in CHECKPOINT_STEPS:
        row = milestones[str(step)]
        if (
            not isinstance(row, Mapping)
            or row.get("state") != "pass"
            or row.get("detail") != "replay_completed"
            or int(row.get("checkpoint_step", -1)) != step
            or row.get("replay_output") != dict(expected_receipts[step])
        ):
            raise ValueError(
                f"dense checkpoint replay waiter milestone differs: {step}"
            )
    return {"identity": dict(identity), "milestones": copy.deepcopy(dict(milestones))}


def _terminal_canonical_state(
    replay: Mapping[str, Any],
    *,
    method: str,
) -> dict[str, dict[str, Any]]:
    latest = replay.get("latest_pointer_replay")
    metrics = replay.get("metrics_replay")
    if not isinstance(latest, Mapping) or not isinstance(metrics, Mapping):
        raise TypeError(f"{method} terminal latest/metrics replay is malformed")
    current_content = latest.get("current_content")
    current_metrics = metrics.get("current_observed_prefix")
    audit_metrics = metrics.get("audit_time_prefix")
    target_row = metrics.get("target_row")
    if not all(
        isinstance(value, Mapping)
        for value in (current_content, current_metrics, audit_metrics, target_row)
    ):
        raise TypeError(f"{method} terminal latest/metrics replay is malformed")
    if (
        latest.get("relation") != "exact_historical_pointer_still_current"
        or int(current_content.get("step", -1)) != EXPECTED_TRAINING_STEPS
        or int(current_metrics.get("last_step", -1)) != EXPECTED_TRAINING_STEPS
        or int(audit_metrics.get("last_step", -1)) != EXPECTED_TRAINING_STEPS
        or int(target_row.get("step", -1)) != EXPECTED_TRAINING_STEPS
        or int(target_row.get("samples_seen", -1))
        != EXPECTED_TRAINING_STEPS * EXPECTED_EFFECTIVE_BATCH
        or current_metrics.get("bytes") != audit_metrics.get("bytes")
        or current_metrics.get("sha256") != audit_metrics.get("sha256")
    ):
        raise ValueError(f"{method} canonical terminal latest/metrics differ")
    return {
        "latest": copy.deepcopy(dict(latest)),
        "metrics": copy.deepcopy(dict(metrics)),
    }


def verify_checkpoint_chain(
    *,
    quality_root: Path,
    audit_dir: Path,
    quality: Mapping[str, Any],
    self_git: Mapping[str, Any],
    training_checkout: Path,
    training_git: Mapping[str, Any],
    auditor_checkout: Path,
    auditor_git: Mapping[str, Any],
    auditor_source_sha256: str,
    replay_checkout: Path,
    replay_git: Mapping[str, Any],
    replay_verifier_source_sha256: str,
    replay_waiter_status_path: Path,
    dataset_sha256: str,
    runtime_sha256: str,
    effective_batch: int,
) -> dict[str, Any]:
    expected_audit_dir = quality_root / "reports" / "checkpoint_audits"
    if audit_dir != expected_audit_dir:
        raise ValueError("checkpoint audit directory is not canonical")
    auditor_source = (
        auditor_checkout / "scripts" / "wait_generation_checkpoint_integrity_audit.py"
    )
    auditor_source_identity = file_identity(auditor_source)
    if auditor_source_identity["sha256"] != auditor_source_sha256:
        raise ValueError("physical checkpoint auditor source SHA256 differs")
    replay_verifier = (
        replay_checkout
        / "scripts"
        / "verify_generation_checkpoint_integrity_audit_replay.py"
    )
    replay_verifier_identity = file_identity(replay_verifier)
    if replay_verifier_identity["sha256"] != replay_verifier_source_sha256:
        raise ValueError("checkpoint replay verifier source SHA256 differs")

    methods: dict[str, Any] = {}
    dense_receipts: dict[int, dict[str, Any]] = {}
    for method in EXPECTED_METHODS:
        alias = AUDIT_ALIASES[method]
        run_dir = quality_root / RUN_NAMES[method]
        terminal_physical = quality["physical_evidence"][method]
        terminal_checkpoint = terminal_physical.get("checkpoint")
        if not isinstance(terminal_checkpoint, Mapping):
            raise TypeError(f"{method} terminal checkpoint identity is missing")
        steps: dict[str, Any] = {}
        for step in CHECKPOINT_STEPS:
            paths = _audit_paths(audit_dir, alias=alias, step=step)
            audit, audit_identity = checkpoint_replay.stable_json(
                paths["audit"],
                name=f"{method} checkpoint {step} physical audit",
            )
            _status, status_identity = checkpoint_replay.stable_json(
                paths["status"],
                name=f"{method} checkpoint {step} waiter status",
            )
            checkpoint = audit.get("checkpoint")
            payload = (
                checkpoint.get("payload") if isinstance(checkpoint, Mapping) else None
            )
            if not isinstance(payload, Mapping) or not _is_sha256(
                payload.get("sha256")
            ):
                raise ValueError(
                    f"{method} checkpoint {step} audit payload is malformed"
                )
            fresh = checkpoint_replay.verify_replay(
                audit_report_path=paths["audit"],
                expected_audit_report_sha256=audit_identity["sha256"],
                waiter_status_path=paths["status"],
                expected_waiter_status_sha256=status_identity["sha256"],
                auditor_source_path=auditor_source,
                expected_auditor_source_sha256=auditor_source_sha256,
                auditor_checkout=auditor_checkout,
                expected_auditor_revision=str(auditor_git["revision"]),
                expected_auditor_tree=str(auditor_git["tree"]),
                expected_auditor_branch=str(auditor_git["branch"]),
                training_checkout=training_checkout,
                expected_training_revision=str(training_git["revision"]),
                expected_training_tree=str(training_git["tree"]),
                expected_training_branch=str(training_git["branch"]),
                expected_checkpoint_step=step,
                expected_checkpoint_sha256=str(payload["sha256"]),
                expected_dataset_sha256=dataset_sha256,
                expected_runtime_sha256=runtime_sha256,
                effective_batch=effective_batch,
                verifier_git=dict(self_git),
            )
            if Path(str(audit.get("run_dir", ""))).resolve() != run_dir.resolve():
                raise ValueError(f"{method} checkpoint {step} run directory differs")
            if step == EXPECTED_TRAINING_STEPS:
                if fresh["checkpoint"]["payload"] != dict(terminal_checkpoint):
                    raise ValueError(
                        f"{method} terminal checkpoint differs from physical audit"
                    )
                _terminal_canonical_state(fresh, method=method)
            row: dict[str, Any] = {
                "audit_report": audit_identity,
                "waiter_status": status_identity,
                "checkpoint": copy.deepcopy(fresh["checkpoint"]),
                "latest_pointer_replay": copy.deepcopy(fresh["latest_pointer_replay"]),
                "metrics_replay": copy.deepcopy(fresh["metrics_replay"]),
                "physical_replay_completed": True,
            }
            if method == "dense_identity":
                receipt, receipt_identity = checkpoint_replay.stable_json(
                    paths["replay"],
                    name=f"dense checkpoint {step} replay receipt",
                )
                _validate_checkout_git_receipt(
                    receipt.get("verifier_git"),
                    expected_git=replay_git,
                    checkout=replay_checkout,
                    label=f"dense checkpoint {step} replay verifier Git",
                )
                if (
                    receipt.get("schema_version") != 1
                    or receipt.get("role") != REPLAY_ROLE
                    or receipt.get("status") != "pass"
                    or receipt.get("verifier_source") != replay_verifier_identity
                    or _receipt_view(receipt) != _receipt_view(fresh)
                ):
                    raise ValueError(f"dense checkpoint {step} replay receipt differs")
                row["independent_replay_receipt"] = receipt_identity
                dense_receipts[step] = receipt_identity
            steps[str(step)] = row
        final_metrics = steps[str(EXPECTED_TRAINING_STEPS)]["metrics_replay"]
        final_latest = steps[str(EXPECTED_TRAINING_STEPS)]["latest_pointer_replay"]
        methods[method] = {
            "run_dir": run_dir.as_posix(),
            "steps": steps,
            "canonical_metrics": copy.deepcopy(
                final_metrics["current_observed_prefix"]
            ),
            "canonical_latest": copy.deepcopy(final_latest["current_identity"]),
            "all_checkpoint_payloads_physically_rehashed": True,
            "all_historical_metrics_prefixes_reconstructed": True,
        }

    replay_status, replay_status_identity = checkpoint_replay.stable_json(
        replay_waiter_status_path,
        name="dense checkpoint replay waiter status",
    )
    replay_waiter = _validate_dense_replay_status(
        identity=replay_status_identity,
        report=replay_status,
        expected_receipts=dense_receipts,
    )
    return {
        "methods": methods,
        "dense_replay_waiter": replay_waiter,
        "physical_auditor": {
            "git": dict(auditor_git),
            "source": auditor_source_identity,
        },
        "checkpoint_replay": {
            "git": dict(replay_git),
            "verifier_source": replay_verifier_identity,
        },
    }


def _canonical_paths(args: argparse.Namespace) -> dict[str, Path]:
    root = reject_symlink_chain(
        args.quality_output_root,
        name="quality bridge output root",
    ).resolve()
    paths = {
        "root": root,
        "terminal_guard": reject_symlink_chain(
            args.terminal_system_guard,
            name="terminal system guard",
        ).resolve(),
        "terminal_status": reject_symlink_chain(
            args.terminal_waiter_status,
            name="terminal system guard waiter status",
        ).resolve(),
        "comparison": reject_symlink_chain(
            args.comparison,
            name="terminal comparison",
        ).resolve(),
        "comparison_status": reject_symlink_chain(
            args.comparison_waiter_status,
            name="terminal comparison waiter status",
        ).resolve(),
        "audit_dir": reject_symlink_chain(
            args.checkpoint_audit_dir,
            name="checkpoint audit directory",
        ).resolve(),
        "replay_status": reject_symlink_chain(
            args.checkpoint_replay_waiter_status,
            name="checkpoint replay waiter status",
        ).resolve(),
        "metrics_trust_receipt": reject_symlink_chain(
            args.metrics_trust_receipt,
            name="terminal metrics trust receipt",
        ).resolve(),
    }
    terminal_dir_name = validated_report_dir_name(
        args.expected_terminal_dir_name,
        prefix="terminal_system_claim_guard_v",
        label="terminal guard directory name",
    )
    comparison_dir_name = validated_report_dir_name(
        args.expected_comparison_dir_name,
        prefix="quality_bridge_comparison_v",
        label="comparison directory name",
    )
    expected = {
        "terminal_guard": root
        / "reports"
        / terminal_dir_name
        / "terminal_system_claim_guard.json",
        "terminal_status": root
        / "reports"
        / terminal_dir_name
        / "waiter_status.json",
        "comparison": root
        / "reports"
        / comparison_dir_name
        / "quality_bridge_comparison.json",
        "comparison_status": root
        / "reports"
        / comparison_dir_name
        / "waiter_status.json",
        "audit_dir": root / "reports" / "checkpoint_audits",
        "replay_status": root
        / "reports"
        / "checkpoint_audits"
        / "dense_checkpoint_integrity_replay_waiter_status.json",
        "metrics_trust_receipt": root
        / "reports"
        / "metrics_trust_boundary_v1"
        / "metrics_trust_receipt.json",
    }
    for name, expected_path in expected.items():
        if paths[name] != expected_path.resolve():
            raise ValueError(f"terminal completion canonical path differs: {name}")
    return paths


def build_audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.effective_batch != EXPECTED_EFFECTIVE_BATCH:
        raise ValueError("terminal completion audit effective batch must be 64")
    for label, value in (
        ("dataset", args.expected_dataset_sha256),
        ("runtime", args.expected_runtime_sha256),
        ("physical auditor source", args.expected_physical_auditor_source_sha256),
        (
            "checkpoint replay verifier source",
            args.expected_checkpoint_replay_verifier_source_sha256,
        ),
        ("metrics trust receipt", args.expected_metrics_trust_receipt_sha256),
    ):
        if not _is_sha256(value):
            raise ValueError(f"terminal completion expected {label} SHA256 is invalid")
    paths = _canonical_paths(args)
    self_git = require_git_identity(
        args.project,
        revision=args.expected_revision,
        tree=args.expected_tree,
        branch=args.expected_branch,
        label="terminal completion audit project",
    )
    expected_self_source = reject_symlink_chain(
        args.project
        / "scripts"
        / "build_generation_quality_bridge_terminal_completion_audit.py",
        name="terminal completion audit source",
    ).resolve()
    actual_self_source = reject_symlink_chain(
        Path(__file__),
        name="loaded terminal completion audit source",
    ).resolve()
    if actual_self_source != expected_self_source:
        raise ValueError("loaded terminal completion audit source is not canonical")
    self_source_identity = file_identity(actual_self_source)
    training_git = require_git_identity(
        args.training_checkout,
        revision=args.expected_training_revision,
        tree=args.expected_training_tree,
        branch=args.expected_training_branch,
        label="quality bridge training checkout",
    )
    auditor_git = require_git_identity(
        args.physical_auditor_checkout,
        revision=args.expected_physical_auditor_revision,
        tree=args.expected_physical_auditor_tree,
        branch=args.expected_physical_auditor_branch,
        label="physical checkpoint auditor checkout",
    )
    replay_git = require_git_identity(
        args.checkpoint_replay_checkout,
        revision=args.expected_checkpoint_replay_revision,
        tree=args.expected_checkpoint_replay_tree,
        branch=args.expected_checkpoint_replay_branch,
        label="checkpoint replay checkout",
    )

    terminal_guard_identity, terminal_guard = _bound_json(
        paths["terminal_guard"],
        expected_sha256=args.expected_terminal_system_guard_sha256,
        label="terminal system claim guard",
    )
    comparison_identity, comparison_report = _bound_json(
        paths["comparison"],
        expected_sha256=args.expected_comparison_sha256,
        label="quality bridge terminal comparison",
    )
    metrics_trust_identity, metrics_trust_receipt = _bound_json(
        paths["metrics_trust_receipt"],
        expected_sha256=args.expected_metrics_trust_receipt_sha256,
        label="terminal metrics trust receipt",
    )
    metrics_trust = verify_metrics_trust_receipt(metrics_trust_receipt)
    terminal_status_identity, terminal_status = _replay_identity(
        file_identity(paths["terminal_status"]),
        label="terminal system guard waiter status",
    )
    comparison_status_identity, comparison_status = _replay_identity(
        file_identity(paths["comparison_status"]),
        label="terminal comparison waiter status",
    )
    status_bindings = validate_status_bindings(
        terminal_status_identity=terminal_status_identity,
        terminal_status=terminal_status,
        terminal_guard_identity=terminal_guard_identity,
        terminal_guard=terminal_guard,
        comparison_status_identity=comparison_status_identity,
        comparison_status=comparison_status,
        comparison_identity=comparison_identity,
    )
    if (
        terminal_guard.get("schema_version") != 1
        or terminal_guard.get("role") != TERMINAL_GUARD_ROLE
        or terminal_guard.get("status") not in {"pass", "hold"}
    ):
        raise ValueError("terminal system guard contract differs")

    quality_identity, quality_report = _replay_identity(
        terminal_guard.get("sources", {}).get("quality_bridge_result"),
        label="terminal quality bridge result",
    )
    quality = replay_quality_result(quality_identity, quality_report)
    class_fidelity_integrity = verify_class_fidelity_classifier_sources(
        quality["source_reports"]
    )
    metrics_report_binding = validate_generation_reports_against_metrics_trust(
        quality["generation_metric_reports"],
        verified_trust=metrics_trust,
    )
    comparison = replay_comparison(
        terminal_guard_identity=terminal_guard_identity,
        terminal_guard=terminal_guard,
        comparison_identity=comparison_identity,
        comparison_report=comparison_report,
    )
    rendered_comparison = validate_comparison_renderings(
        comparison_report,
        status_bindings["comparison_outputs"],
    )
    if (
        comparison["status"] != status_bindings["terminal_status"]
        or comparison["decision"] != status_bindings["terminal_decision"]
        or comparison["replication_scope"]
        != status_bindings["replication_scope"]
    ):
        raise ValueError("terminal comparison and system guard decisions differ")

    checkpoint_chain = verify_checkpoint_chain(
        quality_root=paths["root"],
        audit_dir=paths["audit_dir"],
        quality=quality,
        self_git=self_git,
        training_checkout=args.training_checkout.resolve(),
        training_git=training_git,
        auditor_checkout=args.physical_auditor_checkout.resolve(),
        auditor_git=auditor_git,
        auditor_source_sha256=args.expected_physical_auditor_source_sha256,
        replay_checkout=args.checkpoint_replay_checkout.resolve(),
        replay_git=replay_git,
        replay_verifier_source_sha256=(
            args.expected_checkpoint_replay_verifier_source_sha256
        ),
        replay_waiter_status_path=paths["replay_status"],
        dataset_sha256=args.expected_dataset_sha256,
        runtime_sha256=args.expected_runtime_sha256,
        effective_batch=args.effective_batch,
    )

    final_checkpoint_sources = revalidate_checkpoint_chain_sources(checkpoint_chain)
    final_quality = replay_quality_result(quality_identity, quality_report)
    if final_quality != quality:
        raise ValueError("terminal quality result changed during completion replay")
    final_class_fidelity_integrity = verify_class_fidelity_classifier_sources(
        final_quality["source_reports"]
    )
    if final_class_fidelity_integrity != class_fidelity_integrity:
        raise ValueError(
            "terminal class-fidelity classifier evidence changed during replay"
        )
    final_metrics_trust_identity, final_metrics_trust_receipt = _bound_json(
        paths["metrics_trust_receipt"],
        expected_sha256=args.expected_metrics_trust_receipt_sha256,
        label="terminal metrics trust receipt final replay",
    )
    final_metrics_trust = verify_metrics_trust_receipt(final_metrics_trust_receipt)
    final_metrics_report_binding = validate_generation_reports_against_metrics_trust(
        final_quality["generation_metric_reports"],
        verified_trust=final_metrics_trust,
    )
    if (
        final_metrics_trust_identity != metrics_trust_identity
        or final_metrics_trust_receipt != metrics_trust_receipt
        or final_metrics_trust != metrics_trust
        or final_metrics_report_binding != metrics_report_binding
    ):
        raise ValueError("terminal metrics trust evidence changed during replay")
    final_comparison = replay_comparison(
        terminal_guard_identity=terminal_guard_identity,
        terminal_guard=terminal_guard,
        comparison_identity=comparison_identity,
        comparison_report=comparison_report,
    )
    if final_comparison != comparison:
        raise ValueError("terminal comparison changed during completion replay")
    final_rendered_comparison = validate_comparison_renderings(
        comparison_report,
        status_bindings["comparison_outputs"],
    )
    if final_rendered_comparison != rendered_comparison:
        raise ValueError("terminal comparison renderings changed during replay")
    final_terminal_guard_identity, final_terminal_guard = _bound_json(
        paths["terminal_guard"],
        expected_sha256=args.expected_terminal_system_guard_sha256,
        label="terminal system claim guard final replay",
    )
    final_comparison_identity, final_comparison_report = _bound_json(
        paths["comparison"],
        expected_sha256=args.expected_comparison_sha256,
        label="quality bridge terminal comparison final replay",
    )
    final_terminal_status_identity, final_terminal_status = _replay_identity(
        file_identity(paths["terminal_status"]),
        label="terminal system guard waiter status final replay",
    )
    final_comparison_status_identity, final_comparison_status = _replay_identity(
        file_identity(paths["comparison_status"]),
        label="terminal comparison waiter status final replay",
    )
    final_status_bindings = validate_status_bindings(
        terminal_status_identity=final_terminal_status_identity,
        terminal_status=final_terminal_status,
        terminal_guard_identity=final_terminal_guard_identity,
        terminal_guard=final_terminal_guard,
        comparison_status_identity=final_comparison_status_identity,
        comparison_status=final_comparison_status,
        comparison_identity=final_comparison_identity,
    )
    if (
        final_terminal_guard_identity != terminal_guard_identity
        or final_terminal_guard != terminal_guard
        or final_comparison_identity != comparison_identity
        or final_comparison_report != comparison_report
        or final_terminal_status_identity != terminal_status_identity
        or final_terminal_status != terminal_status
        or final_comparison_status_identity != comparison_status_identity
        or final_comparison_status != comparison_status
        or final_status_bindings != status_bindings
    ):
        raise ValueError(
            "terminal claim/comparison source metadata changed during replay"
        )
    final_git = {
        "auditor": require_git_identity(
            args.project,
            revision=args.expected_revision,
            tree=args.expected_tree,
            branch=args.expected_branch,
            label="terminal completion audit project final replay",
        ),
        "training": require_git_identity(
            args.training_checkout,
            revision=args.expected_training_revision,
            tree=args.expected_training_tree,
            branch=args.expected_training_branch,
            label="quality bridge training checkout final replay",
        ),
        "physical_auditor": require_git_identity(
            args.physical_auditor_checkout,
            revision=args.expected_physical_auditor_revision,
            tree=args.expected_physical_auditor_tree,
            branch=args.expected_physical_auditor_branch,
            label="physical checkpoint auditor checkout final replay",
        ),
        "checkpoint_replay": require_git_identity(
            args.checkpoint_replay_checkout,
            revision=args.expected_checkpoint_replay_revision,
            tree=args.expected_checkpoint_replay_tree,
            branch=args.expected_checkpoint_replay_branch,
            label="checkpoint replay checkout final replay",
        ),
    }
    if (
        final_git["auditor"] != self_git
        or final_git["training"] != training_git
        or final_git["physical_auditor"] != auditor_git
        or final_git["checkpoint_replay"] != replay_git
        or file_identity(actual_self_source) != self_source_identity
    ):
        raise ValueError(
            "terminal completion Git or auditor source changed during replay"
        )

    claim = terminal_claim_outcome(
        status=status_bindings["terminal_status"],
        decision=status_bindings["terminal_decision"],
    )
    advantage = claim["generation_advantage_proven"]
    paired_kid_evaluated = terminal_guard["claim_policy"][
        "paired_kid_statistical_evidence_available"
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "status": "pass",
        "detail": "terminal_quality_bridge_evidence_physically_replayed",
        "terminal_status": claim["terminal_status"],
        "terminal_decision": claim["terminal_decision"],
        "generation_advantage_proven": advantage,
        "replication_scope": comparison["replication_scope"],
        "scope": {
            "quality_output_root": paths["root"].as_posix(),
            "dataset": "imagenet_256",
            "training_steps_per_method": EXPECTED_TRAINING_STEPS,
            "effective_batch_size": args.effective_batch,
            "training_images_per_method": EXPECTED_TRAINING_STEPS
            * args.effective_batch,
            "terminal_samples_per_method": 10_000,
            "training_git": training_git,
            "dataset_identity_sha256": args.expected_dataset_sha256,
            "runtime_environment_sha256": args.expected_runtime_sha256,
        },
        "auditor": {
            "git": self_git,
            "source": self_source_identity,
        },
        "sources": {
            "terminal_system_claim_guard": terminal_guard_identity,
            "terminal_system_waiter_status": status_bindings["terminal_waiter_status"],
            "quality_bridge_comparison": comparison_identity,
            "quality_bridge_comparison_markdown": rendered_comparison["markdown"],
            "quality_bridge_comparison_csv": rendered_comparison["csv"],
            "quality_bridge_comparison_waiter_status": status_bindings[
                "comparison_waiter_status"
            ],
            "quality_bridge_result": quality_identity,
            "metrics_trust_receipt": metrics_trust_identity,
            "dense_checkpoint_replay_waiter_status": checkpoint_chain[
                "dense_replay_waiter"
            ]["identity"],
        },
        "replay": {
            "quality_bridge_result": quality,
            "terminal_metrics_trust": {
                "receipt": metrics_trust,
                "generation_report_binding": metrics_report_binding,
                "physical_cache_and_dependencies_revalidated": True,
            },
            "strong_baseline_comparison": comparison,
            "checkpoint_integrity": checkpoint_chain,
            "final_checkpoint_source_revalidation": final_checkpoint_sources,
            "final_git_revalidation": final_git,
            "terminal_sampling": {
                method: {
                    "sampling_report": quality["physical_evidence"][method][
                        "sampling_report"
                    ],
                    "sampling_manifest": quality["physical_evidence"][method][
                        "sampling_manifest"
                    ],
                    "sampling_progress": quality["physical_evidence"][method][
                        "sampling_progress"
                    ],
                    "sample_set_sha256": quality["physical_evidence"][method][
                        "sample_set_sha256"
                    ],
                    "sample_count": quality["physical_evidence"][method][
                        "sample_count"
                    ],
                    "physical_png_set_revalidated": True,
                }
                for method in EXPECTED_METHODS
            },
            "class_fidelity": {
                "qualification": quality["source_reports"][
                    "class_fidelity_qualification"
                ],
                "cofitok": quality["source_reports"]["cofitok_class_fidelity"],
                "dense_identity": quality["source_reports"]["dense_class_fidelity"],
                "sample_pair_binding_revalidated": True,
                "classifier_integrity": class_fidelity_integrity,
                "physical_classifier_weight_revalidated_before_and_after": True,
            },
        },
        "claim_policy": {
            "terminal_system_evidence_complete": True,
            "matched_distribution_quality_claim_allowed": advantage,
            "absolute_usability_claim_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "independent_replication_claim_allowed": False,
            "multiple_independent_terminal_streams_claim_allowed": False,
            "replication_language_requires_distinct_bound_streams": True,
            "larger_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_or_release_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
        "limitations": [
            (
                "This audit proves physical and source-bound reproducibility only for "
                "the exact ImageNet-256 matched 100K quality bridge and its terminal "
                "10K sample pair."
            ),
            (
                "An operational pass preserves the terminal guard's pass or hold "
                "decision; it cannot convert a hold into an advantage claim."
            ),
            (
                (
                    "FID uses the exact bound terminal sample stream; paired "
                    "block-KID was not evaluated. This audit records zero "
                    "independent replications."
                )
                if not paired_kid_evaluated
                else (
                    "FID and paired block-KID reuse one exact bound terminal sample "
                    "stream. This audit records zero independent replications and "
                    "forbids a multiple-stream replication claim."
                )
            ),
            (
                "This artifact does not authorize training, sampling, 300K scaling, "
                "promotion, export, release, process signaling, broad superiority, or "
                "SOTA claims."
            ),
        ],
    }


def add_common_arguments(
    parser: argparse.ArgumentParser,
    *,
    include_dynamic_hashes: bool = True,
) -> None:
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--terminal-system-guard", type=Path, required=True)
    parser.add_argument(
        "--expected-terminal-dir-name",
        default="terminal_system_claim_guard_v1",
    )
    if include_dynamic_hashes:
        parser.add_argument("--expected-terminal-system-guard-sha256", required=True)
    parser.add_argument("--terminal-waiter-status", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument(
        "--expected-comparison-dir-name",
        default="quality_bridge_comparison_v1",
    )
    if include_dynamic_hashes:
        parser.add_argument("--expected-comparison-sha256", required=True)
    parser.add_argument("--comparison-waiter-status", type=Path, required=True)
    parser.add_argument("--checkpoint-audit-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-replay-waiter-status", type=Path, required=True)
    parser.add_argument("--metrics-trust-receipt", type=Path, required=True)
    parser.add_argument("--expected-metrics-trust-receipt-sha256", required=True)
    parser.add_argument("--physical-auditor-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint-replay-checkout", type=Path, required=True)
    parser.add_argument("--training-checkout", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-tree", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-physical-auditor-revision", required=True)
    parser.add_argument("--expected-physical-auditor-tree", required=True)
    parser.add_argument("--expected-physical-auditor-branch", default="")
    parser.add_argument("--expected-physical-auditor-source-sha256", required=True)
    parser.add_argument("--expected-checkpoint-replay-revision", required=True)
    parser.add_argument("--expected-checkpoint-replay-tree", required=True)
    parser.add_argument("--expected-checkpoint-replay-branch", required=True)
    parser.add_argument(
        "--expected-checkpoint-replay-verifier-source-sha256",
        required=True,
    )
    parser.add_argument("--expected-dataset-sha256", required=True)
    parser.add_argument("--expected-runtime-sha256", required=True)
    parser.add_argument("--effective-batch", type=int, default=EXPECTED_EFFECTIVE_BATCH)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Physically replay and bind all terminal 100K quality-bridge training, "
            "checkpoint, sampling, class-fidelity, claim-guard, and comparison evidence."
        )
    )
    add_common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.effective_batch != EXPECTED_EFFECTIVE_BATCH:
        raise ValueError("terminal completion audit effective batch must be 64")
    output = reject_symlink_chain(
        args.output,
        name="terminal completion audit output",
    ).resolve()
    if output.exists():
        raise FileExistsError(f"terminal completion audit already exists: {output}")
    report = build_audit(args)
    write_json_report(output, report)
    print(
        json.dumps(
            {"status": report["status"], "terminal_status": report["terminal_status"]}
        )
    )


if __name__ == "__main__":
    main()
