from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_runtime_claim_guard_strict_comparison"
GUARD_SCHEMA_VERSION = 1
GUARD_ROLE = "generation_runtime_compute_claim_guard"
WAITER_SCHEMA_VERSION = 1
WAITER_ROLE = "generation_quality_bridge_runtime_claim_guard_waiter"
WRAPPER_SCHEMA_VERSION = 1
WRAPPER_ROLE = "generation_runtime_claim_guard_strict_replay_wrapper_deployment"
RECOVERY_SCHEMA_VERSION = 1
RECOVERY_ROLE = "generation_runtime_claim_guard_strict_replay_recovery"
FAILURE_RECORD_ROLE = "generation_runtime_claim_guard_strict_replay_wrapper_failure_record"
EXPECTED_METHODS = ("cofitok", "dense_identity")
DIRECT_POLICY_FIELDS = (
    "training_wall_clock_direct_comparison_allowed",
    "training_throughput_direct_comparison_allowed",
    "cost_efficiency_ranking_allowed",
)
NON_AUTHORIZING_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "sampling_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
    "quality_claim_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "full_300k_launch_allowed": False,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


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


def _validate_guard(report: Mapping[str, Any], *, label: str) -> None:
    if (
        report.get("schema_version") != GUARD_SCHEMA_VERSION
        or report.get("role") != GUARD_ROLE
        or report.get("status") != "pass"
        or report.get("decision")
        not in {
            "runtime_cost_claims_observational_only",
            "direct_runtime_outcome_comparison_allowed",
        }
        or not isinstance(report.get("sources"), Mapping)
        or not isinstance(report.get("matched_training_contract"), Mapping)
        or not isinstance(report.get("metric_roles"), Mapping)
        or not isinstance(report.get("gpu_contention"), Mapping)
        or not isinstance(report.get("claim_policy"), Mapping)
        or not isinstance(report.get("claim_boundary"), Mapping)
    ):
        raise ValueError(f"{label} runtime claim guard contract differs")


def _validate_waiter_status(
    report: Mapping[str, Any],
    *,
    guard_identity: Mapping[str, Any],
    guard_report: Mapping[str, Any],
    expected_control_revision: str,
    expected_training_revision: str,
    expected_training_branch: str,
    label: str,
) -> dict[str, Any]:
    expected = report.get("expected")
    control = expected.get("control_git") if isinstance(expected, Mapping) else None
    pair = expected.get("pair_monitor") if isinstance(expected, Mapping) else None
    guard = report.get("runtime_claim_guard")
    if (
        report.get("schema_version") != WAITER_SCHEMA_VERSION
        or report.get("role") != WAITER_ROLE
        or report.get("status") != "pass"
        or report.get("phase") != "completed"
        or not isinstance(report.get("pid"), int)
        or not isinstance(control, Mapping)
        or control.get("revision") != expected_control_revision
        or control.get("tracked_dirty") is not False
        or not isinstance(pair, Mapping)
        or pair.get("training_revision") != expected_training_revision
        or pair.get("training_branch") != expected_training_branch
        or not isinstance(guard, Mapping)
        or guard.get("identity") != dict(guard_identity)
        or guard.get("status") != "pass"
        or guard.get("decision") != guard_report.get("decision")
    ):
        raise ValueError(f"{label} runtime claim waiter contract differs")
    return {
        "pid": int(report["pid"]),
        "control_git": copy.deepcopy(dict(control)),
        "source_waiter": copy.deepcopy(dict(expected.get("source_waiter", {}))),
        "deployment_receipt": copy.deepcopy(report.get("deployment_receipt")),
    }


def _validate_wrapper(
    report: Mapping[str, Any],
    *,
    canonical_waiter: Mapping[str, Any],
    strict_waiter: Mapping[str, Any],
    expected_strict_control_revision: str,
    strict_status_path: Path | None = None,
    strict_guard_path: Path | None = None,
    require_recovery_binding: bool = False,
) -> None:
    scope = report.get("scope")
    strict_control = report.get("strict_control")
    wrapper = report.get("wrapper")
    canonical = report.get("old_canonical_waiter")
    source = report.get("runtime_fairness_source_waiter")
    behavior = report.get("behavior")
    if (
        report.get("schema_version") != WRAPPER_SCHEMA_VERSION
        or report.get("role") != WRAPPER_ROLE
        or report.get("status") != "pass"
        or not isinstance(scope, Mapping)
        or scope.get("cpu_only") is not True
        or scope.get("non_authorizing") is not True
        or scope.get("gpu_execution_allowed") is not False
        or scope.get("training_process_signals_allowed") is not False
        or scope.get("unrelated_process_signals_allowed") is not False
        or scope.get("old_waiter_signals_allowed") is not False
        or scope.get("promotion_authorization_allowed") is not False
        or scope.get("release_authorization_allowed") is not False
        or scope.get("full_300k_launch_allowed") is not False
        or not isinstance(strict_control, Mapping)
        or any(
            strict_control.get(key) != strict_waiter["control_git"].get(key)
            for key in ("revision", "tree", "branch", "tracked_dirty")
        )
        or strict_control.get("revision") != expected_strict_control_revision
        or not isinstance(wrapper, Mapping)
        or int(wrapper.get("pid", -1)) != int(strict_waiter["pid"])
        or int(wrapper.get("ppid", -1)) != 1
        or wrapper.get("cuda_visible_devices") != ""
        or wrapper.get("omp_num_threads") != "1"
        or wrapper.get("mkl_num_threads") != "1"
        or wrapper.get("ionice") != "idle"
        or not isinstance(canonical, Mapping)
        or int(canonical.get("pid", -1)) != int(canonical_waiter["pid"])
        or not isinstance(source, Mapping)
        or int(source.get("pid", -1))
        != int(strict_waiter["source_waiter"].get("pid", -2))
        or not isinstance(behavior, Mapping)
        or behavior.get("waits_for_exact_old_pid_start_ticks_and_cmdline") is not True
        or behavior.get("executes_only_after_old_waiter_exits_or_identity_changes")
        is not True
        or behavior.get("writes_independent_output") is not True
        or behavior.get("replaces_canonical_guard") is not False
        or behavior.get("comparison_required_before_runtime_claim_trust") is not True
    ):
        raise ValueError("strict replay wrapper deployment contract differs")
    recovery = report.get("recovery")
    if require_recovery_binding:
        if not isinstance(recovery, Mapping):
            raise ValueError("strict replay recovery binding is missing")
        _validate_recovery_binding(
            report,
            recovery=recovery,
            strict_status_path=strict_status_path,
            strict_guard_path=strict_guard_path,
        )


def _validate_recovery_identity(
    report: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    path = report.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError(f"strict replay recovery {label} path is missing")
    identity = file_identity(
        reject_symlink_chain(Path(path), name=f"strict replay recovery {label}")
    )
    if identity != dict(report):
        raise ValueError(f"strict replay recovery {label} identity differs")
    return identity


def _validate_recovery_binding(
    wrapper_report: Mapping[str, Any],
    *,
    recovery: Mapping[str, Any],
    strict_status_path: Path | None,
    strict_guard_path: Path | None,
) -> None:
    targets = wrapper_report.get("targets")
    wrapper = wrapper_report.get("wrapper")
    failed_targets = recovery.get("failed_targets_absent_before_recovery")
    if (
        recovery.get("schema_version") != RECOVERY_SCHEMA_VERSION
        or recovery.get("role") != RECOVERY_ROLE
        or recovery.get("reason")
        != "original_wrapper_missing_pythonpath_import_failure"
        or recovery.get("new_output_version")
        != "runtime_compute_claim_guard_strict_replay_v3"
        or recovery.get("old_outputs_overwritten") is not False
        or recovery.get("old_process_signaled") is not False
        or not isinstance(targets, Mapping)
        or not isinstance(wrapper, Mapping)
        or wrapper.get("nice") != 10
        or not isinstance(wrapper.get("start_ticks"), int)
        or not _is_sha256(wrapper.get("cmdline_sha256"))
        or not isinstance(failed_targets, Mapping)
    ):
        raise ValueError("strict replay recovery contract differs")
    if strict_status_path is None or strict_guard_path is None:
        raise ValueError("strict replay recovery target paths are missing")
    if (
        targets.get("status_output") != strict_status_path.resolve().as_posix()
        or targets.get("guard_output") != strict_guard_path.resolve().as_posix()
    ):
        raise ValueError("strict replay recovery targets differ")

    original_identity = _validate_recovery_identity(
        recovery.get("original_wrapper_deployment", {}),
        label="original wrapper deployment",
    )
    original = read_json_object(
        Path(original_identity["path"]),
        name="strict replay original wrapper deployment",
    )
    if (
        original.get("schema_version") != WRAPPER_SCHEMA_VERSION
        or original.get("role") != WRAPPER_ROLE
        or original.get("status") != "pass"
    ):
        raise ValueError("strict replay original wrapper contract differs")
    log_identity = _validate_recovery_identity(
        recovery.get("original_failure_log", {}),
        label="original failure log",
    )
    failure_text = Path(log_identity["path"]).read_text(
        encoding="utf-8",
        errors="replace",
    )
    if (
        "ModuleNotFoundError: No module named 'scripts'" not in failure_text
        or "run_generation_quality_bridge_runtime_claim_guard_waiter.py"
        not in failure_text
    ):
        raise ValueError("strict replay original failure reason differs")
    for field, label in (
        ("recovery_source", "recovery source"),
        ("strict_runner_source", "strict runner source"),
        ("strict_builder_source", "strict builder source"),
        ("canonical_status", "canonical status"),
    ):
        _validate_recovery_identity(recovery.get(field, {}), label=label)
    failure_status_identity = _validate_recovery_identity(
        recovery.get("prior_failure_status", {}),
        label="prior failure status",
    )
    failure_status = read_json_object(
        Path(failure_status_identity["path"]),
        name="strict replay prior failure status",
    )
    if (
        failure_status.get("schema_version") != 1
        or failure_status.get("role") != FAILURE_RECORD_ROLE
        or failure_status.get("status") != "failed"
        or failure_status.get("detail")
        != "original_wrapper_missing_pythonpath_import_failure"
        or failure_status.get("original_wrapper_deployment") != original_identity
        or failure_status.get("original_failure_log") != log_identity
        or failure_status.get("scope") != {
            "cpu_only": True,
            "non_authorizing": True,
            "gpu_execution_allowed": False,
            "training_process_signals_allowed": False,
            "unrelated_process_signals_allowed": False,
            "old_waiter_signals_allowed": False,
            "promotion_authorization_allowed": False,
            "release_authorization_allowed": False,
            "full_300k_launch_allowed": False,
        }
    ):
        raise ValueError("strict replay prior failure status contract differs")
    if not _is_sha256(recovery.get("strict_runner_argv_sha256")):
        raise ValueError("strict replay recovered runner argv identity differs")
    for field in ("status_output", "guard_output"):
        state = failed_targets.get(field)
        if (
            not isinstance(state, Mapping)
            or state.get("absent") is not True
            or not isinstance(state.get("path"), str)
            or state.get("path") in {
                targets.get("status_output"),
                targets.get("guard_output"),
            }
        ):
            raise ValueError("strict replay failed target recovery state differs")
    if (
        failed_targets["status_output"]["path"]
        != failure_status_identity["path"]
        or Path(str(failed_targets["guard_output"]["path"])).exists()
        or Path(str(failed_targets["guard_output"]["path"])).is_symlink()
    ):
        raise ValueError("strict replay failed target post-recovery state differs")


def _strict_policy_verified(report: Mapping[str, Any]) -> dict[str, Any]:
    policy = report["claim_policy"]
    elapsed = report["metric_roles"].get("adjusted_training_elapsed_and_throughput")
    if not isinstance(elapsed, Mapping):
        raise ValueError("strict guard elapsed metric role is missing")
    lower_bound_methods = elapsed.get("physical_lower_bound_methods")
    method_evidence = elapsed.get("methods")
    if (
        report.get("decision") != "runtime_cost_claims_observational_only"
        or set(lower_bound_methods or []) != set(EXPECTED_METHODS)
        or not isinstance(method_evidence, Mapping)
        or set(method_evidence) != set(EXPECTED_METHODS)
        or elapsed.get("role") != "observational_physical_lower_bounds_only"
        or policy.get("exclusive_gpu_observation_coverage_verified") is not False
        or policy.get("recovery_adjusted_elapsed_exact_for_both_methods") is not False
        or policy.get("physical_lower_bound_label_required") is not True
        or policy.get("observational_only_label_required") is not True
        or any(policy.get(field) is not False for field in DIRECT_POLICY_FIELDS)
        or policy.get("equal_wall_clock_budget_claim_allowed") is not False
        or policy.get("equal_gpu_hours_budget_claim_allowed") is not False
        or policy.get("equal_training_flops_budget_claim_allowed") is not False
        or policy.get("quality_or_generation_advantage_claim_allowed") is not False
    ):
        raise ValueError("strict runtime claim policy is not fail-closed")
    for method in EXPECTED_METHODS:
        evidence = method_evidence[method]
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("elapsed_seconds_role")
            != "physical_lower_bound_including_orphaned_recovery_compute"
        ):
            raise ValueError(f"strict runtime lower-bound role differs: {method}")
    return {
        "decision": report["decision"],
        "physical_lower_bound_methods": list(lower_bound_methods),
        "elapsed_metric_role": elapsed["role"],
        "direct_ranking_allowed": False,
    }


def _canonical_semantic_equivalence(
    canonical: Mapping[str, Any],
    strict: Mapping[str, Any],
) -> dict[str, Any]:
    canonical_policy = canonical["claim_policy"]
    canonical_elapsed = canonical["metric_roles"].get(
        "adjusted_training_elapsed_and_throughput"
    )
    strict_elapsed = strict["metric_roles"].get(
        "adjusted_training_elapsed_and_throughput"
    )
    checks = {
        "same_sources": canonical["sources"] == strict["sources"],
        "same_matched_training_contract": (
            canonical["matched_training_contract"]
            == strict["matched_training_contract"]
        ),
        "same_gpu_contention_evidence": (
            canonical["gpu_contention"] == strict["gpu_contention"]
        ),
        "same_method_cost_evidence": (
            isinstance(canonical_elapsed, Mapping)
            and isinstance(strict_elapsed, Mapping)
            and canonical_elapsed.get("methods") == strict_elapsed.get("methods")
            and canonical_elapsed.get("descriptive_comparison")
            == strict_elapsed.get("descriptive_comparison")
        ),
        "canonical_observational_decision": (
            canonical.get("decision") == "runtime_cost_claims_observational_only"
        ),
        "canonical_observational_metric_role": (
            isinstance(canonical_elapsed, Mapping)
            and canonical_elapsed.get("role")
            == "observational_physical_lower_bounds_only"
        ),
        "canonical_direct_ranking_disabled": all(
            canonical_policy.get(field) is False for field in DIRECT_POLICY_FIELDS
        ),
        "canonical_equal_budget_claims_disabled": all(
            canonical_policy.get(field) is False
            for field in (
                "equal_wall_clock_budget_claim_allowed",
                "equal_gpu_hours_budget_claim_allowed",
                "equal_training_flops_budget_claim_allowed",
            )
        ),
        "canonical_quality_claim_disabled": (
            canonical_policy.get("quality_or_generation_advantage_claim_allowed")
            is False
        ),
    }
    return {
        "checks": checks,
        "semantically_equivalent_for_observational_runtime_claims": all(
            checks.values()
        ),
        "byte_equivalence_required": False,
        "strict_companion_required": True,
    }


def build_comparison(
    *,
    canonical_status_path: Path,
    expected_canonical_status_sha256: str,
    canonical_guard_path: Path,
    expected_canonical_guard_sha256: str,
    strict_status_path: Path,
    expected_strict_status_sha256: str,
    strict_guard_path: Path,
    expected_strict_guard_sha256: str,
    wrapper_receipt_path: Path,
    expected_wrapper_receipt_sha256: str,
    expected_canonical_control_revision: str,
    expected_strict_control_revision: str,
    expected_training_revision: str,
    expected_training_branch: str,
    require_recovery_binding: bool = False,
) -> dict[str, Any]:
    canonical_guard_identity, canonical_guard = _bound_json(
        canonical_guard_path,
        expected_sha256=expected_canonical_guard_sha256,
        label="canonical runtime claim guard",
    )
    strict_guard_identity, strict_guard = _bound_json(
        strict_guard_path,
        expected_sha256=expected_strict_guard_sha256,
        label="strict runtime claim guard",
    )
    canonical_status_identity, canonical_status = _bound_json(
        canonical_status_path,
        expected_sha256=expected_canonical_status_sha256,
        label="canonical runtime claim waiter status",
    )
    strict_status_identity, strict_status = _bound_json(
        strict_status_path,
        expected_sha256=expected_strict_status_sha256,
        label="strict runtime claim waiter status",
    )
    wrapper_identity, wrapper = _bound_json(
        wrapper_receipt_path,
        expected_sha256=expected_wrapper_receipt_sha256,
        label="strict replay wrapper deployment receipt",
    )
    _validate_guard(canonical_guard, label="canonical")
    _validate_guard(strict_guard, label="strict")
    canonical_waiter = _validate_waiter_status(
        canonical_status,
        guard_identity=canonical_guard_identity,
        guard_report=canonical_guard,
        expected_control_revision=expected_canonical_control_revision,
        expected_training_revision=expected_training_revision,
        expected_training_branch=expected_training_branch,
        label="canonical",
    )
    strict_waiter = _validate_waiter_status(
        strict_status,
        guard_identity=strict_guard_identity,
        guard_report=strict_guard,
        expected_control_revision=expected_strict_control_revision,
        expected_training_revision=expected_training_revision,
        expected_training_branch=expected_training_branch,
        label="strict",
    )
    _validate_wrapper(
        wrapper,
        canonical_waiter=canonical_waiter,
        strict_waiter=strict_waiter,
        expected_strict_control_revision=expected_strict_control_revision,
        strict_status_path=strict_status_path,
        strict_guard_path=strict_guard_path,
        require_recovery_binding=require_recovery_binding,
    )
    strict_policy = _strict_policy_verified(strict_guard)
    equivalence = _canonical_semantic_equivalence(canonical_guard, strict_guard)
    canonical_trusted = equivalence[
        "semantically_equivalent_for_observational_runtime_claims"
    ]
    decision = (
        "canonical_observational_runtime_claim_semantically_verified"
        if canonical_trusted
        else "canonical_runtime_claim_rejected_strict_guard_controls"
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "pass",
        "decision": decision,
        "sources": {
            "canonical_waiter_status": canonical_status_identity,
            "canonical_runtime_claim_guard": canonical_guard_identity,
            "strict_waiter_status": strict_status_identity,
            "strict_runtime_claim_guard": strict_guard_identity,
            "strict_replay_wrapper_deployment": wrapper_identity,
        },
        "training_identity": {
            "revision": expected_training_revision,
            "branch": expected_training_branch,
        },
        "control_revisions": {
            "canonical": expected_canonical_control_revision,
            "strict": expected_strict_control_revision,
        },
        "strict_policy": strict_policy,
        "canonical_equivalence": equivalence,
        "claim_policy": {
            "canonical_runtime_claim_trusted": canonical_trusted,
            "strict_runtime_claim_guard_required": True,
            "runtime_configuration_parity_claim_allowed": True,
            "adjusted_elapsed_point_estimate_reporting_allowed": True,
            "physical_lower_bound_label_required": True,
            "observational_only_label_required": True,
            "training_wall_clock_direct_comparison_allowed": False,
            "training_throughput_direct_comparison_allowed": False,
            "cost_efficiency_ranking_allowed": False,
            "equal_wall_clock_budget_claim_allowed": False,
            "equal_gpu_hours_budget_claim_allowed": False,
            "equal_training_flops_budget_claim_allowed": False,
            "quality_or_generation_advantage_claim_allowed": False,
            "strict_recovery_binding_required": require_recovery_binding,
            "strict_recovery_binding_verified": require_recovery_binding,
        },
        "claim_boundary": copy.deepcopy(NON_AUTHORIZING_BOUNDARY),
        "limitations": [
            (
                "This comparison verifies only the runtime-claim boundary. It "
                "does not establish sample quality, mechanism advantage, broad "
                "generation superiority, or authorization for another run."
            ),
            (
                "Recovery-adjusted elapsed values remain physical lower bounds; "
                "wall-clock, throughput, and cost-efficiency ranking is disabled."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the canonical and independently replayed strict runtime "
            "claim guards without authorizing training, sampling, or release."
        )
    )
    for name in (
        "canonical-status",
        "canonical-guard",
        "strict-status",
        "strict-guard",
        "wrapper-receipt",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    parser.add_argument("--expected-canonical-control-revision", required=True)
    parser.add_argument("--expected-strict-control-revision", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--require-recovery-binding", action="store_true")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="runtime claim guard comparison output root",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="runtime claim guard comparison output",
    ).resolve()
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("runtime claim comparison output is outside output root") from error
    report = build_comparison(
        canonical_status_path=args.canonical_status,
        expected_canonical_status_sha256=args.expected_canonical_status_sha256,
        canonical_guard_path=args.canonical_guard,
        expected_canonical_guard_sha256=args.expected_canonical_guard_sha256,
        strict_status_path=args.strict_status,
        expected_strict_status_sha256=args.expected_strict_status_sha256,
        strict_guard_path=args.strict_guard,
        expected_strict_guard_sha256=args.expected_strict_guard_sha256,
        wrapper_receipt_path=args.wrapper_receipt,
        expected_wrapper_receipt_sha256=args.expected_wrapper_receipt_sha256,
        expected_canonical_control_revision=args.expected_canonical_control_revision,
        expected_strict_control_revision=args.expected_strict_control_revision,
        expected_training_revision=args.expected_training_revision,
        expected_training_branch=args.expected_training_branch,
        require_recovery_binding=args.require_recovery_binding,
    )
    identity = prepare_manifest(
        output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": report["decision"],
                "comparison": identity,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
