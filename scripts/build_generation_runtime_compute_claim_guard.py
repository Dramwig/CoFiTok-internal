from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.gpu_contention import validate_gpu_contention_evidence
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_runtime_compute_claim_guard"
FAIRNESS_SCHEMA_VERSION = 1
FAIRNESS_ROLE = "quality_bridge_runtime_compute_fairness_audit"
EXPECTED_METHODS = ("cofitok", "dense_identity")
CLAIM_BOUNDARY = {
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


def _relative_change(value: float, baseline: float) -> float:
    if baseline <= 0.0:
        raise ValueError("runtime comparison baseline must be positive")
    return (value - baseline) / baseline


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


def _validate_cost(
    method: str,
    evidence: Any,
    *,
    runtime: Mapping[str, Any],
    expected_samples: int,
) -> dict[str, Any]:
    if not isinstance(evidence, Mapping) or evidence.get("status") != "verified":
        raise ValueError(f"{method} runtime fairness evidence is invalid")
    cost = evidence.get("training_cost")
    if not isinstance(cost, Mapping) or cost.get("valid") is not True:
        raise ValueError(f"{method} training cost is invalid")
    expected_runtime = {
        "micro_batch_size": int(runtime["micro_batch_size"]),
        "gradient_accumulation_steps": int(
            runtime["gradient_accumulation_steps"]
        ),
        "effective_batch_size": int(runtime["effective_batch_size"]),
    }
    for key, expected in expected_runtime.items():
        if int(cost.get(key, -1)) != expected:
            raise ValueError(f"{method} {key} differs from the runtime contract")
    if (
        int(cost.get("samples_seen", -1)) != expected_samples
        or int(cost.get("expected_samples_seen", -1)) != expected_samples
    ):
        raise ValueError(f"{method} training images differ from the runtime contract")
    reported = _finite_positive(
        cost.get("reported_elapsed_seconds"),
        label=f"{method} reported elapsed seconds",
    )
    elapsed = _finite_positive(
        cost.get("elapsed_seconds"),
        label=f"{method} adjusted elapsed seconds",
    )
    throughput = _finite_positive(
        cost.get("images_per_second"),
        label=f"{method} images per second",
    )
    peak_vram = int(cost.get("peak_vram_bytes", -1))
    adjustment = cost.get("resume_compute_adjustment")
    if (
        not isinstance(adjustment, Mapping)
        or adjustment.get("valid") is not True
        or peak_vram <= 0
        or not _same_float(throughput, expected_samples / elapsed)
    ):
        raise ValueError(f"{method} adjusted physical cost is inconsistent")
    adjustment_seconds = _finite(
        adjustment.get("seconds"),
        label=f"{method} recovery adjustment seconds",
    )
    if adjustment_seconds < 0.0 or not _same_float(
        elapsed, reported + adjustment_seconds
    ):
        raise ValueError(f"{method} recovery-adjusted elapsed time differs")
    expected_role = (
        "physical_lower_bound_including_orphaned_recovery_compute"
        if adjustment.get("applied") is True
        else "reported_training_elapsed_seconds"
    )
    if cost.get("elapsed_seconds_role") != expected_role:
        raise ValueError(f"{method} elapsed-time role differs")
    return copy.deepcopy(dict(cost))


def _validate_fairness_report(report: Mapping[str, Any]) -> dict[str, Any]:
    if (
        report.get("schema_version") != FAIRNESS_SCHEMA_VERSION
        or report.get("role") != FAIRNESS_ROLE
        or report.get("status") != "pass"
    ):
        raise ValueError("runtime fairness report contract differs")
    scope = report.get("scope")
    if (
        not isinstance(scope, Mapping)
        or scope.get("cpu_only") is not True
        or scope.get("gpu_execution_allowed") is not False
        or scope.get("training_process_signals_allowed") is not False
        or scope.get("unrelated_process_signals_allowed") is not False
        or scope.get("promotion_authorization_allowed") is not False
        or scope.get("release_authorization_allowed") is not False
        or scope.get("full_300k_launch_allowed") is not False
    ):
        raise ValueError("runtime fairness report scope differs")
    contract = report.get("contract")
    if not isinstance(contract, Mapping) or contract.get("status") != "verified":
        raise ValueError("runtime fairness static contract is invalid")
    training_git = contract.get("training_git")
    runtime = contract.get("runtime")
    parameters = contract.get("parameters")
    run_dirs = contract.get("training_run_dirs")
    if (
        not isinstance(training_git, Mapping)
        or training_git.get("tracked_dirty") is not False
        or not isinstance(runtime, Mapping)
        or not isinstance(parameters, Mapping)
        or not isinstance(run_dirs, list)
        or len(run_dirs) != 2
        or contract.get("dataset") != "imagenet_256"
    ):
        raise ValueError("runtime fairness static contract identity differs")
    target_steps = int(contract.get("target_steps_per_method", -1))
    micro_batch = int(runtime.get("micro_batch_size", -1))
    accumulation = int(runtime.get("gradient_accumulation_steps", -1))
    effective_batch = int(runtime.get("effective_batch_size", -1))
    if (
        target_steps < 1
        or micro_batch < 1
        or accumulation < 1
        or effective_batch != micro_batch * accumulation
        or not _is_sha256(runtime.get("runtime_environment_sha256"))
    ):
        raise ValueError("runtime fairness resolved runtime is invalid")
    cofitok_parameters = int(parameters.get("cofitok", -1))
    dense_parameters = int(parameters.get("dense_identity", -1))
    relative_gap = _finite(
        parameters.get("relative_gap"), label="runtime fairness parameter gap"
    )
    if (
        cofitok_parameters < 1
        or dense_parameters < 1
        or abs(relative_gap) > 0.02
        or not _same_float(
            relative_gap,
            (cofitok_parameters - dense_parameters) / dense_parameters,
        )
    ):
        raise ValueError("runtime fairness parameter evidence differs")
    observed = report.get("observed_runtime_parity")
    if (
        not isinstance(observed, Mapping)
        or observed.get("status") != "proven"
        or int(observed.get("micro_batch_size", -1)) != micro_batch
        or int(observed.get("gradient_accumulation_steps", -1))
        != accumulation
        or int(observed.get("effective_batch_size", -1)) != effective_batch
        or int(observed.get("canonical_images_per_method", -1))
        != target_steps * effective_batch
    ):
        raise ValueError("runtime fairness observed configuration parity differs")
    methods = report.get("methods")
    if not isinstance(methods, Mapping) or set(methods) != set(EXPECTED_METHODS):
        raise ValueError("runtime fairness methods differ")
    expected_samples = target_steps * effective_batch
    costs = {
        method: _validate_cost(
            method,
            methods[method],
            runtime=runtime,
            expected_samples=expected_samples,
        )
        for method in EXPECTED_METHODS
    }
    cofitok_cost = costs["cofitok"]
    dense_cost = costs["dense_identity"]
    descriptive = report.get("descriptive_comparison")
    expected_descriptive = {
        "role": "measured_outcomes_not_predeclared_advantage",
        "cofitok_minus_dense_adjusted_elapsed_seconds": (
            cofitok_cost["elapsed_seconds"] - dense_cost["elapsed_seconds"]
        ),
        "cofitok_adjusted_elapsed_relative_change": _relative_change(
            cofitok_cost["elapsed_seconds"], dense_cost["elapsed_seconds"]
        ),
        "cofitok_throughput_relative_change": _relative_change(
            cofitok_cost["images_per_second"], dense_cost["images_per_second"]
        ),
        "cofitok_peak_vram_relative_change": _relative_change(
            float(cofitok_cost["peak_vram_bytes"]),
            float(dense_cost["peak_vram_bytes"]),
        ),
    }
    if not isinstance(descriptive, Mapping):
        raise ValueError("runtime fairness descriptive comparison is missing")
    for key, expected in expected_descriptive.items():
        actual = descriptive.get(key)
        if actual != expected and (
            not isinstance(expected, float) or not _same_float(actual, expected)
        ):
            raise ValueError(f"runtime fairness descriptive field differs: {key}")
    boundary = report.get("claim_boundary")
    required_boundary = {
        "matched_runtime_contract_established": True,
        "observed_pair_runtime_parity_established": True,
        "physical_recovery_compute_included": True,
        "training_speed_advantage_predeclared": False,
        "memory_advantage_predeclared": False,
        "sample_quality_established": False,
        "promotion_authorization_allowed": False,
        "release_authorization_allowed": False,
        "full_300k_launch_allowed": False,
    }
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not expected
        for key, expected in required_boundary.items()
    ):
        raise ValueError("runtime fairness claim boundary differs")
    return {
        "training_git": copy.deepcopy(dict(training_git)),
        "dataset": contract["dataset"],
        "output_root": str(contract.get("output_root", "")),
        "run_dirs": [str(path) for path in run_dirs],
        "target_steps": target_steps,
        "runtime": copy.deepcopy(dict(runtime)),
        "parameters": copy.deepcopy(dict(parameters)),
        "costs": costs,
        "descriptive_comparison": copy.deepcopy(dict(descriptive)),
    }


def _validate_pair_monitor(
    report: Mapping[str, Any],
    *,
    fairness: Mapping[str, Any],
    expected_monitor_name: str,
) -> dict[str, Any]:
    if (
        report.get("schema_version") != 2
        or report.get("monitor") != expected_monitor_name
        or report.get("status") != "pass"
        or report.get("stage") != "complete"
        or report.get("git") != fairness["training_git"]
    ):
        raise ValueError("terminal pair monitor identity differs")
    runs = report.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != set(EXPECTED_METHODS):
        raise ValueError("terminal pair monitor runs differ")
    for index, method in enumerate(EXPECTED_METHODS):
        run = runs[method]
        if (
            not isinstance(run, Mapping)
            or run.get("run_dir") != fairness["run_dirs"][index]
            or run.get("complete") is not True
            or int(run.get("last_step", -1)) != int(fairness["target_steps"])
        ):
            raise ValueError(f"terminal pair monitor run differs: {method}")
    return validate_gpu_contention_evidence(dict(report))


def build_guard(
    *,
    runtime_fairness_report_path: Path,
    expected_runtime_fairness_sha256: str,
    pair_monitor_path: Path,
    expected_pair_monitor_sha256: str,
    expected_monitor_name: str,
) -> dict[str, Any]:
    fairness_identity, fairness_report = _bound_json(
        runtime_fairness_report_path,
        expected_sha256=expected_runtime_fairness_sha256,
        label="runtime compute fairness report",
    )
    pair_identity, pair_report = _bound_json(
        pair_monitor_path,
        expected_sha256=expected_pair_monitor_sha256,
        label="terminal pair monitor",
    )
    fairness = _validate_fairness_report(fairness_report)
    contention = _validate_pair_monitor(
        pair_report,
        fairness=fairness,
        expected_monitor_name=expected_monitor_name,
    )
    direct = contention["direct_comparison_allowed"] is True
    decision = (
        "direct_runtime_outcome_comparison_allowed"
        if direct
        else "runtime_cost_claims_observational_only"
    )
    claim_text = (
        "The matched pair used the same resolved runtime configuration, steps, "
        "and training images, and complete exclusive GPU observation coverage "
        "permits a descriptive direct comparison of recovery-adjusted training "
        "wall-clock and throughput. This remains a measured outcome rather than "
        "an equal wall-clock, GPU-hour, or FLOP training budget."
        if direct
        else (
            "The matched pair used the same resolved runtime configuration, "
            "steps, and training images, and recovery-adjusted physical elapsed "
            "time was accounted for. GPU observation coverage is insufficient "
            "for direct wall-clock, throughput, or cost-efficiency ranking, so "
            "those values are observational physical lower bounds only."
        )
    )
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "pass",
        "decision": decision,
        "sources": {
            "runtime_compute_fairness": fairness_identity,
            "terminal_pair_monitor": pair_identity,
        },
        "matched_training_contract": {
            "status": "verified",
            "training_git": fairness["training_git"],
            "dataset": fairness["dataset"],
            "target_steps_per_method": fairness["target_steps"],
            "runtime": fairness["runtime"],
            "parameters": fairness["parameters"],
            "same_runtime_configuration": True,
            "same_optimizer_steps": True,
            "same_training_images": True,
            "physical_recovery_compute_included": True,
        },
        "metric_roles": {
            "adjusted_training_elapsed_and_throughput": {
                "role": (
                    "descriptive_direct_measured_outcomes"
                    if direct
                    else "observational_physical_lower_bounds_only"
                ),
                "methods": {
                    method: {
                        "elapsed_seconds": fairness["costs"][method][
                            "elapsed_seconds"
                        ],
                        "elapsed_seconds_role": fairness["costs"][method][
                            "elapsed_seconds_role"
                        ],
                        "images_per_second": fairness["costs"][method][
                            "images_per_second"
                        ],
                    }
                    for method in EXPECTED_METHODS
                },
                "descriptive_comparison": fairness["descriptive_comparison"],
            },
            "peak_vram": {
                "role": "descriptive_point_estimate_not_predeclared_advantage",
                "methods": {
                    method: fairness["costs"][method]["peak_vram_bytes"]
                    for method in EXPECTED_METHODS
                },
            },
        },
        "gpu_contention": contention,
        "claim_policy": {
            "runtime_configuration_parity_claim_allowed": True,
            "physical_recovery_compute_accounting_claim_allowed": True,
            "adjusted_elapsed_point_estimate_reporting_allowed": True,
            "training_wall_clock_direct_comparison_allowed": direct,
            "training_throughput_direct_comparison_allowed": direct,
            "cost_efficiency_ranking_allowed": direct,
            "observed_pair_runtime_parity_claim_allowed": False,
            "equal_wall_clock_budget_claim_allowed": False,
            "equal_gpu_hours_budget_claim_allowed": False,
            "equal_training_flops_budget_claim_allowed": False,
            "training_speed_advantage_predeclared": False,
            "peak_vram_point_estimate_reporting_allowed": True,
            "peak_vram_advantage_claim_allowed": False,
            "quality_or_generation_advantage_claim_allowed": False,
        },
        "claim_text": claim_text,
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "limitations": [
            (
                "The source field observed_pair_runtime_parity_established is "
                "interpreted only as equality of the resolved runtime "
                "configuration and canonical training budget; it is not evidence "
                "of directly comparable elapsed time."
            ),
            (
                "Direct wall-clock and throughput comparison requires complete, "
                "continuous GPU observation coverage with no unrelated GPU "
                "compute, as verified from the terminal pair monitor."
            ),
            (
                "Equal optimizer steps and images do not establish an equal "
                "wall-clock, GPU-hour, or FLOP budget."
            ),
            (
                "This artifact does not authorize training, sampling, export, "
                "release, process signaling, generation-quality claims, or broad "
                "superiority claims."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bind the terminal runtime/compute fairness report to GPU-contention "
            "coverage and publish a non-authorizing runtime claim guard."
        )
    )
    parser.add_argument("--runtime-fairness-report", type=Path, required=True)
    parser.add_argument("--expected-runtime-fairness-sha256", required=True)
    parser.add_argument("--pair-monitor", type=Path, required=True)
    parser.add_argument("--expected-pair-monitor-sha256", required=True)
    parser.add_argument("--expected-monitor-name", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="runtime claim guard output root",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="runtime claim guard output",
    ).resolve()
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("runtime claim guard output is outside output root") from error
    report = build_guard(
        runtime_fairness_report_path=args.runtime_fairness_report,
        expected_runtime_fairness_sha256=args.expected_runtime_fairness_sha256,
        pair_monitor_path=args.pair_monitor,
        expected_pair_monitor_sha256=args.expected_pair_monitor_sha256,
        expected_monitor_name=args.expected_monitor_name,
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
                "guard": identity,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
