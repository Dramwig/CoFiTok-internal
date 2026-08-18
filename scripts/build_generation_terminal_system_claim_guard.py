from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_DATASET,
    QUALITY_BRIDGE_RECIPE_STAGE,
    QUALITY_BRIDGE_RESULT_ROLE,
    QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
    QUALITY_BRIDGE_STEPS,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)

try:
    import build_generation_quality_bridge_statistical_claim_qualification as quality_claim
    import build_generation_statistical_claim_language_guard as statistical_guard
    import build_generation_requested_class_visual_audit as visual_audit
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        build_generation_quality_bridge_statistical_claim_qualification as quality_claim,
    )
    from scripts import (
        build_generation_requested_class_visual_audit as visual_audit,
    )
    from scripts import (
        build_generation_statistical_claim_language_guard as statistical_guard,
    )


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_terminal_system_claim_guard"
RUNTIME_GUARD_SCHEMA_VERSION = 1
RUNTIME_GUARD_ROLE = "generation_runtime_compute_claim_guard"
RUNTIME_GUARD_DECISIONS = {
    "direct_runtime_outcome_comparison_allowed",
    "runtime_cost_claims_observational_only",
}
RUNTIME_GUARD_CLAIM_BOUNDARY = {
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
VISUAL_WAITER_SCHEMA_VERSION = 1
VISUAL_WAITER_ROLE = "quality_bridge_terminal_requested_class_visual_audit_waiter"
VISUAL_WAITER_FIXED_INDICES = tuple(range(16))
VISUAL_WAITER_AUTHORIZATION_BOUNDARY = {
    "cpu_only_visual_diagnostic_allowed": True,
    "gpu_use_allowed": False,
    "training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
    "quality_bridge_or_followup_decision_modified": False,
}
CLAIM_BOUNDARY = {
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


def _replay_embedded_identity(
    descriptor: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} identity is missing")
    expected = dict(descriptor)
    if int(expected.get("bytes", 0)) < 1 or not _is_sha256(expected.get("sha256")):
        raise ValueError(f"{label} identity is malformed")
    path = reject_symlink_chain(Path(str(expected.get("path", ""))), name=label)
    actual = file_identity(path)
    if actual != expected:
        raise ValueError(f"{label} changed after its guard was written")
    return actual, read_json_object(path, name=label)


def _validate_quality_result(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    quality_output_root: Path,
) -> dict[str, Any]:
    expected_path = quality_output_root / "reports" / "quality_bridge_result.json"
    if Path(str(identity.get("path", ""))).resolve() != expected_path.resolve():
        raise ValueError("terminal system guard quality-result path differs")
    git = report.get("git")
    if (
        int(report.get("schema_version", -1))
        != QUALITY_BRIDGE_RESULT_SCHEMA_VERSION
        or report.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or report.get("status") != "completed"
        or report.get("stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or report.get("authorization_boundary") != RESULT_AUTHORIZATION_BOUNDARY
        or not isinstance(git, Mapping)
        or git.get("tracked_dirty") is not False
    ):
        raise ValueError("terminal system guard quality-result contract differs")
    evidence = quality_claim._quality_result_evidence(
        report,
        expected_revision=str(git.get("revision", "")),
        expected_branch=str(git.get("branch", "")),
    )
    terminal = report.get("terminal")
    physical = terminal.get("physical_evidence") if isinstance(terminal, Mapping) else None
    methods = terminal.get("methods") if isinstance(terminal, Mapping) else None
    if (
        not isinstance(physical, Mapping)
        or set(physical) != {"cofitok", "dense_identity"}
        or not isinstance(methods, Mapping)
        or set(methods) != {"cofitok", "dense_identity"}
    ):
        raise ValueError("terminal system guard quality-result terminal evidence differs")
    return {
        "git": copy.deepcopy(dict(git)),
        "quality_screen": evidence,
        "terminal": copy.deepcopy(dict(terminal)),
    }


def _validate_statistical_guard(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    quality_identity: Mapping[str, Any],
    quality_git: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        report.get("schema_version") != statistical_guard.REPORT_SCHEMA_VERSION
        or report.get("role") != statistical_guard.REPORT_ROLE
        or report.get("source_kind") != "quality_bridge_100k"
        or report.get("status") not in {"pass", "hold"}
        or report.get("claim_boundary") != statistical_guard.CLAIM_BOUNDARY
    ):
        raise ValueError("terminal system statistical guard contract differs")
    sources = report.get("sources")
    if not isinstance(sources, Mapping):
        raise ValueError("terminal system statistical guard sources are missing")
    claim_identity, claim_report = _replay_embedded_identity(
        sources.get("claim_report"),
        label="terminal statistical qualification",
    )
    replayed = statistical_guard.build_guard(
        source_kind="quality_bridge_100k",
        source_report_path=Path(claim_identity["path"]),
        expected_source_report_sha256=claim_identity["sha256"],
    )
    if replayed != dict(report):
        raise ValueError("terminal statistical guard does not replay from its sources")
    if (
        claim_report.get("quality_result") != dict(quality_identity)
        or claim_report.get("quality_git") != dict(quality_git)
        or claim_report.get("claim_scope")
        != "matched_full_data_quality_bridge_100k_relative_fid_advantage"
    ):
        raise ValueError("terminal statistical guard uses another quality result")
    policy = report.get("claim_policy")
    if not isinstance(policy, Mapping):
        raise ValueError("terminal statistical guard policy is missing")
    allowed = report.get("status") == "pass"
    if (
        policy.get("matched_distribution_quality_claim_allowed") is not allowed
        or policy.get("lower_fid_point_estimate_statement_allowed") is not allowed
        or policy.get("paired_kid_statistical_support_statement_allowed") is not allowed
        or policy.get("fid_statistical_significance_claim_allowed") is not False
        or policy.get("fid_confidence_interval_claim_allowed") is not False
        or policy.get("cross_tier_numeric_ranking_allowed") is not False
        or policy.get("broad_generation_superiority_claim_allowed") is not False
        or policy.get("sota_claim_allowed") is not False
    ):
        raise ValueError("terminal statistical guard claim policy differs")
    return {
        "status": str(report["status"]),
        "allowed": allowed,
        "decision": str(report.get("decision", "")),
        "metric_roles": copy.deepcopy(report.get("metric_roles")),
        "qualification": claim_identity,
        "guard": copy.deepcopy(dict(identity)),
    }


def _normalized_sampling_protocol(method: Mapping[str, Any]) -> dict[str, Any]:
    sampling = method.get("sampling")
    if not isinstance(sampling, Mapping):
        raise ValueError("terminal quality method sampling protocol is missing")
    return {key: copy.deepcopy(value) for key, value in sampling.items() if key != "prefix_budgets"}


def _validate_visual_audit(
    status_identity: Mapping[str, Any],
    status: Mapping[str, Any],
    *,
    quality_identity: Mapping[str, Any],
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        status.get("schema_version") != VISUAL_WAITER_SCHEMA_VERSION
        or status.get("role") != VISUAL_WAITER_ROLE
        or status.get("status") != "completed"
        or status.get("detail") != "terminal_visual_audit_source_revalidated"
        or status.get("authorization_boundary")
        != VISUAL_WAITER_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("terminal visual-audit waiter status differs")
    quality_binding = status.get("quality_result")
    if (
        not isinstance(quality_binding, Mapping)
        or quality_binding.get("identity") != dict(quality_identity)
        or quality_binding.get("quality_screen")
        != quality["quality_screen"]["screen"]
    ):
        raise ValueError("terminal visual audit uses another quality result")
    visual_identity, visual = _replay_embedded_identity(
        status.get("visual_audit"),
        label="terminal requested-class visual audit",
    )
    if (
        visual.get("claim_boundary") != visual_audit.CLAIM_BOUNDARY
        or visual.get("status") != "completed"
        or visual.get("role") != visual_audit.REPORT_ROLE
        or visual.get("indices") != list(VISUAL_WAITER_FIXED_INDICES)
    ):
        raise ValueError("terminal requested-class visual audit contract differs")
    panels = visual.get("panels")
    if not isinstance(panels, list) or len(panels) != 2:
        raise ValueError("terminal requested-class visual panels differ")
    for panel_index, panel in enumerate(panels):
        if not isinstance(panel, Mapping):
            raise ValueError("terminal requested-class visual panel is malformed")
        expected_indices = list(
            VISUAL_WAITER_FIXED_INDICES[panel_index * 8 : (panel_index + 1) * 8]
        )
        observed = file_identity(
            reject_symlink_chain(
                Path(str(panel.get("path", ""))),
                name="terminal requested-class visual panel",
            )
        )
        if (
            any(panel.get(field) != observed.get(field) for field in ("path", "bytes", "sha256"))
            or panel.get("indices") != expected_indices
            or panel.get("row_order")
            != ["real_validation", "cofitok", "dense_identity"]
            or int(panel.get("columns", -1)) != 8
        ):
            raise ValueError("terminal requested-class visual panel identity differs")
    sources = visual.get("sources")
    terminal = quality["terminal"]
    physical = terminal.get("physical_evidence")
    methods = terminal.get("methods")
    protocol = visual.get("sampling_protocol")
    if (
        not isinstance(sources, Mapping)
        or set(sources) != {"cofitok", "dense_identity"}
        or not isinstance(physical, Mapping)
        or not isinstance(methods, Mapping)
        or not isinstance(protocol, Mapping)
    ):
        raise ValueError("terminal requested-class visual source binding is incomplete")
    for method_name in ("cofitok", "dense_identity"):
        source = sources.get(method_name)
        method = methods.get(method_name)
        method_physical = physical.get(method_name)
        if not all(isinstance(value, Mapping) for value in (source, method, method_physical)):
            raise ValueError(f"terminal visual source is malformed: {method_name}")
        sample_set = source.get("sample_set")
        checkpoint = source.get("checkpoint")
        if (
            source.get("report") != method_physical.get("sampling_report")
            or not isinstance(sample_set, Mapping)
            or sample_set.get("sha256") != method.get("sample_set_sha256")
            or int(sample_set.get("count", -1)) != int(method.get("sample_count", -1))
            or not isinstance(checkpoint, Mapping)
            or checkpoint.get("sha256") != method.get("checkpoint_sha256")
            or int(checkpoint.get("step", -1)) != int(method.get("checkpoint_step", -1))
            or dict(protocol) != _normalized_sampling_protocol(method)
        ):
            raise ValueError(f"terminal visual source differs from quality evidence: {method_name}")
    return {
        "status": "completed",
        "quantitative_quality_evidence": False,
        "fixed_indices": list(VISUAL_WAITER_FIXED_INDICES),
        "panel_count": len(visual.get("panels", [])),
        "report": visual_identity,
        "waiter_status": copy.deepcopy(dict(status_identity)),
    }


def _validate_runtime_guard(
    identity: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    quality_output_root: Path,
    quality_git: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        report.get("schema_version") != RUNTIME_GUARD_SCHEMA_VERSION
        or report.get("role") != RUNTIME_GUARD_ROLE
        or report.get("status") != "pass"
        or report.get("decision") not in RUNTIME_GUARD_DECISIONS
        or report.get("claim_boundary") != RUNTIME_GUARD_CLAIM_BOUNDARY
    ):
        raise ValueError("terminal runtime claim guard contract differs")
    sources = report.get("sources")
    contract = report.get("matched_training_contract")
    policy = report.get("claim_policy")
    if (
        not isinstance(sources, Mapping)
        or set(sources) != {"runtime_compute_fairness", "terminal_pair_monitor"}
        or not isinstance(contract, Mapping)
        or not isinstance(policy, Mapping)
    ):
        raise ValueError("terminal runtime claim guard evidence is incomplete")
    fairness_identity, fairness = _replay_embedded_identity(
        sources["runtime_compute_fairness"],
        label="terminal runtime fairness report",
    )
    pair_identity, pair = _replay_embedded_identity(
        sources["terminal_pair_monitor"],
        label="terminal pair monitor",
    )
    expected_fairness = quality_output_root / "reports" / "runtime_compute_fairness" / "final_report.json"
    expected_pair = quality_output_root / "pair_monitor.json"
    direct = report.get("decision") == "direct_runtime_outcome_comparison_allowed"
    if (
        Path(fairness_identity["path"]).resolve() != expected_fairness.resolve()
        or Path(pair_identity["path"]).resolve() != expected_pair.resolve()
        or contract.get("status") != "verified"
        or contract.get("training_git") != dict(quality_git)
        or contract.get("dataset") != QUALITY_BRIDGE_DATASET
        or int(contract.get("target_steps_per_method", -1)) != QUALITY_BRIDGE_STEPS
        or str(contract.get("output_root", quality_output_root.resolve().as_posix()))
        != quality_output_root.resolve().as_posix()
        or pair.get("git") != dict(quality_git)
        or pair.get("status") != "pass"
        or pair.get("stage") != "complete"
        or fairness.get("status") != "pass"
    ):
        raise ValueError("terminal runtime guard uses another matched training pair")
    if (
        policy.get("runtime_configuration_parity_claim_allowed") is not True
        or policy.get("physical_recovery_compute_accounting_claim_allowed") is not True
        or policy.get("training_wall_clock_direct_comparison_allowed") is not direct
        or policy.get("training_throughput_direct_comparison_allowed") is not direct
        or policy.get("cost_efficiency_ranking_allowed") is not direct
        or policy.get("equal_wall_clock_budget_claim_allowed") is not False
        or policy.get("equal_gpu_hours_budget_claim_allowed") is not False
        or policy.get("equal_training_flops_budget_claim_allowed") is not False
        or policy.get("peak_vram_advantage_claim_allowed") is not False
        or policy.get("quality_or_generation_advantage_claim_allowed") is not False
    ):
        raise ValueError("terminal runtime claim policy differs")
    return {
        "status": "verified",
        "decision": str(report["decision"]),
        "direct_runtime_comparison_allowed": direct,
        "runtime_configuration_parity_claim_allowed": True,
        "physical_recovery_compute_accounting_claim_allowed": True,
        "guard": copy.deepcopy(dict(identity)),
        "runtime_fairness": fairness_identity,
        "pair_monitor": pair_identity,
    }


def build_guard(
    *,
    quality_result_path: Path,
    expected_quality_result_sha256: str,
    statistical_claim_guard_path: Path,
    expected_statistical_claim_guard_sha256: str,
    visual_audit_waiter_status_path: Path,
    expected_visual_audit_waiter_status_sha256: str,
    runtime_claim_guard_path: Path,
    expected_runtime_claim_guard_sha256: str,
    quality_output_root: Path,
) -> dict[str, Any]:
    quality_output_root = reject_symlink_chain(
        quality_output_root,
        name="terminal system guard quality output root",
    ).resolve()
    quality_identity, quality_report = _bound_json(
        quality_result_path,
        expected_sha256=expected_quality_result_sha256,
        label="terminal system quality result",
    )
    quality = _validate_quality_result(
        quality_identity,
        quality_report,
        quality_output_root=quality_output_root,
    )
    statistical_identity, statistical_report = _bound_json(
        statistical_claim_guard_path,
        expected_sha256=expected_statistical_claim_guard_sha256,
        label="terminal system statistical claim guard",
    )
    statistical = _validate_statistical_guard(
        statistical_identity,
        statistical_report,
        quality_identity=quality_identity,
        quality_git=quality["git"],
    )
    visual_status_identity, visual_status = _bound_json(
        visual_audit_waiter_status_path,
        expected_sha256=expected_visual_audit_waiter_status_sha256,
        label="terminal system visual-audit waiter status",
    )
    visual = _validate_visual_audit(
        visual_status_identity,
        visual_status,
        quality_identity=quality_identity,
        quality=quality,
    )
    runtime_identity, runtime_report = _bound_json(
        runtime_claim_guard_path,
        expected_sha256=expected_runtime_claim_guard_sha256,
        label="terminal system runtime claim guard",
    )
    runtime = _validate_runtime_guard(
        runtime_identity,
        runtime_report,
        quality_output_root=quality_output_root,
        quality_git=quality["git"],
    )

    quality_pass = quality["quality_screen"]["absolute_quality_passed"] is True
    matched_advantage_allowed = statistical["allowed"] is True
    if matched_advantage_allowed and not quality_pass:
        raise ValueError(
            "terminal statistical claim exceeds the absolute quality screen"
        )
    status = "pass" if matched_advantage_allowed else "hold"
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": status,
        "decision": (
            "matched_quality_advantage_qualified_with_terminal_system_evidence"
            if matched_advantage_allowed
            else "terminal_system_evidence_complete_without_qualified_matched_advantage"
        ),
        "scope": {
            "dataset": QUALITY_BRIDGE_DATASET,
            "training_steps_per_method": QUALITY_BRIDGE_STEPS,
            "quality_output_root": quality_output_root.as_posix(),
            "training_git": quality["git"],
        },
        "sources": {
            "quality_bridge_result": quality_identity,
            "statistical_claim_language_guard": statistical_identity,
            "requested_class_visual_audit_waiter_status": visual_status_identity,
            "runtime_compute_claim_guard": runtime_identity,
        },
        "evidence": {
            "quality_screen": {
                "status": quality["quality_screen"]["status"],
                "absolute_quality_passed": quality_pass,
                "failed_checks": quality["quality_screen"]["failed_checks"],
                "check_count": quality["quality_screen"]["check_count"],
            },
            "matched_statistical_advantage": statistical,
            "requested_class_visual_audit": visual,
            "runtime_compute": runtime,
        },
        "claim_policy": {
            "terminal_system_evidence_complete": True,
            "matched_distribution_quality_claim_allowed": matched_advantage_allowed,
            "lower_fid_point_estimate_statement_allowed": matched_advantage_allowed,
            "paired_kid_statistical_support_statement_allowed": matched_advantage_allowed,
            "absolute_quality_screen_pass_statement_allowed": quality_pass,
            "requested_class_visual_evidence_available": True,
            "requested_class_visual_evidence_is_quantitative": False,
            "runtime_configuration_parity_claim_allowed": True,
            "runtime_direct_comparison_allowed": runtime[
                "direct_runtime_comparison_allowed"
            ],
            "absolute_usability_claim_allowed": False,
            "fid_statistical_significance_claim_allowed": False,
            "fid_confidence_interval_claim_allowed": False,
            "cross_tier_numeric_ranking_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "larger_training_launch_allowed": False,
            "inference_export_authorization_allowed": False,
            "release_authorization_allowed": False,
        },
        "claim_text": (
            "Under the exact bound full-data ImageNet-256 100K matched protocol, "
            "CoFiTok K=8 passed the absolute quality screen, obtained a lower FID "
            "point estimate than dense_identity, and the paired block-KID analysis "
            "supported the same distribution-quality direction. Requested-class "
            "visual panels and runtime-accounting evidence are available under their "
            "separate non-authorizing claim boundaries."
            if matched_advantage_allowed
            else (
                "The exact bound terminal system evidence is complete, but it does "
                "not qualify a matched distribution-quality advantage claim for "
                "CoFiTok K=8 over dense_identity."
            )
        ),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "limitations": [
            (
                "The result is limited to the exact bound ImageNet-256 100K "
                "checkpoints, matched sample stream, evaluator, and protocol."
            ),
            (
                "FID remains a point estimate. Statistical support comes only "
                "from the paired block-KID analysis."
            ),
            (
                "Requested-class panels are auditable visual evidence, not a "
                "quantitative metric or an automated usability judgment."
            ),
            (
                "Runtime wording follows the bound runtime guard and does not "
                "imply equal wall-clock, GPU-hour, or FLOP budgets."
            ),
            (
                "This artifact does not authorize training, sampling, export, "
                "release, process signaling, broad superiority, or SOTA claims."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bind terminal quality, statistical, visual, and runtime evidence into "
            "one non-authorizing fail-closed system claim guard."
        )
    )
    parser.add_argument("--quality-result", type=Path, required=True)
    parser.add_argument("--expected-quality-result-sha256", required=True)
    parser.add_argument("--statistical-claim-guard", type=Path, required=True)
    parser.add_argument("--expected-statistical-claim-guard-sha256", required=True)
    parser.add_argument("--visual-audit-waiter-status", type=Path, required=True)
    parser.add_argument("--expected-visual-audit-waiter-status-sha256", required=True)
    parser.add_argument("--runtime-claim-guard", type=Path, required=True)
    parser.add_argument("--expected-runtime-claim-guard-sha256", required=True)
    parser.add_argument("--quality-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = reject_symlink_chain(
        args.output,
        name="terminal system claim guard output",
    ).resolve()
    report = build_guard(
        quality_result_path=args.quality_result,
        expected_quality_result_sha256=args.expected_quality_result_sha256,
        statistical_claim_guard_path=args.statistical_claim_guard,
        expected_statistical_claim_guard_sha256=(
            args.expected_statistical_claim_guard_sha256
        ),
        visual_audit_waiter_status_path=args.visual_audit_waiter_status,
        expected_visual_audit_waiter_status_sha256=(
            args.expected_visual_audit_waiter_status_sha256
        ),
        runtime_claim_guard_path=args.runtime_claim_guard,
        expected_runtime_claim_guard_sha256=args.expected_runtime_claim_guard_sha256,
        quality_output_root=args.quality_output_root,
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
