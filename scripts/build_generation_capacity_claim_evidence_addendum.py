from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)

try:
    import build_generation_capacity_statistical_claim_qualification as qualification_builder
    import build_large_scale_generation_comparison as comparison_builder
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import (
        build_generation_capacity_statistical_claim_qualification as qualification_builder,
    )
    from scripts import build_large_scale_generation_comparison as comparison_builder


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_capacity_claim_evidence_addendum"
CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "sampling_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
    "replaces_comparison_or_qualification": False,
    "cross_tier_numeric_ranking_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
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


def _finite_positive(value: Any, *, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return result


def _same_float(left: Any, right: Any) -> bool:
    return math.isclose(
        float(left),
        float(right),
        rel_tol=0.0,
        abs_tol=1e-12,
    )


def _validate_embedded_identity(identity: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(identity, Mapping):
        raise ValueError(f"{label} identity is missing")
    expected = dict(identity)
    if not _is_sha256(expected.get("sha256")) or int(expected.get("bytes", 0)) < 1:
        raise ValueError(f"{label} identity is invalid")
    actual = file_identity(Path(str(expected.get("path", ""))))
    if actual != expected:
        raise ValueError(f"{label} changed after qualification")
    return actual


def _matched_rows(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = report.get("matched_training_rows")
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("capacity claim comparison matched rows are invalid")
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("capacity claim comparison matched row is malformed")
        method = str(row.get("method", ""))
        if method in indexed:
            raise ValueError("capacity claim comparison method is duplicated")
        indexed[method] = dict(row)
    if set(indexed) != {"CoFiTok K=8", "Dense identity"}:
        raise ValueError("capacity claim comparison methods are invalid")
    for method, row in indexed.items():
        if (
            row.get("comparison_tier") != "matched_training_direct"
            or row.get("directly_comparable_to_cofitok") is not True
            or row.get("weights") != "ema"
            or int(row.get("sample_count", -1)) != 50_000
            or int(row.get("training_steps", -1)) != 300_000
            or row.get("dataset") != "imagenet_256"
            or int(row.get("resolution", -1)) != 256
        ):
            raise ValueError(f"capacity claim comparison contract differs: {method}")
        _finite_positive(row.get("fid"), label=f"{method} FID")
    return indexed


def _validate_comparison(
    report: Mapping[str, Any],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    if (
        report.get("schema_version")
        != comparison_builder.COMPARISON_REPORT_SCHEMA_VERSION
        or report.get("source_profile") != "capacity_full"
    ):
        raise ValueError("capacity claim comparison schema or profile is invalid")
    policy = report.get("comparison_policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("primary_direct_tier") != "matched_training_direct"
        or policy.get("external_context_tier") != "official_pretrained_contextual"
        or policy.get("cross_tier_numeric_ranking_allowed") is not False
    ):
        raise ValueError("capacity claim comparison policy is unsafe")
    budget = report.get("training_budget_policy")
    if (
        not isinstance(budget, Mapping)
        or budget.get("basis") != comparison_builder.MATCHED_TRAINING_BUDGET_BASIS
        or budget.get("direct_quality_comparison_allowed") is not True
        or budget.get("compute_matched_claim_allowed") is not False
    ):
        raise ValueError("capacity claim comparison budget policy differs")
    verified = comparison_builder.verify_comparison_source_reports(dict(report))
    source_reports = verified["source_reports"]
    final_gate_identity = source_reports["final_gate"]
    final_gate = read_json_object(
        Path(final_gate_identity["path"]),
        name="capacity claim final gate",
    )
    gate_summary = final_gate.get("summary")
    gate_view = report.get("final_gate")
    if (
        final_gate.get("stage") != "full"
        or (final_gate.get("status"), final_gate.get("decision"))
        not in {
            ("pass", "large_scale_generation_ready"),
            ("fail", "hold"),
        }
        or not isinstance(gate_summary, Mapping)
        or not isinstance(gate_view, Mapping)
        or gate_view.get("status") != final_gate.get("status")
        or gate_view.get("decision") != final_gate.get("decision")
    ):
        raise ValueError("capacity claim final gate binding is invalid")
    expected_comparison_status = (
        "ready" if final_gate.get("status") == "pass" else "hold"
    )
    if report.get("status") != expected_comparison_status:
        raise ValueError("capacity claim comparison readiness differs from final gate")
    rows = _matched_rows(report)
    cofitok_fid = rows["CoFiTok K=8"]["fid"]
    dense_fid = rows["Dense identity"]["fid"]
    summary = report.get("matched_summary")
    if (
        not isinstance(summary, Mapping)
        or not _same_float(
            summary.get("cofitok_minus_dense_fid"),
            float(cofitok_fid) - float(dense_fid),
        )
        or not _same_float(
            summary.get("cofitok_relative_fid"),
            float(cofitok_fid) / float(dense_fid) - 1.0,
        )
        or not _same_float(gate_summary.get("cofitok_fid"), cofitok_fid)
        or not _same_float(gate_summary.get("dense_fid"), dense_fid)
    ):
        raise ValueError("capacity claim FID summaries differ")
    official = report.get("official_context_source")
    if not isinstance(official, Mapping) or not _is_sha256(official.get("sha256")):
        raise ValueError("capacity claim official context source is invalid")
    official_identity = file_identity(Path(str(official.get("path", ""))))
    if (
        official_identity["path"] != str(official.get("path", ""))
        or official_identity["sha256"] != official.get("sha256")
    ):
        raise ValueError("capacity claim official context source changed")
    return rows, final_gate_identity, official_identity


def _validate_qualification(
    report: Mapping[str, Any],
    *,
    final_gate_identity: Mapping[str, Any],
    rows: Mapping[str, Mapping[str, Any]],
) -> bool:
    if (
        report.get("schema_version") != qualification_builder.REPORT_SCHEMA_VERSION
        or report.get("role") != qualification_builder.REPORT_ROLE
        or report.get("source_kind") != "capacity_full_300k"
        or report.get("claim_scope")
        != "matched_full_300k_relative_generation_advantage"
        or report.get("claim_boundary") != qualification_builder.CLAIM_BOUNDARY
    ):
        raise ValueError("capacity statistical qualification contract differs")
    if report.get("source_anchor") != dict(final_gate_identity):
        raise ValueError("capacity statistical qualification uses another final gate")
    _validate_embedded_identity(
        report.get("execution_manifest"),
        label="capacity claim execution manifest",
    )
    _validate_embedded_identity(
        report.get("uncertainty_report"),
        label="capacity claim uncertainty report",
    )
    evaluator_git = report.get("evaluator_git")
    evaluator_revision = str(
        evaluator_git.get("revision", "")
        if isinstance(evaluator_git, Mapping)
        else ""
    )
    if (
        not isinstance(evaluator_git, Mapping)
        or len(evaluator_revision) != 40
        or any(
            character not in "0123456789abcdef"
            for character in evaluator_revision
        )
        or evaluator_git.get("tracked_dirty") is not False
    ):
        raise ValueError("capacity claim evaluator identity is invalid")
    quality = report.get("quality_evidence")
    uncertainty = report.get("uncertainty_evidence")
    policy = report.get("claim_policy")
    if (
        not isinstance(quality, Mapping)
        or not isinstance(uncertainty, Mapping)
        or not isinstance(policy, Mapping)
    ):
        raise ValueError("capacity statistical qualification evidence is incomplete")
    if (
        report.get("status") not in {"pass", "hold"}
        or uncertainty.get("status") not in {"pass", "hold"}
        or uncertainty.get("advantage_supported")
        is not (uncertainty.get("status") == "pass")
        or int(uncertainty.get("sample_count", -1)) != 50_000
        or int(uncertainty.get("checkpoint_step", -1)) != 300_000
    ):
        raise ValueError("capacity statistical qualification evidence is invalid")
    final_gate = read_json_object(
        Path(str(final_gate_identity["path"])),
        name="capacity claim final gate",
    )
    expected_quality_status = (
        "pass" if final_gate.get("status") == "pass" else "hold"
    )
    if (
        quality.get("status") != expected_quality_status
        or quality.get("absolute_quality_passed")
        is not (expected_quality_status == "pass")
        or quality.get("decision") != final_gate.get("decision")
    ):
        raise ValueError("capacity statistical qualification quality evidence differs")
    expected_uncertainty_decision = (
        "matched_relative_generation_advantage_supported"
        if uncertainty.get("status") == "pass"
        else "matched_relative_generation_advantage_not_confirmed"
    )
    if uncertainty.get("decision") != expected_uncertainty_decision:
        raise ValueError("capacity statistical qualification uncertainty differs")
    fid_estimates = uncertainty.get("fid_point_estimates")
    if (
        not isinstance(fid_estimates, Mapping)
        or not _same_float(
            fid_estimates.get("cofitok"),
            rows["CoFiTok K=8"]["fid"],
        )
        or not _same_float(
            fid_estimates.get("dense_identity"),
            rows["Dense identity"]["fid"],
        )
    ):
        raise ValueError("capacity statistical qualification FID estimates differ")
    allowed = report.get("status") == "pass"
    expected_decision = (
        "matched_capacity_advantage_statistically_qualified"
        if allowed
        else "matched_capacity_advantage_not_statistically_qualified"
    )
    if (
        report.get("decision") != expected_decision
        or policy.get("matched_relative_generation_advantage_claim_allowed")
        is not allowed
        or policy.get("formal_large_scale_generation_advantage_claim_allowed")
        is not allowed
        or policy.get("broad_generation_superiority_claim_allowed") is not False
        or policy.get("sota_claim_allowed") is not False
        or policy.get("absolute_usability_claim_requires_source_gate_pass")
        is not True
        or policy.get("relative_advantage_claim_requires_uncertainty_pass")
        is not True
        or allowed
        is not (
            quality.get("absolute_quality_passed") is True
            and uncertainty.get("advantage_supported") is True
        )
    ):
        raise ValueError("capacity statistical qualification decision is inconsistent")
    if allowed and not (
        float(rows["CoFiTok K=8"]["fid"])
        < float(rows["Dense identity"]["fid"])
    ):
        raise ValueError("qualified capacity claim does not have lower CoFiTok FID")
    return allowed


def build_addendum(
    *,
    comparison_report_path: Path,
    expected_comparison_report_sha256: str,
    statistical_qualification_path: Path,
    expected_statistical_qualification_sha256: str,
) -> dict[str, Any]:
    comparison_identity, comparison = _bound_json(
        comparison_report_path,
        expected_sha256=expected_comparison_report_sha256,
        label="capacity comparison report",
    )
    qualification_identity, qualification = _bound_json(
        statistical_qualification_path,
        expected_sha256=expected_statistical_qualification_sha256,
        label="capacity statistical qualification",
    )
    rows, final_gate_identity, official_identity = _validate_comparison(comparison)
    allowed = _validate_qualification(
        qualification,
        final_gate_identity=final_gate_identity,
        rows=rows,
    )
    if allowed and comparison.get("status") != "ready":
        raise ValueError("qualified capacity claim requires a ready comparison")
    cofitok_fid = float(rows["CoFiTok K=8"]["fid"])
    dense_fid = float(rows["Dense identity"]["fid"])
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "pass" if allowed else "hold",
        "decision": (
            "matched_capacity_fid_advantage_claim_supported"
            if allowed
            else "matched_capacity_fid_advantage_claim_not_supported"
        ),
        "metric_scope": "frechet_inception_distance",
        "source_profile": "capacity_full",
        "sources": {
            "large_scale_generation_comparison": comparison_identity,
            "statistical_claim_qualification": qualification_identity,
            "final_generation_gate": dict(final_gate_identity),
            "official_context_table": official_identity,
            "execution_manifest": copy.deepcopy(
                qualification["execution_manifest"]
            ),
            "uncertainty_report": copy.deepcopy(
                qualification["uncertainty_report"]
            ),
        },
        "matched_evidence": {
            "dataset": "imagenet_256",
            "resolution": 256,
            "training_steps_per_method": 300_000,
            "sample_count_per_method": 50_000,
            "weights": "ema",
            "cofitok_fid": cofitok_fid,
            "dense_identity_fid": dense_fid,
            "cofitok_minus_dense_fid": cofitok_fid - dense_fid,
            "cofitok_relative_fid": cofitok_fid / dense_fid - 1.0,
            "uncertainty_status": qualification["uncertainty_evidence"]["status"],
            "uncertainty_decision": qualification["uncertainty_evidence"][
                "decision"
            ],
            "stream_id": qualification["uncertainty_evidence"]["stream_id"],
        },
        "claim_policy": {
            "matched_relative_fid_advantage_claim_allowed": allowed,
            "formal_large_scale_relative_fid_claim_allowed": allowed,
            "absolute_usability_claim_allowed_by_this_addendum": False,
            "cross_tier_numeric_ranking_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
        },
        "claim_text": (
            "Under the exact bound ImageNet-256 matched-training and "
            "matched-evaluation protocol, CoFiTok K=8 achieved a statistically "
            "supported lower FID than dense_identity."
            if allowed
            else (
                "The exact bound ImageNet-256 evidence does not statistically "
                "qualify a relative FID advantage for CoFiTok K=8 over "
                "dense_identity."
            )
        ),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "limitations": [
            (
                "The claim is limited to the exact bound dataset, checkpoints, "
                "sample stream, evaluator, and matched protocol."
            ),
            (
                "The official D-AR, MAR, and ReTok rows remain contextual and "
                "must not be ranked numerically against the matched tier."
            ),
            (
                "This addendum neither replaces its source reports nor "
                "authorizes training, sampling, export, release, or process "
                "signaling."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Bind the capacity-full comparison to its terminal statistical "
            "qualification without changing any training or release authority."
        )
    )
    parser.add_argument("--comparison-report", type=Path, required=True)
    parser.add_argument("--expected-comparison-report-sha256", required=True)
    parser.add_argument("--statistical-qualification", type=Path, required=True)
    parser.add_argument(
        "--expected-statistical-qualification-sha256",
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="capacity claim addendum output root",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="capacity claim addendum output",
    ).resolve()
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("capacity claim addendum is outside output root") from error
    report = build_addendum(
        comparison_report_path=args.comparison_report,
        expected_comparison_report_sha256=args.expected_comparison_report_sha256,
        statistical_qualification_path=args.statistical_qualification,
        expected_statistical_qualification_sha256=(
            args.expected_statistical_qualification_sha256
        ),
    )
    identity = prepare_manifest(
        output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(json.dumps({"status": report["status"], "addendum": identity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
