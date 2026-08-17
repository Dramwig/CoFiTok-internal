from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Mapping

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)

try:
    import audit_generation_matched_uncertainty as uncertainty_audit
    import build_generation_capacity_matched_uncertainty_manifest as manifest_builder
    import run_generation_matched_uncertainty_waiter as base_waiter
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import audit_generation_matched_uncertainty as uncertainty_audit
    from scripts import (
        build_generation_capacity_matched_uncertainty_manifest as manifest_builder,
    )
    from scripts import run_generation_matched_uncertainty_waiter as base_waiter


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_capacity_statistical_claim_qualification"
CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "inference_export_authorization_allowed": False,
    "release_authorization_allowed": False,
    "replaces_capacity_result_or_final_gate": False,
    "broad_generation_superiority_claim_allowed": False,
    "matched_relative_advantage_claim_requires_status_pass": True,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _bound_file(
    path: Path,
    *,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} expected SHA256 is invalid")
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs")
    return identity


def _quality_evidence(
    source_kind: str,
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    if source_kind == "capacity_completion_100k":
        screen = anchor.get("quality_screen")
        if not isinstance(screen, Mapping) or screen.get("status") not in {
            "pass",
            "hold",
        }:
            raise ValueError("capacity completion quality screen is invalid")
        return {
            "status": screen["status"],
            "absolute_quality_passed": screen["status"] == "pass",
            "failed_checks": list(screen.get("failed_checks", [])),
            "decision": anchor.get("decision_support", {}).get(
                "recommended_next_stage"
            ),
        }
    if source_kind == "capacity_full_300k":
        status = anchor.get("status")
        decision = anchor.get("decision")
        if (status, decision) not in {
            ("pass", "large_scale_generation_ready"),
            ("fail", "hold"),
        }:
            raise ValueError("capacity-full final gate state is invalid")
        return {
            "status": "pass" if status == "pass" else "hold",
            "absolute_quality_passed": status == "pass",
            "failed_checks": list(anchor.get("failed_checks", [])),
            "decision": decision,
        }
    raise ValueError("capacity qualification source kind is invalid")


def build_qualification(
    *,
    source_kind: str,
    source_anchor_path: Path,
    expected_source_anchor_sha256: str,
    execution_manifest_path: Path,
    expected_execution_manifest_sha256: str,
    uncertainty_report_path: Path,
    expected_uncertainty_report_sha256: str,
    expected_evaluator_revision: str,
    expected_evaluator_branch: str,
    output_root: Path,
) -> dict[str, Any]:
    if source_kind not in manifest_builder.SOURCE_KINDS:
        raise ValueError("capacity qualification source kind is invalid")
    anchor_identity = _bound_file(
        source_anchor_path,
        expected_sha256=expected_source_anchor_sha256,
        label="capacity qualification source anchor",
    )
    manifest_identity = _bound_file(
        execution_manifest_path,
        expected_sha256=expected_execution_manifest_sha256,
        label="capacity qualification execution manifest",
    )
    uncertainty_identity = _bound_file(
        uncertainty_report_path,
        expected_sha256=expected_uncertainty_report_sha256,
        label="capacity qualification uncertainty report",
    )
    output_root = reject_symlink_chain(
        output_root,
        name="capacity qualification output root",
    ).resolve()
    manifest = base_waiter.validate_execution_manifest(
        Path(manifest_identity["path"]),
        expected_sha256=manifest_identity["sha256"],
        output_root=output_root,
    )
    report = manifest["report"]
    capacity_source = report.get("capacity_source")
    if (
        not isinstance(capacity_source, Mapping)
        or capacity_source.get("kind") != source_kind
        or capacity_source.get("anchor") != anchor_identity
    ):
        raise ValueError("capacity qualification manifest anchor differs")
    evaluator_git = {
        "revision": expected_evaluator_revision,
        "branch": expected_evaluator_branch,
        "tracked_dirty": False,
    }
    if (
        len(expected_evaluator_revision) != 40
        or any(
            character not in "0123456789abcdef"
            for character in expected_evaluator_revision
        )
    ):
        raise ValueError("capacity qualification evaluator revision is invalid")
    uncertainty = base_waiter.validate_audit_output(
        Path(uncertainty_identity["path"]),
        manifest_identity=manifest_identity,
        evaluator_git=evaluator_git,
    )
    if uncertainty["claim_boundary"] != uncertainty_audit.CLAIM_BOUNDARY:
        raise ValueError("capacity qualification uncertainty boundary differs")
    anchor = read_json_object(
        Path(anchor_identity["path"]),
        name="capacity qualification source anchor",
    )
    quality = _quality_evidence(source_kind, anchor)
    statistical_advantage = uncertainty["advantage_supported"] is True
    qualified = quality["absolute_quality_passed"] is True and statistical_advantage
    if source_kind == "capacity_full_300k":
        claim_scope = "matched_full_300k_relative_generation_advantage"
        formal_large_scale_claim_allowed = qualified
    else:
        claim_scope = "matched_capacity_completion_100k_relative_generation_advantage"
        formal_large_scale_claim_allowed = False
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "pass" if qualified else "hold",
        "decision": (
            "matched_capacity_advantage_statistically_qualified"
            if qualified
            else "matched_capacity_advantage_not_statistically_qualified"
        ),
        "source_kind": source_kind,
        "claim_scope": claim_scope,
        "source_anchor": anchor_identity,
        "execution_manifest": manifest_identity,
        "uncertainty_report": uncertainty_identity,
        "evaluator_git": evaluator_git,
        "quality_evidence": quality,
        "uncertainty_evidence": {
            "status": uncertainty["status"],
            "decision": uncertainty["decision"],
            "advantage_supported": uncertainty["advantage_supported"],
            "stream_id": manifest["stream_id"],
            "sample_count": int(
                report["expected"]["matched_sampling"]["sample_count"]
            ),
            "checkpoint_step": int(manifest["checkpoint_step"]),
            "fid_point_estimates": copy.deepcopy(
                report["expected"]["fid_point_estimates"]
            ),
        },
        "claim_policy": {
            "matched_relative_generation_advantage_claim_allowed": qualified,
            "formal_large_scale_generation_advantage_claim_allowed": (
                formal_large_scale_claim_allowed
            ),
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "absolute_usability_claim_requires_source_gate_pass": True,
            "relative_advantage_claim_requires_uncertainty_pass": True,
        },
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "limitations": [
            (
                "Qualification is limited to the bound ImageNet-256 matched "
                "pair, checkpoint, evaluator, sampling protocol, and "
                "global-index window."
            ),
            (
                "A pass supports a matched relative claim, not cross-tier "
                "SOTA or broad generation superiority."
            ),
            (
                "This report does not authorize training, sampling, inference "
                "export, release, or process signaling."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-authorizing statistical claim qualification that "
            "requires both the bound quality gate and matched uncertainty pass."
        )
    )
    parser.add_argument(
        "--source-kind",
        choices=sorted(manifest_builder.SOURCE_KINDS),
        required=True,
    )
    parser.add_argument("--source-anchor", type=Path, required=True)
    parser.add_argument("--expected-source-anchor-sha256", required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--expected-execution-manifest-sha256", required=True)
    parser.add_argument("--uncertainty-report", type=Path, required=True)
    parser.add_argument("--expected-uncertainty-report-sha256", required=True)
    parser.add_argument("--expected-evaluator-revision", required=True)
    parser.add_argument("--expected-evaluator-branch", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="capacity qualification output root",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="capacity qualification report",
    ).resolve()
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("capacity qualification report is outside output root") from error
    report = build_qualification(
        source_kind=args.source_kind,
        source_anchor_path=args.source_anchor,
        expected_source_anchor_sha256=args.expected_source_anchor_sha256,
        execution_manifest_path=args.execution_manifest,
        expected_execution_manifest_sha256=args.expected_execution_manifest_sha256,
        uncertainty_report_path=args.uncertainty_report,
        expected_uncertainty_report_sha256=args.expected_uncertainty_report_sha256,
        expected_evaluator_revision=args.expected_evaluator_revision,
        expected_evaluator_branch=args.expected_evaluator_branch,
        output_root=output_root,
    )
    identity = prepare_manifest(
        output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(
        json.dumps(
            {"status": report["status"], "qualification": identity},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
