from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_RECIPE_STAGE,
    QUALITY_BRIDGE_RESULT_ROLE,
    QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
    QUALITY_BRIDGE_STEPS,
    QUALITY_BRIDGE_TERMINAL_SAMPLES,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)

try:
    import audit_generation_matched_uncertainty as uncertainty_audit
    import build_generation_quality_bridge_matched_uncertainty_manifest as manifest_builder
    import run_generation_matched_uncertainty_waiter as base_waiter
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import audit_generation_matched_uncertainty as uncertainty_audit
    from scripts import (
        build_generation_quality_bridge_matched_uncertainty_manifest as manifest_builder,
    )
    from scripts import run_generation_matched_uncertainty_waiter as base_waiter


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_quality_bridge_statistical_claim_qualification"
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
    "replaces_quality_bridge_result_or_uncertainty_report": False,
    "broad_generation_superiority_claim_allowed": False,
    "matched_relative_fid_claim_requires_status_pass": True,
}


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
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


def _finite_positive(value: Any, *, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be finite and positive")
    return number


def _quality_result_evidence(
    result: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    if (
        int(result.get("schema_version", -1))
        != QUALITY_BRIDGE_RESULT_SCHEMA_VERSION
        or result.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or result.get("status") != "completed"
        or result.get("stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or result.get("authorization_boundary") != RESULT_AUTHORIZATION_BOUNDARY
        or result.get("git")
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
    ):
        raise ValueError("quality-bridge qualification result contract differs")
    screen = result.get("quality_screen")
    terminal = result.get("terminal")
    methods = terminal.get("methods") if isinstance(terminal, Mapping) else None
    if (
        not isinstance(screen, Mapping)
        or screen.get("status") not in {"pass", "hold"}
        or screen.get("non_authorizing") is not True
        or not isinstance(screen.get("checks"), list)
        or not isinstance(screen.get("failed_checks"), list)
        or not isinstance(methods, Mapping)
        or not isinstance(methods.get("cofitok"), Mapping)
        or not isinstance(methods.get("dense_identity"), Mapping)
    ):
        raise ValueError("quality-bridge qualification evidence is incomplete")
    failed = [
        str(row.get("name", ""))
        for row in screen["checks"]
        if isinstance(row, Mapping) and row.get("passed") is not True
    ]
    if (
        failed != list(screen["failed_checks"])
        or (screen.get("status") == "pass") is not (not failed)
    ):
        raise ValueError("quality-bridge qualification quality screen is inconsistent")
    cofitok_fid = _finite_positive(
        methods["cofitok"].get("fid"),
        label="quality-bridge CoFiTok FID",
    )
    dense_fid = _finite_positive(
        methods["dense_identity"].get("fid"),
        label="quality-bridge dense FID",
    )
    return {
        "status": str(screen["status"]),
        "absolute_quality_passed": screen["status"] == "pass",
        "failed_checks": list(screen["failed_checks"]),
        "check_count": len(screen["checks"]),
        "cofitok_fid": cofitok_fid,
        "dense_identity_fid": dense_fid,
        "screen": copy.deepcopy(dict(screen)),
    }


def build_qualification(
    *,
    quality_result_path: Path,
    expected_quality_result_sha256: str,
    execution_manifest_path: Path,
    expected_execution_manifest_sha256: str,
    uncertainty_report_path: Path,
    expected_uncertainty_report_sha256: str,
    expected_quality_revision: str,
    expected_quality_branch: str,
    expected_evaluator_revision: str,
    expected_evaluator_branch: str,
    output_root: Path,
) -> dict[str, Any]:
    if (
        not _is_revision(expected_quality_revision)
        or not expected_quality_branch
        or not _is_revision(expected_evaluator_revision)
    ):
        raise ValueError("quality-bridge qualification Git identity is invalid")
    result_identity = _bound_file(
        quality_result_path,
        expected_sha256=expected_quality_result_sha256,
        label="quality-bridge qualification result",
    )
    manifest_identity = _bound_file(
        execution_manifest_path,
        expected_sha256=expected_execution_manifest_sha256,
        label="quality-bridge qualification execution manifest",
    )
    uncertainty_identity = _bound_file(
        uncertainty_report_path,
        expected_sha256=expected_uncertainty_report_sha256,
        label="quality-bridge qualification uncertainty report",
    )
    output_root = reject_symlink_chain(
        output_root,
        name="quality-bridge qualification output root",
    ).resolve()
    manifest = base_waiter.validate_execution_manifest(
        Path(manifest_identity["path"]),
        expected_sha256=manifest_identity["sha256"],
        output_root=output_root,
    )
    manifest_report = manifest["report"]
    quality_binding = manifest_report.get("quality_bridge")
    if (
        not isinstance(quality_binding, Mapping)
        or quality_binding.get("result") != result_identity
        or quality_binding.get("git")
        != {
            "revision": expected_quality_revision,
            "branch": expected_quality_branch,
            "tracked_dirty": False,
        }
        or quality_binding.get("authorization_boundary")
        != RESULT_AUTHORIZATION_BOUNDARY
    ):
        raise ValueError("quality-bridge qualification manifest result differs")
    result = read_json_object(
        Path(result_identity["path"]),
        name="quality-bridge qualification result",
    )
    quality = _quality_result_evidence(
        result,
        expected_revision=expected_quality_revision,
        expected_branch=expected_quality_branch,
    )
    if quality_binding.get("quality_screen") != quality["screen"]:
        raise ValueError("quality-bridge qualification manifest screen differs")
    evaluator_git = {
        "revision": expected_evaluator_revision,
        "branch": expected_evaluator_branch,
        "tracked_dirty": False,
    }
    uncertainty = base_waiter.validate_audit_output(
        Path(uncertainty_identity["path"]),
        manifest_identity=manifest_identity,
        evaluator_git=evaluator_git,
    )
    if uncertainty.get("claim_boundary") != uncertainty_audit.CLAIM_BOUNDARY:
        raise ValueError("quality-bridge qualification uncertainty boundary differs")
    expected = manifest_report.get("expected")
    matched = expected.get("matched_sampling") if isinstance(expected, Mapping) else None
    fid_estimates = expected.get("fid_point_estimates") if isinstance(expected, Mapping) else None
    if (
        int(manifest.get("checkpoint_step", -1)) != QUALITY_BRIDGE_STEPS
        or not isinstance(matched, Mapping)
        or int(matched.get("sample_count", -1))
        != QUALITY_BRIDGE_TERMINAL_SAMPLES
        or not isinstance(fid_estimates, Mapping)
        or not math.isclose(
            float(fid_estimates.get("cofitok", math.nan)),
            quality["cofitok_fid"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            float(fid_estimates.get("dense_identity", math.nan)),
            quality["dense_identity_fid"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("quality-bridge qualification scientific scope differs")
    statistical_advantage = uncertainty.get("advantage_supported") is True
    qualified = quality["absolute_quality_passed"] is True and statistical_advantage
    if qualified and not (
        quality["cofitok_fid"] < quality["dense_identity_fid"]
    ):
        raise ValueError("qualified quality-bridge claim lacks lower CoFiTok FID")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "pass" if qualified else "hold",
        "decision": (
            "matched_quality_bridge_fid_advantage_statistically_qualified"
            if qualified
            else "matched_quality_bridge_fid_advantage_not_statistically_qualified"
        ),
        "claim_scope": "matched_full_data_quality_bridge_100k_relative_fid_advantage",
        "quality_result": result_identity,
        "execution_manifest": manifest_identity,
        "uncertainty_report": uncertainty_identity,
        "quality_git": copy.deepcopy(result["git"]),
        "evaluator_git": evaluator_git,
        "quality_evidence": {
            "status": quality["status"],
            "absolute_quality_passed": quality["absolute_quality_passed"],
            "failed_checks": quality["failed_checks"],
            "check_count": quality["check_count"],
        },
        "uncertainty_evidence": {
            "status": uncertainty["status"],
            "decision": uncertainty["decision"],
            "advantage_supported": uncertainty["advantage_supported"],
            "stream_id": manifest["stream_id"],
            "sample_count": int(matched["sample_count"]),
            "checkpoint_step": int(manifest["checkpoint_step"]),
            "fid_point_estimates": {
                "cofitok": quality["cofitok_fid"],
                "dense_identity": quality["dense_identity_fid"],
            },
        },
        "claim_policy": {
            "matched_relative_fid_advantage_claim_allowed": qualified,
            "formal_large_scale_generation_advantage_claim_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "absolute_usability_claim_allowed_by_this_report": False,
            "larger_training_launch_allowed": False,
            "relative_fid_claim_requires_quality_and_uncertainty_pass": True,
        },
        "claim_text": (
            "Under the exact bound full-data ImageNet-256 100K matched protocol, "
            "CoFiTok K=8 achieved a statistically supported lower FID than "
            "dense_identity."
            if qualified
            else (
                "The exact bound full-data ImageNet-256 100K evidence does not "
                "statistically qualify a relative FID advantage for CoFiTok K=8 "
                "over dense_identity."
            )
        ),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "limitations": [
            (
                "Qualification is limited to the exact full-data quality-bridge "
                "checkpoints, matched sample stream, evaluator, and protocol."
            ),
            (
                "A pass supports only a matched relative FID statement; it is not "
                "a broad generation-superiority or SOTA claim."
            ),
            (
                "This report does not authorize larger training, sampling, "
                "inference export, release, or process signaling."
            ),
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-authorizing full-data 100K quality-bridge statistical "
            "claim qualification."
        )
    )
    parser.add_argument("--quality-result", type=Path, required=True)
    parser.add_argument("--expected-quality-result-sha256", required=True)
    parser.add_argument("--execution-manifest", type=Path, required=True)
    parser.add_argument("--expected-execution-manifest-sha256", required=True)
    parser.add_argument("--uncertainty-report", type=Path, required=True)
    parser.add_argument("--expected-uncertainty-report-sha256", required=True)
    parser.add_argument("--expected-quality-revision", required=True)
    parser.add_argument("--expected-quality-branch", required=True)
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
        name="quality-bridge qualification output root",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="quality-bridge qualification report",
    ).resolve()
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("quality-bridge qualification is outside output root") from error
    report = build_qualification(
        quality_result_path=args.quality_result,
        expected_quality_result_sha256=args.expected_quality_result_sha256,
        execution_manifest_path=args.execution_manifest,
        expected_execution_manifest_sha256=args.expected_execution_manifest_sha256,
        uncertainty_report_path=args.uncertainty_report,
        expected_uncertainty_report_sha256=args.expected_uncertainty_report_sha256,
        expected_quality_revision=args.expected_quality_revision,
        expected_quality_branch=args.expected_quality_branch,
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
