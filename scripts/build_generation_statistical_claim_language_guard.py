from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)

try:
    import audit_generation_matched_uncertainty as uncertainty_audit
    import build_generation_quality_bridge_statistical_claim_qualification as quality_claim
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import audit_generation_matched_uncertainty as uncertainty_audit
    from scripts import (
        build_generation_quality_bridge_statistical_claim_qualification as quality_claim,
    )


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_statistical_claim_language_guard"
SOURCE_KINDS = {"quality_bridge_100k", "capacity_full_300k"}
CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "training_launch_allowed": False,
    "gpu_execution_allowed": False,
    "sampling_launch_allowed": False,
    "inference_export_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
    "replaces_source_report_or_uncertainty": False,
    "cross_tier_numeric_ranking_allowed": False,
    "broad_generation_superiority_claim_allowed": False,
    "independent_replication_claim_allowed": False,
    "multiple_independent_terminal_streams_claim_allowed": False,
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


def _embedded_json(
    descriptor: Any,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(descriptor, Mapping):
        raise ValueError(f"{label} identity is missing")
    expected = dict(descriptor)
    if (
        not _is_sha256(expected.get("sha256"))
        or int(expected.get("bytes", 0)) < 1
    ):
        raise ValueError(f"{label} identity is invalid")
    path = reject_symlink_chain(
        Path(str(expected.get("path", ""))),
        name=label,
    ).resolve()
    actual = file_identity(path)
    if actual != expected:
        raise ValueError(f"{label} changed after source qualification")
    return actual, read_json_object(path, name=label)


def _uncertainty_evidence(
    descriptor: Any,
    *,
    expected_cofitok_fid: float,
    expected_dense_fid: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity, report = _embedded_json(
        descriptor,
        label="paired uncertainty report",
    )
    status = report.get("status")
    advantage_supported = report.get("advantage_supported")
    if (
        report.get("schema_version") != uncertainty_audit.REPORT_SCHEMA_VERSION
        or report.get("role") != uncertainty_audit.REPORT_ROLE
        or report.get("claim_boundary") != uncertainty_audit.CLAIM_BOUNDARY
        or status not in {"pass", "hold"}
        or advantage_supported is not (status == "pass")
    ):
        raise ValueError("paired uncertainty report contract differs")
    fid = report.get("fid_point_estimates")
    paired = report.get("paired_block_kid")
    bootstrap = (
        paired.get("paired_block_bootstrap")
        if isinstance(paired, Mapping)
        else None
    )
    if (
        not isinstance(fid, Mapping)
        or not isinstance(paired, Mapping)
        or not isinstance(bootstrap, Mapping)
    ):
        raise ValueError("paired uncertainty metric evidence is incomplete")
    cofitok_fid = _finite_positive(fid.get("cofitok"), label="uncertainty CoFiTok FID")
    dense_fid = _finite_positive(
        fid.get("dense_identity"),
        label="uncertainty dense FID",
    )
    if (
        not _same_float(cofitok_fid, expected_cofitok_fid)
        or not _same_float(dense_fid, expected_dense_fid)
    ):
        raise ValueError("paired uncertainty FID point estimates differ")
    ci_low = _finite(bootstrap.get("ci_low"), label="paired KID CI low")
    ci_high = _finite(bootstrap.get("ci_high"), label="paired KID CI high")
    mean = _finite(bootstrap.get("mean"), label="paired KID mean")
    sign_test_p = _finite(
        paired.get("one_sided_exact_sign_test_p"),
        label="paired KID sign-test p-value",
    )
    if (
        ci_low > ci_high
        or not 0.0 <= sign_test_p <= 1.0
        or int(paired.get("block_count", 0)) < 8
        or int(paired.get("block_size", 0)) < 2
    ):
        raise ValueError("paired uncertainty statistics are invalid")
    kid_support = ci_high < 0.0 and sign_test_p <= 0.05
    fid_direction = cofitok_fid < dense_fid
    if (
        paired.get("uncertainty_supports_cofitok_advantage") is not kid_support
        or bool(advantage_supported) is not (fid_direction and kid_support)
    ):
        raise ValueError("paired uncertainty decision is inconsistent")
    sources = report.get("sources")
    execution_manifest = (
        sources.get("execution_manifest")
        if isinstance(sources, Mapping)
        else None
    )
    sample_sets = sources.get("sample_sets") if isinstance(sources, Mapping) else None
    matched_sampling = report.get("matched_sampling")
    if (
        not isinstance(execution_manifest, Mapping)
        or execution_manifest.get("status") != "verified"
        or not isinstance(execution_manifest.get("stream_id"), str)
        or not execution_manifest["stream_id"]
        or not isinstance(sample_sets, Mapping)
        or set(sample_sets) != {"cofitok", "dense_identity"}
        or not isinstance(matched_sampling, Mapping)
    ):
        raise ValueError("paired uncertainty stream binding is incomplete")
    for method in ("cofitok", "dense_identity"):
        sample = sample_sets[method]
        if (
            not isinstance(sample, Mapping)
            or not _is_sha256(sample.get("sha256"))
            or not _is_sha256(sample.get("checkpoint_sha256"))
            or int(sample.get("checkpoint_step", 0)) < 1
        ):
            raise ValueError("paired uncertainty sample-set binding is invalid")
    start_index = int(matched_sampling.get("start_index", -1))
    end_index_exclusive = int(matched_sampling.get("end_index_exclusive", -1))
    sample_count = int(matched_sampling.get("sample_count", -1))
    if (
        start_index < 0
        or end_index_exclusive <= start_index
        or sample_count != end_index_exclusive - start_index
    ):
        raise ValueError("paired uncertainty sample window is invalid")
    return identity, {
        "status": status,
        "advantage_supported": bool(advantage_supported),
        "fid": {
            "cofitok": cofitok_fid,
            "dense_identity": dense_fid,
            "cofitok_minus_dense": cofitok_fid - dense_fid,
            "cofitok_relative_to_dense": cofitok_fid / dense_fid - 1.0,
            "direction_supports_cofitok": fid_direction,
        },
        "paired_kid": {
            "block_size": int(paired["block_size"]),
            "block_count": int(paired["block_count"]),
            "mean_cofitok_minus_dense": mean,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "one_sided_exact_sign_test_p": sign_test_p,
            "supports_cofitok": kid_support,
        },
        "replication_scope": {
            "bound_stream_id": execution_manifest["stream_id"],
            "start_index": start_index,
            "end_index_exclusive": end_index_exclusive,
            "sample_count": sample_count,
            "sample_sets": {
                method: {
                    "sha256": sample_sets[method]["sha256"],
                    "checkpoint_sha256": sample_sets[method]["checkpoint_sha256"],
                    "checkpoint_step": int(sample_sets[method]["checkpoint_step"]),
                }
                for method in ("cofitok", "dense_identity")
            },
            "bound_terminal_stream_count": 1,
            "independent_replication_count": 0,
            "independent_replication_supported": False,
            "interpretation": (
                "paired_reanalysis_of_one_exact_bound_terminal_sample_stream"
            ),
        },
    }


def _quality_bridge_source(
    report: Mapping[str, Any],
) -> tuple[bool, float, float, Any]:
    status = report.get("status")
    policy = report.get("claim_policy")
    quality = report.get("quality_evidence")
    uncertainty = report.get("uncertainty_evidence")
    if (
        report.get("schema_version") != quality_claim.REPORT_SCHEMA_VERSION
        or report.get("role") != quality_claim.REPORT_ROLE
        or status not in {"pass", "hold"}
        or not isinstance(policy, Mapping)
        or not isinstance(quality, Mapping)
        or not isinstance(uncertainty, Mapping)
    ):
        raise ValueError("quality-bridge claim source contract differs")
    allowed = status == "pass"
    expected_decision = (
        "matched_quality_bridge_fid_advantage_statistically_qualified"
        if allowed
        else "matched_quality_bridge_fid_advantage_not_statistically_qualified"
    )
    if (
        report.get("decision") != expected_decision
        or policy.get("matched_relative_fid_advantage_claim_allowed") is not allowed
        or policy.get("broad_generation_superiority_claim_allowed") is not False
        or quality.get("absolute_quality_passed")
        is not (quality.get("status") == "pass")
        or uncertainty.get("status") not in {"pass", "hold"}
        or uncertainty.get("advantage_supported")
        is not (uncertainty.get("status") == "pass")
        or allowed
        is not (
            quality.get("absolute_quality_passed") is True
            and uncertainty.get("advantage_supported") is True
        )
        or (allowed and uncertainty.get("advantage_supported") is not True)
    ):
        raise ValueError("quality-bridge claim source decision differs")
    fid = uncertainty.get("fid_point_estimates")
    if not isinstance(fid, Mapping):
        raise ValueError("quality-bridge claim source FID evidence is missing")
    return (
        allowed,
        _finite_positive(fid.get("cofitok"), label="source CoFiTok FID"),
        _finite_positive(fid.get("dense_identity"), label="source dense FID"),
        report.get("uncertainty_report"),
    )


def _capacity_source(
    report: Mapping[str, Any],
) -> tuple[bool, float, float, Any]:
    try:
        import build_generation_capacity_claim_evidence_addendum as capacity_addendum
    except ModuleNotFoundError:
        try:
            from scripts import (
                build_generation_capacity_claim_evidence_addendum as capacity_addendum,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise RuntimeError(
                "capacity_full_300k claim support is unavailable in this "
                "quality-bridge-only integration checkout"
            ) from error
    status = report.get("status")
    policy = report.get("claim_policy")
    evidence = report.get("matched_evidence")
    sources = report.get("sources")
    if (
        report.get("schema_version") != capacity_addendum.REPORT_SCHEMA_VERSION
        or report.get("role") != capacity_addendum.REPORT_ROLE
        or report.get("source_profile") != "capacity_full"
        or status not in {"pass", "hold"}
        or not isinstance(policy, Mapping)
        or not isinstance(evidence, Mapping)
        or not isinstance(sources, Mapping)
    ):
        raise ValueError("capacity claim source contract differs")
    allowed = status == "pass"
    expected_decision = (
        "matched_capacity_fid_advantage_claim_supported"
        if allowed
        else "matched_capacity_fid_advantage_claim_not_supported"
    )
    if (
        report.get("decision") != expected_decision
        or policy.get("matched_relative_fid_advantage_claim_allowed") is not allowed
        or policy.get("cross_tier_numeric_ranking_allowed") is not False
        or policy.get("broad_generation_superiority_claim_allowed") is not False
        or evidence.get("uncertainty_status") not in {"pass", "hold"}
        or (allowed and evidence.get("uncertainty_status") != "pass")
    ):
        raise ValueError("capacity claim source decision differs")
    return (
        allowed,
        _finite_positive(evidence.get("cofitok_fid"), label="source CoFiTok FID"),
        _finite_positive(
            evidence.get("dense_identity_fid"),
            label="source dense FID",
        ),
        sources.get("uncertainty_report"),
    )


def build_guard(
    *,
    source_kind: str,
    source_report_path: Path,
    expected_source_report_sha256: str,
) -> dict[str, Any]:
    if source_kind not in SOURCE_KINDS:
        raise ValueError(f"unsupported claim source kind: {source_kind}")
    source_identity, source = _bound_json(
        source_report_path,
        expected_sha256=expected_source_report_sha256,
        label="statistical claim source report",
    )
    extractor = (
        _quality_bridge_source
        if source_kind == "quality_bridge_100k"
        else _capacity_source
    )
    source_allowed, cofitok_fid, dense_fid, uncertainty_descriptor = extractor(source)
    uncertainty_identity, metrics = _uncertainty_evidence(
        uncertainty_descriptor,
        expected_cofitok_fid=cofitok_fid,
        expected_dense_fid=dense_fid,
    )
    if source_kind == "quality_bridge_100k":
        source_stream_id = source["uncertainty_evidence"].get("stream_id")
        if source_stream_id != metrics["replication_scope"]["bound_stream_id"]:
            raise ValueError("quality-bridge source stream identity differs")
    qualified = (
        source_allowed
        and metrics["fid"]["direction_supports_cofitok"] is True
        and metrics["paired_kid"]["supports_cofitok"] is True
        and metrics["advantage_supported"] is True
    )
    if source_allowed is not qualified:
        raise ValueError("statistical claim source overstates its metric evidence")
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "pass" if qualified else "hold",
        "decision": (
            "lower_fid_point_estimate_with_paired_kid_support"
            if qualified
            else "matched_distribution_quality_claim_not_supported"
        ),
        "source_kind": source_kind,
        "sources": {
            "claim_report": source_identity,
            "paired_uncertainty_report": uncertainty_identity,
        },
        "metric_roles": {
            "fid": {
                "role": "matched_point_estimate_direction",
                **copy.deepcopy(metrics["fid"]),
                "confidence_interval_available": False,
                "statistical_significance_tested": False,
            },
            "paired_block_kid": {
                "role": "matched_distribution_distance_uncertainty",
                **copy.deepcopy(metrics["paired_kid"]),
                "statistical_significance_tested": True,
            },
        },
        "replication_scope": copy.deepcopy(metrics["replication_scope"]),
        "claim_policy": {
            "matched_distribution_quality_claim_allowed": qualified,
            "lower_fid_point_estimate_statement_allowed": qualified,
            "paired_kid_statistical_support_statement_allowed": qualified,
            "fid_statistical_significance_claim_allowed": False,
            "fid_confidence_interval_claim_allowed": False,
            "cross_tier_numeric_ranking_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "independent_replication_claim_allowed": False,
            "multiple_independent_terminal_streams_claim_allowed": False,
            "replication_language_requires_distinct_bound_streams": True,
        },
        "claim_text": (
            "Under the exact bound matched protocol, CoFiTok K=8 obtained a "
            "lower FID point estimate than dense_identity, and a paired "
            "block-KID analysis supported the same distribution-quality "
            "direction. This is a paired re-analysis of one exact bound "
            "terminal sample stream, not an independent replication."
            if qualified
            else (
                "The exact bound matched evidence does not qualify a positive "
                "distribution-quality claim for CoFiTok K=8 over dense_identity."
            )
        ),
        "claim_boundary": copy.deepcopy(CLAIM_BOUNDARY),
        "limitations": [
            (
                "The FID values are point estimates; this artifact does not "
                "provide a confidence interval or significance test for FID itself."
            ),
            (
                "The statistical result is a paired block-KID test over the "
                "exact bound sample stream and shared real reference."
            ),
            (
                "The bound FID and paired block-KID evidence reuse the same "
                "terminal sample sets; this artifact provides zero independent "
                "replications and cannot support a two-stream replication claim."
            ),
            (
                "A pass is limited to the bound dataset, checkpoints, evaluator, "
                "sampling protocol, and sample-index window."
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
            "Separate FID point-estimate language from paired KID statistical "
            "support in a source-bound, non-authorizing claim artifact."
        )
    )
    parser.add_argument("--source-kind", choices=sorted(SOURCE_KINDS), required=True)
    parser.add_argument("--source-report", type=Path, required=True)
    parser.add_argument("--expected-source-report-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="statistical claim language guard output root",
    ).resolve()
    output = reject_symlink_chain(
        args.output,
        name="statistical claim language guard output",
    ).resolve()
    try:
        output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("statistical claim language guard is outside output root") from error
    report = build_guard(
        source_kind=args.source_kind,
        source_report_path=args.source_report,
        expected_source_report_sha256=args.expected_source_report_sha256,
    )
    identity = prepare_manifest(
        output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(json.dumps({"status": report["status"], "guard": identity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
