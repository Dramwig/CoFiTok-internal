from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cofitok.inference_replay import file_identity
from scripts import audit_generation_matched_uncertainty as uncertainty_audit
from scripts import build_generation_capacity_claim_evidence_addendum as capacity_addendum
from scripts import (
    build_generation_quality_bridge_statistical_claim_qualification as quality_claim,
)
from scripts import build_generation_statistical_claim_language_guard as guard


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_identity(path)


def _uncertainty(
    tmp_path: Path,
    *,
    passed: bool,
    cofitok_fid: float = 4.0,
    dense_fid: float = 5.0,
) -> dict[str, Any]:
    ci_high = -0.01 if passed else 0.01
    sign_test_p = 0.01 if passed else 0.2
    stream_id = "quality_bridge_terminal_100k_00000000_00010000"
    payload = {
        "schema_version": uncertainty_audit.REPORT_SCHEMA_VERSION,
        "role": uncertainty_audit.REPORT_ROLE,
        "status": "pass" if passed else "hold",
        "decision": (
            "matched_relative_generation_advantage_supported"
            if passed
            else "matched_relative_generation_advantage_not_confirmed"
        ),
        "claim_boundary": uncertainty_audit.CLAIM_BOUNDARY,
        "sources": {
            "execution_manifest": {
                "status": "verified",
                "stream_id": stream_id,
            },
            "sample_sets": {
                "cofitok": {
                    "sha256": "a" * 64,
                    "checkpoint_sha256": "b" * 64,
                    "checkpoint_step": 100000,
                },
                "dense_identity": {
                    "sha256": "c" * 64,
                    "checkpoint_sha256": "d" * 64,
                    "checkpoint_step": 100000,
                },
            },
        },
        "matched_sampling": {
            "start_index": 0,
            "end_index_exclusive": 10000,
            "sample_count": 10000,
        },
        "fid_point_estimates": {
            "cofitok": cofitok_fid,
            "dense_identity": dense_fid,
            "cofitok_relative_to_dense": cofitok_fid / dense_fid - 1.0,
            "direction_supports_cofitok_advantage": cofitok_fid < dense_fid,
        },
        "paired_block_kid": {
            "block_size": 500,
            "block_count": 20,
            "one_sided_exact_sign_test_p": sign_test_p,
            "uncertainty_supports_cofitok_advantage": passed,
            "paired_block_bootstrap": {
                "mean": -0.02 if passed else 0.0,
                "ci_low": -0.03,
                "ci_high": ci_high,
            },
        },
        "advantage_supported": passed,
    }
    path = tmp_path / "matched_uncertainty.json"
    return {"path": path, "identity": _write(path, payload), "payload": payload}


def _quality_source(
    tmp_path: Path,
    *,
    passed: bool,
    uncertainty_passed: bool | None = None,
) -> dict[str, Any]:
    uncertainty_passed = passed if uncertainty_passed is None else uncertainty_passed
    uncertainty = _uncertainty(tmp_path, passed=uncertainty_passed)
    payload = {
        "schema_version": quality_claim.REPORT_SCHEMA_VERSION,
        "role": quality_claim.REPORT_ROLE,
        "status": "pass" if passed else "hold",
        "decision": (
            "matched_quality_bridge_fid_advantage_statistically_qualified"
            if passed
            else "matched_quality_bridge_fid_advantage_not_statistically_qualified"
        ),
        "uncertainty_report": uncertainty["identity"],
        "quality_evidence": {
            "status": "pass" if passed else "hold",
            "absolute_quality_passed": passed,
        },
        "uncertainty_evidence": {
            "status": "pass" if uncertainty_passed else "hold",
            "advantage_supported": uncertainty_passed,
            "stream_id": "quality_bridge_terminal_100k_00000000_00010000",
            "fid_point_estimates": {
                "cofitok": 4.0,
                "dense_identity": 5.0,
            },
        },
        "claim_policy": {
            "matched_relative_fid_advantage_claim_allowed": passed,
            "broad_generation_superiority_claim_allowed": False,
        },
    }
    path = tmp_path / "quality_claim.json"
    return {"path": path, "identity": _write(path, payload), "payload": payload}


def _capacity_source(tmp_path: Path, *, passed: bool) -> dict[str, Any]:
    uncertainty = _uncertainty(tmp_path, passed=passed)
    payload = {
        "schema_version": capacity_addendum.REPORT_SCHEMA_VERSION,
        "role": capacity_addendum.REPORT_ROLE,
        "source_profile": "capacity_full",
        "status": "pass" if passed else "hold",
        "decision": (
            "matched_capacity_fid_advantage_claim_supported"
            if passed
            else "matched_capacity_fid_advantage_claim_not_supported"
        ),
        "sources": {"uncertainty_report": uncertainty["identity"]},
        "matched_evidence": {
            "cofitok_fid": 4.0,
            "dense_identity_fid": 5.0,
            "uncertainty_status": "pass" if passed else "hold",
        },
        "claim_policy": {
            "matched_relative_fid_advantage_claim_allowed": passed,
            "cross_tier_numeric_ranking_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
        },
    }
    path = tmp_path / "capacity_claim.json"
    return {"path": path, "identity": _write(path, payload), "payload": payload}


def _build(source_kind: str, source: dict[str, Any]) -> dict[str, Any]:
    return guard.build_guard(
        source_kind=source_kind,
        source_report_path=source["path"],
        expected_source_report_sha256=source["identity"]["sha256"],
    )


def test_quality_bridge_guard_separates_fid_and_kid_claims(tmp_path: Path) -> None:
    report = _build("quality_bridge_100k", _quality_source(tmp_path, passed=True))

    assert report["status"] == "pass"
    assert report["metric_roles"]["fid"]["statistical_significance_tested"] is False
    assert report["metric_roles"]["paired_block_kid"][
        "statistical_significance_tested"
    ] is True
    assert report["claim_policy"]["fid_statistical_significance_claim_allowed"] is False
    assert report["claim_policy"]["matched_distribution_quality_claim_allowed"] is True
    assert report["claim_policy"]["independent_replication_claim_allowed"] is False
    assert (
        report["claim_policy"]["multiple_independent_terminal_streams_claim_allowed"]
        is False
    )
    assert report["replication_scope"]["bound_terminal_stream_count"] == 1
    assert report["replication_scope"]["independent_replication_count"] == 0
    assert report["replication_scope"]["independent_replication_supported"] is False
    assert "lower FID point estimate" in report["claim_text"]
    assert "paired block-KID" in report["claim_text"]
    assert "not an independent replication" in report["claim_text"]


def test_guard_holds_when_source_qualification_holds(tmp_path: Path) -> None:
    report = _build("quality_bridge_100k", _quality_source(tmp_path, passed=False))

    assert report["status"] == "hold"
    assert report["claim_policy"]["matched_distribution_quality_claim_allowed"] is False
    assert report["claim_policy"]["lower_fid_point_estimate_statement_allowed"] is False


def test_guard_accepts_quality_hold_with_positive_uncertainty(tmp_path: Path) -> None:
    report = _build(
        "quality_bridge_100k",
        _quality_source(tmp_path, passed=False, uncertainty_passed=True),
    )

    assert report["status"] == "hold"
    assert report["metric_roles"]["fid"]["direction_supports_cofitok"] is True
    assert report["metric_roles"]["paired_block_kid"]["supports_cofitok"] is True
    assert report["claim_policy"]["matched_distribution_quality_claim_allowed"] is False


def test_capacity_guard_uses_the_same_metric_boundary(tmp_path: Path) -> None:
    report = _build("capacity_full_300k", _capacity_source(tmp_path, passed=True))

    assert report["status"] == "pass"
    assert report["source_kind"] == "capacity_full_300k"
    assert report["claim_policy"]["fid_confidence_interval_claim_allowed"] is False
    assert report["claim_policy"]["paired_kid_statistical_support_statement_allowed"] is True


def test_guard_rejects_source_sha_drift(tmp_path: Path) -> None:
    source = _quality_source(tmp_path, passed=True)
    source["path"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source report SHA256 differs"):
        _build("quality_bridge_100k", source)


def test_guard_rejects_fid_drift_between_source_and_uncertainty(tmp_path: Path) -> None:
    source = _quality_source(tmp_path, passed=True)
    source["payload"]["uncertainty_evidence"]["fid_point_estimates"]["cofitok"] = 4.2
    source["identity"] = _write(source["path"], source["payload"])

    with pytest.raises(ValueError, match="FID point estimates differ"):
        _build("quality_bridge_100k", source)


def test_guard_rejects_quality_source_stream_drift(tmp_path: Path) -> None:
    source = _quality_source(tmp_path, passed=True)
    source["payload"]["uncertainty_evidence"]["stream_id"] = "different_stream"
    source["identity"] = _write(source["path"], source["payload"])

    with pytest.raises(ValueError, match="source stream identity differs"):
        _build("quality_bridge_100k", source)


def test_guard_rejects_uncertainty_without_bound_stream(tmp_path: Path) -> None:
    source = _quality_source(tmp_path, passed=True)
    uncertainty_path = Path(source["payload"]["uncertainty_report"]["path"])
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    uncertainty.pop("sources")
    source["payload"]["uncertainty_report"] = _write(uncertainty_path, uncertainty)
    source["identity"] = _write(source["path"], source["payload"])

    with pytest.raises(ValueError, match="stream binding is incomplete"):
        _build("quality_bridge_100k", source)


def test_guard_rejects_forged_authorizing_source_policy(tmp_path: Path) -> None:
    source = _capacity_source(tmp_path, passed=True)
    source["payload"]["claim_policy"]["cross_tier_numeric_ranking_allowed"] = True
    source["identity"] = _write(source["path"], source["payload"])

    with pytest.raises(ValueError, match="source decision differs"):
        _build("capacity_full_300k", source)


def test_guard_rejects_quality_claim_with_failed_absolute_screen(tmp_path: Path) -> None:
    source = _quality_source(tmp_path, passed=True)
    source["payload"]["quality_evidence"] = {
        "status": "hold",
        "absolute_quality_passed": False,
    }
    source["identity"] = _write(source["path"], source["payload"])

    with pytest.raises(ValueError, match="source decision differs"):
        _build("quality_bridge_100k", source)


def test_guard_rejects_uncertainty_pass_without_negative_kid_interval(
    tmp_path: Path,
) -> None:
    source = _quality_source(tmp_path, passed=True)
    uncertainty_path = Path(source["payload"]["uncertainty_report"]["path"])
    uncertainty = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    uncertainty["paired_block_kid"]["paired_block_bootstrap"]["ci_high"] = 0.01
    source["payload"]["uncertainty_report"] = _write(uncertainty_path, uncertainty)
    source["identity"] = _write(source["path"], source["payload"])

    with pytest.raises(ValueError, match="decision is inconsistent"):
        _build("quality_bridge_100k", source)
