from __future__ import annotations

import copy
import json
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from cofitok.inference_replay import file_identity
from scripts import (
    build_generation_capacity_claim_evidence_addendum as addendum,
)
from scripts import (
    build_generation_capacity_statistical_claim_qualification as qualification_builder,
)
from scripts import build_large_scale_generation_comparison as comparison_builder


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_identity(path)


def _source_path(root: Path, suffix: str) -> Path:
    return root.joinpath(*PurePosixPath(suffix).parts)


def _training_report(*, parameter_count: int, token_count: int) -> dict[str, Any]:
    return {
        "target_steps": 300_000,
        "parameter_count": parameter_count,
        "elapsed_seconds": 100_000.0,
        "peak_vram_bytes": 24 * 1024**3,
        "final_metrics": {"samples_seen": 19_200_000},
        "config": {
            "data": {"dataset": "imagenet_256", "batch_size": 16},
            "diffusion": {"num_train_timesteps": 1000},
            "runtime": {"device": "cuda"},
            "optimization": {"gradient_accumulation_steps": 4},
            "model": {"image_size": 256, "token_count": token_count},
        },
    }


def _training_cost_fields(report: dict[str, Any]) -> dict[str, Any]:
    cost = comparison_builder.training_cost_summary(report)
    assert cost["valid"] is True
    adjustment = cost["resume_compute_adjustment"]
    return {
        "training_reported_elapsed_seconds": cost["reported_elapsed_seconds"],
        "training_resume_compute_adjustment_seconds": adjustment["seconds"],
        "training_resume_compute_adjustment_hours": adjustment["hours"],
        "training_resume_compute_adjustment_event_count": adjustment[
            "event_count"
        ],
        "training_orphaned_optimizer_steps_lower_bound": adjustment[
            "orphaned_optimizer_steps_lower_bound"
        ],
        "training_orphaned_images_lower_bound": adjustment[
            "orphaned_images_lower_bound"
        ],
        "training_elapsed_seconds": cost["elapsed_seconds"],
        "training_elapsed_seconds_role": cost["elapsed_seconds_role"],
        "training_images_per_second": cost["images_per_second"],
    }


def _case(
    tmp_path: Path,
    *,
    qualification_pass: bool = True,
) -> dict[str, Any]:
    source_root = tmp_path / "CoFiTok"
    training_reports = {
        "cofitok_training": _training_report(
            parameter_count=62_950_800,
            token_count=8,
        ),
        "dense_training": _training_report(
            parameter_count=62_824_707,
            token_count=1,
        ),
    }
    source_reports: dict[str, dict[str, Any]] = {}
    for name, suffix in comparison_builder.SOURCE_REPORT_PROFILES[
        "capacity_full"
    ].items():
        path = _source_path(source_root, suffix)
        if name == "final_gate":
            payload = {
                "stage": "full",
                "status": "pass",
                "decision": "large_scale_generation_ready",
                "summary": {"cofitok_fid": 4.0, "dense_fid": 5.0},
                "resume_compute_adjustments": {},
            }
        elif name in training_reports:
            payload = training_reports[name]
        else:
            payload = {"role": name}
        source_reports[name] = _write(path, payload)

    official_path = source_root / "official_related_methods_table.json"
    official_identity = _write(official_path, {"schema_version": 1, "rows": []})
    comparison_report = {
        "schema_version": comparison_builder.COMPARISON_REPORT_SCHEMA_VERSION,
        "status": "ready",
        "source_profile": "capacity_full",
        "final_gate": {
            "status": "pass",
            "decision": "large_scale_generation_ready",
        },
        "comparison_policy": {
            "primary_direct_tier": "matched_training_direct",
            "external_context_tier": "official_pretrained_contextual",
            "cross_tier_numeric_ranking_allowed": False,
        },
        "training_budget_policy": {
            "basis": comparison_builder.MATCHED_TRAINING_BUDGET_BASIS,
            "direct_quality_comparison_allowed": True,
            "compute_matched_claim_allowed": False,
        },
        "official_context_source": {
            "path": official_identity["path"],
            "sha256": official_identity["sha256"],
            "schema_version": 1,
        },
        "source_reports": source_reports,
        "resume_compute_adjustments": {},
        "matched_training_rows": [
            {
                "method": "CoFiTok K=8",
                "comparison_tier": "matched_training_direct",
                "directly_comparable_to_cofitok": True,
                "dataset": "imagenet_256",
                "resolution": 256,
                "training_steps": 300_000,
                "sample_count": 50_000,
                "weights": "ema",
                "fid": 4.0,
                **_training_cost_fields(training_reports["cofitok_training"]),
            },
            {
                "method": "Dense identity",
                "comparison_tier": "matched_training_direct",
                "directly_comparable_to_cofitok": True,
                "dataset": "imagenet_256",
                "resolution": 256,
                "training_steps": 300_000,
                "sample_count": 50_000,
                "weights": "ema",
                "fid": 5.0,
                **_training_cost_fields(training_reports["dense_training"]),
            },
        ],
        "official_context_rows": [],
        "matched_summary": {
            "cofitok_minus_dense_fid": -1.0,
            "cofitok_relative_fid": -0.2,
        },
    }
    comparison_path = tmp_path / "comparison" / "large_scale_generation_comparison.json"
    comparison_identity = _write(comparison_path, comparison_report)

    execution_manifest_identity = _write(
        tmp_path / "uncertainty" / "execution_manifest.json",
        {"role": "execution_manifest"},
    )
    uncertainty_identity = _write(
        tmp_path / "uncertainty" / "matched_uncertainty.json",
        {"status": "pass" if qualification_pass else "hold"},
    )
    qualification_report = {
        "schema_version": qualification_builder.REPORT_SCHEMA_VERSION,
        "role": qualification_builder.REPORT_ROLE,
        "status": "pass" if qualification_pass else "hold",
        "decision": (
            "matched_capacity_advantage_statistically_qualified"
            if qualification_pass
            else "matched_capacity_advantage_not_statistically_qualified"
        ),
        "source_kind": "capacity_full_300k",
        "claim_scope": "matched_full_300k_relative_generation_advantage",
        "source_anchor": source_reports["final_gate"],
        "execution_manifest": execution_manifest_identity,
        "uncertainty_report": uncertainty_identity,
        "evaluator_git": {
            "revision": "a" * 40,
            "branch": "analysis/matched-uncertainty",
            "tracked_dirty": False,
        },
        "quality_evidence": {
            "status": "pass",
            "absolute_quality_passed": True,
            "failed_checks": [],
            "decision": "large_scale_generation_ready",
        },
        "uncertainty_evidence": {
            "status": "pass" if qualification_pass else "hold",
            "decision": (
                "matched_relative_generation_advantage_supported"
                if qualification_pass
                else "matched_relative_generation_advantage_not_confirmed"
            ),
            "advantage_supported": qualification_pass,
            "stream_id": "capacity_full_terminal_00000000_00050000",
            "sample_count": 50_000,
            "checkpoint_step": 300_000,
            "fid_point_estimates": {
                "cofitok": 4.0,
                "dense_identity": 5.0,
            },
        },
        "claim_policy": {
            "matched_relative_generation_advantage_claim_allowed": qualification_pass,
            "formal_large_scale_generation_advantage_claim_allowed": qualification_pass,
            "broad_generation_superiority_claim_allowed": False,
            "sota_claim_allowed": False,
            "absolute_usability_claim_requires_source_gate_pass": True,
            "relative_advantage_claim_requires_uncertainty_pass": True,
        },
        "claim_boundary": copy.deepcopy(qualification_builder.CLAIM_BOUNDARY),
        "limitations": [],
    }
    qualification_path = (
        tmp_path / "uncertainty" / "statistical_claim_qualification.json"
    )
    qualification_identity = _write(qualification_path, qualification_report)
    return {
        "comparison_path": comparison_path,
        "comparison_identity": comparison_identity,
        "comparison_report": comparison_report,
        "qualification_path": qualification_path,
        "qualification_identity": qualification_identity,
        "qualification_report": qualification_report,
    }


def _build(case: dict[str, Any]) -> dict[str, Any]:
    return addendum.build_addendum(
        comparison_report_path=case["comparison_path"],
        expected_comparison_report_sha256=case["comparison_identity"]["sha256"],
        statistical_qualification_path=case["qualification_path"],
        expected_statistical_qualification_sha256=case["qualification_identity"][
            "sha256"
        ],
    )


def test_addendum_allows_only_bound_matched_fid_claim(tmp_path: Path) -> None:
    report = _build(_case(tmp_path, qualification_pass=True))

    assert report["status"] == "pass"
    assert report["claim_policy"][
        "matched_relative_fid_advantage_claim_allowed"
    ] is True
    assert report["claim_policy"]["cross_tier_numeric_ranking_allowed"] is False
    assert report["claim_policy"]["broad_generation_superiority_claim_allowed"] is False
    assert report["claim_policy"]["sota_claim_allowed"] is False
    assert report["claim_boundary"]["release_authorization_allowed"] is False
    assert report["matched_evidence"]["cofitok_relative_fid"] == pytest.approx(-0.2)
    assert "statistically supported lower FID" in report["claim_text"]


def test_addendum_preserves_uncertainty_hold(tmp_path: Path) -> None:
    report = _build(_case(tmp_path, qualification_pass=False))

    assert report["status"] == "hold"
    assert report["claim_policy"][
        "matched_relative_fid_advantage_claim_allowed"
    ] is False
    assert "does not statistically qualify" in report["claim_text"]


def test_addendum_rejects_another_final_gate(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["qualification_report"]["source_anchor"] = {
        "path": "/wrong/final_generation_gate.json",
        "bytes": 1,
        "sha256": "0" * 64,
    }
    case["qualification_identity"] = _write(
        case["qualification_path"],
        case["qualification_report"],
    )

    with pytest.raises(ValueError, match="another final gate"):
        _build(case)


def test_addendum_rejects_fid_drift(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["qualification_report"]["uncertainty_evidence"]["fid_point_estimates"][
        "cofitok"
    ] = 4.2
    case["qualification_identity"] = _write(
        case["qualification_path"],
        case["qualification_report"],
    )

    with pytest.raises(ValueError, match="FID estimates differ"):
        _build(case)


def test_addendum_rejects_authorizing_qualification_boundary(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["qualification_report"]["claim_boundary"][
        "release_authorization_allowed"
    ] = True
    case["qualification_identity"] = _write(
        case["qualification_path"],
        case["qualification_report"],
    )

    with pytest.raises(ValueError, match="qualification contract differs"):
        _build(case)


def test_addendum_rejects_quality_evidence_from_another_gate(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["qualification_report"]["quality_evidence"].update(
        {
            "status": "hold",
            "absolute_quality_passed": False,
            "decision": "hold",
        }
    )
    case["qualification_report"]["status"] = "hold"
    case["qualification_report"]["decision"] = (
        "matched_capacity_advantage_not_statistically_qualified"
    )
    case["qualification_report"]["claim_policy"][
        "matched_relative_generation_advantage_claim_allowed"
    ] = False
    case["qualification_report"]["claim_policy"][
        "formal_large_scale_generation_advantage_claim_allowed"
    ] = False
    case["qualification_identity"] = _write(
        case["qualification_path"],
        case["qualification_report"],
    )

    with pytest.raises(ValueError, match="quality evidence differs"):
        _build(case)


def test_addendum_rejects_cross_tier_ranking(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["comparison_report"]["comparison_policy"][
        "cross_tier_numeric_ranking_allowed"
    ] = True
    case["comparison_identity"] = _write(
        case["comparison_path"],
        case["comparison_report"],
    )

    with pytest.raises(ValueError, match="comparison policy is unsafe"):
        _build(case)


def test_addendum_rejects_changed_bound_input(tmp_path: Path) -> None:
    case = _case(tmp_path)
    original_sha256 = case["comparison_identity"]["sha256"]
    case["comparison_report"]["status"] = "hold"
    _write(case["comparison_path"], case["comparison_report"])

    with pytest.raises(ValueError, match="comparison report SHA256 differs"):
        addendum.build_addendum(
            comparison_report_path=case["comparison_path"],
            expected_comparison_report_sha256=original_sha256,
            statistical_qualification_path=case["qualification_path"],
            expected_statistical_qualification_sha256=case[
                "qualification_identity"
            ]["sha256"],
        )
