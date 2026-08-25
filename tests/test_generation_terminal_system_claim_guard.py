from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_RECIPE_STAGE,
    QUALITY_BRIDGE_RESULT_ROLE,
    QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.inference_replay import file_identity
from scripts import audit_generation_matched_uncertainty as uncertainty_audit
from scripts import (
    build_generation_quality_bridge_statistical_claim_qualification as quality_claim,
)
from scripts import build_generation_statistical_claim_language_guard as statistical_guard
from scripts import build_generation_terminal_system_claim_guard as guard
from scripts import build_generation_requested_class_visual_audit as visual_audit


REVISION = "a" * 40
BRANCH = "scale/generation-stability-quality-bridge-100k"
SAMPLE_COUNT = 10_000
CHECKPOINT_STEP = 100_000
TEST_CLASSIFIER_WEIGHTS = b"terminal classifier weights\n"


@pytest.fixture(autouse=True)
def _test_classifier_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        guard,
        "CLASS_FIDELITY_CLASSIFIER_BYTES",
        len(TEST_CLASSIFIER_WEIGHTS),
    )
    monkeypatch.setattr(
        guard,
        "CLASS_FIDELITY_CLASSIFIER_SHA256",
        hashlib.sha256(TEST_CLASSIFIER_WEIGHTS).hexdigest(),
    )


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return file_identity(path)


def _quality_screen(*, passed: bool) -> dict[str, Any]:
    return {
        "status": "pass" if passed else "hold",
        "non_authorizing": True,
        "checks": [
            {"name": "absolute_quality", "passed": passed},
            {"name": "matched_quality", "passed": True},
        ],
        "failed_checks": [] if passed else ["absolute_quality"],
    }


def _uncertainty(tmp_path: Path, *, passed: bool) -> dict[str, Any]:
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
                    "sha256": "1" * 64,
                    "checkpoint_sha256": "2" * 64,
                    "checkpoint_step": CHECKPOINT_STEP,
                },
                "dense_identity": {
                    "sha256": "3" * 64,
                    "checkpoint_sha256": "4" * 64,
                    "checkpoint_step": CHECKPOINT_STEP,
                },
            },
        },
        "matched_sampling": {
            "start_index": 0,
            "end_index_exclusive": SAMPLE_COUNT,
            "sample_count": SAMPLE_COUNT,
        },
        "fid_point_estimates": {
            "cofitok": 4.0,
            "dense_identity": 5.0,
            "cofitok_relative_to_dense": -0.2,
            "direction_supports_cofitok_advantage": True,
        },
        "paired_block_kid": {
            "block_size": 500,
            "block_count": 20,
            "one_sided_exact_sign_test_p": 0.01 if passed else 0.2,
            "uncertainty_supports_cofitok_advantage": passed,
            "paired_block_bootstrap": {
                "mean": -0.02 if passed else 0.0,
                "ci_low": -0.03,
                "ci_high": -0.01 if passed else 0.01,
            },
        },
        "advantage_supported": passed,
    }
    path = tmp_path / "statistical" / "matched_uncertainty.json"
    return {"path": path, "payload": payload, "identity": _write(path, payload)}


def _statistical_sources(
    tmp_path: Path,
    *,
    quality_identity: dict[str, Any],
    quality_passed: bool,
    uncertainty_passed: bool,
) -> dict[str, Any]:
    uncertainty = _uncertainty(tmp_path, passed=uncertainty_passed)
    passed = quality_passed and uncertainty_passed
    qualification_payload = {
        "schema_version": quality_claim.REPORT_SCHEMA_VERSION,
        "role": quality_claim.REPORT_ROLE,
        "status": "pass" if passed else "hold",
        "decision": (
            "matched_quality_bridge_fid_advantage_statistically_qualified"
            if passed
            else "matched_quality_bridge_fid_advantage_not_statistically_qualified"
        ),
        "claim_scope": "matched_full_data_quality_bridge_100k_relative_fid_advantage",
        "quality_result": quality_identity,
        "quality_git": {
            "revision": REVISION,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "uncertainty_report": uncertainty["identity"],
        "quality_evidence": {
            "status": "pass" if quality_passed else "hold",
            "absolute_quality_passed": quality_passed,
        },
        "uncertainty_evidence": {
            "status": "pass" if uncertainty_passed else "hold",
            "advantage_supported": uncertainty_passed,
            "stream_id": "quality_bridge_terminal_100k_00000000_00010000",
            "fid_point_estimates": {"cofitok": 4.0, "dense_identity": 5.0},
        },
        "claim_policy": {
            "matched_relative_fid_advantage_claim_allowed": passed,
            "broad_generation_superiority_claim_allowed": False,
        },
    }
    qualification_path = tmp_path / "statistical" / "qualification.json"
    qualification_identity = _write(qualification_path, qualification_payload)
    guard_payload = statistical_guard.build_guard(
        source_kind="quality_bridge_100k",
        source_report_path=qualification_path,
        expected_source_report_sha256=qualification_identity["sha256"],
    )
    guard_path = tmp_path / "statistical" / "claim_guard.json"
    guard_identity = _write(guard_path, guard_payload)
    return {
        "uncertainty": uncertainty,
        "qualification_path": qualification_path,
        "qualification_payload": qualification_payload,
        "qualification_identity": qualification_identity,
        "guard_path": guard_path,
        "guard_payload": guard_payload,
        "guard_identity": guard_identity,
    }


def _runtime_guard(
    quality_root: Path,
    *,
    direct: bool,
) -> dict[str, Any]:
    fairness_path = quality_root / "reports" / "runtime_compute_fairness" / "final_report.json"
    pair_path = quality_root / "pair_monitor.json"
    fairness_identity = _write(fairness_path, {"status": "pass"})
    pair_identity = _write(
        pair_path,
        {
            "status": "pass",
            "stage": "complete",
            "git": {
                "revision": REVISION,
                "branch": BRANCH,
                "tracked_dirty": False,
            },
        },
    )
    decision = (
        "direct_runtime_outcome_comparison_allowed"
        if direct
        else "runtime_cost_claims_observational_only"
    )
    payload = {
        "schema_version": guard.RUNTIME_GUARD_SCHEMA_VERSION,
        "role": guard.RUNTIME_GUARD_ROLE,
        "status": "pass",
        "decision": decision,
        "sources": {
            "runtime_compute_fairness": fairness_identity,
            "terminal_pair_monitor": pair_identity,
        },
        "matched_training_contract": {
            "status": "verified",
            "training_git": {
                "revision": REVISION,
                "branch": BRANCH,
                "tracked_dirty": False,
            },
            "dataset": "imagenet_256",
            "output_root": quality_root.resolve().as_posix(),
            "target_steps_per_method": CHECKPOINT_STEP,
        },
        "claim_policy": {
            "runtime_configuration_parity_claim_allowed": True,
            "physical_recovery_compute_accounting_claim_allowed": True,
            "training_wall_clock_direct_comparison_allowed": direct,
            "training_throughput_direct_comparison_allowed": direct,
            "cost_efficiency_ranking_allowed": direct,
            "equal_wall_clock_budget_claim_allowed": False,
            "equal_gpu_hours_budget_claim_allowed": False,
            "equal_training_flops_budget_claim_allowed": False,
            "peak_vram_advantage_claim_allowed": False,
            "quality_or_generation_advantage_claim_allowed": False,
        },
        "claim_boundary": guard.RUNTIME_GUARD_CLAIM_BOUNDARY,
    }
    path = quality_root / "reports" / "runtime_compute_claim_guard_v1" / "runtime_compute_claim_guard.json"
    return {"path": path, "payload": payload, "identity": _write(path, payload)}


def _case(
    tmp_path: Path,
    *,
    quality_passed: bool = True,
    uncertainty_passed: bool = True,
    runtime_direct: bool = False,
) -> dict[str, Any]:
    quality_root = (tmp_path / "quality").resolve()
    cofitok_sampling_path = quality_root / "cofitok" / "sampling_report.json"
    dense_sampling_path = quality_root / "dense" / "sampling_report.json"
    cofitok_sampling_identity = _write(cofitok_sampling_path, {"status": "completed"})
    dense_sampling_identity = _write(dense_sampling_path, {"status": "completed"})
    sampling_protocol = {
        "start_index": 0,
        "num_samples": SAMPLE_COUNT,
        "num_classes": 1000,
        "image_shape": [3, 256, 256],
        "class_schedule": "balanced_modulo",
        "sampler": "ddim",
        "sample_steps": 100,
        "seed": 0,
    }
    screen = _quality_screen(passed=quality_passed)
    classifier_weights = quality_root / "classifier" / "resnet50-11ad3fa6.pth"
    classifier_weights.parent.mkdir(parents=True, exist_ok=True)
    classifier_weights.write_bytes(TEST_CLASSIFIER_WEIGHTS)
    classifier = {
        "name": guard.CLASS_FIDELITY_CLASSIFIER_NAME,
        "weights_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "weights_path": classifier_weights.resolve().as_posix(),
        "weights_bytes": guard.CLASS_FIDELITY_CLASSIFIER_BYTES,
        "weights_sha256": guard.CLASS_FIDELITY_CLASSIFIER_SHA256,
        "num_classes": 1000,
        "categories_sha256": guard.CLASS_FIDELITY_CATEGORIES_SHA256,
        "preprocessing": guard.CLASS_FIDELITY_PREPROCESSING,
    }
    cofitok_class_path = quality_root / "cofitok" / "class_fidelity.json"
    dense_class_path = quality_root / "dense" / "class_fidelity.json"
    cofitok_class_identity = _write(
        cofitok_class_path,
        {"classifier": classifier},
    )
    dense_class_identity = _write(
        dense_class_path,
        {"classifier": classifier},
    )
    qualification_path = quality_root / "reports" / "class_fidelity.json"
    qualification_identity = _write(
        qualification_path,
        {
            "classifier": classifier,
            "sources": {
                "cofitok": cofitok_class_identity,
                "dense_identity": dense_class_identity,
            },
        },
    )
    quality_payload = {
        "schema_version": QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
        "role": QUALITY_BRIDGE_RESULT_ROLE,
        "status": "completed",
        "stage": QUALITY_BRIDGE_RECIPE_STAGE,
        "git": {
            "revision": REVISION,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "quality_screen": screen,
        "authorization_boundary": RESULT_AUTHORIZATION_BOUNDARY,
        "source_reports": {
            "class_fidelity_qualification": qualification_identity,
            "cofitok_class_fidelity": cofitok_class_identity,
            "dense_class_fidelity": dense_class_identity,
        },
        "terminal": {
            "methods": {
                "cofitok": {
                    "fid": 4.0,
                    "checkpoint_sha256": "1" * 64,
                    "checkpoint_step": CHECKPOINT_STEP,
                    "sample_set_sha256": "2" * 64,
                    "sample_count": SAMPLE_COUNT,
                    "sampling": {**sampling_protocol, "prefix_budgets": [8]},
                },
                "dense_identity": {
                    "fid": 5.0,
                    "checkpoint_sha256": "3" * 64,
                    "checkpoint_step": CHECKPOINT_STEP,
                    "sample_set_sha256": "4" * 64,
                    "sample_count": SAMPLE_COUNT,
                    "sampling": {**sampling_protocol, "prefix_budgets": [1]},
                },
            },
            "physical_evidence": {
                "cofitok": {"sampling_report": cofitok_sampling_identity},
                "dense_identity": {"sampling_report": dense_sampling_identity},
            },
        },
    }
    quality_path = quality_root / "reports" / "quality_bridge_result.json"
    quality_identity = _write(quality_path, quality_payload)
    statistical = _statistical_sources(
        tmp_path,
        quality_identity=quality_identity,
        quality_passed=quality_passed,
        uncertainty_passed=uncertainty_passed,
    )

    visual_dir = quality_root / "reports" / "requested_class_visual_audit_terminal_100k_v1"
    panels = []
    for panel_index in range(2):
        panel_path = visual_dir / f"requested_class_panel_{panel_index:02d}.png"
        panel_path.parent.mkdir(parents=True, exist_ok=True)
        panel_path.write_bytes(f"panel-{panel_index}".encode("ascii"))
        panels.append(
            {
                **file_identity(panel_path),
                "indices": list(range(panel_index * 8, (panel_index + 1) * 8)),
                "row_order": ["real_validation", "cofitok", "dense_identity"],
                "columns": 8,
            }
        )
    visual_payload = {
        "schema_version": 1,
        "role": visual_audit.REPORT_ROLE,
        "status": "completed",
        "claim_boundary": visual_audit.CLAIM_BOUNDARY,
        "indices": list(range(16)),
        "sampling_protocol": sampling_protocol,
        "sources": {
            "cofitok": {
                "report": cofitok_sampling_identity,
                "sample_set": {"count": SAMPLE_COUNT, "sha256": "2" * 64},
                "checkpoint": {"step": CHECKPOINT_STEP, "sha256": "1" * 64},
            },
            "dense_identity": {
                "report": dense_sampling_identity,
                "sample_set": {"count": SAMPLE_COUNT, "sha256": "4" * 64},
                "checkpoint": {"step": CHECKPOINT_STEP, "sha256": "3" * 64},
            },
        },
        "panels": panels,
    }
    visual_path = visual_dir / visual_audit.REPORT_FILENAME
    visual_identity = _write(visual_path, visual_payload)
    visual_status_payload = {
        "schema_version": guard.VISUAL_WAITER_SCHEMA_VERSION,
        "role": guard.VISUAL_WAITER_ROLE,
        "status": "completed",
        "detail": "terminal_visual_audit_source_revalidated",
        "authorization_boundary": guard.VISUAL_WAITER_AUTHORIZATION_BOUNDARY,
        "quality_result": {
            "identity": quality_identity,
            "quality_screen": screen,
        },
        "visual_audit": visual_identity,
    }
    visual_status_path = quality_root / "reports" / "requested_class_visual_audit_waiter_status.json"
    visual_status_identity = _write(visual_status_path, visual_status_payload)
    runtime = _runtime_guard(quality_root, direct=runtime_direct)
    return {
        "quality_root": quality_root,
        "quality_path": quality_path,
        "quality_payload": quality_payload,
        "quality_identity": quality_identity,
        "statistical": statistical,
        "visual_path": visual_path,
        "visual_payload": visual_payload,
        "visual_status_path": visual_status_path,
        "visual_status_payload": visual_status_payload,
        "visual_status_identity": visual_status_identity,
        "runtime": runtime,
        "classifier_weights": classifier_weights,
    }


def _build(case: dict[str, Any]) -> dict[str, Any]:
    return guard.build_guard(
        quality_result_path=case["quality_path"],
        expected_quality_result_sha256=case["quality_identity"]["sha256"],
        statistical_claim_guard_path=case["statistical"]["guard_path"],
        expected_statistical_claim_guard_sha256=case["statistical"][
            "guard_identity"
        ]["sha256"],
        visual_audit_waiter_status_path=case["visual_status_path"],
        expected_visual_audit_waiter_status_sha256=case["visual_status_identity"][
            "sha256"
        ],
        runtime_claim_guard_path=case["runtime"]["path"],
        expected_runtime_claim_guard_sha256=case["runtime"]["identity"]["sha256"],
        quality_output_root=case["quality_root"],
    )


def test_pass_binds_quality_statistics_visuals_and_runtime(tmp_path: Path) -> None:
    report = _build(_case(tmp_path))

    assert report["status"] == "pass"
    assert report["claim_policy"]["matched_distribution_quality_claim_allowed"] is True
    assert report["claim_policy"]["absolute_quality_screen_pass_statement_allowed"] is True
    assert report["claim_policy"]["requested_class_visual_evidence_available"] is True
    assert report["claim_policy"]["requested_class_visual_evidence_is_quantitative"] is False
    assert (
        report["claim_policy"][
            "class_fidelity_classifier_physical_integrity_verified"
        ]
        is True
    )
    assert (
        report["evidence"]["class_fidelity_classifier_integrity"]["status"]
        == "verified"
    )
    assert report["claim_policy"]["runtime_direct_comparison_allowed"] is False
    assert report["claim_policy"]["absolute_usability_claim_allowed"] is False
    assert report["claim_policy"]["broad_generation_superiority_claim_allowed"] is False
    assert report["claim_policy"]["independent_replication_claim_allowed"] is False
    assert (
        report["claim_policy"]["multiple_independent_terminal_streams_claim_allowed"]
        is False
    )
    assert (
        report["evidence"]["matched_statistical_advantage"]["replication_scope"][
            "bound_terminal_stream_count"
        ]
        == 1
    )
    assert (
        report["evidence"]["matched_statistical_advantage"]["replication_scope"][
            "independent_replication_count"
        ]
        == 0
    )
    assert "not an independent replication" in report["claim_text"]


def test_holds_when_quality_passes_but_statistical_support_does_not(
    tmp_path: Path,
) -> None:
    report = _build(_case(tmp_path, quality_passed=True, uncertainty_passed=False))

    assert report["status"] == "hold"
    assert report["evidence"]["quality_screen"]["absolute_quality_passed"] is True
    assert report["claim_policy"]["matched_distribution_quality_claim_allowed"] is False


def test_holds_with_fail_closed_statistics_after_absolute_quality_failure(
    tmp_path: Path,
) -> None:
    case = _case(tmp_path, quality_passed=False, uncertainty_passed=False)
    qualification_payload = {
        "schema_version": quality_claim.REPORT_SCHEMA_VERSION,
        "role": quality_claim.REPORT_ROLE,
        "status": "hold",
        "decision": (
            "matched_quality_bridge_fid_advantage_not_statistically_qualified"
        ),
        "claim_scope": "matched_full_data_quality_bridge_100k_relative_fid_advantage",
        "statistical_evidence_status": "not_evaluated_prerequisite_failed",
        "quality_result": case["quality_identity"],
        "quality_git": {
            "revision": REVISION,
            "branch": BRANCH,
            "tracked_dirty": False,
        },
        "uncertainty_report": None,
        "quality_evidence": {"status": "hold", "absolute_quality_passed": False},
        "uncertainty_evidence": {
            "status": "not_evaluated",
            "advantage_supported": False,
            "stream_id": "quality_bridge_terminal_100k_00000000_00010000",
            "start_index": 0,
            "end_index_exclusive": SAMPLE_COUNT,
            "sample_count": SAMPLE_COUNT,
            "sample_sets": {
                "cofitok": {"sha256": "2" * 64},
                "dense_identity": {"sha256": "4" * 64},
            },
            "fid_point_estimates": {"cofitok": 4.0, "dense_identity": 5.0},
        },
        "claim_policy": {
            "matched_relative_fid_advantage_claim_allowed": False,
            "broad_generation_superiority_claim_allowed": False,
            "paired_uncertainty_execution_required_after_prerequisite_failure": False,
        },
    }
    qualification_path = tmp_path / "statistical" / "fail_closed.json"
    qualification_identity = _write(qualification_path, qualification_payload)
    statistical_payload = statistical_guard.build_guard(
        source_kind="quality_bridge_100k",
        source_report_path=qualification_path,
        expected_source_report_sha256=qualification_identity["sha256"],
    )
    statistical_path = tmp_path / "statistical" / "fail_closed_guard.json"
    statistical_identity = _write(statistical_path, statistical_payload)
    case["statistical"] = {
        "guard_path": statistical_path,
        "guard_identity": statistical_identity,
    }

    report = _build(case)

    assert report["status"] == "hold"
    assert report["evidence"]["matched_statistical_advantage"]["allowed"] is False
    assert report["evidence"]["matched_statistical_advantage"]["metric_roles"][
        "paired_block_kid"
    ]["statistical_significance_tested"] is False
    assert report["claim_policy"]["paired_kid_statistical_evidence_available"] is False
    assert report["claim_policy"]["matched_distribution_quality_claim_allowed"] is False
    assert report["evidence"]["matched_statistical_advantage"][
        "replication_scope"
    ]["interpretation"] == statistical_guard.UNPAIRED_BOUND_STREAM_INTERPRETATION
    assert "was not evaluated" in report["claim_text"]
    assert any(
        "no paired block-KID re-analysis was executed" in item
        for item in report["limitations"]
    )
    assert all(
        "FID and paired block-KID evidence bind" not in item
        for item in report["limitations"]
    )


def test_rejects_visual_sample_set_drift_from_quality_result(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["visual_payload"]["sources"]["cofitok"]["sample_set"]["sha256"] = "f" * 64
    visual_identity = _write(case["visual_path"], case["visual_payload"])
    case["visual_status_payload"]["visual_audit"] = visual_identity
    case["visual_status_identity"] = _write(
        case["visual_status_path"], case["visual_status_payload"]
    )

    with pytest.raises(ValueError, match="visual source differs"):
        _build(case)


def test_rejects_statistical_source_drift_after_guard_creation(tmp_path: Path) -> None:
    case = _case(tmp_path)
    qualification = copy.deepcopy(case["statistical"]["qualification_payload"])
    qualification["claim_scope"] = "changed"
    _write(case["statistical"]["qualification_path"], qualification)

    with pytest.raises(ValueError, match="changed after its guard was written"):
        _build(case)


def test_rejects_class_fidelity_classifier_weight_drift(tmp_path: Path) -> None:
    case = _case(tmp_path)
    case["classifier_weights"].write_bytes(b"drifted classifier weights\n")

    with pytest.raises(ValueError, match="physical identity differs"):
        _build(case)


def test_rejects_classifier_evidence_drift_during_guard_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(tmp_path)
    original = guard._verify_class_fidelity_classifier_sources
    calls = 0

    def drifting(report: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        evidence = original(report)
        if calls == 2:
            evidence = copy.deepcopy(evidence)
            evidence["physical_weights"]["sha256"] = "f" * 64
        return evidence

    monkeypatch.setattr(
        guard,
        "_verify_class_fidelity_classifier_sources",
        drifting,
    )
    with pytest.raises(ValueError, match="changed during replay"):
        _build(case)
    assert calls == 2


def test_rejects_runtime_direct_claim_without_matching_policy(tmp_path: Path) -> None:
    case = _case(tmp_path, runtime_direct=False)
    case["runtime"]["payload"]["claim_policy"][
        "training_wall_clock_direct_comparison_allowed"
    ] = True
    case["runtime"]["identity"] = _write(
        case["runtime"]["path"], case["runtime"]["payload"]
    )

    with pytest.raises(ValueError, match="runtime claim policy differs"):
        _build(case)
