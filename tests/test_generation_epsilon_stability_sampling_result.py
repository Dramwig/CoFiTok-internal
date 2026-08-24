from __future__ import annotations

from typing import Any

import pytest

import cofitok.generation.quality_repair_result as result_module
from cofitok.generation import build_epsilon_stability_sampling_design


def _identity(name: str) -> dict[str, Any]:
    return {
        "path": f"/bound/{name}.json",
        "bytes": 1,
        "sha256": "a" * 64,
    }


def _metrics(
    *,
    fid: float,
    class_top1: float,
    class_top5: float,
    artifact_offset: float,
) -> dict[str, float]:
    return {
        "fid": fid,
        "inception_score": 1.0,
        "class_top1": class_top1,
        "class_top5": class_top5,
        "channel_saturation_fraction": 0.1 + artifact_offset,
        "total_variation": 0.2 + artifact_offset,
        "median_filter_residual_fraction": 0.3 + artifact_offset,
    }


def _install_normalizers(
    monkeypatch: pytest.MonkeyPatch,
    *,
    observations: list[dict[str, Any]],
) -> None:
    route_identity = _identity("route")
    user_authorization_identity = _identity("user_authorization")
    manifest_identity = _identity("manifest")
    real_identity = _identity("real_reference")
    real_set_identity = _identity("real_set")
    real_set = {
        "digest_schema": "test",
        "sha256": "b" * 64,
        "root": "/real",
        "image_count": 50_000,
    }
    real_metrics = {
        "channel_saturation_fraction": 0.1,
        "total_variation": 0.2,
        "median_filter_residual_fraction": 0.3,
    }

    monkeypatch.setattr(
        result_module,
        "_validate_execution_authorization",
        lambda *_args, **_kwargs: {
            "preparation_identity": _identity("preparation"),
            "terminal_route_receipt_identity": route_identity,
            "separate_execution_authorization_identity": (
                user_authorization_identity
            ),
        },
    )
    monkeypatch.setattr(
        result_module,
        "_validate_observation",
        lambda source, **_kwargs: dict(source),
    )
    monkeypatch.setattr(
        result_module,
        "_validate_observation_manifest",
        lambda *_args, **_kwargs: {"identity": manifest_identity},
    )
    monkeypatch.setattr(
        result_module,
        "_validate_real_reference",
        lambda *_args, **_kwargs: {
            "identity": real_identity,
            "real_set_identity": real_set_identity,
            "real_set": real_set,
            "sample_count": 1_000,
            "sample_set_sha256": "c" * 64,
            "source_reports": {"artifact_report": _identity("real_artifact")},
            "metrics": real_metrics,
        },
    )
    for row in observations:
        row["real_set"] = real_set


def _observations(
    design: dict[str, Any],
    *,
    passing_case: str | None,
) -> list[dict[str, Any]]:
    rows = []
    for case in design["cases"]:
        case_id = str(case["case_id"])
        for method in ("cofitok", "dense_identity"):
            if case_id == "legacy_terminal_hard_clip":
                metrics = _metrics(
                    fid=100.0,
                    class_top1=0.5,
                    class_top5=0.7,
                    artifact_offset=0.1,
                )
            elif case_id == passing_case:
                metrics = _metrics(
                    fid=90.0,
                    class_top1=0.5,
                    class_top5=0.7,
                    artifact_offset=0.05,
                )
            else:
                metrics = _metrics(
                    fid=110.0,
                    class_top1=0.49,
                    class_top5=0.69,
                    artifact_offset=0.2,
                )
            rows.append(
                {"case_id": case_id, "method": method, "metrics": metrics}
            )
    return rows


def test_sampling_result_selects_only_shared_non_authorizing_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = build_epsilon_stability_sampling_design()
    passing_case = "start975_unit_hard_clip"
    observations = _observations(design, passing_case=passing_case)
    _install_normalizers(monkeypatch, observations=observations)

    result = result_module.build_epsilon_stability_sampling_result(
        design=design,
        design_identity=_identity("design"),
        execution_authorization={},
        execution_authorization_identity=_identity("execution"),
        observation_sources=observations,
        observation_manifest={},
        real_artifact_reference={},
    )

    assert result["selected_case_id"] == passing_case
    assert result["selection_status"] == (
        "candidate_identified_for_separately_authorized_10k_confirmation"
    )
    assert result["generation_advantage_proven"] is False
    assert result["claim_boundary"] == {
        "one_thousand_sample_screening_only": True,
        "selected_case_is_not_confirmed": True,
        "independent_matched_10000_confirmation_required": True,
        "min_snr_training_tested": False,
        "cofitok_generation_advantage_claim_allowed": False,
    }
    assert all(
        value is False
        for value in result["authorization_boundary"].values()
    )


def test_sampling_result_does_not_select_a_regressing_case(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    design = build_epsilon_stability_sampling_design()
    observations = _observations(design, passing_case=None)
    _install_normalizers(monkeypatch, observations=observations)

    result = result_module.build_epsilon_stability_sampling_result(
        design=design,
        design_identity=_identity("design"),
        execution_authorization={},
        execution_authorization_identity=_identity("execution"),
        observation_sources=observations,
        observation_manifest={},
        real_artifact_reference={},
    )

    assert result["selected_case_id"] is None
    assert result["selection_status"] == (
        "no_shared_sampling_recovery_candidate"
    )
    assert result["generation_advantage_proven"] is False
