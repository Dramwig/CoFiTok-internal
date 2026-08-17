from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_RECIPE_STAGE,
    QUALITY_BRIDGE_RESULT_ROLE,
    QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
    QUALITY_BRIDGE_STEPS,
    QUALITY_BRIDGE_TERMINAL_SAMPLES,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.inference_replay import file_identity
from scripts import (
    build_generation_quality_bridge_statistical_claim_qualification as qualification,
)


QUALITY_REVISION = "a" * 40
QUALITY_BRANCH = "scale/generation-stability-quality-bridge-100k"
EVALUATOR_REVISION = "b" * 40
EVALUATOR_BRANCH = ""


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_identity(path)


def _screen(*, quality_pass: bool) -> dict[str, Any]:
    return {
        "status": "pass" if quality_pass else "hold",
        "non_authorizing": True,
        "checks": [
            {"name": "absolute_quality", "passed": quality_pass},
            {"name": "matched_quality", "passed": True},
        ],
        "failed_checks": [] if quality_pass else ["absolute_quality"],
    }


def _case(
    tmp_path: Path,
    *,
    quality_pass: bool,
    uncertainty_pass: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    output_root = (tmp_path / "uncertainty").resolve()
    screen = _screen(quality_pass=quality_pass)
    result_payload = {
        "schema_version": QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
        "role": QUALITY_BRIDGE_RESULT_ROLE,
        "status": "completed",
        "stage": QUALITY_BRIDGE_RECIPE_STAGE,
        "git": {
            "revision": QUALITY_REVISION,
            "branch": QUALITY_BRANCH,
            "tracked_dirty": False,
        },
        "terminal": {
            "methods": {
                "cofitok": {"fid": 4.0},
                "dense_identity": {"fid": 5.0},
            }
        },
        "quality_screen": screen,
        "authorization_boundary": RESULT_AUTHORIZATION_BOUNDARY,
    }
    result_path = tmp_path / "quality_bridge_result.json"
    result_identity = _write(result_path, result_payload)
    manifest_payload = {
        "quality_bridge": {
            "result": result_identity,
            "git": result_payload["git"],
            "quality_screen": screen,
            "authorization_boundary": RESULT_AUTHORIZATION_BOUNDARY,
        },
        "expected": {
            "matched_sampling": {
                "sample_count": QUALITY_BRIDGE_TERMINAL_SAMPLES,
            },
            "fid_point_estimates": {
                "cofitok": 4.0,
                "dense_identity": 5.0,
            },
        },
    }
    manifest_path = output_root / "reports" / "execution_manifest.json"
    manifest_identity = _write(manifest_path, manifest_payload)
    uncertainty_path = output_root / "terminal_100k" / "matched_uncertainty.json"
    uncertainty_identity = _write(uncertainty_path, {"status": "placeholder"})
    manifest_record = {
        "identity": manifest_identity,
        "stream_id": "full_data_quality_bridge_terminal_100k_00000000_00010000",
        "report": manifest_payload,
        "checkpoint_step": QUALITY_BRIDGE_STEPS,
    }
    monkeypatch.setattr(
        qualification.base_waiter,
        "validate_execution_manifest",
        lambda *_args, **_kwargs: manifest_record,
    )
    monkeypatch.setattr(
        qualification.base_waiter,
        "validate_audit_output",
        lambda *_args, **_kwargs: {
            "status": "pass" if uncertainty_pass else "hold",
            "decision": (
                "matched_relative_generation_advantage_supported"
                if uncertainty_pass
                else "matched_relative_generation_advantage_not_confirmed"
            ),
            "advantage_supported": uncertainty_pass,
            "claim_boundary": qualification.uncertainty_audit.CLAIM_BOUNDARY,
            "source": uncertainty_identity,
        },
    )
    return {
        "output_root": output_root,
        "result_path": result_path,
        "result_identity": result_identity,
        "result_payload": result_payload,
        "manifest_path": manifest_path,
        "manifest_identity": manifest_identity,
        "manifest_payload": manifest_payload,
        "uncertainty_path": uncertainty_path,
        "uncertainty_identity": uncertainty_identity,
    }


def _build(case: dict[str, Any]) -> dict[str, Any]:
    return qualification.build_qualification(
        quality_result_path=case["result_path"],
        expected_quality_result_sha256=case["result_identity"]["sha256"],
        execution_manifest_path=case["manifest_path"],
        expected_execution_manifest_sha256=case["manifest_identity"]["sha256"],
        uncertainty_report_path=case["uncertainty_path"],
        expected_uncertainty_report_sha256=case["uncertainty_identity"]["sha256"],
        expected_quality_revision=QUALITY_REVISION,
        expected_quality_branch=QUALITY_BRANCH,
        expected_evaluator_revision=EVALUATOR_REVISION,
        expected_evaluator_branch=EVALUATOR_BRANCH,
        output_root=case["output_root"],
    )


def test_qualification_requires_quality_and_uncertainty_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _build(
        _case(
            tmp_path,
            quality_pass=True,
            uncertainty_pass=True,
            monkeypatch=monkeypatch,
        )
    )

    assert report["status"] == "pass"
    assert report["claim_policy"][
        "matched_relative_fid_advantage_claim_allowed"
    ] is True
    assert report["claim_policy"][
        "formal_large_scale_generation_advantage_claim_allowed"
    ] is False
    assert report["claim_policy"]["broad_generation_superiority_claim_allowed"] is False
    assert report["claim_boundary"]["full_300k_launch_allowed"] is False
    assert "statistically supported lower FID" in report["claim_text"]


@pytest.mark.parametrize(
    ("quality_pass", "uncertainty_pass"),
    [(False, True), (True, False), (False, False)],
)
def test_qualification_holds_if_either_evidence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality_pass: bool,
    uncertainty_pass: bool,
) -> None:
    report = _build(
        _case(
            tmp_path,
            quality_pass=quality_pass,
            uncertainty_pass=uncertainty_pass,
            monkeypatch=monkeypatch,
        )
    )

    assert report["status"] == "hold"
    assert report["claim_policy"][
        "matched_relative_fid_advantage_claim_allowed"
    ] is False


def test_qualification_rejects_manifest_bound_to_another_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        quality_pass=True,
        uncertainty_pass=True,
        monkeypatch=monkeypatch,
    )
    case["manifest_payload"]["quality_bridge"]["result"] = {
        "path": "/wrong/quality_bridge_result.json",
        "bytes": 1,
        "sha256": "0" * 64,
    }

    with pytest.raises(ValueError, match="manifest result differs"):
        _build(case)


def test_qualification_rejects_manifest_screen_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        quality_pass=True,
        uncertainty_pass=True,
        monkeypatch=monkeypatch,
    )
    case["manifest_payload"]["quality_bridge"]["quality_screen"] = _screen(
        quality_pass=False
    )

    with pytest.raises(ValueError, match="manifest screen differs"):
        _build(case)


def test_qualification_rejects_fid_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        quality_pass=True,
        uncertainty_pass=True,
        monkeypatch=monkeypatch,
    )
    case["manifest_payload"]["expected"]["fid_point_estimates"]["cofitok"] = 4.2

    with pytest.raises(ValueError, match="scientific scope differs"):
        _build(case)
