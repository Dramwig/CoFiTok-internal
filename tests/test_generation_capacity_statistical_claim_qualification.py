from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from cofitok.inference_replay import file_identity
from scripts import (
    build_generation_capacity_statistical_claim_qualification as qualification,
)


EVALUATOR_REVISION = "a" * 40
EVALUATOR_BRANCH = "analysis/matched-uncertainty"


def _write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return file_identity(path)


def _case(
    tmp_path: Path,
    *,
    source_kind: str,
    quality_pass: bool,
    uncertainty_pass: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    output_root = (tmp_path / "uncertainty").resolve()
    anchor_path = tmp_path / "source.json"
    if source_kind == "capacity_full_300k":
        anchor_payload = {
            "status": "pass" if quality_pass else "fail",
            "decision": "large_scale_generation_ready" if quality_pass else "hold",
        }
        checkpoint_step = 300_000
        sample_count = 50_000
    else:
        anchor_payload = {
            "quality_screen": {
                "status": "pass" if quality_pass else "hold",
                "failed_checks": [] if quality_pass else ["absolute_fid"],
            },
            "decision_support": {"recommended_next_stage": {"id": "review"}},
        }
        checkpoint_step = 100_000
        sample_count = 10_000
    anchor_identity = _write(anchor_path, anchor_payload)
    manifest_path = output_root / "reports" / "manifest.json"
    manifest_payload = {
        "capacity_source": {
            "kind": source_kind,
            "anchor": anchor_identity,
        },
        "expected": {
            "matched_sampling": {"sample_count": sample_count},
            "fid_point_estimates": {
                "cofitok": 4.0,
                "dense_identity": 5.0,
            },
        },
    }
    manifest_identity = _write(manifest_path, manifest_payload)
    uncertainty_path = output_root / "audit.json"
    uncertainty_identity = _write(uncertainty_path, {"status": "placeholder"})
    manifest_record = {
        "identity": manifest_identity,
        "stream_id": f"{source_kind}_stream",
        "report": manifest_payload,
        "checkpoint_step": checkpoint_step,
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
        "anchor_path": anchor_path,
        "anchor_identity": anchor_identity,
        "manifest_path": manifest_path,
        "manifest_identity": manifest_identity,
        "uncertainty_path": uncertainty_path,
        "uncertainty_identity": uncertainty_identity,
        "manifest_payload": manifest_payload,
    }


@pytest.mark.parametrize(
    ("source_kind", "formal_large_scale"),
    [
        ("capacity_completion_100k", False),
        ("capacity_full_300k", True),
    ],
)
def test_qualification_requires_quality_and_uncertainty_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    source_kind: str,
    formal_large_scale: bool,
) -> None:
    case = _case(
        tmp_path,
        source_kind=source_kind,
        quality_pass=True,
        uncertainty_pass=True,
        monkeypatch=monkeypatch,
    )
    report = qualification.build_qualification(
        source_kind=source_kind,
        source_anchor_path=case["anchor_path"],
        expected_source_anchor_sha256=case["anchor_identity"]["sha256"],
        execution_manifest_path=case["manifest_path"],
        expected_execution_manifest_sha256=case["manifest_identity"]["sha256"],
        uncertainty_report_path=case["uncertainty_path"],
        expected_uncertainty_report_sha256=case["uncertainty_identity"]["sha256"],
        expected_evaluator_revision=EVALUATOR_REVISION,
        expected_evaluator_branch=EVALUATOR_BRANCH,
        output_root=case["output_root"],
    )

    assert report["status"] == "pass"
    assert report["claim_policy"][
        "matched_relative_generation_advantage_claim_allowed"
    ] is True
    assert report["claim_policy"][
        "formal_large_scale_generation_advantage_claim_allowed"
    ] is formal_large_scale
    assert report["claim_policy"]["broad_generation_superiority_claim_allowed"] is False
    assert report["claim_boundary"]["release_authorization_allowed"] is False


@pytest.mark.parametrize(
    ("quality_pass", "uncertainty_pass"),
    [(False, True), (True, False), (False, False)],
)
def test_qualification_holds_if_either_required_evidence_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    quality_pass: bool,
    uncertainty_pass: bool,
) -> None:
    case = _case(
        tmp_path,
        source_kind="capacity_full_300k",
        quality_pass=quality_pass,
        uncertainty_pass=uncertainty_pass,
        monkeypatch=monkeypatch,
    )
    report = qualification.build_qualification(
        source_kind="capacity_full_300k",
        source_anchor_path=case["anchor_path"],
        expected_source_anchor_sha256=case["anchor_identity"]["sha256"],
        execution_manifest_path=case["manifest_path"],
        expected_execution_manifest_sha256=case["manifest_identity"]["sha256"],
        uncertainty_report_path=case["uncertainty_path"],
        expected_uncertainty_report_sha256=case["uncertainty_identity"]["sha256"],
        expected_evaluator_revision=EVALUATOR_REVISION,
        expected_evaluator_branch=EVALUATOR_BRANCH,
        output_root=case["output_root"],
    )
    assert report["status"] == "hold"
    assert report["claim_policy"][
        "matched_relative_generation_advantage_claim_allowed"
    ] is False


def test_qualification_rejects_manifest_bound_to_another_anchor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _case(
        tmp_path,
        source_kind="capacity_full_300k",
        quality_pass=True,
        uncertainty_pass=True,
        monkeypatch=monkeypatch,
    )
    case["manifest_payload"]["capacity_source"]["anchor"] = {
        "path": "wrong",
        "bytes": 1,
        "sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="manifest anchor differs"):
        qualification.build_qualification(
            source_kind="capacity_full_300k",
            source_anchor_path=case["anchor_path"],
            expected_source_anchor_sha256=case["anchor_identity"]["sha256"],
            execution_manifest_path=case["manifest_path"],
            expected_execution_manifest_sha256=case["manifest_identity"]["sha256"],
            uncertainty_report_path=case["uncertainty_path"],
            expected_uncertainty_report_sha256=case["uncertainty_identity"]["sha256"],
            expected_evaluator_revision=EVALUATOR_REVISION,
            expected_evaluator_branch=EVALUATOR_BRANCH,
            output_root=case["output_root"],
        )
