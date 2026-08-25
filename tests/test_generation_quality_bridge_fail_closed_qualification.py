from __future__ import annotations

from pathlib import Path

import pytest

from cofitok.generation.quality_bridge import RESULT_AUTHORIZATION_BOUNDARY
from scripts import (
    build_generation_quality_bridge_statistical_claim_qualification as builder,
)


REVISION = "a" * 40
BRANCH = "scale/test-quality"


def _manifest(result_identity: dict, *, absolute_passed: bool) -> dict:
    screen = {
        "status": "pass" if absolute_passed else "hold",
        "non_authorizing": True,
        "checks": [
            {"name": "absolute_quality", "passed": absolute_passed},
        ],
        "failed_checks": [] if absolute_passed else ["absolute_quality"],
    }
    return {
        "checkpoint_step": 100000,
        "stream_id": "quality_bridge_terminal_100k_00000000_00010000",
        "report": {
            "quality_bridge": {
                "result": result_identity,
                "git": {
                    "revision": REVISION,
                    "branch": BRANCH,
                    "tracked_dirty": False,
                },
                "authorization_boundary": RESULT_AUTHORIZATION_BOUNDARY,
                "quality_screen": screen,
            },
            "expected": {
                "matched_sampling": {
                    "start_index": 0,
                    "end_index_exclusive": 10000,
                    "sample_count": 10000,
                },
                "fid_point_estimates": {"cofitok": 115.0, "dense_identity": 123.0},
                "sample_sets": {
                    "cofitok": {"sha256": "b" * 64},
                    "dense_identity": {"sha256": "c" * 64},
                },
            },
        },
    }


def _patch_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    absolute_passed: bool,
) -> None:
    result_identity = {"path": "/quality.json", "bytes": 10, "sha256": "d" * 64}
    manifest_identity = {"path": "/manifest.json", "bytes": 11, "sha256": "e" * 64}
    manifest = _manifest(result_identity, absolute_passed=absolute_passed)
    monkeypatch.setattr(
        builder,
        "_bound_file",
        lambda path, **_kwargs: (
            result_identity if path.name == "quality.json" else manifest_identity
        ),
    )
    monkeypatch.setattr(
        builder.base_waiter,
        "validate_execution_manifest",
        lambda *_args, **_kwargs: manifest,
    )
    monkeypatch.setattr(
        builder,
        "read_json_object",
        lambda *_args, **_kwargs: {
            "git": {
                "revision": REVISION,
                "branch": BRANCH,
                "tracked_dirty": False,
            }
        },
    )
    monkeypatch.setattr(
        builder,
        "_quality_result_evidence",
        lambda *_args, **_kwargs: {
            "status": "pass" if absolute_passed else "hold",
            "absolute_quality_passed": absolute_passed,
            "failed_checks": [] if absolute_passed else ["absolute_quality"],
            "check_count": 1,
            "cofitok_fid": 115.0,
            "dense_identity_fid": 123.0,
            "screen": manifest["report"]["quality_bridge"]["quality_screen"],
        },
    )


def test_fail_closed_qualification_binds_manifest_without_uncertainty_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(monkeypatch, absolute_passed=False)

    report = builder.build_fail_closed_qualification(
        quality_result_path=Path("quality.json"),
        expected_quality_result_sha256="d" * 64,
        execution_manifest_path=Path("manifest.json"),
        expected_execution_manifest_sha256="e" * 64,
        expected_quality_revision=REVISION,
        expected_quality_branch=BRANCH,
        output_root=tmp_path,
    )

    assert report["status"] == "hold"
    assert report["uncertainty_report"] is None
    assert report["statistical_evidence_status"] == (
        "not_evaluated_prerequisite_failed"
    )
    assert report["uncertainty_evidence"]["status"] == "not_evaluated"
    assert report["claim_policy"]["matched_relative_fid_advantage_claim_allowed"] is False
    assert report["claim_policy"][
        "paired_uncertainty_execution_required_after_prerequisite_failure"
    ] is False
    assert "paired-KID report does not exist" in report["limitations"][0]


def test_fail_closed_qualification_rejects_a_passing_absolute_screen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_sources(monkeypatch, absolute_passed=True)

    with pytest.raises(ValueError, match="requires a failed absolute screen"):
        builder.build_fail_closed_qualification(
            quality_result_path=Path("quality.json"),
            expected_quality_result_sha256="d" * 64,
            execution_manifest_path=Path("manifest.json"),
            expected_execution_manifest_sha256="e" * 64,
            expected_quality_revision=REVISION,
            expected_quality_branch=BRANCH,
            output_root=tmp_path,
        )
