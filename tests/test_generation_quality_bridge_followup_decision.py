from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from cofitok.generation.quality_bridge import RESULT_AUTHORIZATION_BOUNDARY
from cofitok.generation.quality_bridge_followup import (
    AUTHORIZATION_BOUNDARY,
    QUALITY_BRIDGE_EXECUTION_BRANCH,
    QUALITY_BRIDGE_EXECUTION_REVISION,
    build_quality_bridge_followup_decision,
)
from cofitok.reporting import file_sha256
from scripts import build_generation_quality_bridge_followup_decision as builder


ROOT = Path(__file__).resolve().parents[1]
DECISION_GIT = {
    "revision": "d" * 40,
    "branch": "scale/generation-quality-bridge-followup-decision-v1",
    "tracked_dirty": False,
}


def _identity(name: str, character: str) -> dict[str, object]:
    return {"path": f"/evidence/{name}.json", "bytes": 100, "sha256": character * 64}


def _checks(*failed: str) -> list[dict[str, object]]:
    names = (
        "cofitok_absolute_fid",
        "matched_fid_tolerance",
        "cofitok_precision_floor",
        "cofitok_recall_floor",
        "matched_precision_tolerance",
        "matched_recall_tolerance",
        "matched_endpoint_tolerance",
        "ordered_prefix_rank",
        "coarse_token_utilization",
        "restricted_synthesis_zero_token",
        "shuffle_mismatch",
        "class_fidelity",
    )
    failed_set = set(failed)
    return [
        {"name": name, "passed": name not in failed_set, "observed": 1.0}
        for name in names
    ]


def _terminal_method(character: str, *, fid: float) -> dict[str, object]:
    return {
        "fid": fid,
        "precision": 0.6,
        "recall": 0.4,
        "checkpoint": f"/checkpoints/{character}.pt",
        "checkpoint_sha256": character * 64,
        "checkpoint_integrity_manifest": f"/checkpoints/{character}.pt.integrity.json",
        "sample_set_sha256": character.upper() * 64,
    }


def _result(*failed: str) -> dict[str, object]:
    rows = _checks(*failed)
    milestone_50 = _identity("milestone_50000", "5")
    milestone_100 = _identity("milestone_100000", "6")
    return {
        "schema_version": 1,
        "status": "completed",
        "role": "stability_full_data_quality_bridge_result",
        "stage": "stability_quality_bridge",
        "git": {
            "revision": QUALITY_BRIDGE_EXECUTION_REVISION,
            "branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
            "tracked_dirty": False,
        },
        "source_reports": {
            "milestone_50000": milestone_50,
            "milestone_100000": milestone_100,
        },
        "milestones": {},
        "quality_screen": {
            "status": "hold" if failed else "pass",
            "non_authorizing": True,
            "checks": rows,
            "failed_checks": [row["name"] for row in rows if not row["passed"]],
        },
        "terminal": {
            "methods": {
                "cofitok": _terminal_method("a", fid=80.0),
                "dense_identity": _terminal_method("b", fid=82.0),
            },
            "class_fidelity": {"status": "pass", "valid": True},
        },
        "authorization_boundary": copy.deepcopy(RESULT_AUTHORIZATION_BOUNDARY),
    }


def _sampling(budget: int) -> dict[str, object]:
    return {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "sampler": "ddim",
        "num_samples": 2048,
        "sample_steps": 50,
        "prefix_budgets": [budget],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "sample_set_digest": {"sha256": "f" * 64},
    }


def _milestone(
    step: int,
    *,
    cofitok_fid: float,
    dense_fid: float,
    cofitok_is: float,
    dense_is: float,
    alerts: list[str] | None = None,
) -> dict[str, object]:
    def row(method: str, *, fid: float, score: float) -> dict[str, object]:
        character = "a" if method == "cofitok" else "b"
        budget = 8 if method == "cofitok" else 1
        return {
            "checkpoint": f"/checkpoints/{character}.pt",
            "checkpoint_sha256": character * 64,
            "checkpoint_integrity_manifest": f"/checkpoints/{character}.pt.integrity.json",
            "checkpoint_step": step,
            "weights": "ema",
            "sample_set_sha256": ("c" if method == "cofitok" else "d") * 64,
            "selected_prefix_budget": budget,
            "sample_count": 2048,
            "fid": fid,
            "inception_score": score,
            "endpoint_clean_mse": 0.1,
            "prefix_path_mse_auc": 0.2,
            "ordered_rank_by_path_auc": 1,
            "order_count": 6 if method == "cofitok" else 1,
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 2.0,
            "sampling": _sampling(budget),
        }

    warnings = list(alerts or [])
    return {
        "schema_version": 2,
        "status": "completed",
        "role": "training_quality_trend_only",
        "source_profile": "quality_bridge",
        "claim_policy": {"formal_generation_claim_allowed": False},
        "milestone_step": step,
        "expected_samples": 2048,
        "methods": {
            "cofitok": row("cofitok", fid=cofitok_fid, score=cofitok_is),
            "dense_identity": row("dense_identity", fid=dense_fid, score=dense_is),
        },
        "quality_alerts": warnings,
        "quality_alert": bool(warnings),
    }


def _decision(
    *failed: str,
    improving: bool = True,
    inception_improving: bool = True,
    alerts: list[str] | None = None,
) -> dict[str, object]:
    result = _result(*failed)
    milestone_50 = _milestone(
        50_000,
        cofitok_fid=100.0,
        dense_fid=105.0,
        cofitok_is=4.0,
        dense_is=3.8,
    )
    milestone_100 = _milestone(
        100_000,
        cofitok_fid=90.0 if improving else 101.0,
        dense_fid=95.0 if improving else 106.0,
        cofitok_is=4.5 if inception_improving else 3.9,
        dense_is=4.2 if inception_improving else 3.7,
        alerts=alerts,
    )
    identities = {
        50_000: result["source_reports"]["milestone_50000"],
        100_000: result["source_reports"]["milestone_100000"],
    }
    verifications = {
        step: {"status": "verified", "source_profile": "quality_bridge"}
        for step in (50_000, 100_000)
    }
    return build_quality_bridge_followup_decision(
        quality_bridge_result=result,
        quality_bridge_result_identity=_identity("result", "e"),
        milestones={50_000: milestone_50, 100_000: milestone_100},
        milestone_identities=identities,
        milestone_verifications=verifications,
        decision_git=DECISION_GIT,
    )


def test_followup_pass_routes_to_source_compatible_formal_gate() -> None:
    report = _decision()
    assert report["recommended_next_stage"]["id"] == (
        "build_source_compatible_formal_quality_gate"
    )
    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False


def test_absolute_hold_with_shared_improvement_routes_to_capacity_probe() -> None:
    report = _decision("cofitok_absolute_fid", "cofitok_recall_floor")
    assert report["milestone_trend"]["shared_quality_trend_strictly_improved"] is True
    assert report["recommended_next_stage"]["id"] == (
        "prepare_matched_250m_capacity_qualification_probe"
    )
    assert report["recommended_next_stage"]["execution_ready"] is False


@pytest.mark.parametrize(
    ("improving", "inception_improving"),
    ((False, True),),
)
def test_absolute_hold_without_consistent_shared_trend_routes_to_recipe_diagnosis(
    improving: bool,
    inception_improving: bool,
) -> None:
    report = _decision(
        "cofitok_absolute_fid",
        "cofitok_recall_floor",
        improving=improving,
        inception_improving=inception_improving,
    )
    assert report["recommended_next_stage"]["id"] == (
        "diagnose_terminal_distribution_support_then_recipe_probe"
    )


def test_inception_noise_does_not_veto_shared_fid_capacity_signal() -> None:
    report = _decision(
        "cofitok_absolute_fid",
        "cofitok_recall_floor",
        improving=True,
        inception_improving=False,
    )
    assert report["milestone_trend"]["shared_quality_trend_strictly_improved"] is True
    assert report["milestone_trend"][
        "shared_inception_score_corroborates_fid_trend"
    ] is False
    assert report["recommended_next_stage"]["id"] == (
        "prepare_matched_250m_capacity_qualification_probe"
    )


@pytest.mark.parametrize(
    ("failed", "expected"),
    (
        (
            ("ordered_prefix_rank",),
            "run_matched_factorization_mechanism_recovery_probe",
        ),
        (
            ("matched_fid_tolerance",),
            "run_matched_factorization_quality_regression_probe",
        ),
        (("class_fidelity",), "run_class_conditioning_fidelity_diagnostic"),
    ),
)
def test_scientific_failures_route_before_capacity(
    failed: tuple[str, ...],
    expected: str,
) -> None:
    report = _decision(*failed)
    assert report["recommended_next_stage"]["id"] == expected
    assert report["recommended_next_stage"]["full_300k_launch_allowed"] is False


def test_terminal_milestone_conflict_fails_over_to_reconciliation() -> None:
    report = _decision(
        alerts=["cofitok_fid_more_than_25pct_above_dense"]
    )
    assert report["recommended_next_stage"]["id"] == (
        "reconcile_100k_cross_protocol_evidence"
    )


def test_100k_milestone_must_use_terminal_checkpoint() -> None:
    result = _result("cofitok_absolute_fid")
    milestone_50 = _milestone(
        50_000,
        cofitok_fid=100.0,
        dense_fid=105.0,
        cofitok_is=4.0,
        dense_is=3.8,
    )
    milestone_100 = _milestone(
        100_000,
        cofitok_fid=90.0,
        dense_fid=95.0,
        cofitok_is=4.5,
        dense_is=4.2,
    )
    milestone_100["methods"]["cofitok"]["checkpoint_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="different checkpoints"):
        build_quality_bridge_followup_decision(
            quality_bridge_result=result,
            quality_bridge_result_identity=_identity("result", "e"),
            milestones={50_000: milestone_50, 100_000: milestone_100},
            milestone_identities={
                50_000: result["source_reports"]["milestone_50000"],
                100_000: result["source_reports"]["milestone_100000"],
            },
            milestone_verifications={
                step: {"status": "verified", "source_profile": "quality_bridge"}
                for step in (50_000, 100_000)
            },
            decision_git=DECISION_GIT,
        )


def test_result_replay_rejects_nonreproducible_terminal_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result()
    result["source_reports"] = {
        name: _identity(name, str(index % 9 + 1))
        for index, name in enumerate(builder._RESULT_SOURCE_ARGUMENTS)
    }
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    monkeypatch.setattr(builder, "build_from_args", lambda args: copy.deepcopy(result))
    actual, identity = builder.replay_quality_bridge_result(
        path,
        expected_sha256=file_sha256(path),
    )
    assert actual == result
    assert identity["sha256"] == file_sha256(path)

    changed = copy.deepcopy(result)
    changed["status"] = "tampered"
    monkeypatch.setattr(builder, "build_from_args", lambda args: changed)
    with pytest.raises(ValueError, match="not reproducible"):
        builder.replay_quality_bridge_result(
            path,
            expected_sha256=file_sha256(path),
        )


@pytest.mark.parametrize(
    "entrypoint",
    (
        "scripts/build_generation_quality_bridge_followup_decision.py",
        "scripts/verify_generation_quality_bridge_followup_decision.py",
    ),
)
def test_followup_entrypoints_import_under_runbook_pythonpath(entrypoint: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join((".", "src"))
    result = subprocess.run(
        [sys.executable, entrypoint, "--help"],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "quality" in result.stdout.lower()
    assert "bridge" in result.stdout.lower()


def test_followup_runbook_is_non_authorizing() -> None:
    runbook = (
        ROOT
        / "artifacts/runbooks/generation_quality_bridge_followup_decision_after_result.sh"
    ).read_text(encoding="utf-8")
    assert "build_generation_quality_bridge_followup_decision.py" in runbook
    assert "verify_generation_quality_bridge_followup_decision.py" in runbook
    assert "train_generation.py" not in runbook
    assert "full_matched_300k" not in runbook
