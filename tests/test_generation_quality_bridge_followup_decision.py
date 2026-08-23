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


def _training_exposure(result: dict[str, object]) -> dict[str, object]:
    train_images = 1_281_167
    epochs = 6_400_000 / train_images
    reference_epochs = 3_200_000 / 128_161

    def row() -> dict[str, object]:
        return {
            "schema_version": 1,
            "status": "complete",
            "dataset": "imagenet_256",
            "dataset_identity_sha256": "7" * 64,
            "train_image_count": train_images,
            "target_steps": 100_000,
            "completed_steps": 100_000,
            "training_complete": True,
            "micro_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "effective_batch_size": 64,
            "samples_seen": 6_400_000,
            "expected_samples_seen_at_completed_step": 6_400_000,
            "target_samples_seen": 6_400_000,
            "completed_fraction": 1.0,
            "completed_equivalent_epochs": epochs,
            "target_equivalent_epochs": epochs,
            "git": {
                "revision": QUALITY_BRIDGE_EXECUTION_REVISION,
                "branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
                "dirty": False,
            },
        }

    matched_comparison = {
        "same_dataset": True,
        "same_dataset_identity": True,
        "same_effective_batch_size": True,
        "same_completed_steps": True,
        "same_images_seen": True,
        "same_equivalent_epochs": True,
        "same_dataset_normalized_exposure": True,
        "step_budget_directly_comparable": True,
        "image_budget_directly_comparable": True,
        "dataset_normalized_budget_directly_comparable": True,
        "quality_metric_comparison_allowed": False,
    }
    terminal_methods = result["terminal"]["methods"]
    return {
        "schema_version": 1,
        "exposure_schema_version": 1,
        "status": "pass",
        "role": "generation_training_exposure_audit",
        "sources": {
            "cofitok": _identity("exposure_cofitok_training", "1"),
            "dense_identity": _identity("exposure_dense_training", "2"),
        },
        "rows": {"cofitok": row(), "dense_identity": row()},
        "comparison": matched_comparison,
        "quality_bridge_plan": {
            "source": _identity("exposure_preparation", "3"),
            "training_exposure": {
                "bridge": {
                    "dataset": "imagenet_256",
                    "effective_batch_size": 64,
                    "equivalent_epochs": epochs,
                    "images_seen_per_method": 6_400_000,
                    "steps": 100_000,
                    "train_image_count": train_images,
                },
                "source": {
                    "dataset": "imagenet_256_10pct",
                    "effective_batch_size": 64,
                    "equivalent_epochs": reference_epochs,
                    "images_seen_per_method": 3_200_000,
                    "steps": 50_000,
                    "train_image_count": 128_161,
                },
                "comparison": {
                    "bridge_to_source_equivalent_epochs_ratio": (
                        epochs / reference_epochs
                    ),
                    "same_step_count_means_same_exposure": False,
                    "cross_dataset_quality_comparison_requires_explicit_protocol_binding": True,
                },
            },
        },
        "milestone_binding": {
            "source": _identity("exposure_milestone", "4"),
            "expected_step": 100_000,
            "source_profile": "quality_bridge",
            "matched_training_exposure_verified": True,
        },
        "terminal_binding": {
            "terminal_result": _identity("exposure_terminal_result", "e"),
            "verified_execution_status": _identity("exposure_execution", "5"),
            "checkpoint_binding": {
                method: {
                    "step": 100_000,
                    "checkpoint_sha256": terminal_methods[method]["checkpoint_sha256"],
                }
                for method in ("cofitok", "dense_identity")
            },
            "quality_screen": copy.deepcopy(result["quality_screen"]),
            "terminal_result_binding_verified": True,
            "active_runbook_verification_completed": True,
        },
        "claim_boundary": {
            "training_scale_claim_allowed": True,
            "sample_quality_claim_allowed": False,
            "method_quality_ranking_allowed": False,
            "formal_gate_substitute": False,
            "milestone_quality_diagnostic_allowed": True,
            "formal_generation_claim_allowed": False,
            "terminal_training_exposure_binding_allowed": True,
            "terminal_quality_result_context_allowed": True,
        },
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
        training_exposure_report=_training_exposure(result),
        training_exposure_report_identity=_identity("exposure", "8"),
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
    assert report["training_exposure"]["full_data_equivalent_epochs"] == pytest.approx(
        4.995445558619602
    )
    assert (
        report["training_exposure"]["insufficient_exposure_is_live_hypothesis"] is True
    )
    assert (
        report["recommended_next_stage"]["fallback_if_capacity_not_supported"]["id"]
        == "prepare_matched_training_exposure_qualification"
    )


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
    assert (
        report["milestone_trend"]["shared_inception_score_corroborates_fid_trend"]
        is False
    )
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


def test_class_conditioning_route_requires_class_only_failure() -> None:
    report = _decision("class_fidelity", "cofitok_absolute_fid")

    assert report["recommended_next_stage"]["id"] == (
        "extend_followup_policy_before_execution"
    )
    assert report["recommended_next_stage"]["category"] == (
        "unclassified_fail_closed"
    )
    assert report["recommended_next_stage"]["execution_ready"] is False
    assert report["recommended_next_stage"]["gpu_execution_allowed"] is False
    assert report["recommended_next_stage"]["full_300k_launch_allowed"] is False


@pytest.mark.parametrize(
    "failed",
    (
        ("matched_fid_tolerance", "class_fidelity"),
        ("matched_fid_tolerance", "cofitok_absolute_fid"),
        ("ordered_prefix_rank", "matched_fid_tolerance"),
        ("ordered_prefix_rank", "class_fidelity"),
        ("ordered_prefix_rank", "cofitok_absolute_fid"),
    ),
)
def test_cross_category_failure_sets_fail_closed(
    failed: tuple[str, ...],
) -> None:
    report = _decision(*failed)

    assert report["recommended_next_stage"]["id"] == (
        "extend_followup_policy_before_execution"
    )
    assert report["recommended_next_stage"]["category"] == (
        "unclassified_fail_closed"
    )
    assert report["recommended_next_stage"]["execution_ready"] is False
    assert report["recommended_next_stage"]["gpu_execution_allowed"] is False
    assert report["recommended_next_stage"]["full_300k_launch_allowed"] is False


def test_terminal_milestone_conflict_fails_over_to_reconciliation() -> None:
    report = _decision(alerts=["cofitok_fid_more_than_25pct_above_dense"])
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
            training_exposure_report=_training_exposure(result),
            training_exposure_report_identity=_identity("exposure", "8"),
            decision_git=DECISION_GIT,
        )


def test_terminal_training_exposure_must_match_both_methods() -> None:
    result = _result("cofitok_absolute_fid")
    exposure = _training_exposure(result)
    exposure["rows"]["dense_identity"]["samples_seen"] -= 64
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
    with pytest.raises(ValueError, match="dense_identity differs"):
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
            training_exposure_report=exposure,
            training_exposure_report_identity=_identity("exposure", "8"),
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


def test_result_replay_accepts_physically_bound_authoritative_verifier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _result("cofitok_absolute_fid")
    result["source_reports"] = {}
    for name in sorted(builder._RESULT_SOURCE_ARGUMENTS):
        source_path = tmp_path / f"terminal_source_{name}.json"
        source_path.write_text(json.dumps({"name": name}), encoding="utf-8")
        result["source_reports"][name] = builder.file_identity(source_path)
    canonical = tmp_path / "quality_bridge_result.json"
    snapshot = tmp_path / "snapshot_quality_bridge_result.json"
    canonical.write_text(json.dumps(result), encoding="utf-8")
    snapshot.write_text(json.dumps(result), encoding="utf-8")

    sources = {}
    for name in ("verifier_source", "builder_source", "python"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        sources[name] = builder.file_identity(path)
    snapshot_identity = builder.file_identity(snapshot)
    verification = {
        "schema_version": 1,
        "role": "generation_quality_bridge_authoritative_terminal_verification",
        "status": "verified",
        "quality_project": {
            "revision": QUALITY_BRIDGE_EXECUTION_REVISION,
            "tree": "f" * 40,
            "branch": QUALITY_BRIDGE_EXECUTION_BRANCH,
            "tracked_dirty": False,
            "path": str(tmp_path),
        },
        **sources,
        "terminal_result": snapshot_identity,
        "verifier_output": {
            "status": "verified",
            "result": snapshot_identity,
            "quality_screen": result["quality_screen"],
            "authorization_boundary": result["authorization_boundary"],
        },
        "execution_policy": {
            "cuda_visible_devices": "-1",
            "omp_num_threads": "1",
            "mkl_num_threads": "1",
            "gpu_use_allowed": False,
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
        },
    }
    monkeypatch.setattr(
        builder,
        "build_from_args",
        lambda unused: (_ for _ in ()).throw(AssertionError("legacy replay used")),
    )
    monkeypatch.setattr(
        builder,
        "_authoritative_quality_project_identity",
        lambda unused: copy.deepcopy(verification["quality_project"]),
    )

    actual, identity = builder.replay_quality_bridge_result(
        canonical,
        expected_sha256=file_sha256(canonical),
        authoritative_terminal_verification=verification,
    )

    assert actual == result
    assert identity["sha256"] == snapshot_identity["sha256"]

    Path(sources["verifier_source"]["path"]).write_text(
        "drifted", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="verifier_source identity differs"):
        builder.replay_quality_bridge_result(
            canonical,
            expected_sha256=file_sha256(canonical),
            authoritative_terminal_verification=verification,
        )


def test_training_exposure_replay_reopens_every_bound_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exposure = _training_exposure(_result("cofitok_absolute_fid"))

    def bind(name: str, payload: dict[str, object]) -> dict[str, object]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return builder.file_identity(path)

    exposure["sources"] = {
        "cofitok": bind("cofitok_training", {"method": "cofitok"}),
        "dense_identity": bind("dense_training", {"method": "dense_identity"}),
    }
    exposure["quality_bridge_plan"]["source"] = bind(
        "preparation", {"role": "preparation"}
    )
    exposure["milestone_binding"]["source"] = bind("milestone", {"step": 100_000})
    exposure["terminal_binding"]["terminal_result"] = bind(
        "terminal_result", {"role": "quality_result"}
    )
    exposure["terminal_binding"]["verified_execution_status"] = bind(
        "execution_status", {"status": "completed"}
    )
    exposure_path = tmp_path / "training_exposure_report.json"
    exposure_path.write_text(json.dumps(exposure), encoding="utf-8")
    monkeypatch.setattr(
        builder,
        "build_training_exposure_report",
        lambda *args, **kwargs: copy.deepcopy(exposure),
    )

    actual, identity = builder.replay_training_exposure_report(
        exposure_path,
        expected_sha256=file_sha256(exposure_path),
    )
    assert actual == exposure
    assert identity["sha256"] == file_sha256(exposure_path)

    bound_training = Path(exposure["sources"]["cofitok"]["path"])
    bound_training.write_text('{"tampered": true}', encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        builder.replay_training_exposure_report(
            exposure_path,
            expected_sha256=file_sha256(exposure_path),
        )


def test_training_exposure_replay_forwards_embedded_authoritative_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exposure = _training_exposure(_result("cofitok_absolute_fid"))

    def bind(name: str, payload: dict[str, object]) -> dict[str, object]:
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        return builder.file_identity(path)

    exposure["sources"] = {
        "cofitok": bind("cofitok_training", {"method": "cofitok"}),
        "dense_identity": bind("dense_training", {"method": "dense_identity"}),
    }
    exposure["quality_bridge_plan"]["source"] = bind(
        "preparation", {"role": "preparation"}
    )
    exposure["milestone_binding"]["source"] = bind(
        "milestone", {"step": 100_000}
    )
    exposure["terminal_binding"]["terminal_result"] = bind(
        "terminal_result", {"role": "quality_result"}
    )
    exposure["terminal_binding"]["verified_execution_status"] = bind(
        "execution_status", {"status": "completed"}
    )
    verification = {"status": "verified", "binding": "authoritative"}
    exposure["terminal_binding"][
        "authoritative_terminal_verification"
    ] = verification
    exposure_path = tmp_path / "training_exposure_report.json"
    exposure_path.write_text(json.dumps(exposure), encoding="utf-8")
    captured = {}

    def fake_build(*args, **kwargs):
        captured.update(kwargs)
        return copy.deepcopy(exposure)

    monkeypatch.setattr(builder, "build_training_exposure_report", fake_build)

    builder.replay_training_exposure_report(
        exposure_path,
        expected_sha256=file_sha256(exposure_path),
    )

    assert captured["quality_bridge_terminal_verification"] == verification


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
    assert "EXPOSURE=${EXPOSURE:?" in runbook
    assert "DECISION=${DECISION:?" in runbook
    assert "LOCK=${LOCK:?" in runbook
    assert "EXPECTED_RESULT_SHA256=${EXPECTED_RESULT_SHA256:?" in runbook
    assert "EXPECTED_EXPOSURE_SHA256=${EXPECTED_EXPOSURE_SHA256:?" in runbook
    assert 'mkdir -p "$(dirname "$DECISION")" "$(dirname "$LOCK")"' in runbook
