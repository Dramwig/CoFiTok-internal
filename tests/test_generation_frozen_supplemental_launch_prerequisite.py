from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from cofitok.reporting import file_sha256
from scripts import verify_generation_stability_frozen_supplemental as verifier


TRAINING_REVISION = "1" * 40
TRAINING_BRANCH = "scale/generation-stability-50k-preflight"
EVALUATION_REVISION = "2" * 40
EVALUATION_BRANCH = "scale/generation-stability-50k-posteval-v4"
SUPPLEMENTAL_REVISION = "3" * 40
SUPPLEMENTAL_BRANCH = "scale/generation-large-capacity"


def _git() -> dict:
    return {
        "revision": SUPPLEMENTAL_REVISION,
        "branch": SUPPLEMENTAL_BRANCH,
        "tracked_dirty": False,
    }


def _write(path: Path, payload: dict) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _gate() -> dict:
    return {
        "schema_version": 4,
        "stage": "scaling",
        "source_profile": "stability_scaling",
        "status": "pass",
        "decision": "promote_to_full_imagenet256",
        "provenance_contract": {
            "training_revision": TRAINING_REVISION,
            "training_branch": TRAINING_BRANCH,
            "evaluation_revision": EVALUATION_REVISION,
            "evaluation_branch": EVALUATION_BRANCH,
        },
    }


def _passing_graph(tmp_path: Path) -> dict[str, object]:
    gate_path = tmp_path / "promotion_gate.json"
    gate_identity = _write(gate_path, _gate())

    posteval_waiter = _write(
        tmp_path / "posteval_waiter.json",
        {"status": "pass", "detail": "formal_ema_postevaluation_completed"},
    )
    posteval_verification = {
        "schema_version": 1,
        "role": "generation_stability_frozen_posteval_verification",
        "status": "verified",
        "source": posteval_waiter,
        "verifier_git": _git(),
        "expected": {
            **_gate()["provenance_contract"],
            "formal_300k_allowed": False,
        },
        "claim_boundary": {
            "supplemental_execution_allowed": True,
            "replaces_postevaluation": False,
            "full_training_launch_allowed": False,
        },
    }
    posteval_verification_path = tmp_path / "posteval_verification.json"
    posteval_verification_identity = _write(
        posteval_verification_path,
        posteval_verification,
    )

    distribution_sources = {
        "promotion_gate": gate_identity,
        "cofitok_generation": _write(
            tmp_path / "nested" / "cofitok_generation.json",
            {"metrics": "cofitok"},
        ),
        "dense_generation": _write(
            tmp_path / "nested" / "dense_generation.json",
            {"metrics": "dense"},
        ),
    }
    distribution_checks = [
        {"name": name, "passed": True, "evidence": {"verified": True}}
        for name in sorted(verifier.EXPECTED_DISTRIBUTION_CHECKS)
    ]
    distribution = {
        "schema_version": 1,
        "role": "generation_stability_distribution_support_qualification",
        "status": "pass",
        "decision": "distribution_support_qualified",
        "thresholds": {
            "min_precision": 0.10,
            "min_recall": 0.10,
            "max_precision_regression": 0.05,
            "max_recall_regression": 0.05,
        },
        "sources": distribution_sources,
        "builder_git": _git(),
        "checks": distribution_checks,
        "decision_boundary": {
            "base_gate_passed": True,
            "distribution_support_passed": True,
            "base_gate_and_distribution_support_passed": True,
            "scaling_authorization_evaluated": False,
            "full_training_launch_allowed": False,
        },
        "claim_boundary": {
            "supplemental_non_authorizing": True,
            "replaces_generation_gate": False,
            "replaces_rollout_stability_qualification": False,
            "full_training_launch_allowed": False,
        },
    }
    distribution_path = tmp_path / "distribution_support.json"
    distribution_identity = _write(distribution_path, distribution)

    rollout_sources = {
        name: _write(
            tmp_path / "nested" / f"{name}.json",
            {"source": name},
        )
        for name in sorted(verifier.ROLLOUT_SOURCE_NAMES)
    }
    rollout = {
        "schema_version": 2,
        "status": "pass",
        "protocol": {
            "weights": "ema",
            "checkpoint_step": 50_000,
            "checkpoint_evaluated_images": 1_024,
            "checkpoint_timestep": 500,
            "rollout": {
                "num_images": 64,
                "batch_size": 8,
                "sample_steps": 100,
                "seed": 2029,
                "guidance_scale": 1.5,
                "guidance_rescale": 0.0,
                "teacher_guidance_scale": 1.0,
                "cfg_batch_mode": "batched",
                "clip_x0": True,
                "precision": "bf16",
            },
        },
        "identity": {
            "git_revision": TRAINING_REVISION,
            "training_git_revision": TRAINING_REVISION,
            "evaluation_git_revision": SUPPLEMENTAL_REVISION,
            "evaluation_git_branch": SUPPLEMENTAL_BRANCH,
            "cofitok_checkpoint_sha256": "a" * 64,
            "dense_checkpoint_sha256": "b" * 64,
        },
        "sources": rollout_sources,
        "metrics": {"endpoint_ratio": 1.0},
        "gates": {
            name: {"passed": True, "value": 0.0}
            for name in sorted(verifier.EXPECTED_ROLLOUT_GATES)
        },
        "pair_contract": {"valid": True, "issues": []},
    }
    rollout_path = tmp_path / "rollout_stability.json"
    rollout_identity = _write(rollout_path, rollout)

    combined = {
        "schema_version": 1,
        "role": "generation_stability_frozen_supplemental_qualification",
        "status": "pass",
        "decision": "supplemental_quality_complete",
        "sources": {
            "promotion_gate": gate_identity,
            "posteval_verification": posteval_verification_identity,
            "distribution_support": distribution_identity,
            "rollout_stability": rollout_identity,
        },
        "builder_git": _git(),
        "provenance_contract": {
            **_gate()["provenance_contract"],
            "supplemental_revision": SUPPLEMENTAL_REVISION,
            "supplemental_branch": SUPPLEMENTAL_BRANCH,
        },
        "checks": {
            "base_gate_passed": True,
            "distribution_support_passed": True,
            "ema_rollout_stability_passed": True,
            "all_supplemental_quality_checks_passed": True,
        },
        "claim_boundary": {
            "supplemental_non_authorizing": True,
            "replaces_generation_gate": False,
            "replaces_readiness": False,
            "scaling_authorization_evaluated": False,
            "full_training_launch_allowed": False,
        },
    }
    report_path = tmp_path / "supplemental_qualification.json"
    _write(report_path, combined)
    return {
        "report": combined,
        "report_path": report_path,
        "gate_path": gate_path,
        "rollout_source_path": Path(
            rollout_sources["cofitok_rollout"]["path"]
        ),
    }


def _verify(graph: dict[str, object]) -> dict:
    report_path = graph["report_path"]
    assert isinstance(report_path, Path)
    gate_path = graph["gate_path"]
    assert isinstance(gate_path, Path)
    report = graph["report"]
    assert isinstance(report, dict)
    return verifier.verify_frozen_supplemental_report(
        report,
        report_path=report_path,
        expected_report_sha256=file_sha256(report_path),
        promotion_gate_path=gate_path,
    )


def test_launch_prerequisite_rehashes_full_quality_evidence_graph(
    tmp_path: Path,
) -> None:
    evidence = _verify(_passing_graph(tmp_path))

    assert evidence["required_for_full_training_launch"] is True
    assert evidence["supplemental_non_authorizing"] is True
    assert evidence["full_training_launch_allowed"] is False
    assert all(evidence["checks"].values())


def test_launch_prerequisite_rejects_held_quality_report(tmp_path: Path) -> None:
    graph = _passing_graph(tmp_path)
    held = deepcopy(graph["report"])
    assert isinstance(held, dict)
    held["status"] = "hold"
    held["decision"] = "hold"
    held["checks"]["all_supplemental_quality_checks_passed"] = False
    report_path = graph["report_path"]
    assert isinstance(report_path, Path)
    _write(report_path, held)
    graph["report"] = held

    with pytest.raises(ValueError, match="quality prerequisite did not pass"):
        _verify(graph)


def test_launch_prerequisite_rejects_nested_source_drift(tmp_path: Path) -> None:
    graph = _passing_graph(tmp_path)
    rollout_source = graph["rollout_source_path"]
    assert isinstance(rollout_source, Path)
    rollout_source.write_text('{"changed":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="EMA rollout-stability cofitok_rollout source changed"):
        _verify(graph)


def test_launch_prerequisite_requires_same_physical_promotion_gate(
    tmp_path: Path,
) -> None:
    graph = _passing_graph(tmp_path)
    alternate = tmp_path / "alternate_gate.json"
    _write(alternate, _gate())
    graph["gate_path"] = alternate

    with pytest.raises(ValueError, match="different promotion gate"):
        _verify(graph)
