from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.conditioning_ranking_probe import (
    STANDING_AUTHORIZATION_INTERPRETATION,
    STANDING_AUTHORIZATION_ROLE,
    STANDING_AUTHORIZATION_SAFETY_BOUNDARIES,
    STANDING_AUTHORIZATION_TEXT,
)
from cofitok.generation.conditioning_ranking_training_confirmation import (
    CLAIM_BOUNDARY,
    EXECUTION_BOUNDARY,
    EXPECTED_OUTPUT_ROOT,
    SOURCE_CLAIM_BOUNDARY,
    SOURCE_DECISION,
    SOURCE_METHOD_GATES,
    SOURCE_SAMPLING_GIT,
    SOURCE_SAMPLING_OUTPUT_ROOT,
    SOURCE_SAMPLING_ROLE,
    SOURCE_SAMPLING_STAGE,
    build_training_confirmation_preparation,
    training_confirmation_contract,
    validate_shared_sampling_recovery,
)
from scripts import prepare_generation_conditioning_ranking_training_confirmation as prepare_cli


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "control_cofitok": (
        "configs/generation/imagenet256_10pct_stability_rgbtail3_"
        "rollout_x0_u2_ema_teacher_k8_probe5k.json"
    ),
    "control_dense_identity": (
        "configs/generation/imagenet256_10pct_stability_"
        "rollout_x0_u2_ema_teacher_dense_probe5k.json"
    ),
    "ranked_cofitok": (
        "configs/generation/imagenet256_10pct_stability_rgbtail3_"
        "rollout_x0_u2_ema_teacher_classrank_k8_confirm5k.json"
    ),
    "ranked_dense_identity": (
        "configs/generation/imagenet256_10pct_stability_"
        "rollout_x0_u2_ema_teacher_classrank_dense_confirm5k.json"
    ),
}


def _configs() -> dict[str, dict]:
    return {
        name: config_to_dict(load_config(PROJECT_ROOT / relative))
        for name, relative in CONFIGS.items()
    }


def _identity(name: str, character: str = "a") -> dict:
    return {"path": f"/evidence/{name}.json", "bytes": 123, "sha256": character * 64}


def _sampling_validation() -> dict:
    gates = {name: True for name in SOURCE_METHOD_GATES}
    return {
        "schema_version": 1,
        "role": SOURCE_SAMPLING_ROLE,
        "status": "completed",
        "stage": SOURCE_SAMPLING_STAGE,
        "output_root": SOURCE_SAMPLING_OUTPUT_ROOT,
        "git": copy.deepcopy(SOURCE_SAMPLING_GIT),
        "sampling_contract": {
            "sample_count_per_arm": 5000,
            "sample_run_name": "samples_5000_ddim50_cfg15",
            "methods": {"cofitok": {}, "dense_identity": {}},
        },
        "methods": {
            "cofitok": {"pass": True, "gates": copy.deepcopy(gates)},
            "dense_identity": {"pass": True, "gates": copy.deepcopy(gates)},
        },
        "decision": copy.deepcopy(SOURCE_DECISION),
        "claim_boundary": copy.deepcopy(SOURCE_CLAIM_BOUNDARY),
    }


def _standing_authorization() -> dict:
    return {
        "schema_version": 1,
        "role": STANDING_AUTHORIZATION_ROLE,
        "status": "active",
        "instruction": {
            "language": "zh-CN",
            "exact_text": STANDING_AUTHORIZATION_TEXT,
            "interpretation": STANDING_AUTHORIZATION_INTERPRETATION,
            "received_at": "2026-08-19T00:00:00+08:00",
        },
        "preserved_safety_boundaries": copy.deepcopy(
            STANDING_AUTHORIZATION_SAFETY_BOUNDARIES
        ),
    }


def test_four_arm_5k_configs_match_outside_ranking_fields() -> None:
    configs = _configs()
    contract = training_confirmation_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense=configs["ranked_dense_identity"],
    )

    assert contract["status"] == "pass"
    assert contract["valid"] is True
    assert contract["issues"] == []
    assert contract["execution_boundary"] == EXECUTION_BOUNDARY
    assert contract["claim_boundary"] == CLAIM_BOUNDARY


def test_confirmation_rejects_drift_outside_ranking_fields() -> None:
    configs = _configs()
    configs["ranked_cofitok"]["optimization"]["learning_rate"] = 2e-4

    contract = training_confirmation_contract(
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense=configs["ranked_dense_identity"],
    )

    assert contract["valid"] is False
    assert any("outside ranking fields" in issue for issue in contract["issues"])


def test_sampling_source_requires_shared_exact_pass() -> None:
    report = _sampling_validation()
    assert validate_shared_sampling_recovery(report) == report

    asymmetric = copy.deepcopy(report)
    asymmetric["decision"]["method_passes"]["cofitok"] = False
    with pytest.raises(ValueError, match="did not select"):
        validate_shared_sampling_recovery(asymmetric)

    authorizing = copy.deepcopy(report)
    authorizing["claim_boundary"]["authorizes_training"] = True
    with pytest.raises(ValueError, match="did not select"):
        validate_shared_sampling_recovery(authorizing)

    non_boolean_gate = copy.deepcopy(report)
    non_boolean_gate["methods"]["cofitok"]["gates"]["exact_provenance"] = 1
    with pytest.raises(ValueError, match="evidence is malformed"):
        validate_shared_sampling_recovery(non_boolean_gate)


def test_preparation_is_non_authorizing_and_source_bound() -> None:
    configs = _configs()
    branch = "scale/generation-label-ranking-5k-training-confirmation-v1"
    revision = "b" * 40
    report = build_training_confirmation_preparation(
        sampling_validation=_sampling_validation(),
        sampling_validation_identity=_identity("sampling", "c"),
        standing_authorization=_standing_authorization(),
        standing_authorization_identity=_identity("standing", "d"),
        control_cofitok=configs["control_cofitok"],
        control_dense=configs["control_dense_identity"],
        ranked_cofitok=configs["ranked_cofitok"],
        ranked_dense=configs["ranked_dense_identity"],
        config_identities={name: _identity(name, "e") for name in CONFIGS},
        parameter_counts={
            "control_cofitok": 100,
            "ranked_cofitok": 100,
            "control_dense_identity": 90,
            "ranked_dense_identity": 90,
        },
        builder_git={
            "revision": revision,
            "branch": branch,
            "tracked_dirty": False,
        },
        expected_revision=revision,
        expected_branch=branch,
        expected_output_root=EXPECTED_OUTPUT_ROOT,
    )

    assert report["status"] == "pass"
    assert report["source_sampling_decision"] == SOURCE_DECISION
    assert report["gpu_execution_authorized"] is False
    assert report["execution_boundary"]["training_allowed"] is False
    assert report["execution_boundary"]["resume_from_1k_checkpoint_allowed"] is False
    assert all(
        report["claim_boundary"][field] is False
        for field in (
            "authorizes_training",
            "authorizes_sampling",
            "authorizes_full_100k_or_300k",
            "authorizes_release",
        )
    )


def test_preparation_rejects_parameter_or_output_drift() -> None:
    configs = _configs()
    kwargs = {
        "sampling_validation": _sampling_validation(),
        "sampling_validation_identity": _identity("sampling", "c"),
        "standing_authorization": _standing_authorization(),
        "standing_authorization_identity": _identity("standing", "d"),
        "control_cofitok": configs["control_cofitok"],
        "control_dense": configs["control_dense_identity"],
        "ranked_cofitok": configs["ranked_cofitok"],
        "ranked_dense": configs["ranked_dense_identity"],
        "config_identities": {name: _identity(name, "e") for name in CONFIGS},
        "parameter_counts": {
            "control_cofitok": 100,
            "ranked_cofitok": 101,
            "control_dense_identity": 90,
            "ranked_dense_identity": 90,
        },
        "builder_git": {
            "revision": "b" * 40,
            "branch": "scale/generation-label-ranking-5k-training-confirmation-v1",
            "tracked_dirty": False,
        },
        "expected_revision": "b" * 40,
        "expected_branch": "scale/generation-label-ranking-5k-training-confirmation-v1",
        "expected_output_root": EXPECTED_OUTPUT_ROOT,
    }
    with pytest.raises(ValueError, match="parameter counts differ"):
        build_training_confirmation_preparation(**kwargs)

    kwargs["parameter_counts"]["ranked_cofitok"] = 100
    kwargs["expected_output_root"] = "/tmp/wrong"
    with pytest.raises(ValueError, match="output root differs"):
        build_training_confirmation_preparation(**kwargs)


def test_prepare_cli_writes_once_and_replays_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampling = tmp_path / "sampling_validation.json"
    standing = tmp_path / "standing_authorization.json"
    output = tmp_path / "preparation.json"
    sampling.write_text(
        json.dumps(_sampling_validation(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    standing.write_text(
        json.dumps(_standing_authorization(), sort_keys=True) + "\n",
        encoding="utf-8",
    )
    revision = "b" * 40
    branch = "scale/generation-label-ranking-5k-training-confirmation-v1"
    monkeypatch.setattr(
        prepare_cli,
        "git_provenance",
        lambda _root: {
            "revision": revision,
            "branch": branch,
            "tracked_dirty": False,
        },
    )
    monkeypatch.setattr(
        prepare_cli,
        "replay_sampling_validation",
        lambda path: (
            json.loads(path.read_text(encoding="utf-8")),
            prepare_cli.file_identity(path),
        ),
    )
    monkeypatch.setattr(
        prepare_cli,
        "_parameter_count",
        lambda path: 90 if "dense" in path.name else 100,
    )
    arguments = [
        "prepare_generation_conditioning_ranking_training_confirmation.py",
        "--sampling-validation",
        str(sampling),
        "--expected-sampling-validation-sha256",
        prepare_cli.file_identity(sampling)["sha256"],
        "--standing-authorization",
        str(standing),
        "--expected-standing-authorization-sha256",
        prepare_cli.file_identity(standing)["sha256"],
        "--control-cofitok",
        str(PROJECT_ROOT / CONFIGS["control_cofitok"]),
        "--control-dense",
        str(PROJECT_ROOT / CONFIGS["control_dense_identity"]),
        "--ranked-cofitok",
        str(PROJECT_ROOT / CONFIGS["ranked_cofitok"]),
        "--ranked-dense",
        str(PROJECT_ROOT / CONFIGS["ranked_dense_identity"]),
        "--expected-revision",
        revision,
        "--expected-branch",
        branch,
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    prepare_cli.main()
    first = output.read_bytes()

    monkeypatch.setattr(sys, "argv", [*arguments, "--resume"])
    prepare_cli.main()

    assert output.read_bytes() == first
    report = json.loads(first)
    assert report["source_reports"]["sampling_validation"] == prepare_cli.file_identity(
        sampling
    )
    assert report["gpu_execution_authorized"] is False
