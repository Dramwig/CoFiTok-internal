from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.training_exposure_qualification import (
    CAPACITY_RESULT_BOUNDARY,
    CAPACITY_RESULT_SOURCE_NAMES,
    EXPOSURE_QUALIFICATION_BOUNDARY,
    EXPOSURE_QUALIFICATION_OUTPUT_ROOT,
    HISTORICAL_EXPOSURE_MATCH_STEPS,
    SOURCE_NAMES,
    TARGET_MILESTONES,
    TREND_EVALUATION_MILESTONES,
    build_training_exposure_qualification_preparation,
    validate_training_exposure_qualification_preparation,
)
from cofitok.inference_replay import file_identity
from scripts.build_generation_training_exposure_qualification import (
    replay_capacity_probe_result,
)
from test_generation_quality_bridge_followup_decision import (
    DECISION_GIT,
    _decision,
)


ROOT = Path(__file__).resolve().parents[1]
CAPACITY_GIT = {
    "revision": "c" * 40,
    "branch": "scale/generation-capacity-probe-v1",
    "tracked_dirty": False,
}
PREPARATION_GIT = {
    "revision": "e" * 40,
    "branch": "analysis/generation-training-exposure-qualification-v1",
    "tracked_dirty": False,
}
CAPACITY_OUTPUT_ROOT = (
    "/root/autodl-tmp/CoFiTok/checkpoints/generation/"
    "stability_full_data_100k_capacity_probe_250m_10k_v1"
)


def _identity(path: str, character: str) -> dict[str, object]:
    return {"path": path, "bytes": 100, "sha256": character * 64}


def _configs() -> dict[str, dict]:
    names = {
        "source_cofitok": (
            "imagenet256_stability_quality_bridge_"
            "rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
        ),
        "source_dense": (
            "imagenet256_stability_quality_bridge_"
            "rollout_x0_u2_ema_teacher_dense_100k.json"
        ),
        "target_cofitok": (
            "imagenet256_stability_exposure_"
            "rgbtail3_rollout_x0_u2_ema_teacher_k8_300k.json"
        ),
        "target_dense": (
            "imagenet256_stability_exposure_"
            "rollout_x0_u2_ema_teacher_dense_300k.json"
        ),
    }
    return {
        key: config_to_dict(load_config(ROOT / "configs/generation" / name))
        for key, name in names.items()
    }


def _source_identities() -> dict[str, dict[str, object]]:
    characters = iter("123456")
    return {
        name: _identity(f"/evidence/{name}.json", next(characters))
        for name in sorted(SOURCE_NAMES)
    }


def _capacity_sources() -> dict[str, dict[str, object]]:
    characters = iter("123456789abcdef0")
    return {
        name: _identity(f"/capacity/{name}.json", next(characters))
        for name in sorted(CAPACITY_RESULT_SOURCE_NAMES)
    }


def _capacity_result() -> dict[str, object]:
    fids = {
        "base128_cofitok": 180.0,
        "base128_dense_identity": 175.0,
        "base256_cofitok": 185.0,
        "base256_dense_identity": 170.0,
    }
    arms = {
        name: {
            "fid": fid,
            "mechanism_invariants_valid": True,
        }
        for name, fid in fids.items()
    }
    return {
        "schema_version": 1,
        "status": "completed",
        "role": "stability_full_data_capacity_probe_result",
        "git": copy.deepcopy(CAPACITY_GIT),
        "output_root": CAPACITY_OUTPUT_ROOT,
        "source_reports": _capacity_sources(),
        "selection": {
            "dataset": "imagenet_256",
            "configured_training_horizon": 100_000,
            "stop_after_step": 10_000,
            "effective_batch_size": 64,
            "base_channels": [128, 256],
        },
        "evaluation": {"arms": arms},
        "estimands": {
            "cofitok_fid_delta_base256_minus_base128": 5.0,
            "dense_fid_delta_base256_minus_base128": -5.0,
            "capacity_by_factorization_fid_interaction": 10.0,
            "cofitok_fid_relative_change": 5.0 / 180.0,
            "dense_fid_relative_change": -5.0 / 175.0,
        },
        "decision": {
            "shared_strict_fid_improvement": False,
            "cofitok_mechanism_invariants_valid": True,
            "capacity_supported": False,
            "recommendation": {
                "id": "hold_capacity_scaling_and_revisit_training_objective",
                "category": "capacity_not_supported",
                "execution_ready": False,
                "full_300k_launch_allowed": False,
            },
        },
        "claim_policy": {
            "role": "non_claim_capacity_causal_diagnostic",
            "sample_count_per_arm": 2_048,
            "formal_generation_claim_allowed": False,
            "cross_stage_numeric_ranking_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(CAPACITY_RESULT_BOUNDARY),
    }


def _kwargs() -> dict:
    configs = _configs()
    return {
        "followup_decision": _decision(
            "cofitok_absolute_fid",
            "cofitok_recall_floor",
        ),
        "capacity_probe_result": _capacity_result(),
        "source_cofitok_config": configs["source_cofitok"],
        "source_dense_identity_config": configs["source_dense"],
        "target_cofitok_config": configs["target_cofitok"],
        "target_dense_identity_config": configs["target_dense"],
        "source_identities": _source_identities(),
        "expected_followup_revision": DECISION_GIT["revision"],
        "expected_followup_branch": DECISION_GIT["branch"],
        "expected_capacity_revision": CAPACITY_GIT["revision"],
        "expected_capacity_branch": CAPACITY_GIT["branch"],
        "expected_preparation_revision": PREPARATION_GIT["revision"],
        "expected_preparation_branch": PREPARATION_GIT["branch"],
        "preparation_git": copy.deepcopy(PREPARATION_GIT),
        "output_root": EXPOSURE_QUALIFICATION_OUTPUT_ROOT,
    }


def test_valid_exposure_preparation_is_matched_and_non_authorizing() -> None:
    report = build_training_exposure_qualification_preparation(**_kwargs())

    assert report["status"] == "prepared"
    assert report["selection"]["target_steps"] == 300_000
    assert report["selection"]["target_images_seen_per_method"] == 19_200_000
    assert report["selection"]["historical_exposure_match_steps"] == (
        HISTORICAL_EXPOSURE_MATCH_STEPS
    )
    assert report["selection"]["residual_underexposure_hypothesis_after_300k"] is True
    assert report["evaluation_contract"]["protected_checkpoint_steps"] == (
        TARGET_MILESTONES
    )
    assert report["evaluation_contract"]["trend_evaluation_steps"] == (
        TREND_EVALUATION_MILESTONES
    )
    assert report["authorization_boundary"] == EXPOSURE_QUALIFICATION_BOUNDARY
    assert report["authorization_boundary"]["training_launch_allowed"] is False
    assert report["authorization_boundary"]["full_300k_launch_allowed"] is False
    selection = validate_training_exposure_qualification_preparation(
        report,
        expected_output_root=EXPOSURE_QUALIFICATION_OUTPUT_ROOT,
        expected_preparation_revision=PREPARATION_GIT["revision"],
        expected_preparation_branch=PREPARATION_GIT["branch"],
    )
    assert selection == report["selection"]


def test_exposure_preparation_rejects_capacity_support_or_mechanism_failure() -> None:
    kwargs = _kwargs()
    capacity = kwargs["capacity_probe_result"]
    capacity["evaluation"]["arms"]["base256_cofitok"]["fid"] = 160.0
    capacity["estimands"].update(
        {
            "cofitok_fid_delta_base256_minus_base128": -20.0,
            "capacity_by_factorization_fid_interaction": -15.0,
        }
    )
    capacity["decision"].update(
        {
            "shared_strict_fid_improvement": True,
            "capacity_supported": True,
        }
    )
    capacity["decision"]["recommendation"] = {
        "id": "prepare_source_compatible_capacity_scaling_decision",
        "category": "capacity_supported",
        "execution_ready": False,
        "full_300k_launch_allowed": False,
    }
    with pytest.raises(ValueError, match="does not support exposure"):
        build_training_exposure_qualification_preparation(**kwargs)

    kwargs = _kwargs()
    kwargs["capacity_probe_result"]["evaluation"]["arms"][
        "base256_cofitok"
    ]["mechanism_invariants_valid"] = False
    kwargs["capacity_probe_result"]["decision"][
        "cofitok_mechanism_invariants_valid"
    ] = False
    with pytest.raises(ValueError, match="does not support exposure"):
        build_training_exposure_qualification_preparation(**kwargs)


def test_exposure_preparation_rejects_config_capacity_or_git_drift() -> None:
    kwargs = _kwargs()
    kwargs["target_cofitok_config"]["optimization"]["learning_rate"] = 2e-4
    with pytest.raises(ValueError, match="target exposure recipe is invalid"):
        build_training_exposure_qualification_preparation(**kwargs)

    kwargs = _kwargs()
    kwargs["capacity_probe_result"]["estimands"][
        "capacity_by_factorization_fid_interaction"
    ] = 9.0
    with pytest.raises(ValueError, match="estimands or trigger"):
        build_training_exposure_qualification_preparation(**kwargs)

    kwargs = _kwargs()
    kwargs["expected_preparation_revision"] = "f" * 40
    with pytest.raises(ValueError, match="preparation Git identity differs"):
        build_training_exposure_qualification_preparation(**kwargs)


def test_capacity_result_replay_reopens_every_direct_source(tmp_path: Path) -> None:
    report = _capacity_result()
    payloads = {
        name: {"role": name, "output_root": CAPACITY_OUTPUT_ROOT}
        for name in CAPACITY_RESULT_SOURCE_NAMES
    }
    identities = {}
    for name in sorted(payloads):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payloads[name]), encoding="utf-8")
        identities[name] = file_identity(path)
    payloads["launch_receipt"]["source_reports"] = {
        "preparation": identities["preparation"]
    }
    launch_path = Path(identities["launch_receipt"]["path"])
    launch_path.write_text(json.dumps(payloads["launch_receipt"]), encoding="utf-8")
    identities["launch_receipt"] = file_identity(launch_path)
    report["source_reports"] = identities
    result_path = tmp_path / "capacity_probe_result.json"
    result_path.write_text(json.dumps(report), encoding="utf-8")
    result_identity = file_identity(result_path)

    replayed, replayed_identity = replay_capacity_probe_result(
        result_path,
        expected_sha256=result_identity["sha256"],
    )
    assert replayed == report
    assert replayed_identity == result_identity

    tampered = Path(identities["base128_cofitok_generation"]["path"])
    tampered.write_text(json.dumps({"tampered": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="identity differs"):
        replay_capacity_probe_result(
            result_path,
            expected_sha256=result_identity["sha256"],
        )


@pytest.mark.parametrize(
    "entrypoint",
    [
        "scripts.build_generation_training_exposure_qualification",
        "scripts.verify_generation_training_exposure_qualification",
    ],
)
def test_exposure_qualification_entrypoints_import(entrypoint: str) -> None:
    __import__(entrypoint)


def test_exposure_qualification_runbook_is_cpu_only_and_non_authorizing() -> None:
    runbook = (
        ROOT
        / "artifacts/runbooks/generation_training_exposure_qualification_prepare.sh"
    ).read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=-1" in runbook
    assert "build_generation_training_exposure_qualification.py" in runbook
    assert "verify_generation_training_exposure_qualification.py" in runbook
    for forbidden in (
        "train_generation.py",
        "generate_samples.py",
        "evaluate_generation_checkpoint.py",
        "evaluate_generation_metrics.py",
        "build_generation_full_launch_receipt.py",
        "generation_stability_full_matched_300k",
    ):
        assert forbidden not in runbook
