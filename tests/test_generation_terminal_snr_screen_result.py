from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.generation import capacity_screen_arm as shared_arm
from cofitok.generation import terminal_snr_screen_arm as arm_validation
from cofitok.generation import terminal_snr_screen_result as result
from cofitok.generation.terminal_snr_screen import ARM_NAMES, SCREEN_THRESHOLDS


GIT = {
    "revision": "a" * 40,
    "tree": "b" * 40,
    "branch": "analysis/terminal-snr",
    "tracked_dirty": False,
}


def _identity(name: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": len(name) + 10,
        "sha256": (name.encode().hex() + "0" * 64)[:64],
    }


def _summaries() -> dict[str, dict[str, object]]:
    common = {
        "validation_epsilon_mse": 0.1,
        "sample_count": 1_000,
        "inception_score_mean": 2.0,
        "real_set": {
            "path": "/tmp/imagenet-val",
            "image_count": 50_000,
            "digest_schema": "cofitok_image_tree_sha256_v1",
            "sha256": "e" * 64,
        },
        "classifier": {
            "architecture": "torchvision_resnet50_imagenet1k_v2",
            "checkpoint_sha256": "f" * 64,
        },
        "training_runtime_environment_sha256": "d" * 64,
        "sampling_runtime_environment_sha256": "d" * 64,
        "distribution_runtime_environment_sha256": "d" * 64,
        "class_fidelity_runtime_environment_sha256": "d" * 64,
        "execution_git": copy.deepcopy(GIT),
    }
    rows: dict[str, dict[str, object]] = {}
    for index, method in enumerate(("cofitok", "dense_identity")):
        rows[f"control_{method}"] = {
            **copy.deepcopy(common),
            "arm": f"control_{method}",
            "condition": "control",
            "method": method,
            "endpoint_fraction": 1.0,
            "fid": 100.0,
            "precision": 0.50,
            "recall": 0.40,
            "class_top1": 0.20,
            "class_top5": 0.40,
            "terminal_raw_x0_clip_fraction": 0.999,
            "checkpoint": {"sha256": f"{index + 1}" * 64},
            "sample_set_sha256": f"{index + 3}" * 64,
        }
        rows[f"endpoint0975_{method}"] = {
            **copy.deepcopy(common),
            "arm": f"endpoint0975_{method}",
            "condition": "endpoint0975",
            "method": method,
            "endpoint_fraction": 0.975,
            "fid": 90.0,
            "precision": 0.48,
            "recall": 0.395,
            "class_top1": 0.195,
            "class_top5": 0.39,
            "terminal_raw_x0_clip_fraction": 0.90,
            "checkpoint": {"sha256": f"{index + 5}" * 64},
            "sample_set_sha256": f"{index + 7}" * 64,
        }
    diagnostics = {
        "ordered_rank_by_path_auc": 1,
        "zero_token_max_abs": 0.0,
        "shuffled_to_ordered_endpoint_ratio": 3.0,
        "coarse_token_energy_ratio": 0.20,
        "tail_two_energy_ratio": 0.50,
    }
    rows["control_cofitok"]["cofitok_diagnostics"] = copy.deepcopy(diagnostics)
    rows["endpoint0975_cofitok"]["cofitok_diagnostics"] = diagnostics
    return rows


def _report(summaries: dict[str, dict[str, object]]) -> dict[str, object]:
    evaluated = result._evaluate(summaries)
    cross_arm = result._cross_arm_evidence(summaries, "d" * 64)
    passed = evaluated["screen_pass"]
    return {
        "schema_version": result.RESULT_SCHEMA,
        "role": result.RESULT_ROLE,
        "status": "completed",
        "operational_status": "pass",
        "scientific_status": "pass" if passed else "hold",
        "terminal_status": "hold",
        "screen_pass": passed,
        "generation_advantage_proven": False,
        "decision": (
            "prepare_separately_authorized_frozen_10k_confirmation"
            if passed
            else "hold_terminal_snr_intervention"
        ),
        "result_git": copy.deepcopy(GIT),
        "runtime_environment_sha256": "d" * 64,
        "source_evidence": {
            "preparation": _identity("preparation"),
            "launch_receipt": _identity("launch"),
            "arm_validations": {
                arm: _identity(f"arm_{arm}") for arm in ARM_NAMES
            },
        },
        "thresholds": copy.deepcopy(SCREEN_THRESHOLDS),
        "arm_summaries": summaries,
        "cross_arm_evidence": cross_arm,
        "comparisons": evaluated["comparisons"],
        "checks": evaluated["checks"],
        "failed_checks": evaluated["failed_checks"],
        "next_stage": {
            "route": "frozen_10k_confirmation_preparation" if passed else "hold",
            "frozen_confirmation_preparation_allowed": passed,
            "frozen_confirmation_launch_allowed": False,
            "separate_exact_authorization_required": True,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(result.RESULT_BOUNDARY),
    }


def test_passes_only_with_both_method_and_mechanism_checks() -> None:
    report = _report(_summaries())

    validated = result.validate_terminal_snr_screen_result_contract(report)

    assert validated["screen_pass"] is True
    assert validated["scientific_status"] == "pass"
    assert validated["generation_advantage_proven"] is False
    assert validated["next_stage"]["frozen_confirmation_launch_allowed"] is False
    assert len(validated["checks"]) == 19


@pytest.mark.parametrize(
    ("arm", "field", "value", "failed"),
    [
        (
            "endpoint0975_dense_identity",
            "fid",
            96.0,
            "dense_identity.relative_fid_improvement",
        ),
        (
            "endpoint0975_cofitok",
            "terminal_raw_x0_clip_fraction",
            0.96,
            "cofitok.endpoint_terminal_raw_x0_clip_fraction",
        ),
    ],
)
def test_holds_when_any_predeclared_method_check_fails(
    arm: str, field: str, value: float, failed: str
) -> None:
    summaries = _summaries()
    summaries[arm][field] = value
    report = _report(summaries)

    validated = result.validate_terminal_snr_screen_result_contract(report)

    assert validated["screen_pass"] is False
    assert validated["scientific_status"] == "hold"
    assert failed in validated["failed_checks"]


def test_rejects_threshold_weakening() -> None:
    report = _report(_summaries())
    report["thresholds"]["both_methods_min_relative_fid_improvement"] = 0.0

    with pytest.raises(ValueError, match="result contract differs"):
        result.validate_terminal_snr_screen_result_contract(report)


def test_rejects_duplicate_sample_sets() -> None:
    report = _report(_summaries())
    report["arm_summaries"]["endpoint0975_cofitok"][
        "sample_set_sha256"
    ] = report["arm_summaries"]["control_cofitok"]["sample_set_sha256"]

    with pytest.raises(ValueError, match="sample sets are missing or duplicated"):
        result.validate_terminal_snr_screen_result_contract(report)


def test_rejects_different_real_set() -> None:
    report = _report(_summaries())
    report["arm_summaries"]["endpoint0975_dense_identity"]["real_set"][
        "sha256"
    ] = "0" * 64

    with pytest.raises(ValueError, match="one identical real set"):
        result.validate_terminal_snr_screen_result_contract(report)


def test_rejects_evaluator_runtime_mismatch() -> None:
    report = _report(_summaries())
    report["arm_summaries"]["endpoint0975_cofitok"][
        "distribution_runtime_environment_sha256"
    ] = "0" * 64

    with pytest.raises(ValueError, match="runtime differs from launch"):
        result.validate_terminal_snr_screen_result_contract(report)


def test_arm_runtime_binding_rejects_mismatch() -> None:
    with pytest.raises(ValueError, match="sampling runtime differs"):
        arm_validation._require_launch_runtime(
            {"runtime_environment_sha256": "0" * 64},
            arm="endpoint0975_cofitok",
            evidence_name="sampling",
            expected_runtime_environment_sha256="d" * 64,
        )


def test_terminal_clip_source_is_exact_first_rollout_step(tmp_path: Path) -> None:
    path = tmp_path / "rollout.json"
    path.write_text(json.dumps({
        "free_sampling_rollout": {
            "steps": [
                {
                    "step_index": 0,
                    "timestep": 999,
                    "raw_x0_clip_fraction": 0.901,
                }
            ]
        }
    }), encoding="utf-8")

    extracted = arm_validation._terminal_clip_fraction(path, "endpoint0975_cofitok")

    assert extracted == {
        "source": "free_sampling_rollout.steps[0].raw_x0_clip_fraction",
        "step_index": 0,
        "timestep": 999,
        "raw_x0_clip_fraction": 0.901,
    }


def test_shared_helper_binding_restores_original_globals() -> None:
    original_specs = shared_arm.ARM_SPECS
    original_names = shared_arm.ARM_NAMES
    original_seed = shared_arm.SAMPLE_SEED

    with arm_validation._terminal_helper_contract():
        assert shared_arm.ARM_NAMES == ARM_NAMES
        assert shared_arm.SAMPLE_SEED == 2027

    assert shared_arm.ARM_SPECS is original_specs
    assert shared_arm.ARM_NAMES is original_names
    assert shared_arm.SAMPLE_SEED == original_seed
