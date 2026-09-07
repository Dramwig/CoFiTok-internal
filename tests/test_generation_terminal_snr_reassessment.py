from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from cofitok.configs import DiffusionConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.generation import terminal_snr_reassessment as reassessment


ROOT = Path(__file__).resolve().parents[1]


def _read_config(name: str) -> dict[str, object]:
    path = ROOT / "configs" / "generation" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _identity(name: str, character: str) -> dict[str, object]:
    return {
        "path": f"/tmp/{name}.json",
        "bytes": 100 + len(name),
        "sha256": character * 64,
    }


def _candidate(case_id: str, cofitok_gain: float, dense_gain: float) -> dict[str, object]:
    def method(gain: float) -> dict[str, object]:
        return {
            "versus_legacy": {
                "fid_relative_improvement": gain,
                "checks": {
                    "fid_strictly_lower": True,
                    "class_top1_non_regression": False,
                },
                "passes": False,
            }
        }

    return {
        "case_id": case_id,
        "passes_shared_recovery_screen": False,
        "methods": {
            "cofitok": method(cofitok_gain),
            "dense_identity": method(dense_gain),
        },
    }


def _observations(case_id: str, noise_scale: str) -> list[dict[str, object]]:
    return [
        {
            "case_id": case_id,
            "method": method,
            "sampling": {
                "start_timestep": 975,
                "initial_noise_scale": noise_scale,
                "x0_constraint": "clip",
                "recompute_epsilon_after_x0_constraint": False,
            },
        }
        for method in ("cofitok", "dense_identity")
    ]


def _fixture(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    hold_id = _identity("hold", "a")
    hold_validation_id = _identity("hold-validation", "b")
    sampling_id = _identity("sampling", "c")
    hold = {
        "source_evidence": {"sampling_recovery": {"result": sampling_id}},
        "evidence_interpretation": {
            "terminal_start": {
                "all_at_least_0_99": True,
                "causal_root_cause_is_proven": False,
            }
        },
    }
    hold_validation = {"decision": hold_id}
    sampling = {
        "candidates": [
            _candidate("start975_unit_hard_clip", 0.007, 0.12),
            _candidate("start975_sigma_hard_clip", 0.057, 0.13),
        ],
        "observations": [
            *_observations("start975_unit_hard_clip", "unit"),
            *_observations("start975_sigma_hard_clip", "schedule_sigma"),
        ],
    }
    monkeypatch.setattr(
        reassessment, "validate_hold_recovery_decision", lambda value: copy.deepcopy(value)
    )
    monkeypatch.setattr(
        reassessment,
        "validate_hold_recovery_validation",
        lambda receipt, **kwargs: copy.deepcopy(receipt),
    )
    monkeypatch.setattr(
        reassessment,
        "validate_sampling_recovery_result",
        lambda value, **kwargs: {
            "selection_status": "no_shared_sampling_recovery_candidate",
            "candidate_count": 7,
            "physical_identity_count": 107,
        },
    )
    controls = {
        "cofitok": _read_config(
            "imagenet256_capacity_reference_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
        ),
        "dense_identity": _read_config(
            "imagenet256_capacity_reference_rollout_x0_u2_ema_teacher_dense_100k.json"
        ),
    }
    interventions = {
        "cofitok": _read_config(
            "imagenet256_terminal_snr_endpoint0975_rgbtail3_rollout_x0_u2_ema_teacher_k8_100k.json"
        ),
        "dense_identity": _read_config(
            "imagenet256_terminal_snr_endpoint0975_rollout_x0_u2_ema_teacher_dense_100k.json"
        ),
    }
    return {
        "hold_decision": hold,
        "hold_decision_identity": hold_id,
        "hold_validation": hold_validation,
        "hold_validation_identity": hold_validation_id,
        "sampling_recovery_result": sampling,
        "sampling_recovery_identity": sampling_id,
        "control_configs": controls,
        "control_config_identities": {
            "cofitok": _identity("cofitok-control", "d"),
            "dense_identity": _identity("dense-control", "e"),
        },
        "intervention_configs": interventions,
        "intervention_config_identities": {
            "cofitok": _identity("cofitok-intervention", "f"),
            "dense_identity": _identity("dense-intervention", "0"),
        },
        "decision_git": {
            "branch": "analysis/terminal-snr",
            "revision": "1" * 40,
            "tree": "2" * 40,
            "tracked_dirty": False,
        },
        "allowed_sampling_output_prefix": "/root/autodl-tmp/CoFiTok/checkpoints/generation/",
    }


def test_default_cosine_endpoint_is_backward_compatible() -> None:
    implicit = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=1_000, schedule_type="cosine"), "cpu"
    )
    explicit = DiffusionSchedule(
        DiffusionConfig(
            num_train_timesteps=1_000,
            schedule_type="cosine",
            cosine_endpoint_fraction=1.0,
        ),
        "cpu",
    )
    assert torch.equal(implicit.betas, explicit.betas)
    assert torch.equal(implicit.alphas_cumprod, explicit.alphas_cumprod)


def test_endpoint0975_reduces_terminal_conditioning_factor() -> None:
    control = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=1_000, schedule_type="cosine"), "cpu"
    )
    intervention = DiffusionSchedule(
        DiffusionConfig(
            num_train_timesteps=1_000,
            schedule_type="cosine",
            cosine_endpoint_fraction=0.975,
        ),
        "cpu",
    )
    assert float(control.sqrt_alphas_cumprod[-1]) < 1e-4
    assert 0.03 < float(intervention.sqrt_alphas_cumprod[-1]) < 0.05
    assert float(
        control.sqrt_alphas_cumprod[-1]
        / intervention.sqrt_alphas_cumprod[-1]
    ) < 0.002


@pytest.mark.parametrize("value", [0.0, -0.1, 1.001, float("inf"), float("nan")])
def test_invalid_cosine_endpoint_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="cosine_endpoint_fraction"):
        DiffusionSchedule(
            DiffusionConfig(
                num_train_timesteps=10,
                schedule_type="cosine",
                cosine_endpoint_fraction=value,
            ),
            "cpu",
        )


def test_linear_schedule_rejects_nondefault_cosine_endpoint() -> None:
    with pytest.raises(ValueError, match="only for a cosine schedule"):
        DiffusionSchedule(
            DiffusionConfig(
                num_train_timesteps=10,
                schedule_type="linear",
                cosine_endpoint_fraction=0.975,
            ),
            "cpu",
        )


def test_reassessment_selects_one_non_authorizing_intervention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(monkeypatch)
    report = reassessment.build_terminal_snr_reassessment(**inputs)
    assert reassessment.validate_terminal_snr_reassessment(report) == report
    assert report["selected_intervention"]["id"] == reassessment.SELECTED_INTERVENTION
    assert report["selected_intervention"]["prediction_target"] == "epsilon"
    assert report["source_evidence"]["config_contract"][
        "scientific_difference_paths"
    ] == ["diffusion.cosine_endpoint_fraction"]
    assert report["numerical_discriminator"]["condition_factor_reduction"] > 500
    assert not any(report["authorization_boundary"].values())
    assert report["next_stage"]["source_bound_preparation_may_be_built"] is True
    assert report["next_stage"]["gpu_execution_allowed"] is False


def test_reassessment_rejects_extra_intervention_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(monkeypatch)
    inputs["intervention_configs"]["dense_identity"]["optimization"][
        "learning_rate"
    ] = 2e-4
    with pytest.raises(ValueError, match="valid matched pairs|endpoint fraction"):
        reassessment.build_terminal_snr_reassessment(**inputs)


def test_reassessment_rejects_nonpositive_t975_fid_direction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(monkeypatch)
    inputs["sampling_recovery_result"]["candidates"][0]["methods"]["cofitok"][
        "versus_legacy"
    ]["fid_relative_improvement"] = 0.0
    with pytest.raises(ValueError, match="did not improve FID directionally"):
        reassessment.build_terminal_snr_reassessment(**inputs)


def test_reassessment_rejects_velocity_output_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(monkeypatch)
    inputs["intervention_configs"]["cofitok"]["diffusion"][
        "prediction_target"
    ] = "velocity"
    with pytest.raises(ValueError, match="valid matched pairs|endpoint fraction"):
        reassessment.build_terminal_snr_reassessment(**inputs)


def test_validation_receipt_is_reproducible_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(monkeypatch)
    report = reassessment.build_terminal_snr_reassessment(**inputs)
    report_id = _identity("reassessment", "9")
    validator_git = {
        "branch": "analysis/terminal-snr",
        "revision": "3" * 40,
        "tree": "4" * 40,
        "tracked_dirty": False,
    }
    receipt = reassessment.build_terminal_snr_reassessment_validation(
        decision=report,
        decision_identity=report_id,
        validator_git=validator_git,
    )
    assert (
        reassessment.validate_terminal_snr_reassessment_validation(
            receipt, decision=report, decision_identity=report_id
        )
        == receipt
    )
    tampered = copy.deepcopy(report)
    tampered["authorization_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="contract differs"):
        reassessment.validate_terminal_snr_reassessment(tampered)
