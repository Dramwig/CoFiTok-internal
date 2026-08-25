from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import build_generation_min_snr_early_warning as early


GAMMA5_REVISION = "a" * 40
GAMMA0_REVISION = "b" * 40
GAMMA5_BRANCH = "scale/min-snr"
GAMMA0_BRANCH = "scale/gamma-zero"
DATASET_IDENTITY = "c" * 64
RUNTIME_IDENTITY = "d" * 64


def _config(*, dense: bool, gamma: float | None, name: str) -> dict:
    loss = {
        "epsilon_weight": 1.0,
        "rollout_consistency_weight": 0.1,
        "rollout_consistency_start_step": 0,
        "rollout_consistency_warmup_steps": 200,
        "rollout_consistency_timestep_delta": 10,
        "rollout_consistency_unroll_steps": 2,
        "rollout_consistency_batch_fraction": 0.125,
        "rollout_consistency_clip_x0": True,
        "rollout_consistency_mode": "clipped_x0",
        "ema_teacher_consistency_weight": 0.25,
        "ema_teacher_consistency_start_step": 100,
        "ema_teacher_consistency_warmup_steps": 100,
        "ema_teacher_consistency_batch_fraction": 0.0625,
        "prefix_weight": 0.0 if dense else 0.2,
    }
    if gamma is not None:
        loss["min_snr_gamma"] = gamma
    return {
        "name": name,
        "data": {"batch_size": 4, "dataset": "imagenet_256"},
        "diffusion": {"prediction_target": "epsilon"},
        "optimization": {
            "gradient_accumulation_steps": 2,
            "log_interval": 50,
        },
        "runtime": {"evaluation_interval": 100, "steps": 500},
        "model": {
            "base_channels": 32,
            "token_count": 1 if dense else 8,
            "token_channels": 3 if dense else 8,
            "predictor_use_feedback": not dense,
            "synthesis_mode": "dense_identity" if dense else "fixed_basis",
        },
        "loss": loss,
    }


def _source(*, dense: bool, gamma5: bool) -> dict:
    return {
        "git": {
            "revision": GAMMA5_REVISION if gamma5 else GAMMA0_REVISION,
            "branch": GAMMA5_BRANCH if gamma5 else GAMMA0_BRANCH,
            "dirty": False,
        },
        "parameter_count": 999 if dense else 1_000,
        "runtime_environment_sha256": RUNTIME_IDENTITY,
        "dataset_provenance": {
            "status": "pass",
            "formal": True,
            "dataset": "imagenet_256",
            "identity_sha256": DATASET_IDENTITY,
        },
        "config": _config(
            dense=dense,
            gamma=5.0 if gamma5 else None,
            name="gamma5" if gamma5 else "gamma0_dense" if dense else "gamma0",
        ),
    }


def _rows(*, gamma5: bool, dense: bool = False) -> list[dict]:
    rows = []
    for step in (1, 50, 100, 150, 200):
        epsilon = 0.3 / step + (0.0002 if dense else 0.0)
        row = {
            "step": step,
            "samples_seen": step * 8,
            "epsilon": epsilon,
            "rollout_consistency": 0.1,
            "rollout_consistency_scale": min(step / 200, 1.0),
            "ema_teacher_consistency": 0.05,
            "ema_teacher_consistency_scale": (
                0.0 if step < 100 else min((step - 100) / 100, 1.0)
            ),
        }
        if gamma5:
            row.update(
                {
                    "epsilon_unweighted": epsilon * 1.5,
                    "min_snr_weight_mean": 0.8,
                }
            )
        if step in (100, 200):
            event = step // 100 - 1
            control = 0.03 + event * 0.001 + (0.0002 if dense else 0.0)
            row.update(
                {
                    "validation_epsilon_mse": control + (0.004 if gamma5 else 0.0),
                    "validation_event_index": event,
                    "validation_batch_index": event,
                    "validation_num_images": 64,
                    "validation_noise_seed": 102030,
                }
            )
        rows.append(row)
    return rows


def _build(**overrides) -> dict:
    values = {
        "gamma5_metrics": {"rows": _rows(gamma5=True)},
        "gamma5_manifest": _source(dense=False, gamma5=True),
        "gamma0_cofitok_metrics": {"rows": _rows(gamma5=False)},
        "gamma0_cofitok_report": _source(dense=False, gamma5=False),
        "gamma0_dense_metrics": {"rows": _rows(gamma5=False, dense=True)},
        "gamma0_dense_report": _source(dense=True, gamma5=False),
        "cutoff_step": 200,
        "expected_gamma5_revision": GAMMA5_REVISION,
        "expected_gamma5_branch": GAMMA5_BRANCH,
        "expected_gamma0_revision": GAMMA0_REVISION,
        "expected_gamma0_branch": GAMMA0_BRANCH,
    }
    values.update(overrides)
    return early.build_report(**values)


def test_report_binds_controlled_treatment_without_claiming_quality() -> None:
    report = _build()

    assert report["status"] == "pass"
    assert report["images_seen"] == 1_600
    assert report["controlled_treatment"]["only_config_differences"] == [
        "name",
        "loss.min_snr_gamma",
    ]
    assert report["controlled_treatment"]["legacy_pair_contract"]["valid"] is True
    paired = report["paired_fixed_validation"]
    assert paired["vs_gamma0_cofitok"]["event_count"] == 2
    assert paired["vs_gamma0_cofitok"]["gamma5_lower_event_count"] == 0
    assert paired["vs_gamma0_dense"]["gamma5_higher_event_count"] == 2
    assert report["trajectory_contract"]["min_snr_delivery"][
        "downweighted_row_count"
    ] == 5
    assert report["scientific_interpretation"]["quality_claim_allowed"] is False
    assert report["scientific_interpretation"]["sample_quality_metrics_present"] is False
    assert all(value is False for value in report["authorization_boundary"].values())


def test_report_rejects_non_treatment_config_drift() -> None:
    manifest = deepcopy(_source(dense=False, gamma5=True))
    manifest["config"]["model"]["base_channels"] = 64

    with pytest.raises(ValueError, match="differ beyond treatment"):
        _build(gamma5_manifest=manifest)


def test_report_rejects_fixed_validation_provenance_drift() -> None:
    metrics = {"rows": _rows(gamma5=False, dense=True)}
    metrics["rows"][-1]["validation_noise_seed"] += 1

    with pytest.raises(ValueError, match="provenance differs"):
        _build(gamma0_dense_metrics=metrics)


def test_report_rejects_missing_min_snr_delivery() -> None:
    metrics = {"rows": _rows(gamma5=True)}
    for row in metrics["rows"]:
        row["min_snr_weight_mean"] = 1.0

    with pytest.raises(ValueError, match="never apply Min-SNR downweighting"):
        _build(gamma5_metrics=metrics)


def test_report_rejects_three_source_identity_drift() -> None:
    dense = deepcopy(_source(dense=True, gamma5=False))
    dense["runtime_environment_sha256"] = "e" * 64

    with pytest.raises(ValueError, match="runtime_environment_sha256 differs"):
        _build(gamma0_dense_report=dense)


def test_report_rejects_legacy_pair_contract_drift() -> None:
    dense = deepcopy(_source(dense=True, gamma5=False))
    dense["config"]["optimization"]["gradient_accumulation_steps"] = 1

    with pytest.raises(ValueError, match="legacy gamma0 pair contract failed"):
        _build(gamma0_dense_report=dense)
