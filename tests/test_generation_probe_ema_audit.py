from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts import audit_generation_probe_ema as audit


REVISION = "a" * 40
ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _candidate(run_dir: Path, *, ema_endpoint: float, model_endpoint: float) -> None:
    evaluation_git = {
        "revision": REVISION,
        "branch": "scale/generative-system",
        "tracked_dirty": False,
    }
    training_git = {
        "revision": REVISION,
        "branch": "scale/generative-system",
        "dirty": False,
    }
    config = {
        "optimization": {"ema_decay": 0.9999, "ema_warmup_steps": 2_000}
    }
    _write(
        run_dir / "training_report.json",
        {
            "training_complete": True,
            "completed_steps": 5_000,
            "target_steps": 5_000,
            "config": config,
            "git": training_git,
        },
    )
    for weights, endpoint, path in (
        ("ema", ema_endpoint, audit.EMA_EVAL),
        ("model", model_endpoint, audit.MODEL_EVAL),
    ):
        _write(
            run_dir / path,
            {
                "status": "completed",
                "weights": weights,
                "checkpoint_step": 5_000,
                "checkpoint_sha256": "b" * 64,
                "config": config,
                "git": evaluation_git,
                "metrics": {
                    "evaluated_images": 512,
                    "timestep": 500,
                    "order_count": 18,
                    "ordered_rank_by_path_auc": 1,
                    "component_energy": [0.4, 0.6],
                    "orders": {
                        "ordered": {
                            "endpoint_clean_mse": endpoint,
                            "endpoint_clean_psnr": 20.0,
                            "prefix_path_mse_auc": 0.1,
                        }
                    },
                },
            },
        )


def test_linear_warmup_profile_exposes_short_horizon_ema_age() -> None:
    profile = audit.ema_weight_profile(
        steps=5_000,
        decay=0.9999,
        warmup_steps=2_000,
    )

    assert profile["weighted_mean_step"] == pytest.approx(2367.2010706)
    assert profile["median_step"] == 1961
    assert profile["last_1000_steps_weight"] == pytest.approx(0.0951671064)


def test_rank_recovery_runbook_builds_model_and_ema_audit() -> None:
    source = (
        ROOT / "artifacts/runbooks/generation_rank_recovery_probe_2026-07-19.sh"
    ).read_text(encoding="utf-8")

    assert "--weights model" in source
    assert "scripts/audit_generation_probe_ema.py" in source


def test_equal_progress_probe_targets_balanced_component_work() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/generation/"
            "imagenet256_10pct_rankcomplete_equal_progress_k8_probe5k.json"
        ).read_text(encoding="utf-8")
    )

    assert config["model"]["token_channel_schedule"] == [4, 4, 8, 8, 8, 8, 1, 2]
    assert config["model"]["token_spatial_strides"] == [16, 16, 8, 8, 4, 4, 1, 1]
    assert config["loss"]["denoise_path_progress_power"] == 1.0
    assert config["loss"]["denoise_path_prefix_weight"] == 0.15
    assert config["loss"]["denoise_path_component_weight"] == 0.3
    assert config["loss"]["energy_budget_weight"] == 0.5
    assert config["loss"]["energy_target"] == [1.0] * 8

    runbook = (
        ROOT
        / "artifacts/runbooks/"
        "generation_rank_recovery_equal_progress_probe_2026-07-20.sh"
    ).read_text(encoding="utf-8")
    assert "imagenet256_10pct_rankcomplete_equal_progress_k8_probe5k_v3" in runbook
    assert "--candidate \"equal_progress=$RUN\"" in runbook
    assert "--weights ema" in runbook
    assert "--weights model" in runbook
    assert "--num-samples 512" in runbook
    assert "--sample-steps 50" in runbook


def test_timestep_diagnostic_covers_fixed_schedule_without_training() -> None:
    runbook = (
        ROOT
        / "artifacts/runbooks/generation_rank_recovery_timestep_diagnostic_2026-07-20.sh"
    ).read_text(encoding="utf-8")

    assert "for timestep in 50 250 500 750 950" in runbook
    assert "--num-images 256" in runbook
    assert "--random-orders 16" in runbook
    assert "checkpoint_eval_ema_t${timestep}_256_energy_scope" in runbook
    assert "train_generation.py \\" not in runbook


def test_target_energy_probe_matches_path_energy_per_sample() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/generation/"
            "imagenet256_10pct_rankcomplete_target_energy_k8_probe5k.json"
        ).read_text(encoding="utf-8")
    )

    assert config["model"]["token_channel_schedule"] == [4, 4, 8, 8, 8, 8, 1, 2]
    assert config["model"]["token_spatial_strides"] == [16, 16, 8, 8, 4, 4, 1, 1]
    assert config["loss"]["energy_budget_weight"] == 0.0
    assert config["loss"]["energy_target"] == []
    assert config["loss"]["denoise_path_prefix_weight"] == 0.15
    assert config["loss"]["denoise_path_component_weight"] == 0.3
    assert config["loss"]["denoise_path_energy_weight"] == 0.5
    assert config["loss"]["denoise_path_progress_power"] == 1.0

    runbook = (
        ROOT
        / "artifacts/runbooks/generation_rank_recovery_target_energy_probe_2026-07-20.sh"
    ).read_text(encoding="utf-8")
    assert "imagenet256_10pct_rankcomplete_target_energy_k8_probe5k_v4" in runbook
    assert "--candidate \"target_energy=$RUN\"" in runbook
    assert "for timestep in 50 250 750 950" in runbook
    assert "--weights ema" in runbook
    assert "--weights model" in runbook
    assert "--num-samples 512" in runbook
    assert "--sample-steps 50" in runbook


def test_capacity_path_probe_matches_targets_to_restricted_token_layout() -> None:
    config = json.loads(
        (
            ROOT
            / "configs/generation/"
            "imagenet256_10pct_rankcomplete_capacity_path_k8_probe5k.json"
        ).read_text(encoding="utf-8")
    )

    assert config["model"]["token_channel_schedule"] == [4, 4, 8, 8, 8, 8, 1, 2]
    assert config["model"]["token_spatial_strides"] == [16, 16, 8, 8, 4, 4, 1, 1]
    assert config["loss"]["denoise_path_progress_mode"] == "token_capacity"
    assert config["loss"]["denoise_path_energy_weight"] == 0.5

    runbook = (
        ROOT
        / "artifacts/runbooks/"
        "generation_rank_recovery_capacity_path_probe_2026-07-20.sh"
    ).read_text(encoding="utf-8")
    assert "imagenet256_10pct_rankcomplete_capacity_path_k8_probe5k_v5" in runbook
    assert "--candidate \"capacity_path=$RUN\"" in runbook
    assert "for timestep in 50 250 750 950" in runbook
    assert "--weights ema" in runbook
    assert "--weights model" in runbook
    assert "--num-samples 512" in runbook
    assert "--sample-steps 50" in runbook


def test_posthoc_waiter_is_revision_locked_and_waits_for_training_exit() -> None:
    source = (
        ROOT
        / "artifacts/runbooks/"
        "generation_rank_recovery_posthoc_model_eval_waiter_2026-07-20.sh"
    ).read_text(encoding="utf-8")

    assert f"EXPECTED_REVISION={REVISION}" not in source
    assert "EXPECTED_REVISION=05bbb4af63a9f1d9b7f11bc4222d50875e382e1d" in source
    assert "[g]eneration_rank_recovery_probe_2026-07-19.sh" in source
    assert "[s]cripts/train_generation.py" in source
    assert "--weights model" in source
    assert "TIMEOUT_SECONDS=43200" in source


def test_probe_ema_audit_binds_model_and_ema_without_authorizing_scale(
    tmp_path: Path, monkeypatch
) -> None:
    denoise = tmp_path / "denoise"
    band = tmp_path / "band"
    _candidate(denoise, ema_endpoint=0.08, model_endpoint=0.04)
    _candidate(band, ema_endpoint=0.07, model_endpoint=0.05)
    output = tmp_path / "ema_audit.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_generation_probe_ema.py",
            "--candidate",
            f"denoise_path={denoise}",
            "--candidate",
            f"epsilon_band={band}",
            "--output",
            str(output),
        ],
    )

    audit.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["formal_claim_allowed"] is False
    assert report["automatic_50k_or_300k_launch_allowed"] is False
    assert report["candidates"][0]["comparison"][
        "ema_to_model_endpoint_mse_ratio"
    ] == pytest.approx(2.0)


def test_probe_ema_audit_rejects_different_checkpoint_identity(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _candidate(run, ema_endpoint=0.08, model_endpoint=0.04)
    model_path = run / audit.MODEL_EVAL
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["checkpoint_sha256"] = "c" * 64
    _write(model_path, model)

    with pytest.raises(ValueError, match="identity differs"):
        audit._candidate("candidate", run)


def test_probe_ema_audit_rejects_git_cleanliness_mismatch(tmp_path: Path) -> None:
    run = tmp_path / "run"
    _candidate(run, ema_endpoint=0.08, model_endpoint=0.04)
    model_path = run / audit.MODEL_EVAL
    model = json.loads(model_path.read_text(encoding="utf-8"))
    model["git"]["tracked_dirty"] = True
    _write(model_path, model)

    with pytest.raises(ValueError, match="identity differs"):
        audit._candidate("candidate", run)
