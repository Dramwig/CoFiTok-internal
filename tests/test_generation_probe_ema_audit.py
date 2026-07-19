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
    git = {
        "revision": REVISION,
        "branch": "scale/generative-system",
        "tracked_dirty": False,
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
            "git": git,
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
                "git": git,
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
