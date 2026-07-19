from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.configs import config_to_dict, load_config
from cofitok.training.checkpointing import checkpoint_integrity_path
from cofitok.training.completion import validate_completed_generation_training


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/generation/smoke_random_cpu.json"
REVISION = "a" * 40


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _completed_run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    checkpoint = run / "checkpoint_step_00000002.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_integrity_path(checkpoint).write_text("{}", encoding="utf-8")
    latest = {"checkpoint": checkpoint.name, "step": 2}
    _write(run / "latest.json", latest)
    _write(
        run / "training_report.json",
        {
            "training_complete": True,
            "completed_steps": 2,
            "target_steps": 2,
            "config": config_to_dict(load_config(CONFIG)),
            "git": {
                "revision": REVISION,
                "branch": "scale/generative-system",
                "dirty": False,
            },
            "latest_checkpoint": latest,
        },
    )
    return run


def test_completed_training_is_safe_to_skip_only_with_current_identity(
    tmp_path: Path,
) -> None:
    run = _completed_run(tmp_path)

    report = validate_completed_generation_training(
        report_path=run / "training_report.json",
        config_path=CONFIG,
        expected_steps=2,
        expected_revision=REVISION,
    )

    assert report["completed_steps"] == 2


@pytest.mark.parametrize("mutation", ["revision", "config", "latest"])
def test_completed_training_rejects_stale_or_inconsistent_identity(
    tmp_path: Path, mutation: str
) -> None:
    run = _completed_run(tmp_path)
    report_path = run / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if mutation == "revision":
        report["git"]["revision"] = "b" * 40
    elif mutation == "config":
        report["config"] = copy.deepcopy(report["config"])
        report["config"]["loss"]["epsilon_weight"] = 0.5
    else:
        latest = json.loads((run / "latest.json").read_text(encoding="utf-8"))
        latest["step"] = 1
        _write(run / "latest.json", latest)
    _write(report_path, report)

    with pytest.raises(ValueError):
        validate_completed_generation_training(
            report_path=report_path,
            config_path=CONFIG,
            expected_steps=2,
            expected_revision=REVISION,
        )
