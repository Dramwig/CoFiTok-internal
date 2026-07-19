from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.training.checkpointing import checkpoint_integrity_path


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return payload


def validate_completed_generation_training(
    *,
    report_path: str | Path,
    config_path: str | Path,
    expected_steps: int,
    expected_revision: str,
) -> dict[str, Any]:
    if expected_steps < 1 or len(expected_revision) != 40:
        raise ValueError("completion expectation is invalid")
    report_file = Path(report_path).resolve()
    config_file = Path(config_path).resolve()
    report = _read_object(report_file)
    if report.get("training_complete") is not True:
        raise ValueError("training report is not complete")
    if (
        int(report.get("completed_steps", -1)) != expected_steps
        or int(report.get("target_steps", -1)) != expected_steps
    ):
        raise ValueError("training report step identity differs")

    expected_config = json.loads(json.dumps(config_to_dict(load_config(config_file))))
    if report.get("config") != expected_config:
        raise ValueError("training report resolved config differs from the current config")
    git = report.get("git", {})
    if (
        git.get("revision") != expected_revision
        or git.get("branch") != "scale/generative-system"
        or git.get("dirty") is not False
    ):
        raise ValueError("training report Git identity differs")

    latest = report.get("latest_checkpoint")
    if not isinstance(latest, dict) or int(latest.get("step", -1)) != expected_steps:
        raise ValueError("training report latest checkpoint differs")
    run_dir = report_file.parent
    latest_path = run_dir / "latest.json"
    if _read_object(latest_path) != latest:
        raise ValueError("training report and latest.json differ")
    checkpoint_name = str(latest.get("checkpoint", ""))
    checkpoint = run_dir / checkpoint_name
    if (
        not checkpoint_name
        or checkpoint.parent != run_dir
        or not checkpoint.is_file()
        or not checkpoint_integrity_path(checkpoint).is_file()
    ):
        raise ValueError("completed training checkpoint or integrity sidecar is missing")
    return report
