from __future__ import annotations

import os
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/generation/smoke_random_cpu.json"


def _run(
    output: Path,
    *extra: str,
    environment_overrides: dict[str, str] | None = None,
) -> None:
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.update(environment_overrides or {})
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/train_generation.py"),
            "--config",
            str(CONFIG),
            "--output-dir",
            str(output),
            *extra,
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _assert_nested_equal(left: Any, right: Any) -> None:
    assert type(left) is type(right)
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        assert np.array_equal(left, right)
    elif isinstance(left, Mapping):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, Sequence) and not isinstance(left, (str, bytes)):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            _assert_nested_equal(left_item, right_item)
    else:
        assert left == right


def test_segmented_resume_matches_uninterrupted_training_exactly(tmp_path) -> None:
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    _run(uninterrupted)
    _run(resumed, "--stop-after-steps", "1")
    _run(resumed, "--resume", "auto")

    uninterrupted_checkpoint = torch.load(
        uninterrupted / "checkpoint_step_00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    resumed_checkpoint = torch.load(
        resumed / "checkpoint_step_00000002.pt",
        map_location="cpu",
        weights_only=False,
    )
    for key in ("model", "ema", "optimizer", "scheduler", "rng_state"):
        _assert_nested_equal(uninterrupted_checkpoint[key], resumed_checkpoint[key])
    _assert_nested_equal(
        uninterrupted_checkpoint["extra_state"]["sampler"],
        resumed_checkpoint["extra_state"]["sampler"],
    )
    assert resumed_checkpoint["extra_state"]["cumulative_elapsed_seconds"] > 0.0
    assert resumed_checkpoint["extra_state"]["cumulative_peak_vram_bytes"] == 0
    for key in (
        "total",
        "epsilon",
        "grad_norm",
        "learning_rate",
        "validation_epsilon_mse",
    ):
        assert uninterrupted_checkpoint["metrics"][key] == resumed_checkpoint["metrics"][key]

    steps = [
        int(json.loads(line)["step"])
        for line in (resumed / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert steps == [1, 2]


@pytest.mark.parametrize(
    ("override", "mismatch_path"),
    [
        (("--micro-batch-size", "1"), "config.data.batch_size"),
        (
            ("--gradient-accumulation-steps", "1"),
            "config.optimization.gradient_accumulation_steps",
        ),
    ],
)
def test_resume_rejects_changed_training_config_before_advancing(
    tmp_path, override: tuple[str, str], mismatch_path: str
) -> None:
    output = tmp_path / mismatch_path.replace(".", "_")
    _run(output, "--stop-after-steps", "1")

    with pytest.raises(subprocess.CalledProcessError) as error:
        _run(output, "--resume", "auto", *override)

    assert f"Checkpoint config mismatch at: {mismatch_path}" in error.value.stderr
    latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    assert latest["step"] == 1
    assert not (output / "checkpoint_step_00000002.pt").exists()


def test_resume_rejects_runtime_environment_drift_before_advancing(tmp_path) -> None:
    output = tmp_path / "environment_drift"
    _run(output, "--stop-after-steps", "1")
    checkpoint_path = output / "checkpoint_step_00000001.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    integrity = json.loads(
        (output / "checkpoint_step_00000001.pt.integrity.json").read_text(
            encoding="utf-8"
        )
    )
    latest = json.loads((output / "latest.json").read_text(encoding="utf-8"))
    environment_sha = checkpoint["extra_state"]["runtime_environment_sha256"]

    assert checkpoint["extra_state"]["runtime_environment"]["schema_version"] == 1
    assert integrity["runtime_environment_sha256"] == environment_sha
    assert latest["runtime_environment_sha256"] == environment_sha

    with pytest.raises(subprocess.CalledProcessError) as error:
        _run(
            output,
            "--resume",
            "auto",
            environment_overrides={"PYTHONHASHSEED": "314159"},
        )

    assert (
        "Checkpoint runtime environment mismatch at: "
        "runtime_environment.environment_variables.PYTHONHASHSEED"
    ) in error.value.stderr
    assert json.loads((output / "latest.json").read_text(encoding="utf-8"))["step"] == 1
    assert not (output / "checkpoint_step_00000002.pt").exists()


def test_runtime_benchmark_executes_training_without_checkpoint(tmp_path) -> None:
    output = tmp_path / "benchmark_run"
    report_path = tmp_path / "benchmark_report.json"
    _run(
        output,
        "--benchmark-steps",
        "2",
        "--benchmark-warmup-steps",
        "1",
        "--benchmark-output",
        str(report_path),
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    assert report["role"] == "training_runtime_selection_only"
    assert report["benchmark_steps"] == 2
    assert report["measured_steps"] == 1
    assert report["effective_batch_size"] == 4
    assert report["mean_optimizer_step_seconds"] > 0.0
    assert report["images_per_second"] > 0.0
    assert report["checkpoint_written"] is False
    assert not list(output.glob("checkpoint_step_*.pt"))
    assert not (output / "training_report.json").exists()
