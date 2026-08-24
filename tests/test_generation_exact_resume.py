from __future__ import annotations

import os
import json
import subprocess
import sys
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from cofitok.configs import config_from_dict, config_to_dict, load_config
from cofitok.environment import runtime_environment_sha256
from cofitok.training.checkpointing import (
    _config_mismatch_paths,
    _normalize_exact_resume_config,
)
from scripts.train_generation import (
    _augment_training_images,
    _resolve_resume_git_provenance,
    _validate_config,
    _validate_existing_resume_revision_transition,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/generation/smoke_random_cpu.json"


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _run(
    output: Path,
    *extra: str,
    config: Path = CONFIG,
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
            str(config),
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
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["data"]["random_horizontal_flip_prob"] = 0.5
    config["loss"].update(
        {
            "class_conditioning_ranking_weight": 0.05,
            "class_conditioning_ranking_start_step": 0,
            "class_conditioning_ranking_warmup_steps": 1,
            "class_conditioning_ranking_batch_fraction": 0.5,
            "class_conditioning_ranking_margin": 0.01,
            "class_conditioning_ranking_wrong_label_offset": 5,
            "class_conditioning_ranking_min_timestep": 8,
        }
    )
    config_path = tmp_path / "smoke_random_cpu_flip.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    uninterrupted = tmp_path / "uninterrupted"
    resumed = tmp_path / "resumed"
    _run(uninterrupted, config=config_path)
    _run(resumed, "--stop-after-steps", "1", config=config_path)
    _run(resumed, "--resume", "auto", config=config_path)

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
        "validation_batch_index",
        "validation_epsilon_mse",
        "validation_event_index",
        "validation_noise_seed",
        "validation_num_images",
        "class_conditioning_ranking",
        "class_conditioning_ranking_scale",
        "class_conditioning_correct_mse",
        "class_conditioning_wrong_mse",
        "class_conditioning_null_mse",
    ):
        assert uninterrupted_checkpoint["metrics"][key] == resumed_checkpoint["metrics"][key]

    rows = [
        json.loads(line)
        for line in (resumed / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [int(row["step"]) for row in rows] == [1, 2]
    assert [row["validation_event_index"] for row in rows] == [0, 1]
    assert [row["validation_batch_index"] for row in rows] == [0, 1]
    assert {row["validation_noise_seed"] for row in rows} == {100_020}
    assert {row["validation_num_images"] for row in rows} == {2}


def test_residual_alignment_segmented_resume_matches_uninterrupted(tmp_path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["loss"].update(
        {
            "class_conditioning_residual_alignment_weight": 0.05,
            "class_conditioning_residual_alignment_start_step": 0,
            "class_conditioning_residual_alignment_warmup_steps": 1,
            "class_conditioning_residual_alignment_batch_fraction": 0.5,
            "class_conditioning_residual_alignment_margin": 0.1,
            "class_conditioning_residual_alignment_temperature": 0.1,
            "class_conditioning_residual_alignment_wrong_label_offsets": [1, 2],
            "class_conditioning_residual_alignment_min_timestep": 0,
            "class_conditioning_residual_alignment_pooling_factors": [4, 8],
            "class_conditioning_residual_alignment_reconstruction_weight": 0.25,
        }
    )
    config_path = tmp_path / "smoke_residual_alignment.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    uninterrupted = tmp_path / "residual_uninterrupted"
    resumed = tmp_path / "residual_resumed"

    _run(uninterrupted, config=config_path)
    _run(resumed, "--stop-after-steps", "1", config=config_path)
    _run(resumed, "--resume", "auto", config=config_path)

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
    for key in (
        "total",
        "epsilon",
        "class_conditioning_residual_alignment",
        "class_conditioning_residual_alignment_scale",
        "class_conditioning_residual_direction",
        "class_conditioning_residual_contrastive",
        "class_conditioning_residual_reconstruction",
        "class_conditioning_residual_correct_x0_mse",
        "class_conditioning_residual_wrong_x0_mse",
        "class_conditioning_residual_null_x0_mse",
    ):
        assert uninterrupted_checkpoint["metrics"][key] == resumed_checkpoint["metrics"][
            key
        ]

    rows = [
        json.loads(line)
        for line in (resumed / "train_metrics.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [int(row["step"]) for row in rows] == [1, 2]
    assert all(row["class_conditioning_residual_alignment_scale"] == 1.0 for row in rows)
    assert all(row["class_conditioning_residual_alignment"] > 0.0 for row in rows)


def test_controlled_resume_accepts_only_a_clean_ancestor_revision() -> None:
    current_revision = _git("rev-parse", "HEAD")
    source_revision = _git("rev-parse", "HEAD^")
    current_git = {
        "revision": current_revision,
        "branch": "scale/generative-system",
        "dirty": False,
    }

    checkpoint_git, transition = _resolve_resume_git_provenance(
        current_git,
        source_revision,
    )

    assert checkpoint_git == {
        "revision": source_revision,
        "branch": "scale/generative-system",
        "dirty": False,
    }
    assert transition == {
        "schema_version": 1,
        "reason": "sampler_rng_state_device_compatibility",
        "source_revision": source_revision,
        "target_revision": current_revision,
        "branch": "scale/generative-system",
    }

    with pytest.raises(ValueError, match="must differ"):
        _resolve_resume_git_provenance(current_git, current_revision)
    with pytest.raises(ValueError, match="clean named"):
        _resolve_resume_git_provenance(
            {**current_git, "dirty": True},
            source_revision,
        )


def test_existing_resume_transition_must_target_current_revision() -> None:
    current_git = {
        "revision": "b" * 40,
        "branch": "scale/generative-system",
        "dirty": False,
    }
    transition = {
        "schema_version": 1,
        "reason": "sampler_rng_state_device_compatibility",
        "source_revision": "a" * 40,
        "target_revision": "b" * 40,
        "branch": "scale/generative-system",
        "source_checkpoint": {
            "bytes": 10,
            "step": 25000,
            "sha256": "c" * 64,
            "integrity_manifest_sha256": "d" * 64,
        },
    }

    assert (
        _validate_existing_resume_revision_transition(transition, current_git)
        == transition
    )
    with pytest.raises(ValueError, match="transition is invalid"):
        _validate_existing_resume_revision_transition(
            {**transition, "target_revision": "e" * 40},
            current_git,
        )


def test_training_horizontal_flip_probability_boundaries_and_rng_restore() -> None:
    images = torch.arange(8 * 3 * 2 * 4, dtype=torch.float32).reshape(8, 3, 2, 4)

    assert _augment_training_images(images, 0.0) is images
    assert torch.equal(_augment_training_images(images, 1.0), images.flip(dims=(-1,)))

    torch.manual_seed(2027)
    rng_state = torch.get_rng_state()
    first = _augment_training_images(images, 0.5)
    torch.set_rng_state(rng_state)
    restored = _augment_training_images(images, 0.5)

    assert torch.equal(first, restored)
    changed = (first != images).flatten(1).any(dim=1)
    assert changed.any()
    assert (~changed).any()


def test_legacy_config_defaults_to_no_random_horizontal_flip() -> None:
    config = config_from_dict({"data": {"dataset": "random"}})
    assert config.data.random_horizontal_flip_prob == 0.0


def test_legacy_checkpoint_config_accepts_only_disabled_semantic_defaults() -> None:
    expected = config_to_dict(load_config(CONFIG))
    legacy = deepcopy(expected)
    for field in (
        "class_conditioning_ranking_weight",
        "class_conditioning_ranking_start_step",
        "class_conditioning_ranking_warmup_steps",
        "class_conditioning_ranking_batch_fraction",
        "class_conditioning_ranking_margin",
        "class_conditioning_ranking_wrong_label_offset",
        "class_conditioning_ranking_min_timestep",
        "class_conditioning_residual_alignment_weight",
        "class_conditioning_residual_alignment_start_step",
        "class_conditioning_residual_alignment_warmup_steps",
        "class_conditioning_residual_alignment_batch_fraction",
        "class_conditioning_residual_alignment_margin",
        "class_conditioning_residual_alignment_temperature",
        "class_conditioning_residual_alignment_wrong_label_offsets",
        "class_conditioning_residual_alignment_min_timestep",
        "class_conditioning_residual_alignment_pooling_factors",
        "class_conditioning_residual_alignment_reconstruction_weight",
    ):
        legacy["loss"].pop(field)

    assert _config_mismatch_paths(
        _normalize_exact_resume_config(expected),
        _normalize_exact_resume_config(legacy),
    ) == []

    enabled = deepcopy(expected)
    enabled["loss"]["class_conditioning_ranking_weight"] = 0.05
    assert _config_mismatch_paths(
        _normalize_exact_resume_config(enabled),
        _normalize_exact_resume_config(legacy),
    ) == ["config.loss.class_conditioning_ranking_weight"]

    residual_enabled = deepcopy(expected)
    residual_enabled["loss"]["class_conditioning_residual_alignment_weight"] = 0.05
    assert _config_mismatch_paths(
        _normalize_exact_resume_config(residual_enabled),
        _normalize_exact_resume_config(legacy),
    ) == ["config.loss.class_conditioning_residual_alignment_weight"]


def test_training_accepts_fixed_basis_restricted_synthesis() -> None:
    config = load_config(CONFIG)
    fixed_basis = replace(
        config,
        model=replace(config.model, synthesis_mode="fixed_basis"),
    )

    _validate_config(fixed_basis)


@pytest.mark.parametrize("probability", [-0.01, 1.01, float("nan")])
def test_training_rejects_invalid_horizontal_flip_probability(probability: float) -> None:
    config = load_config(CONFIG)
    invalid = replace(
        config,
        data=replace(config.data, random_horizontal_flip_prob=probability),
    )

    with pytest.raises(ValueError, match="random_horizontal_flip_prob"):
        _validate_config(invalid)


@pytest.mark.parametrize(
    "target",
    ([1.0], [1.0, 0.0, 1.0, 1.0], [1.0, float("nan"), 1.0, 1.0]),
)
def test_training_rejects_invalid_enabled_energy_target(target: list[float]) -> None:
    config = load_config(CONFIG)
    invalid = replace(
        config,
        loss=replace(
            config.loss,
            energy_budget_weight=0.5,
            energy_target=target,
        ),
    )

    with pytest.raises(ValueError, match="energy_target"):
        _validate_config(invalid)


def test_training_rejects_invalid_enabled_energy_scope() -> None:
    config = load_config(CONFIG)
    invalid = replace(
        config,
        loss=replace(
            config.loss,
            energy_budget_weight=0.5,
            energy_target=[1.0] * config.model.token_count,
            energy_budget_scope="timestep",
        ),
    )

    with pytest.raises(ValueError, match="energy_budget_scope"):
        _validate_config(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("ema_teacher_consistency_weight", -0.1),
        ("ema_teacher_consistency_start_step", -1),
        ("ema_teacher_consistency_warmup_steps", -1),
        ("ema_teacher_consistency_batch_fraction", 0.0),
        ("ema_teacher_consistency_batch_fraction", 1.1),
    ],
)
def test_training_rejects_invalid_ema_teacher_config(
    field: str,
    value: float,
) -> None:
    config = load_config(CONFIG)
    invalid = replace(
        config,
        loss=replace(config.loss, **{field: value}),
    )

    with pytest.raises(ValueError, match=field):
        _validate_config(invalid)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("class_conditioning_ranking_weight", -0.1),
        ("class_conditioning_ranking_start_step", -1),
        ("class_conditioning_ranking_warmup_steps", -1),
        ("class_conditioning_ranking_batch_fraction", 0.0),
        ("class_conditioning_ranking_batch_fraction", 1.1),
        ("class_conditioning_ranking_margin", -0.1),
        ("class_conditioning_ranking_margin", 1.0),
        ("class_conditioning_ranking_wrong_label_offset", 0),
        ("class_conditioning_ranking_min_timestep", -1),
    ],
)
def test_training_rejects_invalid_class_conditioning_ranking_config(
    field: str,
    value: float,
) -> None:
    config = load_config(CONFIG)
    invalid = replace(
        config,
        loss=replace(config.loss, **{field: value}),
    )

    with pytest.raises(ValueError, match=field):
        _validate_config(invalid)


def test_training_accepts_enabled_class_conditioning_ranking_config() -> None:
    config = load_config(CONFIG)
    enabled = replace(
        config,
        loss=replace(
            config.loss,
            class_conditioning_ranking_weight=0.05,
            class_conditioning_ranking_start_step=10,
            class_conditioning_ranking_warmup_steps=20,
            class_conditioning_ranking_batch_fraction=0.5,
            class_conditioning_ranking_margin=0.01,
            class_conditioning_ranking_wrong_label_offset=5,
            class_conditioning_ranking_min_timestep=8,
        ),
    )

    _validate_config(enabled)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("class_conditioning_residual_alignment_weight", -0.1),
        ("class_conditioning_residual_alignment_weight", float("nan")),
        ("class_conditioning_residual_alignment_start_step", -1),
        ("class_conditioning_residual_alignment_start_step", 1.5),
        ("class_conditioning_residual_alignment_warmup_steps", -1),
        ("class_conditioning_residual_alignment_warmup_steps", 1.5),
        ("class_conditioning_residual_alignment_batch_fraction", 0.0),
        ("class_conditioning_residual_alignment_batch_fraction", 1.1),
        ("class_conditioning_residual_alignment_batch_fraction", True),
        ("class_conditioning_residual_alignment_margin", -0.1),
        ("class_conditioning_residual_alignment_margin", 2.0),
        ("class_conditioning_residual_alignment_margin", float("nan")),
        ("class_conditioning_residual_alignment_temperature", 0.0),
        ("class_conditioning_residual_alignment_temperature", True),
        ("class_conditioning_residual_alignment_temperature", float("nan")),
        ("class_conditioning_residual_alignment_wrong_label_offsets", []),
        ("class_conditioning_residual_alignment_wrong_label_offsets", [0]),
        ("class_conditioning_residual_alignment_wrong_label_offsets", [1, 11]),
        ("class_conditioning_residual_alignment_wrong_label_offsets", [1.5]),
        ("class_conditioning_residual_alignment_wrong_label_offsets", [True]),
        ("class_conditioning_residual_alignment_min_timestep", -1),
        ("class_conditioning_residual_alignment_min_timestep", 1.5),
        ("class_conditioning_residual_alignment_pooling_factors", []),
        ("class_conditioning_residual_alignment_pooling_factors", [0]),
        ("class_conditioning_residual_alignment_pooling_factors", [2, 1]),
        ("class_conditioning_residual_alignment_pooling_factors", [3]),
        ("class_conditioning_residual_alignment_pooling_factors", [1.5]),
        ("class_conditioning_residual_alignment_pooling_factors", [True]),
        ("class_conditioning_residual_alignment_reconstruction_weight", -0.1),
        (
            "class_conditioning_residual_alignment_reconstruction_weight",
            float("nan"),
        ),
    ],
)
def test_training_rejects_invalid_residual_alignment_config(
    field: str,
    value: Any,
) -> None:
    config = load_config(CONFIG)
    values = {
        "class_conditioning_residual_alignment_weight": 0.05,
        field: value,
    }
    invalid = replace(
        config,
        loss=replace(config.loss, **values),
    )

    with pytest.raises(ValueError, match=field):
        _validate_config(invalid)


def test_training_accepts_enabled_residual_alignment_config() -> None:
    config = load_config(CONFIG)
    enabled = replace(
        config,
        loss=replace(
            config.loss,
            class_conditioning_residual_alignment_weight=0.05,
            class_conditioning_residual_alignment_start_step=10,
            class_conditioning_residual_alignment_warmup_steps=20,
            class_conditioning_residual_alignment_batch_fraction=0.5,
            class_conditioning_residual_alignment_margin=0.1,
            class_conditioning_residual_alignment_temperature=0.1,
            class_conditioning_residual_alignment_wrong_label_offsets=[1, 5],
            class_conditioning_residual_alignment_min_timestep=8,
            class_conditioning_residual_alignment_pooling_factors=[4, 8],
            class_conditioning_residual_alignment_reconstruction_weight=0.25,
        ),
    )

    _validate_config(enabled)


def test_training_rejects_two_semantic_alignment_objectives() -> None:
    config = load_config(CONFIG)
    invalid = replace(
        config,
        loss=replace(
            config.loss,
            class_conditioning_ranking_weight=0.05,
            class_conditioning_residual_alignment_weight=0.05,
            class_conditioning_residual_alignment_pooling_factors=[4, 8],
        ),
    )

    with pytest.raises(ValueError, match="cannot both be enabled"):
        _validate_config(invalid)


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
    checkpoint_git = checkpoint["extra_state"]["git"]

    assert checkpoint["extra_state"]["runtime_environment"]["schema_version"] == 1
    assert integrity["runtime_environment_sha256"] == environment_sha
    assert latest["runtime_environment_sha256"] == environment_sha
    assert integrity["git_revision"] == checkpoint_git["revision"]
    assert integrity["git_branch"] == checkpoint_git["branch"]
    assert integrity["git_dirty"] == checkpoint_git["dirty"]
    assert latest["git_revision"] == checkpoint_git["revision"]

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
    assert report["runtime_environment_sha256"] == runtime_environment_sha256(
        report["runtime_environment"]
    )
    assert report["checkpoint_written"] is False
    assert not list(output.glob("checkpoint_step_*.pt"))
    assert not (output / "training_report.json").exists()
