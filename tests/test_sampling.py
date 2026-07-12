import os

import torch

from scripts import evaluate_generated_samples_stream
from scripts.evaluate_generated_samples_stream import _ensure_project_torch_home, collect_generated_features
from scripts.sample_checkpoint import parse_prefix_budgets, save_individual_samples, select_sampling_timesteps


def test_select_sampling_timesteps_descends_to_zero() -> None:
    timesteps = select_sampling_timesteps(num_train_timesteps=10, sample_steps=4)

    assert timesteps == sorted(timesteps, reverse=True)
    assert timesteps[0] == 9
    assert timesteps[-1] == 0


def test_select_sampling_timesteps_caps_to_schedule_length() -> None:
    timesteps = select_sampling_timesteps(num_train_timesteps=4, sample_steps=99)

    assert timesteps == [3, 2, 1, 0]


def test_parse_prefix_budgets_keeps_valid_unique_values_and_full_budget() -> None:
    assert parse_prefix_budgets("1,4,4,99", token_count=8) == [1, 4, 8]
    assert parse_prefix_budgets("", token_count=4) == [1, 2, 4]


def test_save_individual_samples_writes_prefix_directory(tmp_path) -> None:
    samples = torch.zeros(2, 3, 4, 4)

    paths = save_individual_samples(samples, tmp_path, prefix_budget=8)

    assert len(paths) == 2
    assert (tmp_path / "samples_prefix_8" / "sample_00000.png").exists()
    assert (tmp_path / "samples_prefix_8" / "sample_00001.png").exists()


def test_collect_generated_features_streams_sample_batches(monkeypatch) -> None:
    calls = []

    def fake_ddim_sample(**kwargs):
        calls.append(kwargs["shape"])
        return torch.zeros(kwargs["shape"])

    monkeypatch.setattr(evaluate_generated_samples_stream, "ddim_sample", fake_ddim_sample)

    lowres, inception, count = collect_generated_features(
        model=object(),
        schedule=object(),
        image_shape=(3, 4, 4),
        total_samples=5,
        batch_size=2,
        sample_steps=3,
        prefix_budget=1,
        eta=0.0,
        clip_x0=True,
        feature_size=2,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
        inception_model=None,
    )

    assert calls == [(2, 3, 4, 4), (2, 3, 4, 4), (1, 3, 4, 4)]
    assert lowres.shape[0] == 5
    assert inception is None
    assert count == 5


def test_stream_evaluator_infers_torch_home_from_checkpoint(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "CoFiTok" / "checkpoints" / "run" / "checkpoint_final.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"placeholder")
    monkeypatch.delenv("TORCH_HOME", raising=False)

    _ensure_project_torch_home(checkpoint)

    assert os.environ["TORCH_HOME"] == str((tmp_path / "CoFiTok" / "checkpoints" / "torch_cache").resolve())
