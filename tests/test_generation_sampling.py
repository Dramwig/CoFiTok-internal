from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cofitok.configs import DiffusionConfig
from cofitok.diffusion import DiffusionSchedule, ddim_sample, predict_epsilon
from cofitok.models.cofitok import CoFiTokOutput
from scripts.generate_samples import (
    _batch_complete,
    _prepare_sampling_manifest,
    _sample_generators,
    _sample_seed,
    _save_batch,
    _validate_numbered_output,
)


class _ZeroModel(torch.nn.Module):
    def __init__(self, token_count: int = 2) -> None:
        super().__init__()
        self.token_count = token_count
        self.first_inputs: list[torch.Tensor] = []

    def forward(self, images, timesteps, class_labels=None, force_unconditional=False):
        if not self.first_inputs:
            self.first_inputs.append(images.detach().clone())
        zeros = torch.zeros_like(images)
        components = [zeros for _ in range(self.token_count)]
        prefixes = [zeros for _ in range(self.token_count)]
        return CoFiTokOutput(
            tokens=components,
            components=components,
            prefix_epsilons=prefixes,
            epsilon=zeros,
        )


class _ClassModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.config = SimpleNamespace(num_classes=5)

    def forward(self, images, timesteps, class_labels=None, force_unconditional=False):
        del timesteps
        self.calls += 1
        if force_unconditional:
            values = torch.zeros(images.shape[0], device=images.device)
        else:
            values = torch.where(
                class_labels == self.config.num_classes,
                torch.zeros_like(class_labels),
                class_labels + 1,
            ).to(dtype=images.dtype)
        epsilon = values.view(-1, 1, 1, 1).expand_as(images)
        return CoFiTokOutput(
            tokens=[epsilon],
            components=[epsilon],
            prefix_epsilons=[epsilon],
            epsilon=epsilon,
        )


class _OracleX0Model(torch.nn.Module):
    def __init__(self, schedule: DiffusionSchedule, target_x0: torch.Tensor) -> None:
        super().__init__()
        self.schedule = schedule
        self.target_x0 = target_x0

    def forward(self, images, timesteps, class_labels=None, force_unconditional=False):
        del class_labels, force_unconditional
        alpha = self.schedule.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        sigma = self.schedule.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
        target = self.target_x0.expand_as(images)
        epsilon = (images - alpha * target) / sigma
        return CoFiTokOutput(
            tokens=[epsilon],
            components=[epsilon],
            prefix_epsilons=[epsilon],
            epsilon=epsilon,
        )


def _sample(
    model: torch.nn.Module,
    *,
    start: int,
    count: int,
    budget: int,
    eta: float = 0.0,
) -> torch.Tensor:
    device = torch.device("cpu")
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=8), device=device)
    return ddim_sample(
        model,
        schedule,
        (count, 1, 4, 4),
        sample_steps=4,
        prefix_budget=budget,
        eta=eta,
        clip_x0=False,
        device=device,
        sample_generators=_sample_generators(17, start, count, device),
    )


def test_sample_seed_is_stable_per_global_index() -> None:
    assert _sample_seed(17, 23) == 40
    assert _sample_seed(2**63 - 1, 1) == 0


def test_per_sample_stream_is_batch_and_resume_invariant() -> None:
    for eta in (0.0, 0.5):
        batched = _sample(_ZeroModel(), start=0, count=4, budget=2, eta=eta)
        resumed = torch.cat(
            [
                _sample(_ZeroModel(), start=0, count=1, budget=2, eta=eta),
                _sample(_ZeroModel(), start=1, count=3, budget=2, eta=eta),
            ]
        )
        torch.testing.assert_close(batched, resumed, rtol=0.0, atol=0.0)


def test_prefix_budgets_share_the_same_initial_noise() -> None:
    first = _ZeroModel()
    second = _ZeroModel()
    _sample(first, start=11, count=3, budget=1)
    _sample(second, start=11, count=3, budget=2)
    torch.testing.assert_close(first.first_inputs[0], second.first_inputs[0], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("eta", [0.0, 0.5])
@pytest.mark.parametrize("clip_x0", [False, True])
def test_ddim_oracle_epsilon_reconstructs_x0(eta: float, clip_x0: bool) -> None:
    device = torch.device("cpu")
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=32, schedule_type="cosine"),
        device=device,
    )
    target = torch.linspace(-0.8, 0.8, 16).reshape(1, 1, 4, 4)
    model = _OracleX0Model(schedule, target)

    sampled = ddim_sample(
        model,
        schedule,
        tuple(target.shape),
        sample_steps=7,
        prefix_budget=1,
        eta=eta,
        clip_x0=clip_x0,
        device=device,
        generator=torch.Generator(device=device).manual_seed(29),
    )

    torch.testing.assert_close(sampled, target, rtol=1e-5, atol=1e-5)


def test_batched_cfg_matches_sequential_cfg_with_one_forward() -> None:
    images = torch.randn(3, 1, 4, 4)
    timesteps = torch.tensor([7, 6, 5])
    labels = torch.tensor([0, 2, 4])
    batched_model = _ClassModel()
    sequential_model = _ClassModel()

    batched = predict_epsilon(
        batched_model,
        images,
        timesteps,
        prefix_budget=1,
        class_labels=labels,
        guidance_scale=1.5,
        guidance_rescale=0.0,
        cfg_batch_mode="batched",
    )
    sequential = predict_epsilon(
        sequential_model,
        images,
        timesteps,
        prefix_budget=1,
        class_labels=labels,
        guidance_scale=1.5,
        guidance_rescale=0.0,
        cfg_batch_mode="sequential",
    )

    torch.testing.assert_close(batched, sequential, rtol=0.0, atol=0.0)
    assert batched_model.calls == 1
    assert sequential_model.calls == 2


def test_sampling_manifest_allows_only_an_exact_resume(tmp_path) -> None:
    path = tmp_path / "sampling_manifest.json"
    manifest = {"checkpoint_sha256": "a" * 64, "sampling": {"seed": 17}}
    _prepare_sampling_manifest(path, manifest, resume=True, has_existing_images=False)
    _prepare_sampling_manifest(path, manifest, resume=True, has_existing_images=True)

    try:
        _prepare_sampling_manifest(
            path,
            {"checkpoint_sha256": "b" * 64, "sampling": {"seed": 17}},
            resume=True,
            has_existing_images=True,
        )
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("mismatched checkpoint manifest was accepted")


def test_sampling_manifest_rejects_unproven_existing_images(tmp_path) -> None:
    try:
        _prepare_sampling_manifest(
            tmp_path / "sampling_manifest.json",
            {"checkpoint_sha256": "a" * 64},
            resume=True,
            has_existing_images=True,
        )
    except FileExistsError as error:
        assert "without a sampling manifest" in str(error)
    else:
        raise AssertionError("unproven partial sample set was accepted")


def test_completed_sampling_report_requires_exact_numbered_output(tmp_path) -> None:
    directory = tmp_path / "prefix_8"
    _save_batch(
        torch.zeros(2, 3, 4, 4),
        directory,
        0,
        overwrite=False,
        skip_existing=False,
    )
    _validate_numbered_output(directory, 0, 2, image_size=4, image_channels=3)
    (directory / "000003.png").touch()

    try:
        _validate_numbered_output(directory, 0, 2, image_size=4, image_channels=3)
    except RuntimeError as error:
        assert "extra=1" in str(error)
    else:
        raise AssertionError("extra stale image was accepted")


def test_resume_regenerates_corrupt_png_instead_of_skipping_it(tmp_path) -> None:
    images = torch.zeros(2, 3, 4, 4)
    _save_batch(images, tmp_path, 0, overwrite=False, skip_existing=False)
    assert _batch_complete(tmp_path, 0, 2, image_size=4, image_channels=3)

    (tmp_path / "000001.png").write_bytes(b"corrupt")
    assert not _batch_complete(tmp_path, 0, 2, image_size=4, image_channels=3)

    _save_batch(images, tmp_path, 0, overwrite=False, skip_existing=True)
    assert _batch_complete(tmp_path, 0, 2, image_size=4, image_channels=3)


def test_sample_png_is_published_only_after_encoding_completes(tmp_path, monkeypatch) -> None:
    final_path = tmp_path / "000000.png"
    calls = []

    def fake_save_image(image, path, *, format):
        del image
        calls.append((path, format))
        assert not final_path.exists()
        path.write_bytes(b"complete-png")

    monkeypatch.setattr("scripts.generate_samples.save_image", fake_save_image)
    _save_batch(torch.zeros(1, 3, 4, 4), tmp_path, 0, overwrite=False, skip_existing=False)

    assert final_path.read_bytes() == b"complete-png"
    assert calls == [(tmp_path / ".000000.png.part", "png")]
    assert not (tmp_path / ".000000.png.part").exists()


def test_failed_png_encoding_leaves_no_published_or_partial_file(tmp_path, monkeypatch) -> None:
    def failing_save_image(image, path, *, format):
        del image, format
        path.write_bytes(b"partial")
        raise RuntimeError("encoding interrupted")

    monkeypatch.setattr("scripts.generate_samples.save_image", failing_save_image)
    with pytest.raises(RuntimeError, match="encoding interrupted"):
        _save_batch(torch.zeros(1, 3, 4, 4), tmp_path, 0, overwrite=False, skip_existing=False)

    assert not (tmp_path / "000000.png").exists()
    assert not (tmp_path / ".000000.png.part").exists()
