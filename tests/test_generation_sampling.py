from __future__ import annotations

import torch

from cofitok.configs import DiffusionConfig
from cofitok.diffusion import DiffusionSchedule, ddim_sample
from cofitok.models.cofitok import CoFiTokOutput
from scripts.generate_samples import (
    _prepare_sampling_manifest,
    _sample_generators,
    _sample_seed,
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
    directory.mkdir()
    (directory / "000000.png").touch()
    (directory / "000001.png").touch()
    _validate_numbered_output(directory, 0, 2)
    (directory / "000003.png").touch()

    try:
        _validate_numbered_output(directory, 0, 2)
    except RuntimeError as error:
        assert "extra=1" in str(error)
    else:
        raise AssertionError("extra stale image was accepted")
