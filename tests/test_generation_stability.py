from __future__ import annotations

from types import SimpleNamespace

import torch

from cofitok.configs import DiffusionConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.generation.stability import (
    component_energy_ratios,
    deterministic_ddim_step,
    high_frequency_energy_ratio,
    oracle_epsilon,
    predict_guided_components,
)
from cofitok.models.cofitok import CoFiTokOutput


class _TwoComponentClassModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=3)

    def forward(self, images, timesteps, class_labels=None, force_unconditional=False):
        del timesteps
        if force_unconditional:
            values = torch.zeros(images.shape[0], device=images.device)
        else:
            values = torch.where(
                class_labels == self.config.num_classes,
                torch.zeros_like(class_labels),
                class_labels + 1,
            ).to(dtype=images.dtype)
        first = values.view(-1, 1, 1, 1).expand_as(images)
        second = 2.0 * first
        return CoFiTokOutput(
            tokens=[first, second],
            components=[first, second],
            prefix_epsilons=[first, first + second],
            epsilon=first + second,
        )


def test_component_energy_ratios_are_per_sample_and_normalized() -> None:
    first = torch.ones(2, 1, 2, 2)
    second = torch.stack([2.0 * first[0], 3.0 * first[0]])
    ratios = component_energy_ratios([first, second])

    torch.testing.assert_close(ratios.sum(dim=1), torch.ones(2))
    torch.testing.assert_close(ratios[0], torch.tensor([0.2, 0.8]))
    torch.testing.assert_close(ratios[1], torch.tensor([0.1, 0.9]))


def test_high_frequency_ratio_distinguishes_constant_and_checkerboard() -> None:
    constant = torch.ones(1, 1, 8, 8)
    checkerboard = torch.tensor(
        [[[(1.0 if (row + column) % 2 == 0 else -1.0) for column in range(8)] for row in range(8)]]
    )

    assert float(high_frequency_energy_ratio(constant)) == 0.0
    assert float(high_frequency_energy_ratio(checkerboard)) > 0.5


def test_guided_components_sum_to_cfg_epsilon() -> None:
    model = _TwoComponentClassModel()
    images = torch.zeros(2, 1, 2, 2)
    timesteps = torch.tensor([4, 4])
    labels = torch.tensor([0, 2])
    prediction = predict_guided_components(
        model,
        images,
        timesteps,
        class_labels=labels,
        guidance_scale=1.5,
        guidance_rescale=0.0,
        cfg_batch_mode="batched",
    )

    expected = torch.tensor([4.5, 13.5]).view(2, 1, 1, 1).expand_as(images)
    torch.testing.assert_close(prediction.epsilon, expected)
    torch.testing.assert_close(torch.stack(prediction.components).sum(dim=0), expected)
    assert len(prediction.reference_output.tokens) == 2


def test_oracle_epsilon_ddim_step_recovers_clean_image() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=16, schedule_type="cosine"),
        device="cpu",
    )
    clean = torch.linspace(-0.8, 0.8, 16).reshape(1, 1, 4, 4)
    noise = torch.randn(clean.shape, generator=torch.Generator().manual_seed(7))
    timestep = 15
    time_batch = torch.tensor([timestep])
    noisy = schedule.add_noise(clean, noise, time_batch)
    epsilon = oracle_epsilon(schedule, noisy, clean, time_batch)

    endpoint, raw_x0, predicted_x0 = deterministic_ddim_step(
        schedule,
        noisy,
        epsilon,
        timestep=timestep,
        previous_timestep=-1,
        clip_x0=True,
    )

    torch.testing.assert_close(raw_x0, clean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(predicted_x0, clean, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(endpoint, clean, rtol=1e-4, atol=1e-4)
