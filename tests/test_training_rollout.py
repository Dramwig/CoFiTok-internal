from types import SimpleNamespace

import pytest
import torch

from cofitok.configs import DiffusionConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.training.rollout import (
    one_step_rollout_consistency_loss,
    rollout_consistency_weight_scale,
)


class _ScaledImagePredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        class_labels: torch.Tensor | None,
    ) -> SimpleNamespace:
        del timesteps, class_labels
        return SimpleNamespace(epsilon=images * self.scale)


@pytest.mark.parametrize(
    ("step", "expected"),
    [(0, 0.0), (10, 0.0), (15, 0.5), (20, 1.0), (25, 1.0)],
)
def test_rollout_consistency_weight_scale(step: int, expected: float) -> None:
    assert rollout_consistency_weight_scale(
        step,
        start_step=10,
        warmup_steps=10,
    ) == pytest.approx(expected)


def test_rollout_consistency_uses_detached_generated_state() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=32, schedule_type="cosine"),
        device="cpu",
    )
    model = _ScaledImagePredictor()
    clean = torch.randn(4, 3, 8, 8)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([31, 24, 16, 8])
    noisy = schedule.add_noise(clean, noise, timesteps)
    first_epsilon = torch.randn_like(clean, requires_grad=True)

    loss = one_step_rollout_consistency_loss(
        model,
        first_epsilon=first_epsilon,
        schedule=schedule,
        noisy_images=noisy,
        clean_images=clean,
        timesteps=timesteps,
        class_labels=torch.tensor([0, 1, 2, 3]),
        timestep_delta=4,
        batch_fraction=0.5,
        clip_x0=True,
        mode="epsilon",
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert model.scale.grad is not None
    assert first_epsilon.grad is None


def test_rollout_consistency_returns_connected_zero_without_valid_timesteps() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=16, schedule_type="cosine"),
        device="cpu",
    )
    model = _ScaledImagePredictor()
    clean = torch.randn(2, 3, 4, 4)
    first_epsilon = torch.randn_like(clean, requires_grad=True)

    loss = one_step_rollout_consistency_loss(
        model,
        first_epsilon=first_epsilon,
        schedule=schedule,
        noisy_images=clean,
        clean_images=clean,
        timesteps=torch.tensor([0, 1]),
        class_labels=None,
        timestep_delta=4,
        batch_fraction=0.5,
        clip_x0=True,
        mode="epsilon",
    )
    loss.backward()

    assert loss.item() == 0.0
    assert first_epsilon.grad is not None
    torch.testing.assert_close(first_epsilon.grad, torch.zeros_like(first_epsilon))


def test_rollout_consistency_supports_bounded_x0_loss() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=32, schedule_type="cosine"),
        device="cpu",
    )
    model = _ScaledImagePredictor()
    clean = torch.randn(4, 3, 8, 8).clamp(-1.0, 1.0)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([31, 24, 16, 8])
    noisy = schedule.add_noise(clean, noise, timesteps)

    loss = one_step_rollout_consistency_loss(
        model,
        first_epsilon=torch.randn_like(clean),
        schedule=schedule,
        noisy_images=noisy,
        clean_images=clean,
        timesteps=timesteps,
        class_labels=None,
        timestep_delta=4,
        batch_fraction=0.5,
        clip_x0=True,
        mode="clipped_x0",
    )
    loss.backward()

    assert 0.0 <= loss.item() <= 4.0
    assert model.scale.grad is not None
