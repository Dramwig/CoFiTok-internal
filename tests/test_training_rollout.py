from types import SimpleNamespace

import pytest
import torch

from cofitok.configs import DiffusionConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.training.rollout import (
    ema_teacher_consistency_loss,
    one_step_rollout_consistency_loss,
    rollout_consistency_loss,
    rollout_consistency_weight_scale,
)


class _ScaledImagePredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.register_buffer("fixed_offset", torch.tensor(0.0), persistent=False)
        self.calls = 0

    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        class_labels: torch.Tensor | None,
    ) -> SimpleNamespace:
        del timesteps, class_labels
        self.calls += 1
        return SimpleNamespace(epsilon=images * self.scale + self.fixed_offset)


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


def test_ema_teacher_consistency_uses_shadow_and_restores_training_mode() -> None:
    model = _ScaledImagePredictor()
    model.train()
    images = torch.randn(4, 3, 8, 8)
    timesteps = torch.arange(4)
    labels = torch.arange(4)
    student = model(images, timesteps, class_labels=labels).epsilon
    ema_state = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    ema_state["scale"].fill_(1.0)

    loss = ema_teacher_consistency_loss(
        model,
        ema_state=ema_state,
        student_epsilon=student,
        noisy_images=images,
        timesteps=timesteps,
        class_labels=labels,
        batch_fraction=0.5,
    )
    loss.backward()

    expected = ((images[:2] * 0.5) - images[:2]).square().mean()
    torch.testing.assert_close(loss, expected)
    assert model.scale.grad is not None
    assert model.training is True
    assert ema_state["scale"].grad is None


@pytest.mark.parametrize("batch_fraction", [0.0, -0.1, 1.1])
def test_ema_teacher_consistency_rejects_invalid_fraction(
    batch_fraction: float,
) -> None:
    model = _ScaledImagePredictor()
    images = torch.randn(2, 3, 4, 4)
    output = model(images, torch.arange(2), class_labels=None)

    with pytest.raises(ValueError, match="batch_fraction"):
        ema_teacher_consistency_loss(
            model,
            ema_state=model.state_dict(),
            student_epsilon=output.epsilon,
            noisy_images=images,
            timesteps=torch.arange(2),
            class_labels=None,
            batch_fraction=batch_fraction,
        )


def test_ema_teacher_consistency_rejects_mismatched_state() -> None:
    model = _ScaledImagePredictor()
    images = torch.randn(2, 3, 4, 4)
    output = model(images, torch.arange(2), class_labels=None)

    with pytest.raises(ValueError, match="state structure"):
        ema_teacher_consistency_loss(
            model,
            ema_state={},
            student_epsilon=output.epsilon,
            noisy_images=images,
            timesteps=torch.arange(2),
            class_labels=None,
            batch_fraction=0.5,
        )


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


def test_rollout_consistency_supports_two_detached_generated_steps() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=32, schedule_type="cosine"),
        device="cpu",
    )
    model = _ScaledImagePredictor()
    clean = torch.randn(4, 3, 8, 8).clamp(-1.0, 1.0)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([31, 24, 16, 8])
    noisy = schedule.add_noise(clean, noise, timesteps)
    first_epsilon = torch.randn_like(clean, requires_grad=True)

    loss = rollout_consistency_loss(
        model,
        first_epsilon=first_epsilon,
        schedule=schedule,
        noisy_images=noisy,
        clean_images=clean,
        timesteps=timesteps,
        class_labels=None,
        timestep_delta=4,
        unroll_steps=2,
        batch_fraction=0.5,
        clip_x0=True,
        mode="clipped_x0",
    )
    loss.backward()

    assert 0.0 <= loss.item() <= 4.0
    assert model.scale.grad is not None
    assert first_epsilon.grad is None
    assert model.calls == 2
