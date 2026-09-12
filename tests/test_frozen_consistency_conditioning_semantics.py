"""Characterize frozen loss semantics; these tests do not propose a recipe fix.

The counterexamples isolate mathematical possibilities. They neither measure
their prevalence in a trained checkpoint nor establish the collapse's cause.
"""

from types import SimpleNamespace

import pytest
import torch

from cofitok.configs import DiffusionConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.models.scalable_unet import ScalableUNetTokenPredictor
from cofitok.training.rollout import ema_teacher_consistency_loss, rollout_consistency_loss


class ConditioningOnly(torch.nn.Module):
    # Reuse the production label/null/dropout path, isolating it from the trunk.
    _conditioning = ScalableUNetTokenPredictor._conditioning

    def __init__(self, dropout):
        super().__init__()
        self.class_dropout_prob = dropout
        self.null_class = 2
        self.class_embed = torch.nn.Embedding(3, 1)
        self.time_embed = lambda timesteps: torch.zeros(len(timesteps), 1)
        with torch.no_grad():
            self.class_embed.weight.copy_(torch.tensor([[1.0], [-1.0], [0.0]]))

    def forward(self, images, timesteps, *, class_labels):
        embedding = self._conditioning(timesteps, class_labels, False)
        return SimpleNamespace(epsilon=embedding[:, :, None, None].expand_as(images))


@pytest.mark.parametrize("dropout,expected_loss,expected_null_gradient", [(0.0, 0.0, 0.0), (1.0, 1.0, -2.0)])
def test_identical_ema_can_penalize_different_conditioning(dropout, expected_loss, expected_null_gradient):
    model = ConditioningOnly(dropout).train()
    images = torch.zeros(1, 1, 2, 2)
    timesteps, labels = torch.tensor([500]), torch.tensor([0])
    original = {key: value.detach().clone() for key, value in model.state_dict().items()}
    student = model(images, timesteps, class_labels=labels).epsilon
    loss = ema_teacher_consistency_loss(
        model, ema_state=original, student_epsilon=student, noisy_images=images,
        timesteps=timesteps, class_labels=labels, batch_fraction=1.0,
    )
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss)
    assert model.class_embed.weight.grad[2].item() == pytest.approx(expected_null_gradient)
    assert model.class_embed.weight.grad[0].item() == 0.0  # teacher is detached
    assert model.training
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, original[key])


class TimestepScaledEpsilon(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, images, timesteps, *, class_labels):
        # Keep both unrolled predictions strictly outside the clipping interval,
        # rather than landing exactly on a nondifferentiable clamp boundary.
        epsilon = self.value * (32 - timesteps).view(-1, 1, 1, 1)
        return SimpleNamespace(epsilon=epsilon.expand_as(images))


@pytest.mark.parametrize("mode", ["clipped_x0", "epsilon"])
def test_positive_saturated_rollout_loss_can_have_zero_gradient(mode):
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=32, schedule_type="cosine"), "cpu")
    model = TimestepScaledEpsilon(-100)
    zeros = torch.zeros(1, 1, 2, 2)
    loss = rollout_consistency_loss(
        model, first_epsilon=zeros, schedule=schedule, noisy_images=zeros,
        clean_images=zeros, timesteps=torch.tensor([24]), class_labels=None,
        timestep_delta=4, unroll_steps=2, batch_fraction=1.0,
        clip_x0=True, mode=mode,
    )
    loss.backward()
    assert loss.item() > 0
    if mode == "clipped_x0":
        assert loss.item() == pytest.approx(1)
        assert model.value.grad.item() == 0
    else:
        assert abs(model.value.grad.item()) > 0
