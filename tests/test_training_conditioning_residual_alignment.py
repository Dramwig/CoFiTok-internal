from types import SimpleNamespace

import pytest
import torch

from cofitok.configs import DiffusionConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.training.conditioning import (
    class_conditioning_residual_alignment_loss,
)


class _ResidualPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.class_values = torch.nn.Parameter(
            torch.tensor([-0.5, 1.0, 0.75, 0.25, 0.0])
        )
        self.call_training_modes: list[bool] = []
        self.call_labels: list[torch.Tensor] = []

    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        class_labels: torch.Tensor | None,
        force_unconditional: bool = False,
    ) -> SimpleNamespace:
        del timesteps
        self.call_training_modes.append(self.training)
        if force_unconditional:
            labels = torch.full(
                (images.shape[0],),
                4,
                dtype=torch.long,
                device=images.device,
            )
        else:
            assert class_labels is not None
            labels = class_labels
        self.call_labels.append(labels.detach().cpu())
        values = self.class_values[labels].reshape(-1, 1, 1, 1)
        return SimpleNamespace(epsilon=values.expand_as(images))


def _schedule() -> DiffusionSchedule:
    return DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=10),
        device="cpu",
    )


def _aligned_clean_images(
    schedule: DiffusionSchedule,
    timesteps: torch.Tensor,
    *,
    image_size: int = 4,
) -> torch.Tensor:
    alpha = schedule.sqrt_alphas_cumprod[timesteps]
    sigma = schedule.sqrt_one_minus_alphas_cumprod[timesteps]
    values = (sigma / alpha).reshape(-1, 1, 1, 1)
    return values.expand(-1, 1, image_size, image_size).clone()


def test_residual_alignment_uses_correct_direction_and_detached_references() -> None:
    model = _ResidualPredictor()
    model.train()
    schedule = _schedule()
    timesteps = torch.tensor([5, 6])
    images = torch.zeros(2, 1, 4, 4)

    result = class_conditioning_residual_alignment_loss(
        model,
        schedule=schedule,
        noisy_images=images,
        clean_images=_aligned_clean_images(schedule, timesteps),
        timesteps=timesteps,
        class_labels=torch.tensor([0, 0]),
        num_classes=4,
        batch_fraction=1.0,
        margin=0.1,
        temperature=0.1,
        wrong_label_offsets=[1, 2],
        min_timestep=0,
        pooling_factors=[1, 2],
        reconstruction_weight=0.25,
    )
    result.loss.backward()

    assert model.training is True
    assert model.call_training_modes == [False, False, False, False]
    assert [labels.tolist() for labels in model.call_labels] == [
        [0, 0],
        [4, 4],
        [1, 1],
        [2, 2],
    ]
    assert result.correct_alignment_cosine.item() == pytest.approx(1.0)
    assert result.wrong_alignment_cosine.item() == pytest.approx(-1.0)
    assert result.direction_loss.item() == pytest.approx(0.0, abs=1e-6)
    assert result.correct_x0_mse.item() < result.null_x0_mse.item()
    assert result.null_x0_mse.item() < result.wrong_x0_mse.item()
    assert result.correct_better_wrong_fraction.item() == 1.0
    assert result.correct_better_null_fraction.item() == 1.0
    assert result.selected_fraction.item() == 1.0
    assert result.wrong_condition_count.item() == 2.0
    assert model.class_values.grad is not None
    assert model.class_values.grad[0].abs().item() > 0.0
    assert torch.equal(
        model.class_values.grad[1:],
        torch.zeros_like(model.class_values.grad[1:]),
    )


def test_residual_alignment_returns_connected_zero_without_eligible_timestep() -> None:
    model = _ResidualPredictor()
    model.eval()
    result = class_conditioning_residual_alignment_loss(
        model,
        schedule=_schedule(),
        noisy_images=torch.zeros(2, 1, 4, 4),
        clean_images=torch.zeros(2, 1, 4, 4),
        timesteps=torch.tensor([1, 2]),
        class_labels=torch.tensor([0, 1]),
        num_classes=4,
        batch_fraction=0.5,
        margin=0.0,
        temperature=0.1,
        wrong_label_offsets=[1, 2],
        min_timestep=5,
        pooling_factors=[1, 2],
        reconstruction_weight=0.25,
    )
    result.loss.backward()

    assert model.training is False
    assert model.call_training_modes == []
    assert result.loss.item() == 0.0
    assert result.selected_fraction.item() == 0.0
    assert model.class_values.grad is not None
    assert torch.equal(model.class_values.grad, torch.zeros_like(model.class_values))


class _FailingReferencePredictor(_ResidualPredictor):
    def forward(
        self,
        images: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        class_labels: torch.Tensor | None,
        force_unconditional: bool = False,
    ) -> SimpleNamespace:
        if force_unconditional:
            raise RuntimeError("reference failure")
        return super().forward(
            images,
            timesteps,
            class_labels=class_labels,
            force_unconditional=force_unconditional,
        )


def test_residual_alignment_restores_mode_after_reference_failure() -> None:
    model = _FailingReferencePredictor()
    model.train()

    with pytest.raises(RuntimeError, match="reference failure"):
        class_conditioning_residual_alignment_loss(
            model,
            schedule=_schedule(),
            noisy_images=torch.zeros(1, 1, 4, 4),
            clean_images=torch.zeros(1, 1, 4, 4),
            timesteps=torch.tensor([5]),
            class_labels=torch.tensor([0]),
            num_classes=4,
            batch_fraction=1.0,
            margin=0.0,
            temperature=0.1,
            wrong_label_offsets=[1],
            min_timestep=0,
            pooling_factors=[1, 2],
            reconstruction_weight=0.0,
        )

    assert model.training is True


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"batch_fraction": 0.0}, "batch_fraction"),
        ({"batch_fraction": True}, "batch_fraction"),
        ({"num_classes": 2.5}, "num_classes"),
        ({"margin": -0.1}, "margin"),
        ({"margin": 2.0}, "margin"),
        ({"margin": float("nan")}, "margin"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": True}, "temperature"),
        ({"temperature": float("nan")}, "temperature"),
        ({"wrong_label_offsets": []}, "wrong_label_offsets"),
        ({"wrong_label_offsets": [0]}, "wrong_label_offsets"),
        ({"wrong_label_offsets": [1, 5]}, "wrong_label_offsets"),
        ({"wrong_label_offsets": [1.5]}, "wrong_label_offsets"),
        ({"wrong_label_offsets": [True]}, "wrong_label_offsets"),
        ({"min_timestep": -1}, "min_timestep"),
        ({"min_timestep": 1.5}, "min_timestep"),
        ({"pooling_factors": []}, "pooling_factors"),
        ({"pooling_factors": [0]}, "pooling_factors"),
        ({"pooling_factors": [2, 1]}, "pooling_factors"),
        ({"pooling_factors": [3]}, "pooling_factors"),
        ({"pooling_factors": [1.5]}, "pooling_factors"),
        ({"pooling_factors": [True]}, "pooling_factors"),
        ({"reconstruction_weight": -0.1}, "reconstruction_weight"),
        ({"reconstruction_weight": True}, "reconstruction_weight"),
        ({"reconstruction_weight": float("nan")}, "reconstruction_weight"),
    ],
)
def test_residual_alignment_rejects_invalid_contract(
    override: dict,
    message: str,
) -> None:
    model = _ResidualPredictor()
    arguments = {
        "schedule": _schedule(),
        "noisy_images": torch.zeros(2, 1, 4, 4),
        "clean_images": torch.zeros(2, 1, 4, 4),
        "timesteps": torch.tensor([5, 6]),
        "class_labels": torch.tensor([0, 1]),
        "num_classes": 4,
        "batch_fraction": 0.5,
        "margin": 0.0,
        "temperature": 0.1,
        "wrong_label_offsets": [1, 2],
        "min_timestep": 0,
        "pooling_factors": [1, 2],
        "reconstruction_weight": 0.25,
    }
    arguments.update(override)

    with pytest.raises(ValueError, match=message):
        class_conditioning_residual_alignment_loss(model, **arguments)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("noisy_images", torch.zeros(1, 4, 4), "images must have shape"),
        ("timesteps", torch.tensor([5.0, 6.0]), "timesteps must contain integers"),
        ("class_labels", torch.tensor([0.0, 1.0]), "labels must contain integers"),
    ],
)
def test_residual_alignment_rejects_invalid_tensor_contract(
    field: str,
    value: torch.Tensor,
    message: str,
) -> None:
    arguments = {
        "schedule": _schedule(),
        "noisy_images": torch.zeros(2, 1, 4, 4),
        "clean_images": torch.zeros(2, 1, 4, 4),
        "timesteps": torch.tensor([5, 6]),
        "class_labels": torch.tensor([0, 1]),
        "num_classes": 4,
        "batch_fraction": 0.5,
        "margin": 0.0,
        "temperature": 0.1,
        "wrong_label_offsets": [1, 2],
        "min_timestep": 0,
        "pooling_factors": [1, 2],
        "reconstruction_weight": 0.25,
    }
    arguments[field] = value
    if field == "noisy_images":
        arguments["clean_images"] = torch.zeros_like(value)
        arguments["timesteps"] = torch.tensor([5])
        arguments["class_labels"] = torch.tensor([0])

    with pytest.raises(ValueError, match=message):
        class_conditioning_residual_alignment_loss(
            _ResidualPredictor(),
            **arguments,
        )
