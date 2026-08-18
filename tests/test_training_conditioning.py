from types import SimpleNamespace

import pytest
import torch

from cofitok.training.conditioning import class_conditioning_ranking_loss


class _LabelPredictor(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.class_values = torch.nn.Parameter(torch.tensor([1.0, 0.5, 0.25]))
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
                2,
                dtype=torch.long,
                device=images.device,
            )
        else:
            assert class_labels is not None
            labels = class_labels
        self.call_labels.append(labels.detach().cpu())
        values = self.class_values[labels].reshape(-1, 1, 1, 1)
        return SimpleNamespace(epsilon=values.expand_as(images))


def test_ranking_updates_only_the_correct_label_branch_and_restores_mode() -> None:
    model = _LabelPredictor()
    model.train()
    images = torch.zeros(4, 1, 2, 2)
    noise = torch.zeros_like(images)
    timesteps = torch.tensor([1, 8, 9, 2])
    labels = torch.tensor([1, 0, 1, 0])

    result = class_conditioning_ranking_loss(
        model,
        noisy_images=images,
        noise=noise,
        timesteps=timesteps,
        class_labels=labels,
        num_classes=2,
        batch_fraction=0.25,
        margin=0.01,
        wrong_label_offset=1,
        min_timestep=5,
    )
    result.loss.backward()

    assert model.training is True
    assert model.call_training_modes == [False, False, False]
    assert [row.tolist() for row in model.call_labels] == [[0], [1], [2]]
    assert result.loss.item() > 0.0
    assert result.selected_fraction.item() == pytest.approx(0.25)
    assert model.class_values.grad is not None
    assert model.class_values.grad[0].item() > 0.0
    assert model.class_values.grad[1].item() == 0.0
    assert model.class_values.grad[2].item() == 0.0


def test_ranking_returns_connected_zero_when_no_timestep_is_eligible() -> None:
    model = _LabelPredictor()
    result = class_conditioning_ranking_loss(
        model,
        noisy_images=torch.zeros(2, 1, 2, 2),
        noise=torch.zeros(2, 1, 2, 2),
        timesteps=torch.tensor([1, 2]),
        class_labels=torch.tensor([0, 1]),
        num_classes=2,
        batch_fraction=0.5,
        margin=0.0,
        wrong_label_offset=1,
        min_timestep=5,
    )
    result.loss.backward()

    assert result.loss.item() == 0.0
    assert result.selected_fraction.item() == 0.0
    assert model.call_training_modes == []
    assert model.class_values.grad is not None
    assert torch.equal(model.class_values.grad, torch.zeros_like(model.class_values))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"batch_fraction": 0.0}, "batch_fraction"),
        ({"margin": 1.0}, "margin"),
        ({"wrong_label_offset": 2}, "wrong_label_offset"),
        ({"min_timestep": -1}, "min_timestep"),
    ],
)
def test_ranking_rejects_invalid_contract(kwargs: dict, message: str) -> None:
    model = _LabelPredictor()
    arguments = {
        "noisy_images": torch.zeros(2, 1, 2, 2),
        "noise": torch.zeros(2, 1, 2, 2),
        "timesteps": torch.tensor([5, 6]),
        "class_labels": torch.tensor([0, 1]),
        "num_classes": 2,
        "batch_fraction": 0.5,
        "margin": 0.0,
        "wrong_label_offset": 1,
        "min_timestep": 0,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError, match=message):
        class_conditioning_ranking_loss(model, **arguments)
