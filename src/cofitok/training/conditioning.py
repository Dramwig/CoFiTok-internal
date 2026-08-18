from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ClassConditioningRankingResult:
    loss: torch.Tensor
    correct_mse: torch.Tensor
    wrong_mse: torch.Tensor
    null_mse: torch.Tensor
    correct_better_wrong_fraction: torch.Tensor
    correct_better_null_fraction: torch.Tensor
    selected_fraction: torch.Tensor


def _per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("class-conditioning prediction and target shapes differ")
    return F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)


def _connected_zero(model: torch.nn.Module, reference: torch.Tensor) -> torch.Tensor:
    parameter = next(model.parameters(), None)
    if parameter is None:
        return reference.sum() * 0.0
    return parameter.reshape(-1)[0] * 0.0


def class_conditioning_ranking_loss(
    model: torch.nn.Module,
    *,
    noisy_images: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor,
    num_classes: int,
    batch_fraction: float,
    margin: float,
    wrong_label_offset: int,
    min_timestep: int,
) -> ClassConditioningRankingResult:
    """Rank the correct class above detached wrong and unconditional targets.

    The auxiliary correct-label forward is the only branch that retains gradients.
    Wrong-label and classifier-free-null references are evaluated under ``no_grad``
    and detached before the relative hinge is formed.  The model is temporarily put
    in evaluation mode so class dropout cannot change the three compared conditions.
    """

    if noisy_images.shape != noise.shape:
        raise ValueError("class-conditioning noisy images and noise shapes differ")
    if timesteps.ndim != 1 or timesteps.shape[0] != noisy_images.shape[0]:
        raise ValueError("class-conditioning timesteps must have shape [batch]")
    if class_labels.ndim != 1 or class_labels.shape[0] != noisy_images.shape[0]:
        raise ValueError("class-conditioning labels must have shape [batch]")
    if num_classes < 2:
        raise ValueError("class-conditioning ranking requires at least two classes")
    if not 0.0 < batch_fraction <= 1.0:
        raise ValueError("class-conditioning ranking batch_fraction must be in (0, 1]")
    if not math.isfinite(margin) or not 0.0 <= margin < 1.0:
        raise ValueError("class-conditioning ranking margin must be finite and in [0, 1)")
    if wrong_label_offset % num_classes == 0:
        raise ValueError("class-conditioning wrong_label_offset must change the class")
    if min_timestep < 0:
        raise ValueError("class-conditioning min_timestep must be non-negative")

    labels = class_labels.to(device=timesteps.device, dtype=torch.long)
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        raise ValueError("class-conditioning labels are outside the configured class range")
    valid_indices = torch.nonzero(timesteps >= min_timestep, as_tuple=False).flatten()
    selected_count = min(
        valid_indices.numel(),
        max(1, math.ceil(noisy_images.shape[0] * batch_fraction)),
    )
    scalar_zero = noisy_images.new_zeros((), dtype=torch.float32)
    if selected_count == 0:
        return ClassConditioningRankingResult(
            loss=_connected_zero(model, noisy_images),
            correct_mse=scalar_zero,
            wrong_mse=scalar_zero,
            null_mse=scalar_zero,
            correct_better_wrong_fraction=scalar_zero,
            correct_better_null_fraction=scalar_zero,
            selected_fraction=scalar_zero,
        )

    selected = valid_indices[:selected_count]
    selected_images = noisy_images[selected]
    selected_noise = noise[selected]
    selected_timesteps = timesteps[selected]
    selected_labels = labels[selected]
    wrong_labels = (selected_labels + wrong_label_offset) % num_classes

    was_training = model.training
    model.eval()
    try:
        correct_output = model(
            selected_images,
            selected_timesteps,
            class_labels=selected_labels,
        )
        correct_mse = _per_sample_mse(correct_output.epsilon, selected_noise)
        with torch.no_grad():
            wrong_output = model(
                selected_images,
                selected_timesteps,
                class_labels=wrong_labels,
            )
            wrong_mse = _per_sample_mse(
                wrong_output.epsilon,
                selected_noise,
            ).detach()
            null_output = model(
                selected_images,
                selected_timesteps,
                class_labels=None,
                force_unconditional=True,
            )
            null_mse = _per_sample_mse(
                null_output.epsilon,
                selected_noise,
            ).detach()
    finally:
        model.train(was_training)

    wrong_scale = wrong_mse.clamp_min(1e-8)
    null_scale = null_mse.clamp_min(1e-8)
    wrong_violation = (correct_mse - wrong_mse) / wrong_scale + margin
    null_violation = (correct_mse - null_mse) / null_scale + margin
    loss = 0.5 * (F.relu(wrong_violation) + F.relu(null_violation)).mean()
    return ClassConditioningRankingResult(
        loss=loss,
        correct_mse=correct_mse.detach().mean(),
        wrong_mse=wrong_mse.mean(),
        null_mse=null_mse.mean(),
        correct_better_wrong_fraction=(correct_mse.detach() < wrong_mse).float().mean(),
        correct_better_null_fraction=(correct_mse.detach() < null_mse).float().mean(),
        selected_fraction=torch.as_tensor(
            selected_count / noisy_images.shape[0],
            dtype=torch.float32,
            device=noisy_images.device,
        ),
    )
