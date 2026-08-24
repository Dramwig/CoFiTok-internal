from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from numbers import Integral, Real

import torch
import torch.nn.functional as F

from cofitok.diffusion import DiffusionSchedule


@dataclass(frozen=True)
class ClassConditioningRankingResult:
    loss: torch.Tensor
    correct_mse: torch.Tensor
    wrong_mse: torch.Tensor
    null_mse: torch.Tensor
    correct_better_wrong_fraction: torch.Tensor
    correct_better_null_fraction: torch.Tensor
    selected_fraction: torch.Tensor


@dataclass(frozen=True)
class ClassConditioningResidualAlignmentResult:
    loss: torch.Tensor
    direction_loss: torch.Tensor
    contrastive_loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    correct_alignment_cosine: torch.Tensor
    wrong_alignment_cosine: torch.Tensor
    correct_x0_mse: torch.Tensor
    wrong_x0_mse: torch.Tensor
    null_x0_mse: torch.Tensor
    correct_better_wrong_fraction: torch.Tensor
    correct_better_null_fraction: torch.Tensor
    selected_fraction: torch.Tensor
    wrong_condition_count: torch.Tensor


def _per_sample_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("class-conditioning prediction and target shapes differ")
    return F.mse_loss(prediction.float(), target.float(), reduction="none").flatten(1).mean(1)


def _connected_zero(model: torch.nn.Module, reference: torch.Tensor) -> torch.Tensor:
    parameter = next(model.parameters(), None)
    if parameter is None:
        return reference.sum() * 0.0
    return parameter.reshape(-1)[0] * 0.0


def _pooled(tensor: torch.Tensor, factor: int) -> torch.Tensor:
    if factor == 1:
        return tensor
    return F.avg_pool2d(tensor, kernel_size=factor, stride=factor)


def _multiscale_per_sample_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pooling_factors: Sequence[int],
) -> torch.Tensor:
    return torch.stack(
        [
            _per_sample_mse(_pooled(prediction, factor), _pooled(target, factor))
            for factor in pooling_factors
        ],
        dim=0,
    ).mean(dim=0)


def _multiscale_per_sample_cosine(
    update: torch.Tensor,
    target: torch.Tensor,
    pooling_factors: Sequence[int],
) -> torch.Tensor:
    if update.shape != target.shape:
        raise ValueError("class-conditioning update and target shapes differ")
    return torch.stack(
        [
            F.cosine_similarity(
                _pooled(update.float(), factor).flatten(1),
                _pooled(target.float(), factor).flatten(1),
                dim=1,
                eps=1e-8,
            )
            for factor in pooling_factors
        ],
        dim=0,
    ).mean(dim=0)


def _validate_residual_alignment_contract(
    *,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor,
    num_classes: int,
    batch_fraction: float,
    margin: float,
    temperature: float,
    wrong_label_offsets: Sequence[int],
    min_timestep: int,
    pooling_factors: Sequence[int],
    reconstruction_weight: float,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    if noisy_images.ndim != 4:
        raise ValueError("class-conditioning images must have shape [batch, channels, H, W]")
    if noisy_images.shape != clean_images.shape:
        raise ValueError("class-conditioning noisy and clean image shapes differ")
    if timesteps.ndim != 1 or timesteps.shape[0] != noisy_images.shape[0]:
        raise ValueError("class-conditioning timesteps must have shape [batch]")
    if class_labels.ndim != 1 or class_labels.shape[0] != noisy_images.shape[0]:
        raise ValueError("class-conditioning labels must have shape [batch]")
    if (
        timesteps.dtype == torch.bool
        or timesteps.is_floating_point()
        or timesteps.is_complex()
    ):
        raise ValueError("class-conditioning timesteps must contain integers")
    if (
        class_labels.dtype == torch.bool
        or class_labels.is_floating_point()
        or class_labels.is_complex()
    ):
        raise ValueError("class-conditioning labels must contain integers")
    if not isinstance(num_classes, Integral) or isinstance(num_classes, bool):
        raise ValueError("class-conditioning num_classes must be an integer")
    if num_classes < 2:
        raise ValueError("class-conditioning residual alignment requires at least two classes")
    if (
        not isinstance(batch_fraction, Real)
        or isinstance(batch_fraction, bool)
        or not math.isfinite(batch_fraction)
        or not 0.0 < batch_fraction <= 1.0
    ):
        raise ValueError(
            "class-conditioning residual alignment batch_fraction must be in (0, 1]"
        )
    if (
        not isinstance(margin, Real)
        or isinstance(margin, bool)
        or not math.isfinite(margin)
        or not 0.0 <= margin < 2.0
    ):
        raise ValueError(
            "class-conditioning residual alignment margin must be finite and in [0, 2)"
        )
    if (
        not isinstance(temperature, Real)
        or isinstance(temperature, bool)
        or not math.isfinite(temperature)
        or temperature <= 0.0
    ):
        raise ValueError(
            "class-conditioning residual alignment temperature must be finite and positive"
        )
    if not isinstance(min_timestep, Integral) or isinstance(min_timestep, bool):
        raise ValueError(
            "class-conditioning residual alignment min_timestep must be an integer"
        )
    if min_timestep < 0:
        raise ValueError(
            "class-conditioning residual alignment min_timestep must be non-negative"
        )
    if (
        not isinstance(reconstruction_weight, Real)
        or isinstance(reconstruction_weight, bool)
        or not math.isfinite(reconstruction_weight)
        or reconstruction_weight < 0.0
    ):
        raise ValueError(
            "class-conditioning residual alignment reconstruction_weight must be "
            "finite and non-negative"
        )

    raw_offsets = tuple(wrong_label_offsets)
    if not raw_offsets:
        raise ValueError(
            "class-conditioning residual alignment wrong_label_offsets must not be empty"
        )
    if any(
        not isinstance(offset, Integral) or isinstance(offset, bool)
        for offset in raw_offsets
    ):
        raise ValueError(
            "class-conditioning residual alignment wrong_label_offsets must contain integers"
        )
    offsets = tuple(int(offset) for offset in raw_offsets)
    residues = [offset % num_classes for offset in offsets]
    if any(residue == 0 for residue in residues):
        raise ValueError(
            "class-conditioning residual alignment wrong_label_offsets must change the class"
        )
    if len(set(residues)) != len(residues):
        raise ValueError(
            "class-conditioning residual alignment wrong_label_offsets must be "
            "unique modulo num_classes"
        )

    raw_factors = tuple(pooling_factors)
    if not raw_factors or any(
        not isinstance(factor, Integral) or isinstance(factor, bool)
        for factor in raw_factors
    ):
        raise ValueError(
            "class-conditioning residual alignment pooling_factors must contain integers"
        )
    factors = tuple(int(factor) for factor in raw_factors)
    if any(factor < 1 for factor in factors):
        raise ValueError(
            "class-conditioning residual alignment pooling_factors must be positive"
        )
    if list(factors) != sorted(set(factors)):
        raise ValueError(
            "class-conditioning residual alignment pooling_factors must be sorted and unique"
        )
    height, width = noisy_images.shape[-2:]
    if any(
        factor > min(height, width)
        or height % factor != 0
        or width % factor != 0
        for factor in factors
    ):
        raise ValueError(
            "class-conditioning residual alignment pooling_factors must divide the image shape"
        )
    return offsets, factors


def class_conditioning_residual_alignment_loss(
    model: torch.nn.Module,
    *,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor,
    num_classes: int,
    batch_fraction: float,
    margin: float,
    temperature: float,
    wrong_label_offsets: Sequence[int],
    min_timestep: int,
    pooling_factors: Sequence[int],
    reconstruction_weight: float,
) -> ClassConditioningResidualAlignmentResult:
    """Align the correct-class low-frequency x0 update with the null residual.

    The correct branch is the only differentiable branch. Wrong-class and
    classifier-free-null predictions are detached references. Conditioning is
    still consumed only by the predictor; the synthesis operator receives the
    predictor's token outputs exactly as it does for the primary epsilon loss.
    """

    offsets, factors = _validate_residual_alignment_contract(
        noisy_images=noisy_images,
        clean_images=clean_images,
        timesteps=timesteps,
        class_labels=class_labels,
        num_classes=num_classes,
        batch_fraction=batch_fraction,
        margin=margin,
        temperature=temperature,
        wrong_label_offsets=wrong_label_offsets,
        min_timestep=min_timestep,
        pooling_factors=pooling_factors,
        reconstruction_weight=reconstruction_weight,
    )
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
        connected_zero = _connected_zero(model, noisy_images)
        return ClassConditioningResidualAlignmentResult(
            loss=connected_zero,
            direction_loss=scalar_zero,
            contrastive_loss=scalar_zero,
            reconstruction_loss=scalar_zero,
            correct_alignment_cosine=scalar_zero,
            wrong_alignment_cosine=scalar_zero,
            correct_x0_mse=scalar_zero,
            wrong_x0_mse=scalar_zero,
            null_x0_mse=scalar_zero,
            correct_better_wrong_fraction=scalar_zero,
            correct_better_null_fraction=scalar_zero,
            selected_fraction=scalar_zero,
            wrong_condition_count=scalar_zero,
        )

    selected = valid_indices[:selected_count]
    selected_images = noisy_images[selected]
    selected_clean = clean_images[selected]
    selected_timesteps = timesteps[selected]
    selected_labels = labels[selected]

    was_training = model.training
    model.eval()
    try:
        correct_output = model(
            selected_images,
            selected_timesteps,
            class_labels=selected_labels,
        )
        correct_x0 = schedule.predict_x0_from_epsilon(
            selected_images.float(),
            correct_output.epsilon.float(),
            selected_timesteps,
        )
        with torch.no_grad():
            null_output = model(
                selected_images,
                selected_timesteps,
                class_labels=None,
                force_unconditional=True,
            )
            null_x0 = schedule.predict_x0_from_epsilon(
                selected_images.float(),
                null_output.epsilon.float(),
                selected_timesteps,
            ).detach()
            wrong_x0s = []
            for offset in offsets:
                wrong_labels = (selected_labels + offset) % num_classes
                wrong_output = model(
                    selected_images,
                    selected_timesteps,
                    class_labels=wrong_labels,
                )
                wrong_x0s.append(
                    schedule.predict_x0_from_epsilon(
                        selected_images.float(),
                        wrong_output.epsilon.float(),
                        selected_timesteps,
                    ).detach()
                )
    finally:
        model.train(was_training)

    target_residual = selected_clean.float() - null_x0
    correct_update = correct_x0 - null_x0
    correct_cosine = _multiscale_per_sample_cosine(
        correct_update,
        target_residual,
        factors,
    )
    wrong_cosines = torch.stack(
        [
            _multiscale_per_sample_cosine(
                wrong_x0 - null_x0,
                target_residual,
                factors,
            )
            for wrong_x0 in wrong_x0s
        ],
        dim=0,
    )
    direction_loss = (1.0 - correct_cosine).mean()
    contrastive_loss = (
        temperature
        * F.softplus(
            (wrong_cosines - correct_cosine.unsqueeze(0) + margin) / temperature
        )
    ).mean()

    correct_mse = _multiscale_per_sample_mse(
        correct_x0,
        selected_clean.float(),
        factors,
    )
    null_mse = _multiscale_per_sample_mse(
        null_x0,
        selected_clean.float(),
        factors,
    )
    wrong_mses = torch.stack(
        [
            _multiscale_per_sample_mse(
                wrong_x0,
                selected_clean.float(),
                factors,
            )
            for wrong_x0 in wrong_x0s
        ],
        dim=0,
    )
    reference_mse = 0.5 * (null_mse + wrong_mses.mean(dim=0))
    reconstruction_loss = (
        correct_mse / reference_mse.detach().clamp_min(1e-8)
    ).mean()
    loss = (
        direction_loss
        + contrastive_loss
        + reconstruction_weight * reconstruction_loss
    )
    return ClassConditioningResidualAlignmentResult(
        loss=loss,
        direction_loss=direction_loss.detach(),
        contrastive_loss=contrastive_loss.detach(),
        reconstruction_loss=reconstruction_loss.detach(),
        correct_alignment_cosine=correct_cosine.detach().mean(),
        wrong_alignment_cosine=wrong_cosines.mean(),
        correct_x0_mse=correct_mse.detach().mean(),
        wrong_x0_mse=wrong_mses.mean(),
        null_x0_mse=null_mse.mean(),
        correct_better_wrong_fraction=(
            correct_mse.detach().unsqueeze(0) < wrong_mses
        ).float().mean(),
        correct_better_null_fraction=(
            correct_mse.detach() < null_mse
        ).float().mean(),
        selected_fraction=torch.as_tensor(
            selected_count / noisy_images.shape[0],
            dtype=torch.float32,
            device=noisy_images.device,
        ),
        wrong_condition_count=torch.as_tensor(
            len(offsets),
            dtype=torch.float32,
            device=noisy_images.device,
        ),
    )


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
