from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from cofitok.diffusion import DiffusionSchedule


def rollout_consistency_weight_scale(
    step: int,
    *,
    start_step: int,
    warmup_steps: int,
) -> float:
    if step <= start_step:
        return 0.0
    if warmup_steps <= 0:
        return 1.0
    return min((step - start_step) / warmup_steps, 1.0)


def one_step_rollout_consistency_loss(
    model: torch.nn.Module,
    *,
    first_epsilon: torch.Tensor,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor | None,
    timestep_delta: int,
    batch_fraction: float,
    clip_x0: bool,
) -> torch.Tensor:
    if timestep_delta < 1:
        raise ValueError("rollout consistency timestep_delta must be positive")
    if not 0.0 < batch_fraction <= 1.0:
        raise ValueError("rollout consistency batch_fraction must be in (0, 1]")

    valid_indices = torch.nonzero(timesteps >= timestep_delta, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return first_epsilon.sum() * 0.0
    sample_count = min(
        valid_indices.numel(),
        max(1, math.ceil(first_epsilon.shape[0] * batch_fraction)),
    )
    selected = valid_indices[:sample_count]
    selected_timesteps = timesteps[selected]
    previous_timesteps = selected_timesteps - timestep_delta

    with torch.no_grad():
        selected_noisy = noisy_images[selected].float()
        selected_clean = clean_images[selected].float()
        selected_epsilon = first_epsilon[selected].detach().float()
        predicted_x0 = schedule.predict_x0_from_epsilon(
            selected_noisy,
            selected_epsilon,
            selected_timesteps,
        )
        if clip_x0:
            predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
        alpha_previous = schedule.sqrt_alphas_cumprod[previous_timesteps].view(-1, 1, 1, 1)
        sigma_previous = schedule.sqrt_one_minus_alphas_cumprod[previous_timesteps].view(
            -1,
            1,
            1,
            1,
        )
        previous_images = alpha_previous * predicted_x0 + sigma_previous * selected_epsilon
        target_epsilon = (
            previous_images - alpha_previous * selected_clean
        ) / sigma_previous.clamp_min(1e-8)

    selected_labels = class_labels[selected] if class_labels is not None else None
    second_output = model(
        previous_images.to(dtype=noisy_images.dtype),
        previous_timesteps,
        class_labels=selected_labels,
    )
    return F.mse_loss(second_output.epsilon.float(), target_epsilon)
