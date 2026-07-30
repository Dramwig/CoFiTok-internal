from __future__ import annotations

import math
from collections.abc import Mapping

import torch
import torch.nn.functional as F
from torch.func import functional_call

from cofitok.diffusion import DiffusionSchedule


def consistency_weight_scale(
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


def rollout_consistency_weight_scale(
    step: int,
    *,
    start_step: int,
    warmup_steps: int,
) -> float:
    return consistency_weight_scale(
        step,
        start_step=start_step,
        warmup_steps=warmup_steps,
    )


def ema_teacher_consistency_loss(
    model: torch.nn.Module,
    *,
    ema_state: Mapping[str, torch.Tensor],
    student_epsilon: torch.Tensor,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor | None,
    batch_fraction: float,
) -> torch.Tensor:
    if not 0.0 < batch_fraction <= 1.0:
        raise ValueError("EMA teacher consistency batch_fraction must be in (0, 1]")
    if student_epsilon.shape[0] != noisy_images.shape[0]:
        raise ValueError("EMA teacher consistency batch dimensions do not match")
    if ema_state.keys() != model.state_dict().keys():
        raise ValueError("EMA teacher state does not match model state structure")
    sample_count = max(1, math.ceil(student_epsilon.shape[0] * batch_fraction))
    selected = slice(0, sample_count)
    selected_labels = class_labels[selected] if class_labels is not None else None
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            teacher_output = functional_call(
                model,
                ema_state,
                (noisy_images[selected], timesteps[selected]),
                {"class_labels": selected_labels},
                strict=False,
            )
    finally:
        model.train(was_training)
    return F.mse_loss(
        student_epsilon[selected].float(),
        teacher_output.epsilon.detach().float(),
    )


def rollout_consistency_loss(
    model: torch.nn.Module,
    *,
    first_epsilon: torch.Tensor,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    class_labels: torch.Tensor | None,
    timestep_delta: int,
    unroll_steps: int,
    batch_fraction: float,
    clip_x0: bool,
    mode: str,
) -> torch.Tensor:
    if timestep_delta < 1:
        raise ValueError("rollout consistency timestep_delta must be positive")
    if unroll_steps < 1:
        raise ValueError("rollout consistency unroll_steps must be positive")
    if not 0.0 < batch_fraction <= 1.0:
        raise ValueError("rollout consistency batch_fraction must be in (0, 1]")
    if mode not in {"epsilon", "clipped_x0"}:
        raise ValueError("rollout consistency mode must be epsilon or clipped_x0")

    minimum_timestep = timestep_delta * unroll_steps
    valid_indices = torch.nonzero(timesteps >= minimum_timestep, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return first_epsilon.sum() * 0.0
    sample_count = min(
        valid_indices.numel(),
        max(1, math.ceil(first_epsilon.shape[0] * batch_fraction)),
    )
    selected = valid_indices[:sample_count]
    selected_clean = clean_images[selected].float()
    selected_labels = class_labels[selected] if class_labels is not None else None
    current_images = noisy_images[selected].float()
    current_timesteps = timesteps[selected]
    current_epsilon = first_epsilon[selected].detach().float()
    step_losses = []
    for _ in range(unroll_steps):
        previous_timesteps = current_timesteps - timestep_delta
        with torch.no_grad():
            predicted_x0 = schedule.predict_x0_from_epsilon(
                current_images,
                current_epsilon,
                current_timesteps,
            )
            if clip_x0:
                predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
            alpha_previous = schedule.sqrt_alphas_cumprod[previous_timesteps].view(
                -1,
                1,
                1,
                1,
            )
            sigma_previous = schedule.sqrt_one_minus_alphas_cumprod[
                previous_timesteps
            ].view(-1, 1, 1, 1)
            previous_images = (
                alpha_previous * predicted_x0 + sigma_previous * current_epsilon
            )
            target_epsilon = (
                previous_images - alpha_previous * selected_clean
            ) / sigma_previous.clamp_min(1e-8)

        next_output = model(
            previous_images.to(dtype=noisy_images.dtype),
            previous_timesteps,
            class_labels=selected_labels,
        )
        if mode == "epsilon":
            step_loss = F.mse_loss(next_output.epsilon.float(), target_epsilon)
        else:
            next_x0 = schedule.predict_x0_from_epsilon(
                previous_images,
                next_output.epsilon.float(),
                previous_timesteps,
            ).clamp(-1.0, 1.0)
            step_loss = F.mse_loss(next_x0, selected_clean)
        step_losses.append(step_loss)
        current_images = previous_images
        current_timesteps = previous_timesteps
        current_epsilon = next_output.epsilon.detach().float()
    return torch.stack(step_losses).mean()


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
    mode: str,
) -> torch.Tensor:
    return rollout_consistency_loss(
        model,
        first_epsilon=first_epsilon,
        schedule=schedule,
        noisy_images=noisy_images,
        clean_images=clean_images,
        timesteps=timesteps,
        class_labels=class_labels,
        timestep_delta=timestep_delta,
        unroll_steps=1,
        batch_fraction=batch_fraction,
        clip_x0=clip_x0,
        mode=mode,
    )
