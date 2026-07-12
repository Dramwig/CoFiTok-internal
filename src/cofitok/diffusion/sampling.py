from __future__ import annotations

from collections.abc import Sequence

import torch

from cofitok.diffusion.schedule import DiffusionSchedule
from cofitok.models.cofitok import CoFiTokOutput


def select_sampling_timesteps(num_train_timesteps: int, sample_steps: int) -> list[int]:
    if sample_steps < 1:
        raise ValueError("sample_steps must be positive")
    count = min(num_train_timesteps, sample_steps)
    values = torch.linspace(0, num_train_timesteps - 1, steps=count).round().long().unique()
    timesteps = sorted((int(value) for value in values), reverse=True)
    if timesteps[-1] != 0:
        timesteps.append(0)
    return timesteps


def prefix_epsilon(output: CoFiTokOutput, prefix_budget: int) -> torch.Tensor:
    if not 1 <= prefix_budget <= len(output.prefix_epsilons):
        raise ValueError("prefix_budget is outside the model token range")
    return output.prefix_epsilons[prefix_budget - 1]


def _guidance_rescale(
    guided: torch.Tensor,
    conditional: torch.Tensor,
    amount: float,
) -> torch.Tensor:
    if amount <= 0.0:
        return guided
    dims = tuple(range(1, guided.ndim))
    guided_std = guided.std(dim=dims, keepdim=True).clamp_min(1e-6)
    conditional_std = conditional.std(dim=dims, keepdim=True)
    rescaled = guided * (conditional_std / guided_std)
    return torch.lerp(guided, rescaled, amount)


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    generator: torch.Generator | None,
    sample_generators: Sequence[torch.Generator] | None,
) -> torch.Tensor:
    if sample_generators is None:
        if generator is None:
            raise ValueError("generator or sample_generators must be provided")
        return torch.randn(shape, device=device, generator=generator)
    if len(sample_generators) != shape[0]:
        raise ValueError("sample_generators must match the batch dimension")
    return torch.stack(
        [
            torch.randn(shape[1:], device=device, generator=sample_generator)
            for sample_generator in sample_generators
        ],
        dim=0,
    )


@torch.no_grad()
def predict_epsilon(
    model: torch.nn.Module,
    images: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    prefix_budget: int,
    class_labels: torch.Tensor | None,
    guidance_scale: float,
    guidance_rescale: float,
) -> torch.Tensor:
    conditional = prefix_epsilon(
        model(images, timesteps, class_labels=class_labels),
        prefix_budget,
    )
    if class_labels is None or guidance_scale == 1.0:
        return conditional
    unconditional = prefix_epsilon(
        model(
            images,
            timesteps,
            class_labels=class_labels,
            force_unconditional=True,
        ),
        prefix_budget,
    )
    guided = unconditional + guidance_scale * (conditional - unconditional)
    return _guidance_rescale(guided, conditional, guidance_rescale)


@torch.no_grad()
def ddim_sample(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int],
    *,
    sample_steps: int,
    prefix_budget: int,
    eta: float,
    clip_x0: bool,
    device: torch.device,
    generator: torch.Generator | None = None,
    sample_generators: Sequence[torch.Generator] | None = None,
    class_labels: torch.Tensor | None = None,
    guidance_scale: float = 1.0,
    guidance_rescale: float = 0.0,
) -> torch.Tensor:
    if eta < 0.0:
        raise ValueError("eta must be non-negative")
    if guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    model.eval()
    images = _randn(
        shape,
        device=device,
        generator=generator,
        sample_generators=sample_generators,
    )
    timesteps = select_sampling_timesteps(schedule.num_train_timesteps, sample_steps)
    for index, timestep in enumerate(timesteps):
        previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        time_batch = torch.full((shape[0],), timestep, dtype=torch.long, device=device)
        epsilon = predict_epsilon(
            model,
            images,
            time_batch,
            prefix_budget=prefix_budget,
            class_labels=class_labels,
            guidance_scale=guidance_scale,
            guidance_rescale=guidance_rescale,
        )
        predicted_x0 = schedule.predict_x0_from_epsilon(images, epsilon, time_batch)
        if clip_x0:
            predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
        if previous < 0:
            images = predicted_x0
            continue
        alpha_t = schedule.alphas_cumprod[timestep]
        alpha_previous = schedule.alphas_cumprod[previous]
        sigma = eta * torch.sqrt((1.0 - alpha_previous) / (1.0 - alpha_t).clamp_min(1e-12))
        sigma = sigma * torch.sqrt((1.0 - alpha_t / alpha_previous).clamp_min(0.0))
        direction = torch.sqrt((1.0 - alpha_previous - sigma.square()).clamp_min(0.0))
        images = torch.sqrt(alpha_previous) * predicted_x0 + direction * epsilon
        if eta > 0.0:
            images = images + sigma * _randn(
                tuple(images.shape),
                device=device,
                generator=generator,
                sample_generators=sample_generators,
            )
    return images
