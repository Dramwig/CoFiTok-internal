from __future__ import annotations

from dataclasses import dataclass

import torch

from cofitok.configs import ExperimentConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokTiny
from cofitok.training.losses import LossBreakdown, compute_losses


@dataclass
class SmokeStepOutput:
    losses: LossBreakdown
    output_shapes: dict[str, object]
    component_energy: list[torch.Tensor]


def _shape_list(tensors: list[torch.Tensor]) -> list[list[int]]:
    return [list(tensor.shape) for tensor in tensors]


def run_smoke_step(
    config: ExperimentConfig,
    model: CoFiTokTiny,
    schedule: DiffusionSchedule,
    clean_images: torch.Tensor,
) -> SmokeStepOutput:
    device = clean_images.device
    noise = torch.randn_like(clean_images)
    timesteps = schedule.sample_timesteps(clean_images.shape[0], device=device)
    noisy_images = schedule.add_noise(clean_images, noise, timesteps)
    output = model(noisy_images, timesteps)
    zero_components = model.synthesis.zero_components_like(output.tokens)
    losses = compute_losses(
        config.loss,
        output,
        schedule,
        noisy_images,
        clean_images,
        noise,
        timesteps,
        zero_components=zero_components,
    )
    component_energy = [component.pow(2).mean().detach() for component in output.components]
    shapes = {
        "tokens": _shape_list(output.tokens),
        "components": _shape_list(output.components),
        "prefix_epsilons": _shape_list(output.prefix_epsilons),
        "epsilon": list(output.epsilon.shape),
    }
    return SmokeStepOutput(losses=losses, output_shapes=shapes, component_energy=component_energy)
