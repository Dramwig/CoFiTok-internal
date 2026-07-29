from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cofitok.diffusion.schedule import DiffusionSchedule
from cofitok.models.cofitok import CoFiTokOutput


@dataclass(frozen=True)
class GuidedComponentPrediction:
    components: list[torch.Tensor]
    epsilon: torch.Tensor
    reference_output: CoFiTokOutput


def per_sample_mse(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return (first.float() - second.float()).square().flatten(1).mean(dim=1)


def per_sample_rms(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float().square().flatten(1).mean(dim=1).sqrt()


def high_frequency_energy_ratio(tensor: torch.Tensor) -> torch.Tensor:
    values = tensor.float()
    lowpass = F.avg_pool2d(
        values,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False,
    )
    high_frequency_energy = (values - lowpass).square().flatten(1).mean(dim=1)
    total_energy = values.square().flatten(1).mean(dim=1)
    return high_frequency_energy / total_energy.clamp_min(1e-12)


def component_energy_ratios(components: list[torch.Tensor]) -> torch.Tensor:
    if not components:
        raise ValueError("component energy ratios require at least one component")
    energies = torch.stack(
        [component.float().square().flatten(1).mean(dim=1) for component in components],
        dim=1,
    )
    return energies / energies.sum(dim=1, keepdim=True).clamp_min(1e-12)


def component_high_frequency_ratios(components: list[torch.Tensor]) -> torch.Tensor:
    if not components:
        raise ValueError("component high-frequency ratios require at least one component")
    return torch.stack(
        [high_frequency_energy_ratio(component) for component in components],
        dim=1,
    )


def tail_energy_ratio(components: list[torch.Tensor], tail_count: int = 2) -> torch.Tensor:
    ratios = component_energy_ratios(components)
    count = min(max(tail_count, 1), ratios.shape[1])
    return ratios[:, -count:].sum(dim=1)


def _guided_components(
    conditional: list[torch.Tensor],
    unconditional: list[torch.Tensor],
    guidance_scale: float,
    guidance_rescale: float,
) -> list[torch.Tensor]:
    if len(conditional) != len(unconditional):
        raise ValueError("conditional and unconditional component counts differ")
    guided = [
        unconditional_component
        + guidance_scale * (conditional_component - unconditional_component)
        for conditional_component, unconditional_component in zip(
            conditional,
            unconditional,
        )
    ]
    if guidance_rescale <= 0.0:
        return guided
    conditional_epsilon = torch.stack(conditional).sum(dim=0)
    guided_epsilon = torch.stack(guided).sum(dim=0)
    dims = tuple(range(1, guided_epsilon.ndim))
    guided_std = guided_epsilon.std(dim=dims, keepdim=True).clamp_min(1e-6)
    conditional_std = conditional_epsilon.std(dim=dims, keepdim=True)
    scale = torch.lerp(
        torch.ones_like(guided_std),
        conditional_std / guided_std,
        guidance_rescale,
    )
    return [component * scale for component in guided]


@torch.no_grad()
def predict_guided_components(
    model: torch.nn.Module,
    images: torch.Tensor,
    timesteps: torch.Tensor,
    *,
    class_labels: torch.Tensor | None,
    guidance_scale: float,
    guidance_rescale: float,
    cfg_batch_mode: str = "batched",
) -> GuidedComponentPrediction:
    if cfg_batch_mode not in {"batched", "sequential"}:
        raise ValueError("cfg_batch_mode must be batched or sequential")
    if guidance_scale < 0.0:
        raise ValueError("guidance_scale must be non-negative")
    if not 0.0 <= guidance_rescale <= 1.0:
        raise ValueError("guidance_rescale must be in [0, 1]")

    if class_labels is None or guidance_scale == 1.0:
        output = model(images, timesteps, class_labels=class_labels)
        return GuidedComponentPrediction(
            components=output.components,
            epsilon=output.epsilon,
            reference_output=output,
        )

    if cfg_batch_mode == "batched":
        model_config = getattr(model, "config", None)
        null_class = int(getattr(model_config, "num_classes", 0))
        if null_class < 1:
            raise ValueError("batched CFG requires a class-conditional model")
        joint_output = model(
            torch.cat([images, images], dim=0),
            torch.cat([timesteps, timesteps], dim=0),
            class_labels=torch.cat(
                [class_labels, torch.full_like(class_labels, null_class)],
                dim=0,
            ),
        )
        conditional = []
        unconditional = []
        for component in joint_output.components:
            conditional_component, unconditional_component = component.chunk(2)
            conditional.append(conditional_component)
            unconditional.append(unconditional_component)
        reference = CoFiTokOutput(
            tokens=[token.chunk(2)[0] for token in joint_output.tokens],
            components=conditional,
            prefix_epsilons=[],
            epsilon=torch.stack(conditional).sum(dim=0),
        )
    else:
        reference = model(images, timesteps, class_labels=class_labels)
        unconditional_output = model(
            images,
            timesteps,
            class_labels=class_labels,
            force_unconditional=True,
        )
        conditional = reference.components
        unconditional = unconditional_output.components

    components = _guided_components(
        conditional,
        unconditional,
        guidance_scale,
        guidance_rescale,
    )
    return GuidedComponentPrediction(
        components=components,
        epsilon=torch.stack(components).sum(dim=0),
        reference_output=reference,
    )


def oracle_epsilon(
    schedule: DiffusionSchedule,
    images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alpha = schedule.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sigma = schedule.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    return (images - alpha * clean_images) / sigma.clamp_min(1e-8)


def deterministic_ddim_step(
    schedule: DiffusionSchedule,
    images: torch.Tensor,
    epsilon: torch.Tensor,
    *,
    timestep: int,
    previous_timestep: int,
    clip_x0: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    time_batch = torch.full(
        (images.shape[0],),
        timestep,
        dtype=torch.long,
        device=images.device,
    )
    raw_x0 = schedule.predict_x0_from_epsilon(images, epsilon, time_batch)
    predicted_x0 = raw_x0.clamp(-1.0, 1.0) if clip_x0 else raw_x0
    if previous_timestep < 0:
        return predicted_x0, raw_x0, predicted_x0
    alpha_previous = schedule.alphas_cumprod[previous_timestep]
    direction = torch.sqrt((1.0 - alpha_previous).clamp_min(0.0))
    previous_images = torch.sqrt(alpha_previous) * predicted_x0 + direction * epsilon
    return previous_images, raw_x0, predicted_x0
