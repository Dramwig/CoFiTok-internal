from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F

from cofitok.configs import LossConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    epsilon: torch.Tensor
    prefix: torch.Tensor
    monotonic: torch.Tensor
    zero_token: torch.Tensor
    energy_budget: torch.Tensor
    residual_component: torch.Tensor
    component_decorrelation: torch.Tensor
    tail_floor: torch.Tensor
    sampled_prefix: torch.Tensor
    sampled_component: torch.Tensor
    sampled_budget: torch.Tensor
    group_residual: torch.Tensor
    epsilon_band_prefix: torch.Tensor
    epsilon_band_component: torch.Tensor
    denoise_path_prefix: torch.Tensor
    denoise_path_component: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total.detach(),
            "epsilon": self.epsilon.detach(),
            "prefix": self.prefix.detach(),
            "monotonic": self.monotonic.detach(),
            "zero_token": self.zero_token.detach(),
            "energy_budget": self.energy_budget.detach(),
            "residual_component": self.residual_component.detach(),
            "component_decorrelation": self.component_decorrelation.detach(),
            "tail_floor": self.tail_floor.detach(),
            "sampled_prefix": self.sampled_prefix.detach(),
            "sampled_component": self.sampled_component.detach(),
            "sampled_budget": self.sampled_budget.detach(),
            "group_residual": self.group_residual.detach(),
            "epsilon_band_prefix": self.epsilon_band_prefix.detach(),
            "epsilon_band_component": self.epsilon_band_component.detach(),
            "denoise_path_prefix": self.denoise_path_prefix.detach(),
            "denoise_path_component": self.denoise_path_component.detach(),
        }


def _prefix_targets(clean_images: torch.Tensor, count: int) -> list[torch.Tensor]:
    targets = []
    max_scale = min(clean_images.shape[-2:])
    for index in range(count):
        if index == count - 1:
            targets.append(clean_images)
            continue
        scale = min(2 ** (count - index - 1), max_scale)
        pooled = F.avg_pool2d(clean_images, kernel_size=scale, stride=scale, ceil_mode=True)
        restored = F.interpolate(pooled, size=clean_images.shape[-2:], mode="bilinear", align_corners=False)
        targets.append(restored)
    return targets


def _coarse_to_fine_scales(count: int, shape: tuple[int, int]) -> list[int]:
    if count <= 0:
        return []
    max_scale = max(1, min(shape))
    if count == 1 or max_scale <= 1:
        return [1 for _ in range(count)]
    max_exponent = math.log2(max_scale)
    if count == 2:
        return [max(2, int(round(2**max_exponent))), 1]
    scales = []
    for index in range(count - 1):
        fraction = index / max(count - 2, 1)
        exponent = max_exponent + (1.0 - max_exponent) * fraction
        scales.append(min(max_scale, max(2, int(round(2**exponent)))))
    scales.append(1)
    return scales


def _lowpass_like(tensor: torch.Tensor, scale: int) -> torch.Tensor:
    if scale <= 1:
        return tensor
    pooled = F.avg_pool2d(tensor, kernel_size=scale, stride=scale, ceil_mode=True)
    return F.interpolate(pooled, size=tensor.shape[-2:], mode="bilinear", align_corners=False)


def _epsilon_band_prefix_targets(noise: torch.Tensor, count: int) -> list[torch.Tensor]:
    scales = _coarse_to_fine_scales(count, noise.shape[-2:])
    return [_lowpass_like(noise, scale) for scale in scales]


def _prefix_loss(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    targets = _prefix_targets(clean_images, len(output.prefix_epsilons))
    losses = []
    for prefix_epsilon, target in zip(output.prefix_epsilons, targets):
        pred_x0 = schedule.predict_x0_from_epsilon(noisy_images, prefix_epsilon, timesteps)
        losses.append(F.mse_loss(pred_x0, target))
    return torch.stack(losses).mean()


def _monotonic_loss(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    if len(output.prefix_epsilons) < 2:
        return output.epsilon.new_zeros(())
    errors = []
    for prefix_epsilon in output.prefix_epsilons:
        pred_x0 = schedule.predict_x0_from_epsilon(noisy_images, prefix_epsilon, timesteps)
        errors.append(F.mse_loss(pred_x0, clean_images))
    penalties = []
    for previous, current in zip(errors[:-1], errors[1:]):
        penalties.append(torch.relu(current - previous + margin))
    return torch.stack(penalties).mean()


def _energy_budget_loss(
    output: CoFiTokOutput,
    target: list[float],
) -> torch.Tensor:
    if not target:
        return output.epsilon.new_zeros(())
    if len(target) != len(output.components):
        raise ValueError(f"Energy target has {len(target)} entries for {len(output.components)} components")
    energies = torch.stack([component.pow(2).mean() for component in output.components])
    ratios = energies / energies.sum().clamp_min(1e-12)
    target_tensor = torch.tensor(target, dtype=ratios.dtype, device=ratios.device)
    target_tensor = target_tensor / target_tensor.sum().clamp_min(1e-12)
    return F.mse_loss(ratios, target_tensor)


def _component_energy_ratios(output: CoFiTokOutput) -> torch.Tensor:
    energies = torch.stack([component.pow(2).mean() for component in output.components])
    return energies / energies.sum().clamp_min(1e-12)


def _component_decorrelation_loss(output: CoFiTokOutput) -> torch.Tensor:
    if len(output.components) < 2:
        return output.epsilon.new_zeros(())
    flattened = torch.stack([component.flatten(1) for component in output.components], dim=0)
    normalized = F.normalize(flattened, dim=-1, eps=1e-12)
    correlations = torch.einsum("kbn,lbn->klb", normalized, normalized).abs()
    pair_losses = []
    for first in range(len(output.components)):
        for second in range(first + 1, len(output.components)):
            pair_losses.append(correlations[first, second].mean())
    return torch.stack(pair_losses).mean()


def _tail_floor_loss(output: CoFiTokOutput, min_ratio: float) -> torch.Tensor:
    if min_ratio <= 0.0 or len(output.components) < 2:
        return output.epsilon.new_zeros(())
    ratios = _component_energy_ratios(output)
    deficits = torch.relu(output.epsilon.new_tensor(min_ratio) - ratios[1:])
    return deficits.pow(2).mean()


def _target_prefix_epsilon(
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    target_x0: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    alpha = schedule.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    sigma = schedule.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    return (noisy_images - alpha * target_x0) / sigma.clamp_min(0.05)


def _residual_component_loss(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    targets = _prefix_targets(clean_images, len(output.components))
    target_prefix_epsilons = [
        _target_prefix_epsilon(schedule, noisy_images, target, timesteps)
        for target in targets
    ]
    previous = torch.zeros_like(target_prefix_epsilons[0])
    losses = []
    for component, target_prefix in zip(output.components, target_prefix_epsilons):
        target_component = target_prefix - previous
        losses.append(F.mse_loss(component, target_component))
        previous = target_prefix
    return torch.stack(losses).mean()


def _sampled_progressive_losses(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    count = len(output.prefix_epsilons)
    if count == 0:
        zero = output.epsilon.new_zeros(())
        return zero, zero, zero
    budget = torch.randint(1, count + 1, (), device=output.epsilon.device)
    index = int(budget.detach().cpu().item()) - 1
    targets = _prefix_targets(clean_images, count)
    pred_x0 = schedule.predict_x0_from_epsilon(noisy_images, output.prefix_epsilons[index], timesteps)
    sampled_prefix_loss = F.mse_loss(pred_x0, targets[index])

    target_prefix = _target_prefix_epsilon(schedule, noisy_images, targets[index], timesteps)
    if index == 0:
        previous_target_prefix = torch.zeros_like(target_prefix)
    else:
        previous_target_prefix = _target_prefix_epsilon(schedule, noisy_images, targets[index - 1], timesteps)
    target_component = target_prefix - previous_target_prefix
    sampled_component_loss = F.mse_loss(output.components[index], target_component)
    return sampled_prefix_loss, sampled_component_loss, budget.to(dtype=output.epsilon.dtype)


def _group_residual_loss(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if group_size <= 0:
        return output.epsilon.new_zeros(())
    count = len(output.components)
    targets = _prefix_targets(clean_images, count)
    target_prefix_epsilons = [
        _target_prefix_epsilon(schedule, noisy_images, target, timesteps)
        for target in targets
    ]
    losses = []
    for start in range(0, count, group_size):
        end = min(start + group_size, count) - 1
        predicted_group = torch.stack(output.components[start : end + 1]).sum(dim=0)
        previous_target = (
            torch.zeros_like(target_prefix_epsilons[end])
            if start == 0
            else target_prefix_epsilons[start - 1]
        )
        target_group = target_prefix_epsilons[end] - previous_target
        losses.append(F.mse_loss(predicted_group, target_group))
    return torch.stack(losses).mean()


def _epsilon_band_losses(
    output: CoFiTokOutput,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    targets = _epsilon_band_prefix_targets(noise, len(output.prefix_epsilons))
    prefix_losses = [
        F.mse_loss(prefix_epsilon, target)
        for prefix_epsilon, target in zip(output.prefix_epsilons, targets)
    ]
    previous_target = torch.zeros_like(targets[0])
    component_losses = []
    for component, target in zip(output.components, targets):
        component_losses.append(F.mse_loss(component, target - previous_target))
        previous_target = target
    return torch.stack(prefix_losses).mean(), torch.stack(component_losses).mean()


def _denoise_path_prefix_targets(
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    count: int,
    progress_power: float,
) -> list[torch.Tensor]:
    spatial_targets = _prefix_targets(clean_images, count)
    zero_epsilon = torch.zeros_like(noisy_images)
    start_x0 = schedule.predict_x0_from_epsilon(noisy_images, zero_epsilon, timesteps)
    targets = []
    power = max(progress_power, 1e-6)
    for index, spatial_target in enumerate(spatial_targets):
        progress = ((index + 1) / count) ** power
        targets.append(torch.lerp(start_x0, spatial_target, progress))
    return targets


def _denoise_path_losses(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    progress_power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    spatial_targets = _prefix_targets(clean_images, len(output.prefix_epsilons))
    power = max(progress_power, 1e-6)
    # The denoise path starts at epsilon=0 and is linear in x0. Construct
    # targets directly in epsilon space to avoid catastrophic cancellation at
    # the near-zero terminal alpha of cosine schedules.
    target_prefix_epsilons = []
    for index, spatial_target in enumerate(spatial_targets):
        progress = ((index + 1) / len(spatial_targets)) ** power
        target_prefix_epsilons.append(
            progress * _target_prefix_epsilon(
                schedule,
                noisy_images,
                spatial_target,
                timesteps,
            )
        )
    prefix_losses = [
        F.mse_loss(prefix_epsilon, target_prefix)
        for prefix_epsilon, target_prefix in zip(output.prefix_epsilons, target_prefix_epsilons)
    ]
    previous_target = torch.zeros_like(target_prefix_epsilons[0])
    component_losses = []
    for component, target_prefix in zip(output.components, target_prefix_epsilons):
        component_losses.append(F.mse_loss(component, target_prefix - previous_target))
        previous_target = target_prefix
    return torch.stack(prefix_losses).mean(), torch.stack(component_losses).mean()


def compute_losses(
    config: LossConfig,
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    zero_components: list[torch.Tensor] | None = None,
) -> LossBreakdown:
    epsilon_loss = F.mse_loss(output.epsilon, noise)
    prefix_loss = _prefix_loss(output, schedule, noisy_images, clean_images, timesteps)
    monotonic_loss = _monotonic_loss(
        output,
        schedule,
        noisy_images,
        clean_images,
        timesteps,
        margin=config.monotonic_margin,
    )
    if zero_components:
        zero_token_loss = torch.stack([component.pow(2).mean() for component in zero_components]).mean()
    else:
        zero_token_loss = output.epsilon.new_zeros(())
    energy_budget_loss = _energy_budget_loss(output, config.energy_target)
    if config.component_decorrelation_weight > 0.0:
        component_decorrelation_loss = _component_decorrelation_loss(output)
    else:
        component_decorrelation_loss = output.epsilon.new_zeros(())
    tail_floor_loss = _tail_floor_loss(output, config.tail_floor_min_ratio)
    if config.residual_component_weight > 0.0:
        residual_component_loss = _residual_component_loss(
            output,
            schedule,
            noisy_images,
            clean_images,
            timesteps,
        )
    else:
        residual_component_loss = output.epsilon.new_zeros(())
    if config.sampled_prefix_weight > 0.0 or config.sampled_component_weight > 0.0:
        sampled_prefix_loss, sampled_component_loss, sampled_budget = _sampled_progressive_losses(
            output,
            schedule,
            noisy_images,
            clean_images,
            timesteps,
        )
    else:
        sampled_prefix_loss = output.epsilon.new_zeros(())
        sampled_component_loss = output.epsilon.new_zeros(())
        sampled_budget = output.epsilon.new_zeros(())
    if config.group_residual_weight > 0.0:
        group_residual_loss = _group_residual_loss(
            output,
            schedule,
            noisy_images,
            clean_images,
            timesteps,
            group_size=config.group_residual_size,
        )
    else:
        group_residual_loss = output.epsilon.new_zeros(())
    if config.epsilon_band_prefix_weight > 0.0 or config.epsilon_band_component_weight > 0.0:
        epsilon_band_prefix_loss, epsilon_band_component_loss = _epsilon_band_losses(
            output,
            noise,
        )
    else:
        epsilon_band_prefix_loss = output.epsilon.new_zeros(())
        epsilon_band_component_loss = output.epsilon.new_zeros(())
    if config.denoise_path_prefix_weight > 0.0 or config.denoise_path_component_weight > 0.0:
        denoise_path_prefix_loss, denoise_path_component_loss = _denoise_path_losses(
            output,
            schedule,
            noisy_images,
            clean_images,
            timesteps,
            progress_power=config.denoise_path_progress_power,
        )
    else:
        denoise_path_prefix_loss = output.epsilon.new_zeros(())
        denoise_path_component_loss = output.epsilon.new_zeros(())
    total = (
        config.epsilon_weight * epsilon_loss
        + config.prefix_weight * prefix_loss
        + config.monotonic_weight * monotonic_loss
        + config.zero_token_weight * zero_token_loss
        + config.energy_budget_weight * energy_budget_loss
        + config.residual_component_weight * residual_component_loss
        + config.component_decorrelation_weight * component_decorrelation_loss
        + config.tail_floor_weight * tail_floor_loss
        + config.sampled_prefix_weight * sampled_prefix_loss
        + config.sampled_component_weight * sampled_component_loss
        + config.group_residual_weight * group_residual_loss
        + config.epsilon_band_prefix_weight * epsilon_band_prefix_loss
        + config.epsilon_band_component_weight * epsilon_band_component_loss
        + config.denoise_path_prefix_weight * denoise_path_prefix_loss
        + config.denoise_path_component_weight * denoise_path_component_loss
    )
    return LossBreakdown(
        total=total,
        epsilon=epsilon_loss,
        prefix=prefix_loss,
        monotonic=monotonic_loss,
        zero_token=zero_token_loss,
        energy_budget=energy_budget_loss,
        residual_component=residual_component_loss,
        component_decorrelation=component_decorrelation_loss,
        tail_floor=tail_floor_loss,
        sampled_prefix=sampled_prefix_loss,
        sampled_component=sampled_component_loss,
        sampled_budget=sampled_budget,
        group_residual=group_residual_loss,
        epsilon_band_prefix=epsilon_band_prefix_loss,
        epsilon_band_component=epsilon_band_component_loss,
        denoise_path_prefix=denoise_path_prefix_loss,
        denoise_path_component=denoise_path_component_loss,
    )
