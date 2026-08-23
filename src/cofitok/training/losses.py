from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from cofitok.configs import LossConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokOutput


@dataclass
class LossBreakdown:
    total: torch.Tensor
    epsilon: torch.Tensor
    epsilon_unweighted: torch.Tensor
    min_snr_weight_mean: torch.Tensor
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
    denoise_path_energy: torch.Tensor
    low_snr_high_frequency: torch.Tensor
    rollout_consistency: torch.Tensor
    rollout_consistency_scale: torch.Tensor
    ema_teacher_consistency: torch.Tensor
    ema_teacher_consistency_scale: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "total": self.total.detach(),
            "epsilon": self.epsilon.detach(),
            "epsilon_unweighted": self.epsilon_unweighted.detach(),
            "min_snr_weight_mean": self.min_snr_weight_mean.detach(),
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
            "denoise_path_energy": self.denoise_path_energy.detach(),
            "low_snr_high_frequency": self.low_snr_high_frequency.detach(),
            "rollout_consistency": self.rollout_consistency.detach(),
            "rollout_consistency_scale": self.rollout_consistency_scale.detach(),
            "ema_teacher_consistency": (self.ema_teacher_consistency.detach()),
            "ema_teacher_consistency_scale": (
                self.ema_teacher_consistency_scale.detach()
            ),
        }


def _prefix_targets(clean_images: torch.Tensor, count: int) -> list[torch.Tensor]:
    targets = []
    max_scale = min(clean_images.shape[-2:])
    for index in range(count):
        if index == count - 1:
            targets.append(clean_images)
            continue
        scale = min(2 ** (count - index - 1), max_scale)
        pooled = F.avg_pool2d(
            clean_images, kernel_size=scale, stride=scale, ceil_mode=True
        )
        restored = F.interpolate(
            pooled, size=clean_images.shape[-2:], mode="bilinear", align_corners=False
        )
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
    return F.interpolate(
        pooled, size=tensor.shape[-2:], mode="bilinear", align_corners=False
    )


def denoise_path_schedule(
    clean_images: torch.Tensor,
    tokens: list[torch.Tensor],
    progress_power: float,
    progress_mode: str,
) -> tuple[list[torch.Tensor], list[float]]:
    count = len(tokens)
    if count == 0:
        return [], []
    if progress_mode == "power":
        power = max(progress_power, 1e-6)
        return (
            _prefix_targets(clean_images, count),
            [((index + 1) / count) ** power for index in range(count)],
        )
    if progress_mode != "token_capacity":
        raise ValueError(f"Unknown denoise path progress mode: {progress_mode}")

    output_channels = clean_images.shape[1]
    capacities = [
        min(token.shape[1], output_channels) * math.prod(token.shape[-2:])
        for token in tokens
    ]
    # Component energy is quadratic in its amplitude, so sqrt(rank) increments
    # make target energy proportional to each restricted synthesis subspace.
    progress_increments = [math.sqrt(capacity) for capacity in capacities]
    total_progress = sum(progress_increments)
    if total_progress <= 0:
        raise ValueError("Denoise path token capacities must be positive")
    progress = []
    cumulative = 0
    spatial_targets = []
    output_height, output_width = clean_images.shape[-2:]
    for token, progress_increment in zip(tokens, progress_increments):
        token_height, token_width = token.shape[-2:]
        if token_height <= 0 or token_width <= 0:
            raise ValueError("Denoise path token spatial dimensions must be positive")
        scale = max(
            1,
            math.ceil(output_height / token_height),
            math.ceil(output_width / token_width),
        )
        spatial_targets.append(_lowpass_like(clean_images, scale))
        cumulative += progress_increment
        progress.append(cumulative / total_progress)
    progress[-1] = 1.0
    return spatial_targets, progress


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
        pred_x0 = schedule.predict_x0_from_epsilon(
            noisy_images, prefix_epsilon, timesteps
        )
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
        pred_x0 = schedule.predict_x0_from_epsilon(
            noisy_images, prefix_epsilon, timesteps
        )
        errors.append(F.mse_loss(pred_x0, clean_images))
    penalties = []
    for previous, current in zip(errors[:-1], errors[1:]):
        penalties.append(torch.relu(current - previous + margin))
    return torch.stack(penalties).mean()


def _energy_budget_loss(
    output: CoFiTokOutput,
    target: list[float],
    scope: str,
) -> torch.Tensor:
    if not target:
        return output.epsilon.new_zeros(())
    if len(target) != len(output.components):
        raise ValueError(
            f"Energy target has {len(target)} entries for {len(output.components)} components"
        )
    if scope == "batch":
        energies = torch.stack(
            [component.pow(2).mean() for component in output.components]
        )
        ratios = energies / energies.sum().clamp_min(1e-12)
        target_tensor = torch.tensor(target, dtype=ratios.dtype, device=ratios.device)
        target_tensor = target_tensor / target_tensor.sum().clamp_min(1e-12)
        return F.mse_loss(ratios, target_tensor)
    if scope == "sample":
        ratios = _per_sample_component_energy_ratios(output.components)
        target_tensor = torch.tensor(target, dtype=ratios.dtype, device=ratios.device)
        target_tensor = target_tensor / target_tensor.sum().clamp_min(1e-12)
        return F.mse_loss(ratios, target_tensor.unsqueeze(0).expand_as(ratios))
    raise ValueError(f"Unknown energy budget scope: {scope}")


def _per_sample_component_energy_ratios(
    components: list[torch.Tensor],
) -> torch.Tensor:
    energies = torch.stack(
        [component.float().square().flatten(1).mean(dim=1) for component in components],
        dim=1,
    )
    return energies / energies.sum(dim=1, keepdim=True).clamp_min(1e-12)


def _component_energy_distribution_loss(
    components: list[torch.Tensor],
    target_components: list[torch.Tensor],
    mode: str,
    *,
    tokens: list[torch.Tensor] | None = None,
    output_channels: int | None = None,
    capacity_weight: float = 0.0,
    capacity_power: float = 0.5,
) -> torch.Tensor:
    predicted = _per_sample_component_energy_ratios(components)
    target = component_energy_target_ratios(
        target_components,
        tokens=tokens,
        output_channels=output_channels,
        capacity_weight=capacity_weight,
        capacity_power=capacity_power,
    )
    if mode == "mse":
        return F.mse_loss(predicted, target)
    if mode in {"hellinger", "hellinger_stable"}:
        epsilon = (
            predicted.new_tensor(1e-4)
            if mode == "hellinger_stable"
            else predicted.new_tensor(torch.finfo(predicted.dtype).eps)
        )
        distances = (
            torch.sqrt(predicted + epsilon) - torch.sqrt(target + epsilon)
        ).square()
        return 0.5 * distances.sum(dim=1).mean()
    raise ValueError(f"Unknown denoise path energy mode: {mode}")


def component_energy_target_ratios(
    target_components: list[torch.Tensor],
    *,
    tokens: list[torch.Tensor] | None = None,
    output_channels: int | None = None,
    capacity_weight: float = 0.0,
    capacity_power: float = 0.5,
) -> torch.Tensor:
    target = _per_sample_component_energy_ratios(target_components)
    if capacity_weight > 0.0:
        if tokens is None or output_channels is None:
            raise ValueError(
                "capacity energy prior requires tokens and output channels"
            )
        capacities = torch.tensor(
            [
                min(token.shape[1], output_channels) * math.prod(token.shape[-2:])
                for token in tokens
            ],
            dtype=target.dtype,
            device=target.device,
        )
        prior = capacities.pow(capacity_power)
        prior = prior / prior.sum().clamp_min(1e-12)
        target = torch.lerp(
            target,
            prior.unsqueeze(0).expand_as(target),
            capacity_weight,
        )
    return target


def _highpass_like(tensor: torch.Tensor) -> torch.Tensor:
    lowpass = F.avg_pool2d(
        tensor,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False,
    )
    return tensor - lowpass


def _low_snr_high_frequency_loss(
    output: CoFiTokOutput,
    noise: torch.Tensor,
    schedule: DiffusionSchedule,
    timesteps: torch.Tensor,
    power: float,
) -> torch.Tensor:
    per_sample = (
        (_highpass_like(output.epsilon.float()) - _highpass_like(noise.float()))
        .square()
        .flatten(1)
        .mean(dim=1)
    )
    low_snr_weight = schedule.sqrt_one_minus_alphas_cumprod[timesteps].square()
    return (per_sample * low_snr_weight.pow(power)).mean()


def _component_energy_ratios(output: CoFiTokOutput) -> torch.Tensor:
    energies = torch.stack([component.pow(2).mean() for component in output.components])
    return energies / energies.sum().clamp_min(1e-12)


def _component_decorrelation_loss(output: CoFiTokOutput) -> torch.Tensor:
    if len(output.components) < 2:
        return output.epsilon.new_zeros(())
    flattened = torch.stack(
        [component.flatten(1) for component in output.components], dim=0
    )
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
    pred_x0 = schedule.predict_x0_from_epsilon(
        noisy_images, output.prefix_epsilons[index], timesteps
    )
    sampled_prefix_loss = F.mse_loss(pred_x0, targets[index])

    target_prefix = _target_prefix_epsilon(
        schedule, noisy_images, targets[index], timesteps
    )
    if index == 0:
        previous_target_prefix = torch.zeros_like(target_prefix)
    else:
        previous_target_prefix = _target_prefix_epsilon(
            schedule, noisy_images, targets[index - 1], timesteps
        )
    target_component = target_prefix - previous_target_prefix
    sampled_component_loss = F.mse_loss(output.components[index], target_component)
    return (
        sampled_prefix_loss,
        sampled_component_loss,
        budget.to(dtype=output.epsilon.dtype),
    )


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


def denoise_path_prefix_x0_targets(
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    count: int,
    progress_power: float,
    tokens: list[torch.Tensor] | None = None,
    progress_mode: str = "power",
) -> list[torch.Tensor]:
    if tokens is None:
        if progress_mode != "power":
            raise ValueError(
                "Denoise path tokens are required for token_capacity progress"
            )
        tokens = [
            clean_images.new_empty((clean_images.shape[0], 1, 1, 1))
            for _ in range(count)
        ]
    if len(tokens) != count:
        raise ValueError(f"Expected {count} denoise path tokens, got {len(tokens)}")
    spatial_targets, progress_values = denoise_path_schedule(
        clean_images,
        tokens,
        progress_power,
        progress_mode,
    )
    zero_epsilon = torch.zeros_like(noisy_images)
    start_x0 = schedule.predict_x0_from_epsilon(noisy_images, zero_epsilon, timesteps)
    return [
        torch.lerp(start_x0, spatial_target, progress)
        for spatial_target, progress in zip(spatial_targets, progress_values)
    ]


_denoise_path_prefix_targets = denoise_path_prefix_x0_targets


def denoise_path_prefix_epsilon_targets(
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    count: int,
    progress_power: float,
    tokens: list[torch.Tensor] | None = None,
    progress_mode: str = "power",
) -> list[torch.Tensor]:
    if tokens is None:
        if progress_mode != "power":
            raise ValueError(
                "Denoise path tokens are required for token_capacity progress"
            )
        tokens = [
            clean_images.new_empty((clean_images.shape[0], 1, 1, 1))
            for _ in range(count)
        ]
    if len(tokens) != count:
        raise ValueError(f"Expected {count} denoise path tokens, got {len(tokens)}")
    spatial_targets, progress_values = denoise_path_schedule(
        clean_images,
        tokens,
        progress_power,
        progress_mode,
    )
    return [
        progress
        * _target_prefix_epsilon(
            schedule,
            noisy_images,
            spatial_target,
            timesteps,
        )
        for spatial_target, progress in zip(spatial_targets, progress_values)
    ]


def _denoise_path_losses(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    progress_power: float,
    progress_mode: str,
    energy_mode: str,
    energy_capacity_weight: float,
    energy_capacity_power: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The denoise path starts at epsilon=0 and is linear in x0. Construct
    # targets directly in epsilon space to avoid catastrophic cancellation at
    # the near-zero terminal alpha of cosine schedules.
    target_prefix_epsilons = denoise_path_prefix_epsilon_targets(
        schedule,
        noisy_images,
        clean_images,
        timesteps,
        len(output.prefix_epsilons),
        progress_power,
        tokens=output.tokens,
        progress_mode=progress_mode,
    )
    prefix_losses = [
        F.mse_loss(prefix_epsilon, target_prefix)
        for prefix_epsilon, target_prefix in zip(
            output.prefix_epsilons, target_prefix_epsilons
        )
    ]
    previous_target = torch.zeros_like(target_prefix_epsilons[0])
    component_losses = []
    target_components = []
    for component, target_prefix in zip(output.components, target_prefix_epsilons):
        target_component = target_prefix - previous_target
        target_components.append(target_component)
        component_losses.append(F.mse_loss(component, target_component))
        previous_target = target_prefix
    energy_loss = _component_energy_distribution_loss(
        output.components,
        target_components,
        energy_mode,
        tokens=output.tokens,
        output_channels=output.epsilon.shape[1],
        capacity_weight=energy_capacity_weight,
        capacity_power=energy_capacity_power,
    )
    return (
        torch.stack(prefix_losses).mean(),
        torch.stack(component_losses).mean(),
        energy_loss,
    )


def compute_losses(
    config: LossConfig,
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    zero_components: list[torch.Tensor] | None = None,
    rollout_consistency: torch.Tensor | None = None,
    rollout_consistency_scale: float = 1.0,
    ema_teacher_consistency: torch.Tensor | None = None,
    ema_teacher_consistency_scale: float = 1.0,
) -> LossBreakdown:
    def float32_scalar(value: float) -> torch.Tensor:
        return torch.as_tensor(
            value,
            dtype=torch.float32,
            device=output.epsilon.device,
        )

    if config.min_snr_gamma <= 0.0:
        # Preserve the exact legacy reduction and dtype when Min-SNR is off so
        # existing epsilon checkpoints retain their original training path.
        epsilon_loss = F.mse_loss(output.epsilon, noise)
        epsilon_unweighted_loss = epsilon_loss
        min_snr_weight_mean = float32_scalar(1.0)
    else:
        per_sample_epsilon_mse = (
            (output.epsilon.float() - noise.float()).square().flatten(1).mean(dim=1)
        )
        epsilon_unweighted_loss = per_sample_epsilon_mse.mean()
        min_snr_weights = schedule.min_snr_loss_weights(
            timesteps,
            config.min_snr_gamma,
        )
        epsilon_loss = (per_sample_epsilon_mse * min_snr_weights).mean()
        min_snr_weight_mean = min_snr_weights.mean()
    if config.prefix_weight > 0.0:
        prefix_loss = _prefix_loss(
            output, schedule, noisy_images, clean_images, timesteps
        )
    else:
        prefix_loss = output.epsilon.new_zeros(())
    if config.monotonic_weight > 0.0:
        monotonic_loss = _monotonic_loss(
            output,
            schedule,
            noisy_images,
            clean_images,
            timesteps,
            margin=config.monotonic_margin,
        )
    else:
        monotonic_loss = output.epsilon.new_zeros(())
    if config.zero_token_weight > 0.0 and zero_components:
        zero_token_loss = torch.stack(
            [component.pow(2).mean() for component in zero_components]
        ).mean()
    else:
        zero_token_loss = output.epsilon.new_zeros(())
    if config.energy_budget_weight > 0.0:
        energy_budget_loss = _energy_budget_loss(
            output,
            config.energy_target,
            config.energy_budget_scope,
        )
    else:
        energy_budget_loss = output.epsilon.new_zeros(())
    if config.component_decorrelation_weight > 0.0:
        component_decorrelation_loss = _component_decorrelation_loss(output)
    else:
        component_decorrelation_loss = output.epsilon.new_zeros(())
    if config.tail_floor_weight > 0.0:
        tail_floor_loss = _tail_floor_loss(output, config.tail_floor_min_ratio)
    else:
        tail_floor_loss = output.epsilon.new_zeros(())
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
        sampled_prefix_loss, sampled_component_loss, sampled_budget = (
            _sampled_progressive_losses(
                output,
                schedule,
                noisy_images,
                clean_images,
                timesteps,
            )
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
    if (
        config.epsilon_band_prefix_weight > 0.0
        or config.epsilon_band_component_weight > 0.0
    ):
        epsilon_band_prefix_loss, epsilon_band_component_loss = _epsilon_band_losses(
            output,
            noise,
        )
    else:
        epsilon_band_prefix_loss = output.epsilon.new_zeros(())
        epsilon_band_component_loss = output.epsilon.new_zeros(())
    if (
        config.denoise_path_prefix_weight > 0.0
        or config.denoise_path_component_weight > 0.0
        or config.denoise_path_energy_weight > 0.0
    ):
        (
            denoise_path_prefix_loss,
            denoise_path_component_loss,
            denoise_path_energy_loss,
        ) = _denoise_path_losses(
            output,
            schedule,
            noisy_images,
            clean_images,
            timesteps,
            progress_power=config.denoise_path_progress_power,
            progress_mode=config.denoise_path_progress_mode,
            energy_mode=config.denoise_path_energy_mode,
            energy_capacity_weight=config.denoise_path_energy_capacity_weight,
            energy_capacity_power=config.denoise_path_energy_capacity_power,
        )
    else:
        denoise_path_prefix_loss = output.epsilon.new_zeros(())
        denoise_path_component_loss = output.epsilon.new_zeros(())
        denoise_path_energy_loss = output.epsilon.new_zeros(())
    if config.low_snr_high_frequency_weight > 0.0:
        low_snr_high_frequency_loss = _low_snr_high_frequency_loss(
            output,
            noise,
            schedule,
            timesteps,
            power=config.low_snr_high_frequency_power,
        )
    else:
        low_snr_high_frequency_loss = output.epsilon.new_zeros(())
    if (
        config.rollout_consistency_weight > 0.0
        and rollout_consistency is not None
        and rollout_consistency_scale > 0.0
    ):
        rollout_consistency_loss = rollout_consistency
        rollout_scale = float32_scalar(rollout_consistency_scale)
    else:
        rollout_consistency_loss = output.epsilon.new_zeros(())
        rollout_scale = float32_scalar(0.0)
    if (
        config.ema_teacher_consistency_weight > 0.0
        and ema_teacher_consistency is not None
        and ema_teacher_consistency_scale > 0.0
    ):
        ema_teacher_consistency_loss = ema_teacher_consistency
        ema_teacher_scale = float32_scalar(ema_teacher_consistency_scale)
    else:
        ema_teacher_consistency_loss = output.epsilon.new_zeros(())
        ema_teacher_scale = float32_scalar(0.0)
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
        + config.denoise_path_energy_weight * denoise_path_energy_loss
        + config.low_snr_high_frequency_weight * low_snr_high_frequency_loss
        + config.rollout_consistency_weight * rollout_scale * rollout_consistency_loss
        + config.ema_teacher_consistency_weight
        * ema_teacher_scale
        * ema_teacher_consistency_loss
    )
    return LossBreakdown(
        total=total,
        epsilon=epsilon_loss,
        epsilon_unweighted=epsilon_unweighted_loss,
        min_snr_weight_mean=min_snr_weight_mean,
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
        denoise_path_energy=denoise_path_energy_loss,
        low_snr_high_frequency=low_snr_high_frequency_loss,
        rollout_consistency=rollout_consistency_loss,
        rollout_consistency_scale=rollout_scale,
        ema_teacher_consistency=ema_teacher_consistency_loss,
        ema_teacher_consistency_scale=ema_teacher_scale,
    )
