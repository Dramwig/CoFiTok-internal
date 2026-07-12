from __future__ import annotations

import argparse
import math
import time
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid, save_image

from cofitok.configs import ExperimentConfig, LossConfig, config_to_dict, load_config
from cofitok.data import build_dataloader
from cofitok.diagnostics import run_synthesis_diagnostics, shuffle_tokens_across_batch
from cofitok.diffusion import DiffusionSchedule
from cofitok.metrics import energy_distribution_metrics, normalized_curve_auc
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.training import compute_losses
from cofitok.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short real-data CoFiTok training loop.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--output-dir", required=True, help="Directory for report, checkpoint, and prefix grid.")
    return parser.parse_args()


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _cycle(loader: torch.utils.data.DataLoader) -> Iterator[object]:
    while True:
        for batch in loader:
            yield batch


def _batch_images(batch: object, device: torch.device) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        images = batch[0]
    else:
        images = batch
    return images.to(device=device, dtype=torch.float32)


def _denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(-1.0, 1.0) + 1.0) * 0.5


def _component_energy_summary(components: list[torch.Tensor]) -> dict[str, object]:
    energies = [component.pow(2).mean().detach().cpu().item() for component in components]
    total = sum(energies)
    if total > 0:
        ratios = [energy / total for energy in energies]
    else:
        ratios = [0.0 for _ in energies]
    tail_ratio = sum(ratios[1:]) if len(ratios) > 1 else 0.0
    half = len(ratios) // 2
    return {
        "component_energy": energies,
        "component_energy_ratio": ratios,
        "head_energy_ratio": ratios[0] if ratios else 0.0,
        "tail_energy_ratio": tail_ratio,
        "late_half_energy_ratio": sum(ratios[half:]) if ratios else 0.0,
        "active_tail_tokens": sum(1 for ratio in ratios[1:] if ratio >= 0.01),
        **energy_distribution_metrics(ratios),
    }


def _prefix_targets(clean_images: torch.Tensor, count: int) -> list[torch.Tensor]:
    targets = []
    max_scale = min(clean_images.shape[-2:])
    for index in range(count):
        if index == count - 1:
            targets.append(clean_images)
            continue
        scale = min(2 ** (count - index - 1), max_scale)
        pooled = torch.nn.functional.avg_pool2d(clean_images, kernel_size=scale, stride=scale, ceil_mode=True)
        restored = torch.nn.functional.interpolate(
            pooled,
            size=clean_images.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        targets.append(restored)
    return targets


def _denoise_path_targets(
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    count: int,
    progress_power: float,
) -> list[torch.Tensor]:
    zero_epsilon = torch.zeros_like(noisy_images)
    start_x0 = schedule.predict_x0_from_epsilon(noisy_images, zero_epsilon, timesteps)
    power = max(progress_power, 1e-6)
    targets = []
    for index, spatial_target in enumerate(_prefix_targets(clean_images, count)):
        progress = math.pow((index + 1) / count, power)
        targets.append(torch.lerp(start_x0, spatial_target, progress))
    return targets


def _prefix_epsilons_from_components(components: list[torch.Tensor]) -> list[torch.Tensor]:
    prefix_epsilons = []
    running = torch.zeros_like(components[0])
    for component in components:
        running = running + component
        prefix_epsilons.append(running)
    return prefix_epsilons


def _select_component_order(
    components: list[torch.Tensor],
    mode: str,
    random_seed: int,
) -> tuple[list[torch.Tensor], list[int]]:
    count = len(components)
    if mode == "ordered":
        order = list(range(count))
    elif mode == "reverse":
        order = list(reversed(range(count)))
    elif mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(random_seed)
        order = torch.randperm(count, generator=generator).tolist()
    else:
        raise ValueError(f"Unknown component order mode: {mode}")
    return [components[index] for index in order], order


def _scheduled_loss_config(loss: LossConfig, step: int) -> LossConfig:
    if loss.component_decorrelation_weight <= 0.0:
        return loss
    if loss.component_decorrelation_start_step < 0:
        raise ValueError("component_decorrelation_start_step must be non-negative")
    if loss.component_decorrelation_warmup_steps < 0:
        raise ValueError("component_decorrelation_warmup_steps must be non-negative")

    start_step = loss.component_decorrelation_start_step
    warmup_steps = loss.component_decorrelation_warmup_steps
    if step <= start_step:
        effective_weight = 0.0
    elif warmup_steps == 0:
        effective_weight = loss.component_decorrelation_weight
    else:
        progress = min(1.0, (step - start_step) / warmup_steps)
        effective_weight = loss.component_decorrelation_weight * progress
    if effective_weight == loss.component_decorrelation_weight:
        return loss
    return replace(loss, component_decorrelation_weight=effective_weight)


@torch.no_grad()
def _write_prefix_grid(
    path: Path,
    model: CoFiTokTiny,
    schedule: DiffusionSchedule,
    clean_images: torch.Tensor,
    visualization_count: int,
    denoise_path_progress_power: float,
    component_order: str = "ordered",
    random_order_seed: int = 0,
) -> dict[str, object]:
    model.eval()
    images = clean_images[:visualization_count]
    timesteps = torch.full(
        (images.shape[0],),
        schedule.num_train_timesteps // 2,
        dtype=torch.long,
        device=images.device,
    )
    noise = torch.randn_like(images)
    noisy_images = schedule.add_noise(images, noise, timesteps)
    output = model(noisy_images, timesteps)
    ordered_components, component_order_indices = _select_component_order(
        output.components,
        mode=component_order,
        random_seed=random_order_seed,
    )
    prefix_epsilons = _prefix_epsilons_from_components(ordered_components)
    prefix_images = [
        schedule.predict_x0_from_epsilon(noisy_images, prefix_epsilon, timesteps)
        for prefix_epsilon in prefix_epsilons
    ]

    rows = [images, noisy_images, *prefix_images]
    grid_items = torch.cat([_denormalize(row).cpu() for row in rows], dim=0)
    grid = make_grid(grid_items, nrow=images.shape[0], padding=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)

    shuffled_tokens = shuffle_tokens_across_batch(output.tokens)
    shuffled_components = model.synthesis(shuffled_tokens)
    shuffled_ordered_components, _ = _select_component_order(
        shuffled_components,
        mode=component_order,
        random_seed=random_order_seed,
    )
    shuffled_prefix_epsilons = _prefix_epsilons_from_components(shuffled_ordered_components)
    shuffled_prefix_images = [
        schedule.predict_x0_from_epsilon(noisy_images, prefix_epsilon, timesteps)
        for prefix_epsilon in shuffled_prefix_epsilons
    ]
    shuffled_rows = [images, noisy_images, *shuffled_prefix_images]
    shuffled_grid_items = torch.cat([_denormalize(row).cpu() for row in shuffled_rows], dim=0)
    shuffled_grid = make_grid(shuffled_grid_items, nrow=images.shape[0], padding=2)
    shuffled_grid_path = path.with_name(f"{path.stem}_shuffled{path.suffix}")
    save_image(shuffled_grid, shuffled_grid_path)

    prefix_mse = [
        torch.nn.functional.mse_loss(prefix_image, images).detach().cpu().item()
        for prefix_image in prefix_images
    ]
    path_targets = _denoise_path_targets(
        schedule,
        noisy_images,
        images,
        timesteps,
        count=len(prefix_images),
        progress_power=denoise_path_progress_power,
    )
    path_mse = [
        torch.nn.functional.mse_loss(prefix_image, target).detach().cpu().item()
        for prefix_image, target in zip(prefix_images, path_targets)
    ]
    shuffled_prefix_mse = [
        torch.nn.functional.mse_loss(prefix_image, images).detach().cpu().item()
        for prefix_image in shuffled_prefix_images
    ]
    final_mse = prefix_mse[-1] if prefix_mse else 0.0
    shuffled_final_mse = shuffled_prefix_mse[-1] if shuffled_prefix_mse else 0.0
    diagnostics = run_synthesis_diagnostics(model, output.tokens)
    energy_summary = _component_energy_summary(output.components)
    return {
        "grid_path": str(path),
        "shuffled_grid_path": str(shuffled_grid_path),
        "component_order": component_order,
        "component_order_indices": component_order_indices,
        "random_order_seed": random_order_seed,
        "rows": ["clean", "noisy", *[f"prefix_{index + 1}" for index in range(len(prefix_images))]],
        "timestep": int(timesteps[0].detach().cpu().item()),
        "prefix_mse_to_clean": prefix_mse,
        "prefix_mse_to_clean_auc": normalized_curve_auc(prefix_mse),
        "prefix_mse_to_denoise_path": path_mse,
        "prefix_mse_to_denoise_path_auc": normalized_curve_auc(path_mse),
        "shuffled_prefix_mse_to_clean": shuffled_prefix_mse,
        "shuffled_prefix_mse_to_clean_auc": normalized_curve_auc(shuffled_prefix_mse),
        "shuffled_final_clean_mse": shuffled_final_mse,
        "shuffled_final_mse_ratio": shuffled_final_mse / max(final_mse, 1e-12),
        **energy_summary,
        "diagnostics": diagnostics,
    }


def _train_step(
    config: ExperimentConfig,
    model: CoFiTokTiny,
    schedule: DiffusionSchedule,
    optimizer: torch.optim.Optimizer,
    clean_images: torch.Tensor,
    step: int,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    noise = torch.randn_like(clean_images)
    timesteps = schedule.sample_timesteps(clean_images.shape[0], device=clean_images.device)
    noisy_images = schedule.add_noise(clean_images, noise, timesteps)
    output = model(noisy_images, timesteps)
    loss_config = _scheduled_loss_config(config.loss, step)
    losses = compute_losses(
        loss_config,
        output,
        schedule,
        noisy_images,
        clean_images,
        noise,
        timesteps,
        zero_components=model.synthesis.zero_components_like(output.tokens),
    )
    losses.total.backward()
    grad_norm = clip_grad_norm_(model.parameters(), config.optimization.grad_clip_norm)
    optimizer.step()
    values = {key: float(value.detach().cpu().item()) for key, value in losses.as_dict().items()}
    values["component_decorrelation_effective_weight"] = float(loss_config.component_decorrelation_weight)
    values["grad_norm"] = float(grad_norm.detach().cpu().item())
    return values


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.runtime.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(config.runtime.device)
    train_loader = build_dataloader(config.data, split="train")
    eval_loader = build_dataloader(config.data, split="val")
    train_batches = _cycle(train_loader)

    model = CoFiTokTiny(config.model).to(device)
    schedule = DiffusionSchedule(config.diffusion, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.optimization.learning_rate,
        betas=tuple(config.optimization.betas),
        weight_decay=config.optimization.weight_decay,
    )

    start = time.time()
    history = []
    latest_losses = {}
    for step in range(1, config.runtime.steps + 1):
        clean_images = _batch_images(next(train_batches), device)
        latest_losses = _train_step(config, model, schedule, optimizer, clean_images, step)
        if step == 1 or step % config.optimization.log_interval == 0 or step == config.runtime.steps:
            entry = {"step": step, **latest_losses}
            history.append(entry)
            print(
                f"step {step:04d} "
                f"total={latest_losses['total']:.4f} "
                f"epsilon={latest_losses['epsilon']:.4f} "
                f"prefix={latest_losses['prefix']:.4f} "
                f"energy={latest_losses['energy_budget']:.4f} "
                f"residual={latest_losses['residual_component']:.4f} "
                f"decor={latest_losses['component_decorrelation']:.4f} "
                f"decor_w={latest_losses['component_decorrelation_effective_weight']:.5f} "
                f"tail={latest_losses['tail_floor']:.4f} "
                f"sample_m={latest_losses['sampled_budget']:.0f} "
                f"group={latest_losses['group_residual']:.4f} "
                f"band_p={latest_losses['epsilon_band_prefix']:.4f} "
                f"band_c={latest_losses['epsilon_band_component']:.4f} "
                f"path_p={latest_losses['denoise_path_prefix']:.4f} "
                f"path_c={latest_losses['denoise_path_component']:.4f}"
            )

    eval_images = _batch_images(next(iter(eval_loader)), device)
    prefix_summary = _write_prefix_grid(
        output_dir / "prefix_final.png",
        model,
        schedule,
        eval_images,
        visualization_count=config.optimization.visualization_count,
        denoise_path_progress_power=config.loss.denoise_path_progress_power,
    )

    checkpoint_path = output_dir / "checkpoint_final.pt"
    torch.save(
        {
            "config": config_to_dict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": config.runtime.steps,
            "history": history,
        },
        checkpoint_path,
    )

    elapsed = time.time() - start
    report = {
        "config": config_to_dict(config),
        "runtime": {
            "requested_device": config.runtime.device,
            "actual_device": str(device),
            "elapsed_seconds": elapsed,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        },
        "final_losses": latest_losses,
        "history": history,
        "prefix_summary": prefix_summary,
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "prefix_grid": str(output_dir / "prefix_final.png"),
        },
    }
    report_path = output_dir / "report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")
    print(f"wrote {checkpoint_path}")
    print(f"wrote {output_dir / 'prefix_final.png'}")


if __name__ == "__main__":
    main()
