from __future__ import annotations

import argparse
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from cofitok.configs import config_to_dict
from cofitok.data import build_dataset
from cofitok.diffusion import DiffusionSchedule, select_sampling_timesteps
from cofitok.generation import load_generation_model
from cofitok.generation.stability import (
    component_energy_ratios,
    component_high_frequency_ratios,
    deterministic_ddim_step,
    high_frequency_energy_ratio,
    oracle_epsilon,
    per_sample_mse,
    per_sample_rms,
    predict_guided_components,
    tail_energy_ratio,
)
from cofitok.reporting import git_provenance, write_json_report
from cofitok.training.losses import denoise_path_prefix_epsilon_targets
from cofitok.training.runtime import autocast_context


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose timestep error and DDIM rollout stability for a generation checkpoint."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument(
        "--teacher-timesteps",
        default="999,900,750,500,250,100,10",
        help="Comma-separated training timesteps for teacher-forced evaluation.",
    )
    parser.add_argument("--seed", type=int, default=2029)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--teacher-guidance-scale", type=float, default=1.0)
    parser.add_argument("--cfg-batch-mode", choices=["batched", "sequential"], default="batched")
    parser.add_argument(
        "--clip-x0",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def _parse_timesteps(value: str, limit: int) -> list[int]:
    timesteps = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not timesteps:
        raise ValueError("teacher-timesteps cannot be empty")
    if len(set(timesteps)) != len(timesteps):
        raise ValueError("teacher-timesteps must be unique")
    if any(timestep < 0 or timestep >= limit for timestep in timesteps):
        raise ValueError("teacher-timesteps are outside the diffusion schedule")
    return sorted(timesteps, reverse=True)


def _batch(
    batch: object,
    device: torch.device,
    class_conditional: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if not isinstance(batch, (list, tuple)) or not batch:
        images = batch
        labels = None
    else:
        images = batch[0]
        labels = batch[1] if len(batch) > 1 else None
    if not isinstance(images, torch.Tensor):
        raise TypeError("evaluation batch does not contain an image tensor")
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    if not class_conditional:
        return images, None
    if not isinstance(labels, torch.Tensor):
        raise ValueError("class-conditional evaluation batch is missing labels")
    return images, labels.to(device=device, dtype=torch.long, non_blocking=True)


def _append(accumulator: dict[str, list[Any]], name: str, value: torch.Tensor) -> None:
    accumulator[name].append(value.detach().float().cpu())


def _mean(values: list[torch.Tensor]) -> float | list[float]:
    combined = torch.cat(values, dim=0)
    result = combined.mean(dim=0)
    if result.ndim == 0:
        return float(result)
    return [float(value) for value in result]


def _finalize(accumulator: dict[str, list[Any]]) -> dict[str, Any]:
    return {name: _mean(values) for name, values in accumulator.items()}


def _target_components(
    *,
    schedule: DiffusionSchedule,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    reference_output,
    progress_power: float,
    progress_mode: str,
) -> list[torch.Tensor]:
    targets = denoise_path_prefix_epsilon_targets(
        schedule,
        noisy,
        clean,
        timesteps,
        len(reference_output.components),
        progress_power,
        tokens=reference_output.tokens,
        progress_mode=progress_mode,
    )
    components = []
    previous = torch.zeros_like(targets[0])
    for target in targets:
        components.append(target - previous)
        previous = target
    return components


@torch.no_grad()
def evaluate_teacher_forced(
    *,
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    batches: list[tuple[torch.Tensor, torch.Tensor | None]],
    timesteps: list[int],
    guidance_scale: float,
    guidance_rescale: float,
    cfg_batch_mode: str,
    precision: str,
    seed: int,
    progress_power: float,
    progress_mode: str,
) -> list[dict[str, Any]]:
    accumulators = {timestep: defaultdict(list) for timestep in timesteps}
    generator = torch.Generator(device=batches[0][0].device).manual_seed(seed)
    for clean, labels in batches:
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        for timestep in timesteps:
            time_batch = torch.full(
                (clean.shape[0],),
                timestep,
                device=clean.device,
                dtype=torch.long,
            )
            noisy = schedule.add_noise(clean, noise, time_batch)
            with autocast_context(clean.device, precision):
                prediction = predict_guided_components(
                    model,
                    noisy,
                    time_batch,
                    class_labels=labels,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    cfg_batch_mode=cfg_batch_mode,
                )
            raw_x0 = schedule.predict_x0_from_epsilon(noisy, prediction.epsilon, time_batch)
            clipped_x0 = raw_x0.clamp(-1.0, 1.0)
            target_components = _target_components(
                schedule=schedule,
                noisy=noisy,
                clean=clean,
                timesteps=time_batch,
                reference_output=prediction.reference_output,
                progress_power=progress_power,
                progress_mode=progress_mode,
            )
            metrics = accumulators[timestep]
            _append(metrics, "epsilon_mse", per_sample_mse(prediction.epsilon, noise))
            _append(metrics, "raw_x0_mse", per_sample_mse(raw_x0, clean))
            _append(metrics, "clipped_x0_mse", per_sample_mse(clipped_x0, clean))
            _append(metrics, "epsilon_rms", per_sample_rms(prediction.epsilon))
            _append(
                metrics,
                "epsilon_high_frequency_ratio",
                high_frequency_energy_ratio(prediction.epsilon),
            )
            _append(
                metrics,
                "raw_x0_clip_fraction",
                (raw_x0.float().abs() > 1.0).flatten(1).float().mean(dim=1),
            )
            _append(
                metrics,
                "component_energy_ratio",
                component_energy_ratios(prediction.components),
            )
            _append(
                metrics,
                "component_high_frequency_ratio",
                component_high_frequency_ratios(prediction.components),
            )
            _append(
                metrics,
                "tail_two_energy_ratio",
                tail_energy_ratio(prediction.components, tail_count=2),
            )
            _append(
                metrics,
                "target_component_energy_ratio",
                component_energy_ratios(target_components),
            )
            _append(
                metrics,
                "target_tail_two_energy_ratio",
                tail_energy_ratio(target_components, tail_count=2),
            )
    return [
        {"timestep": timestep, **_finalize(accumulators[timestep])}
        for timestep in timesteps
    ]


def _trajectory_summary(steps: list[dict[str, Any]], reconstruction: bool) -> dict[str, Any]:
    tail_values = [float(step["tail_two_energy_ratio"]) for step in steps]
    clip_values = [float(step["raw_x0_clip_fraction"]) for step in steps]
    summary: dict[str, Any] = {
        "step_count": len(steps),
        "tail_two_energy_ratio_mean": sum(tail_values) / len(tail_values),
        "tail_two_energy_ratio_max": max(tail_values),
        "raw_x0_clip_fraction_mean": sum(clip_values) / len(clip_values),
        "raw_x0_clip_fraction_max": max(clip_values),
        "early_tail_two_energy_ratio_mean": sum(tail_values[: max(1, len(steps) // 3)])
        / max(1, len(steps) // 3),
        "late_tail_two_energy_ratio_mean": sum(tail_values[-max(1, len(steps) // 3) :])
        / max(1, len(steps) // 3),
    }
    if reconstruction:
        x0_errors = [float(step["clipped_x0_mse"]) for step in steps]
        local_errors = [float(step["local_step_mse"]) for step in steps]
        epsilon_errors = [float(step["epsilon_oracle_mse"]) for step in steps]
        minimum = min(x0_errors)
        summary.update(
            {
                "initial_clipped_x0_mse": x0_errors[0],
                "final_clipped_x0_mse": x0_errors[-1],
                "peak_clipped_x0_mse": max(x0_errors),
                "final_to_best_x0_mse_amplification": x0_errors[-1] / max(minimum, 1e-12),
                "peak_local_step_mse": max(local_errors),
                "early_epsilon_oracle_mse_mean": sum(
                    epsilon_errors[: max(1, len(steps) // 3)]
                )
                / max(1, len(steps) // 3),
                "late_epsilon_oracle_mse_mean": sum(
                    epsilon_errors[-max(1, len(steps) // 3) :]
                )
                / max(1, len(steps) // 3),
            }
        )
    return summary


@torch.no_grad()
def evaluate_rollout(
    *,
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    batches: list[tuple[torch.Tensor, torch.Tensor | None]],
    sample_steps: int,
    guidance_scale: float,
    guidance_rescale: float,
    cfg_batch_mode: str,
    precision: str,
    clip_x0: bool,
    seed: int,
    reconstruction: bool,
) -> dict[str, Any]:
    timesteps = select_sampling_timesteps(schedule.num_train_timesteps, sample_steps)
    accumulators = [defaultdict(list) for _ in timesteps]
    generator = torch.Generator(device=batches[0][0].device).manual_seed(seed)
    for clean, labels in batches:
        initial_noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        if reconstruction:
            start_times = torch.full(
                (clean.shape[0],),
                timesteps[0],
                device=clean.device,
                dtype=torch.long,
            )
            images = schedule.add_noise(clean, initial_noise, start_times)
        else:
            images = initial_noise
        previous_x0 = None
        for index, timestep in enumerate(timesteps):
            previous_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            time_batch = torch.full(
                (clean.shape[0],),
                timestep,
                device=clean.device,
                dtype=torch.long,
            )
            with autocast_context(clean.device, precision):
                prediction = predict_guided_components(
                    model,
                    images,
                    time_batch,
                    class_labels=labels,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    cfg_batch_mode=cfg_batch_mode,
                )
            next_images, raw_x0, predicted_x0 = deterministic_ddim_step(
                schedule,
                images,
                prediction.epsilon,
                timestep=timestep,
                previous_timestep=previous_timestep,
                clip_x0=clip_x0,
            )
            metrics = accumulators[index]
            _append(metrics, "state_rms", per_sample_rms(images))
            _append(metrics, "epsilon_rms", per_sample_rms(prediction.epsilon))
            _append(metrics, "predicted_x0_rms", per_sample_rms(predicted_x0))
            _append(
                metrics,
                "epsilon_high_frequency_ratio",
                high_frequency_energy_ratio(prediction.epsilon),
            )
            _append(
                metrics,
                "predicted_x0_high_frequency_ratio",
                high_frequency_energy_ratio(predicted_x0),
            )
            _append(
                metrics,
                "raw_x0_clip_fraction",
                (raw_x0.float().abs() > 1.0).flatten(1).float().mean(dim=1),
            )
            _append(
                metrics,
                "component_energy_ratio",
                component_energy_ratios(prediction.components),
            )
            _append(
                metrics,
                "component_high_frequency_ratio",
                component_high_frequency_ratios(prediction.components),
            )
            _append(
                metrics,
                "tail_two_energy_ratio",
                tail_energy_ratio(prediction.components, tail_count=2),
            )
            if previous_x0 is None:
                x0_drift = torch.zeros(clean.shape[0], device=clean.device)
            else:
                x0_drift = per_sample_mse(predicted_x0, previous_x0)
            _append(metrics, "predicted_x0_drift_mse", x0_drift)
            if reconstruction:
                target_epsilon = oracle_epsilon(schedule, images, clean, time_batch)
                oracle_next, _, _ = deterministic_ddim_step(
                    schedule,
                    images,
                    target_epsilon,
                    timestep=timestep,
                    previous_timestep=previous_timestep,
                    clip_x0=clip_x0,
                )
                _append(
                    metrics,
                    "epsilon_oracle_mse",
                    per_sample_mse(prediction.epsilon, target_epsilon),
                )
                _append(metrics, "raw_x0_mse", per_sample_mse(raw_x0, clean))
                _append(metrics, "clipped_x0_mse", per_sample_mse(predicted_x0, clean))
                _append(metrics, "local_step_mse", per_sample_mse(next_images, oracle_next))
            previous_x0 = predicted_x0
            images = next_images
    steps = [
        {
            "step_index": index,
            "timestep": timestep,
            "previous_timestep": (
                timesteps[index + 1] if index + 1 < len(timesteps) else -1
            ),
            **_finalize(accumulators[index]),
        }
        for index, timestep in enumerate(timesteps)
    ]
    return {
        "mode": "real_image_reconstruction" if reconstruction else "free_noise_sampling",
        "timesteps": timesteps,
        "steps": steps,
        "summary": _trajectory_summary(steps, reconstruction),
    }


def _materialize_batches(
    *,
    loader: DataLoader,
    num_images: int,
    device: torch.device,
    class_conditional: bool,
) -> list[tuple[torch.Tensor, torch.Tensor | None]]:
    batches = []
    count = 0
    for raw_batch in loader:
        if count >= num_images:
            break
        images, labels = _batch(raw_batch, device, class_conditional)
        remaining = num_images - count
        images = images[:remaining]
        if labels is not None:
            labels = labels[:remaining]
        batches.append((images, labels))
        count += images.shape[0]
    if count != num_images:
        raise RuntimeError(f"requested {num_images} images but loader yielded {count}")
    return batches


def main() -> None:
    args = parse_args()
    if args.num_images < 1 or args.batch_size < 1 or args.sample_steps < 1:
        raise ValueError("num-images, batch-size, and sample-steps must be positive")
    if not math.isfinite(args.guidance_scale) or args.guidance_scale < 0.0:
        raise ValueError("guidance-scale must be finite and non-negative")
    if not math.isfinite(args.teacher_guidance_scale) or args.teacher_guidance_scale < 0.0:
        raise ValueError("teacher-guidance-scale must be finite and non-negative")
    loaded = load_generation_model(args.checkpoint, weights=args.weights)
    config = loaded.config
    device = loaded.device
    schedule = DiffusionSchedule(config.diffusion, device=device)
    teacher_timesteps = _parse_timesteps(
        args.teacher_timesteps,
        config.diffusion.num_train_timesteps,
    )
    dataset = build_dataset(config.data, split="val")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    batches = _materialize_batches(
        loader=loader,
        num_images=args.num_images,
        device=device,
        class_conditional=config.data.class_conditional,
    )
    start = time.time()
    teacher_forced = evaluate_teacher_forced(
        model=loaded.model,
        schedule=schedule,
        batches=batches,
        timesteps=teacher_timesteps,
        guidance_scale=args.teacher_guidance_scale,
        guidance_rescale=args.guidance_rescale,
        cfg_batch_mode=args.cfg_batch_mode,
        precision=args.precision,
        seed=args.seed,
        progress_power=config.loss.denoise_path_progress_power,
        progress_mode=config.loss.denoise_path_progress_mode,
    )
    reconstruction = evaluate_rollout(
        model=loaded.model,
        schedule=schedule,
        batches=batches,
        sample_steps=args.sample_steps,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
        cfg_batch_mode=args.cfg_batch_mode,
        precision=args.precision,
        clip_x0=args.clip_x0,
        seed=args.seed + 1,
        reconstruction=True,
    )
    free_sampling = evaluate_rollout(
        model=loaded.model,
        schedule=schedule,
        batches=batches,
        sample_steps=args.sample_steps,
        guidance_scale=args.guidance_scale,
        guidance_rescale=args.guidance_rescale,
        cfg_batch_mode=args.cfg_batch_mode,
        precision=args.precision,
        clip_x0=args.clip_x0,
        seed=args.seed + 2,
        reconstruction=False,
    )
    elapsed = time.time() - start
    report = {
        "schema_version": 1,
        "status": "completed",
        "git": git_provenance(PROJECT_ROOT),
        "checkpoint": loaded.checkpoint_path.as_posix(),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_integrity_manifest": loaded.checkpoint_integrity_manifest.as_posix(),
        "checkpoint_step": loaded.checkpoint_step,
        "weights": loaded.weights,
        "config": config_to_dict(config),
        "protocol": {
            "num_images": args.num_images,
            "batch_size": args.batch_size,
            "teacher_timesteps": teacher_timesteps,
            "sample_steps": args.sample_steps,
            "guidance_scale": args.guidance_scale,
            "teacher_guidance_scale": args.teacher_guidance_scale,
            "guidance_rescale": args.guidance_rescale,
            "cfg_batch_mode": args.cfg_batch_mode,
            "clip_x0": args.clip_x0,
            "precision": args.precision,
            "seed": args.seed,
        },
        "teacher_forced": teacher_forced,
        "reconstruction_rollout": reconstruction,
        "free_sampling_rollout": free_sampling,
        "runtime": {
            "elapsed_seconds": elapsed,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_peak_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
            ),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "rollout_stability_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
