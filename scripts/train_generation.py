from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from cofitok.configs import ExperimentConfig, config_to_dict, load_config
from cofitok.data import StatefulRandomSampler, build_dataloader, build_dataset
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.training import ExponentialMovingAverage, compute_losses
from cofitok.training.checkpointing import (
    load_training_checkpoint,
    prune_checkpoints,
    resolve_latest_checkpoint,
    save_training_checkpoint,
)
from cofitok.training.runtime import (
    autocast_context,
    build_grad_scaler,
    build_warmup_cosine_scheduler,
)
from cofitok.utils.seed import seed_everything


class StopController:
    def __init__(self) -> None:
        self.requested = False
        self.signal_number: int | None = None

    def request(self, signal_number: int, _frame: object) -> None:
        self.requested = True
        self.signal_number = signal_number
        print(f"received signal {signal_number}; checkpointing after the current optimizer step", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the scalable CoFiTok generation system.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", default="", help="Checkpoint path or 'auto' for latest.json.")
    parser.add_argument("--max-steps", type=int, default=0, help="Override steps for smoke runs.")
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=0,
        help="Stop cleanly after N optimizer steps without changing the configured LR schedule.",
    )
    return parser.parse_args()


def _validate_config(config: ExperimentConfig) -> None:
    if config.model.predictor_type not in {"scalable_unet", "adm_unet", "generation_unet"}:
        raise ValueError("train_generation.py requires the scalable generation predictor")
    if config.model.synthesis_mode not in {"restricted", "dense_identity"}:
        raise ValueError("production training permits only restricted CoFiTok or dense_identity control")
    if config.diffusion.prediction_target != "epsilon":
        raise ValueError("production training currently supports epsilon prediction only")
    if config.data.class_conditional != (config.model.num_classes > 0):
        raise ValueError("data.class_conditional and model.num_classes must agree")
    if config.optimization.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.runtime.precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("runtime.precision must be fp32, bf16, or fp16")


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for production training but is unavailable")
    return torch.device(requested)


def _build_train_loader(config: ExperimentConfig) -> tuple[DataLoader, StatefulRandomSampler]:
    dataset = build_dataset(config.data, split="train")
    sampler = StatefulRandomSampler(dataset, seed=config.runtime.seed)
    worker_options: dict[str, object] = {}
    if config.data.num_workers > 0:
        worker_options = {
            "persistent_workers": config.data.persistent_workers,
            "prefetch_factor": config.data.prefetch_factor,
        }
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        sampler=sampler,
        num_workers=config.data.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
        generator=torch.Generator().manual_seed(config.runtime.seed + 71),
        **worker_options,
    )
    return loader, sampler


def _next_batch(
    loader: DataLoader,
    sampler: StatefulRandomSampler,
    iterator: object,
) -> tuple[object, object]:
    try:
        batch = next(iterator)
    except StopIteration:
        sampler.start_next_epoch()
        iterator = iter(loader)
        batch = next(iterator)
    images = batch[0] if isinstance(batch, (list, tuple)) else batch
    sampler.mark_consumed(int(images.shape[0]))
    return batch, iterator


def _move_batch(
    batch: object,
    device: torch.device,
    class_conditional: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(batch, (list, tuple)):
        images = batch[0]
        labels = batch[1] if len(batch) > 1 else None
    else:
        images = batch
        labels = None
    images = images.to(device=device, dtype=torch.float32, non_blocking=True)
    if not class_conditional:
        return images, None
    if labels is None:
        raise ValueError("class-conditional training batch is missing labels")
    return images, labels.to(device=device, dtype=torch.long, non_blocking=True)


def _optimizer_groups(model: torch.nn.Module, weight_decay: float) -> list[dict[str, object]]:
    decay = []
    no_decay = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias") or "norm" in name or "embed" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def _resolve_resume(output_dir: Path, requested: str) -> Path | None:
    if not requested:
        return None
    if requested != "auto":
        return Path(requested)
    return resolve_latest_checkpoint(output_dir)


def _git_revision() -> dict[str, str | bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"revision": revision, "branch": branch, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"revision": "unknown", "branch": "unknown", "dirty": True}


@torch.no_grad()
def _evaluate_batch(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    batch: object,
    config: ExperimentConfig,
    device: torch.device,
) -> float:
    model.eval()
    clean, labels = _move_batch(batch, device, config.data.class_conditional)
    generator = torch.Generator(device=device).manual_seed(config.runtime.seed + 100_003)
    noise = torch.randn(clean.shape, device=device, generator=generator)
    timesteps = torch.randint(
        0,
        schedule.num_train_timesteps,
        (clean.shape[0],),
        device=device,
        generator=generator,
    )
    noisy = schedule.add_noise(clean, noise, timesteps)
    output = model(noisy, timesteps, class_labels=labels)
    return float(torch.nn.functional.mse_loss(output.epsilon.float(), noise.float()).item())


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.max_steps > 0:
        config = replace(config, runtime=replace(config.runtime, steps=args.max_steps))
    if args.micro_batch_size > 0:
        config = replace(config, data=replace(config.data, batch_size=args.micro_batch_size))
    if args.gradient_accumulation_steps > 0:
        config = replace(
            config,
            optimization=replace(
                config.optimization,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
            ),
        )
    _validate_config(config)
    stop = StopController()
    signal.signal(signal.SIGTERM, stop.request)
    signal.signal(signal.SIGINT, stop.request)
    seed_everything(config.runtime.seed)
    device = _resolve_device(config.runtime.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
        torch.backends.cudnn.benchmark = config.runtime.cudnn_benchmark
    torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_loader, sampler = _build_train_loader(config)
    eval_loader = build_dataloader(
        config.data,
        split="val",
        drop_last=False,
        generator=torch.Generator().manual_seed(config.runtime.seed + 97),
    )
    train_iterator = iter(train_loader)
    eval_iterator = iter(eval_loader)

    base_model = CoFiTokTiny(config.model).to(device)
    optimizer = torch.optim.AdamW(
        _optimizer_groups(base_model, config.optimization.weight_decay),
        lr=config.optimization.learning_rate,
        betas=tuple(config.optimization.betas),
        eps=1e-8,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=config.runtime.steps,
        warmup_steps=config.optimization.warmup_steps,
        min_learning_rate=config.optimization.min_learning_rate,
    )
    scaler = build_grad_scaler(device, config.runtime.precision)
    ema = ExponentialMovingAverage(
        base_model,
        decay=config.optimization.ema_decay,
        warmup_steps=config.optimization.ema_warmup_steps,
    )
    schedule = DiffusionSchedule(config.diffusion, device=device)
    start_step = 0
    resume_path = _resolve_resume(output_dir, args.resume)
    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path,
            model=base_model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            restore_rng=True,
            map_location=device,
        )
        start_step = int(checkpoint["step"])
        sampler_state = checkpoint.get("extra_state", {}).get("sampler")
        if sampler_state is None:
            raise ValueError("production checkpoint is missing sampler state")
        sampler.load_state_dict(sampler_state)
        train_iterator = iter(train_loader)

    model: torch.nn.Module = base_model
    if config.runtime.compile_model:
        model = torch.compile(base_model, dynamic=False)

    manifest = {
        "config": config_to_dict(config),
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "git": _git_revision(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "parameter_count": sum(parameter.numel() for parameter in base_model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad
        ),
        "resume": str(resume_path) if resume_path is not None else None,
    }
    write_json_report(output_dir / "run_manifest.json", manifest)

    metrics_path = output_dir / "train_metrics.jsonl"
    training_start = time.time()
    last_metrics: dict[str, float | int] = {}
    completed_step = start_step
    accumulation = config.optimization.gradient_accumulation_steps
    loop_end = config.runtime.steps
    if args.stop_after_steps > 0:
        loop_end = min(loop_end, start_step + args.stop_after_steps)
    for step in range(start_step + 1, loop_end + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        micro_samples = 0
        for _ in range(accumulation):
            batch, train_iterator = _next_batch(train_loader, sampler, train_iterator)
            clean, labels = _move_batch(batch, device, config.data.class_conditional)
            micro_samples += clean.shape[0]
            noise = torch.randn_like(clean)
            timesteps = schedule.sample_timesteps(clean.shape[0], device=device)
            noisy = schedule.add_noise(clean, noise, timesteps)
            with autocast_context(device, config.runtime.precision):
                output = model(noisy, timesteps, class_labels=labels)
                losses = compute_losses(
                    config.loss,
                    output,
                    schedule,
                    noisy,
                    clean,
                    noise,
                    timesteps,
                    zero_components=(
                        base_model.synthesis.zero_components_like(output.tokens)
                        if config.loss.zero_token_weight > 0.0
                        else None
                    ),
                )
                scaled_loss = losses.total / accumulation
            if not torch.isfinite(scaled_loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            for name, value in losses.as_dict().items():
                aggregate[name] = aggregate.get(name, 0.0) + float(value.detach().float().item()) / accumulation

        if scaler is not None:
            scaler.unscale_(optimizer)
        grad_norm = clip_grad_norm_(base_model.parameters(), config.optimization.grad_clip_norm)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        ema.update(base_model)
        completed_step = step

        last_metrics = {
            "step": step,
            **aggregate,
            "grad_norm": float(grad_norm.detach().float().item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "ema_decay": ema._effective_decay(),
            "samples_seen": step * micro_samples,
            "elapsed_seconds": time.time() - training_start,
        }
        if not stop.requested and (
            step % config.runtime.evaluation_interval == 0 or step == loop_end
        ):
            try:
                eval_batch = next(eval_iterator)
            except StopIteration:
                eval_iterator = iter(eval_loader)
                eval_batch = next(eval_iterator)
            last_metrics["validation_epsilon_mse"] = _evaluate_batch(
                model,
                schedule,
                eval_batch,
                config,
                device,
            )

        should_log = step == start_step + 1 or step % config.optimization.log_interval == 0
        if should_log:
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(last_metrics, sort_keys=True) + "\n")
            validation = last_metrics.get("validation_epsilon_mse")
            validation_text = "" if validation is None else f" val={validation:.6f}"
            print(
                f"step={step} total={aggregate['total']:.6f} "
                f"epsilon={aggregate['epsilon']:.6f} grad={last_metrics['grad_norm']:.4f} "
                f"lr={last_metrics['learning_rate']:.3e}{validation_text}"
            )

        should_checkpoint = (
            step % config.runtime.checkpoint_interval == 0
            or step == loop_end
            or stop.requested
        )
        if should_checkpoint:
            checkpoint_path = output_dir / f"checkpoint_step_{step:08d}.pt"
            save_training_checkpoint(
                checkpoint_path,
                model=model,
                ema=ema,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                config=config_to_dict(config),
                metrics=last_metrics,
                extra_state={"sampler": sampler.state_dict()},
            )
            prune_checkpoints(output_dir, config.runtime.keep_last_checkpoints)
        if stop.requested:
            break

    report = {
        **manifest,
        "completed_steps": completed_step,
        "target_steps": config.runtime.steps,
        "training_complete": completed_step == config.runtime.steps,
        "stop_requested": stop.requested,
        "stop_signal": stop.signal_number,
        "final_metrics": last_metrics,
        "elapsed_seconds": time.time() - training_start,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
        "latest_checkpoint": json.loads((output_dir / "latest.json").read_text(encoding="utf-8")),
    }
    write_json_report(output_dir / "training_report.json", report)
    print(f"wrote {output_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
