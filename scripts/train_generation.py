from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from cofitok.configs import ExperimentConfig, config_to_dict, load_config
from cofitok.data import (
    StatefulRandomSampler,
    build_dataloader,
    build_dataset,
    capture_dataset_provenance,
)
from cofitok.diffusion import DiffusionSchedule
from cofitok.environment import (
    capture_runtime_environment,
    runtime_environment_sha256,
)
from cofitok.models import CoFiTokTiny
from cofitok.reporting import file_sha256, write_json_report
from cofitok.training import (
    ExponentialMovingAverage,
    capture_generation_training_authorization,
    class_conditioning_ranking_loss,
    compute_losses,
    consistency_weight_scale,
    ema_teacher_consistency_loss,
    ensure_fresh_training_output,
    reconcile_metrics_for_resume,
    rollout_consistency_loss,
    rollout_consistency_weight_scale,
)
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_GIT_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


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
    parser.add_argument(
        "--resume-source-revision",
        default="",
        help=(
            "Explicit clean ancestor revision allowed for one controlled resume. "
            "The transition is bound into subsequent checkpoints and reports."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=0, help="Override steps for smoke runs.")
    parser.add_argument("--micro-batch-size", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=0)
    parser.add_argument(
        "--stop-after-steps",
        type=int,
        default=0,
        help="Stop cleanly after N optimizer steps without changing the configured LR schedule.",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=0,
        help="Run optimizer steps for runtime measurement without writing checkpoints.",
    )
    parser.add_argument("--benchmark-warmup-steps", type=int, default=2)
    parser.add_argument("--benchmark-output", default="")
    parser.add_argument(
        "--authorization-gate",
        default="",
        help="Passing scaling gate that authorizes formal full ImageNet-256 training.",
    )
    return parser.parse_args()


def _validate_config(config: ExperimentConfig) -> None:
    if config.model.predictor_type not in {"scalable_unet", "adm_unet", "generation_unet"}:
        raise ValueError("train_generation.py requires the scalable generation predictor")
    if config.model.synthesis_mode not in {
        "restricted",
        "fixed_basis",
        "dense_identity",
    }:
        raise ValueError(
            "production training permits only restricted/fixed-basis CoFiTok "
            "or dense_identity control"
        )
    if config.diffusion.prediction_target != "epsilon":
        raise ValueError("production training currently supports epsilon prediction only")
    if config.data.class_conditional != (config.model.num_classes > 0):
        raise ValueError("data.class_conditional and model.num_classes must agree")
    if not 0.0 <= config.data.random_horizontal_flip_prob <= 1.0:
        raise ValueError("data.random_horizontal_flip_prob must be between 0 and 1")
    if config.optimization.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if config.runtime.precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("runtime.precision must be fp32, bf16, or fp16")
    if config.loss.energy_budget_weight > 0.0:
        target = config.loss.energy_target
        if len(target) != config.model.token_count:
            raise ValueError(
                "energy_target length must equal model.token_count when the "
                "energy budget is enabled"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in target):
            raise ValueError("energy_target values must be finite and positive")
        if config.loss.energy_budget_scope not in {"batch", "sample"}:
            raise ValueError("energy_budget_scope must be batch or sample")
    if not 0.0 <= config.loss.denoise_path_energy_capacity_weight <= 1.0:
        raise ValueError("denoise_path_energy_capacity_weight must be in [0, 1]")
    if (
        not math.isfinite(config.loss.denoise_path_energy_capacity_power)
        or config.loss.denoise_path_energy_capacity_power <= 0.0
    ):
        raise ValueError("denoise_path_energy_capacity_power must be finite and positive")
    if (
        not math.isfinite(config.loss.low_snr_high_frequency_power)
        or config.loss.low_snr_high_frequency_power <= 0.0
    ):
        raise ValueError("low_snr_high_frequency_power must be finite and positive")
    if config.loss.rollout_consistency_weight < 0.0:
        raise ValueError("rollout_consistency_weight must be non-negative")
    if config.loss.rollout_consistency_start_step < 0:
        raise ValueError("rollout_consistency_start_step must be non-negative")
    if config.loss.rollout_consistency_warmup_steps < 0:
        raise ValueError("rollout_consistency_warmup_steps must be non-negative")
    if config.loss.rollout_consistency_timestep_delta < 1:
        raise ValueError("rollout_consistency_timestep_delta must be positive")
    if config.loss.rollout_consistency_unroll_steps < 1:
        raise ValueError("rollout_consistency_unroll_steps must be positive")
    if (
        config.loss.rollout_consistency_timestep_delta
        * config.loss.rollout_consistency_unroll_steps
        >= config.diffusion.num_train_timesteps
    ):
        raise ValueError(
            "rollout consistency horizon must be shorter than the diffusion schedule"
        )
    if not 0.0 < config.loss.rollout_consistency_batch_fraction <= 1.0:
        raise ValueError("rollout_consistency_batch_fraction must be in (0, 1]")
    if config.loss.rollout_consistency_mode not in {"epsilon", "clipped_x0"}:
        raise ValueError("rollout_consistency_mode must be epsilon or clipped_x0")
    if config.loss.ema_teacher_consistency_weight < 0.0:
        raise ValueError("ema_teacher_consistency_weight must be non-negative")
    if config.loss.ema_teacher_consistency_start_step < 0:
        raise ValueError("ema_teacher_consistency_start_step must be non-negative")
    if config.loss.ema_teacher_consistency_warmup_steps < 0:
        raise ValueError("ema_teacher_consistency_warmup_steps must be non-negative")
    if not 0.0 < config.loss.ema_teacher_consistency_batch_fraction <= 1.0:
        raise ValueError(
            "ema_teacher_consistency_batch_fraction must be in (0, 1]"
        )
    if config.loss.class_conditioning_ranking_weight < 0.0:
        raise ValueError("class_conditioning_ranking_weight must be non-negative")
    if config.loss.class_conditioning_ranking_start_step < 0:
        raise ValueError("class_conditioning_ranking_start_step must be non-negative")
    if config.loss.class_conditioning_ranking_warmup_steps < 0:
        raise ValueError("class_conditioning_ranking_warmup_steps must be non-negative")
    if not 0.0 < config.loss.class_conditioning_ranking_batch_fraction <= 1.0:
        raise ValueError(
            "class_conditioning_ranking_batch_fraction must be in (0, 1]"
        )
    if (
        not math.isfinite(config.loss.class_conditioning_ranking_margin)
        or not 0.0 <= config.loss.class_conditioning_ranking_margin < 1.0
    ):
        raise ValueError("class_conditioning_ranking_margin must be in [0, 1)")
    if config.loss.class_conditioning_ranking_wrong_label_offset < 1:
        raise ValueError(
            "class_conditioning_ranking_wrong_label_offset must be positive"
        )
    if config.loss.class_conditioning_ranking_min_timestep < 0:
        raise ValueError(
            "class_conditioning_ranking_min_timestep must be non-negative"
        )
    if config.loss.class_conditioning_ranking_weight > 0.0:
        if not config.data.class_conditional or config.model.num_classes < 2:
            raise ValueError(
                "class_conditioning_ranking_weight requires class-conditional training"
            )
        if (
            config.loss.class_conditioning_ranking_wrong_label_offset
            % config.model.num_classes
            == 0
        ):
            raise ValueError(
                "class_conditioning_ranking_wrong_label_offset must change the class"
            )
        if (
            config.loss.class_conditioning_ranking_min_timestep
            >= config.diffusion.num_train_timesteps
        ):
            raise ValueError(
                "class_conditioning_ranking_min_timestep is outside the diffusion schedule"
            )
    protected_steps = config.runtime.protected_checkpoint_steps
    if protected_steps != sorted(set(protected_steps)):
        raise ValueError("protected_checkpoint_steps must be sorted and unique")
    if any(step < 1 or step > config.runtime.steps for step in protected_steps):
        raise ValueError("protected checkpoint step is outside the configured training range")
    if any(
        step != config.runtime.steps and step % config.runtime.checkpoint_interval != 0
        for step in protected_steps
    ):
        raise ValueError("protected checkpoint steps must align with checkpoint_interval")


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


def _advance_eval_iterator(
    loader: DataLoader,
    iterator: object,
    batches: int,
) -> object:
    for _ in range(batches):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)
    return iterator


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


def _augment_training_images(images: torch.Tensor, flip_probability: float) -> torch.Tensor:
    if flip_probability <= 0.0:
        return images
    flipped = images.flip(dims=(-1,))
    if flip_probability >= 1.0:
        return flipped
    flip_mask = torch.rand(
        (images.shape[0], 1, 1, 1),
        device=images.device,
    ) < flip_probability
    return torch.where(flip_mask, flipped, images)


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


def _resolve_resume_git_provenance(
    current_git: dict[str, str | bool],
    source_revision: str,
) -> tuple[dict[str, str | bool], dict[str, object] | None]:
    if not source_revision:
        return dict(current_git), None
    current_revision = str(current_git.get("revision", ""))
    current_branch = str(current_git.get("branch", ""))
    if (
        not FULL_GIT_REVISION_PATTERN.fullmatch(source_revision)
        or not FULL_GIT_REVISION_PATTERN.fullmatch(current_revision)
    ):
        raise ValueError("controlled resume requires full lowercase Git revisions")
    if current_git.get("dirty") is not False or not current_branch:
        raise ValueError("controlled resume requires a clean named Git branch")
    if source_revision == current_revision:
        raise ValueError("resume source revision must differ from the current revision")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_revision, current_revision],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise ValueError("resume source revision is not an ancestor of the current revision")
    return (
        {
            "revision": source_revision,
            "branch": current_branch,
            "dirty": False,
        },
        {
            "schema_version": 1,
            "reason": "sampler_rng_state_device_compatibility",
            "source_revision": source_revision,
            "target_revision": current_revision,
            "branch": current_branch,
        },
    )


def _build_resume_revision_transition(
    transition: dict[str, object],
    resume_path: Path,
) -> dict[str, object]:
    integrity_path = checkpoint_integrity_path(resume_path)
    with integrity_path.open(encoding="utf-8") as handle:
        integrity = json.load(handle)
    if not isinstance(integrity, dict):
        raise ValueError("resume checkpoint integrity manifest must be an object")
    if (
        integrity.get("git_revision") != transition["source_revision"]
        or integrity.get("git_branch") != transition["branch"]
        or integrity.get("git_dirty") is not False
    ):
        raise ValueError("resume checkpoint does not match the controlled source revision")
    return {
        **transition,
        "source_checkpoint": {
            "path": resume_path.resolve().as_posix(),
            "filename": resume_path.name,
            "bytes": int(integrity["checkpoint_bytes"]),
            "sha256": str(integrity["checkpoint_sha256"]),
            "step": int(integrity["step"]),
            "integrity_manifest": integrity_path.resolve().as_posix(),
            "integrity_manifest_bytes": integrity_path.stat().st_size,
            "integrity_manifest_sha256": file_sha256(integrity_path),
        },
    }


def _validate_existing_resume_revision_transition(
    transition: object,
    current_git: dict[str, str | bool],
) -> dict[str, object]:
    if not isinstance(transition, dict):
        raise ValueError("checkpoint resume revision transition must be an object")
    if (
        transition.get("schema_version") != 1
        or transition.get("reason") != "sampler_rng_state_device_compatibility"
        or transition.get("target_revision") != current_git.get("revision")
        or transition.get("branch") != current_git.get("branch")
        or not FULL_GIT_REVISION_PATTERN.fullmatch(
            str(transition.get("source_revision", ""))
        )
    ):
        raise ValueError("checkpoint resume revision transition is invalid")
    source_checkpoint = transition.get("source_checkpoint")
    if (
        not isinstance(source_checkpoint, dict)
        or int(source_checkpoint.get("bytes", 0)) < 1
        or int(source_checkpoint.get("step", 0)) < 1
        or not re.fullmatch(r"[0-9a-f]{64}", str(source_checkpoint.get("sha256", "")))
        or not re.fullmatch(
            r"[0-9a-f]{64}",
            str(source_checkpoint.get("integrity_manifest_sha256", "")),
        )
    ):
        raise ValueError("checkpoint resume transition source identity is invalid")
    return dict(transition)


@torch.no_grad()
def _evaluate_batch(
    model: torch.nn.Module,
    schedule: DiffusionSchedule,
    batch: object,
    config: ExperimentConfig,
    device: torch.device,
) -> tuple[float, int, int]:
    model.eval()
    clean, labels = _move_batch(batch, device, config.data.class_conditional)
    validation_noise_seed = config.runtime.seed + 100_003
    generator = torch.Generator(device=device).manual_seed(validation_noise_seed)
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
    mse = float(torch.nn.functional.mse_loss(output.epsilon.float(), noise.float()).item())
    return mse, int(clean.shape[0]), validation_noise_seed


def main() -> None:
    args = parse_args()
    benchmark_mode = args.benchmark_steps > 0
    if benchmark_mode:
        if not args.benchmark_output:
            raise ValueError("benchmark mode requires --benchmark-output")
        if args.benchmark_warmup_steps < 0 or args.benchmark_warmup_steps >= args.benchmark_steps:
            raise ValueError("benchmark warmup must leave at least one measured step")
        if (
            args.resume
            or args.resume_source_revision
            or args.max_steps > 0
            or args.stop_after_steps > 0
        ):
            raise ValueError("benchmark mode cannot resume or override the training horizon")
    elif args.benchmark_output:
        raise ValueError("--benchmark-output requires --benchmark-steps")
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
    formal_full_training = (
        not benchmark_mode
        and config.data.dataset == "imagenet_256"
        and config.runtime.steps == 300_000
    )
    if benchmark_mode and args.authorization_gate:
        raise ValueError("benchmark mode does not consume a training authorization gate")
    if formal_full_training and not args.authorization_gate:
        raise ValueError("formal full ImageNet-256 training requires --authorization-gate")
    training_authorization = (
        capture_generation_training_authorization(args.authorization_gate)
        if args.authorization_gate
        else None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "train_metrics.jsonl"
    resume_path = _resolve_resume(output_dir, args.resume)
    if args.resume_source_revision and resume_path is None:
        raise ValueError("--resume-source-revision requires --resume")
    if resume_path is None:
        ensure_fresh_training_output(output_dir)
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
    runtime_environment = capture_runtime_environment(
        device,
        project_root=PROJECT_ROOT,
    )
    runtime_environment_sha = runtime_environment_sha256(runtime_environment)
    git_provenance = _git_revision()
    checkpoint_git_provenance, requested_resume_transition = (
        _resolve_resume_git_provenance(
            git_provenance,
            args.resume_source_revision,
        )
    )

    train_loader, sampler = _build_train_loader(config)
    eval_loader = build_dataloader(
        config.data,
        split="val",
        drop_last=False,
        generator=torch.Generator().manual_seed(config.runtime.seed + 97),
    )
    dataset_provenance = capture_dataset_provenance(
        config.data,
        train_images=len(train_loader.dataset),
        val_images=len(eval_loader.dataset),
    )
    if dataset_provenance.get("status") == "fail":
        raise ValueError(
            "formal dataset provenance failed: "
            + "; ".join(dataset_provenance.get("issues", []))
        )

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
    cumulative_elapsed_before_segment = 0.0
    cumulative_peak_vram_before_segment = 0
    metrics_resume_reconciliation = None
    resume_revision_transition = None
    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path,
            model=base_model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            restore_rng=True,
            expected_config=config_to_dict(config),
            expected_runtime_environment=runtime_environment,
            expected_git_provenance=checkpoint_git_provenance,
            expected_dataset_provenance=dataset_provenance,
            expected_training_authorization=training_authorization,
            map_location=device,
        )
        start_step = int(checkpoint["step"])
        checkpoint_extra_state = checkpoint.get("extra_state", {})
        sampler_state = checkpoint_extra_state.get("sampler")
        if sampler_state is None:
            raise ValueError("production checkpoint is missing sampler state")
        cumulative_elapsed_before_segment = float(
            checkpoint_extra_state.get("cumulative_elapsed_seconds", 0.0)
        )
        cumulative_peak_vram_before_segment = int(
            checkpoint_extra_state.get("cumulative_peak_vram_bytes", 0)
        )
        if cumulative_elapsed_before_segment < 0.0 or cumulative_peak_vram_before_segment < 0:
            raise ValueError("checkpoint cumulative compute accounting is invalid")
        sampler.load_state_dict(sampler_state)
        existing_resume_transition = checkpoint_extra_state.get(
            "resume_revision_transition"
        )
        if requested_resume_transition is not None:
            if existing_resume_transition is not None:
                raise ValueError(
                    "checkpoint already contains a resume revision transition"
                )
            resume_revision_transition = _build_resume_revision_transition(
                requested_resume_transition,
                resume_path,
            )
        elif existing_resume_transition is not None:
            resume_revision_transition = _validate_existing_resume_revision_transition(
                existing_resume_transition,
                git_provenance,
            )
        metrics_resume_reconciliation = reconcile_metrics_for_resume(
            metrics_path,
            resume_step=start_step,
        )

    train_iterator = iter(train_loader)
    eval_iterator = iter(eval_loader)
    eval_iterator = _advance_eval_iterator(
        eval_loader,
        eval_iterator,
        start_step // config.runtime.evaluation_interval,
    )
    completed_validation_events = start_step // config.runtime.evaluation_interval

    model: torch.nn.Module = base_model
    if config.runtime.compile_model:
        model = torch.compile(base_model, dynamic=False)

    manifest = {
        "config": config_to_dict(config),
        "config_path": str(Path(args.config).resolve()),
        "output_dir": str(output_dir.resolve()),
        "git": git_provenance,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha,
        "dataset_provenance": dataset_provenance,
        "training_authorization": training_authorization,
        "parameter_count": sum(parameter.numel() for parameter in base_model.parameters()),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in base_model.parameters() if parameter.requires_grad
        ),
        "resume": str(resume_path) if resume_path is not None else None,
        "resume_revision_transition": resume_revision_transition,
        "metrics_resume_reconciliation": metrics_resume_reconciliation,
    }
    write_json_report(output_dir / "run_manifest.json", manifest)

    training_start = time.time()
    benchmark_durations: list[float] = []
    if benchmark_mode and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    last_metrics: dict[str, float | int] = {}
    completed_step = start_step
    accumulation = config.optimization.gradient_accumulation_steps
    loop_end = config.runtime.steps
    if args.stop_after_steps > 0:
        loop_end = min(loop_end, start_step + args.stop_after_steps)
    if benchmark_mode:
        loop_end = args.benchmark_steps
    for step in range(start_step + 1, loop_end + 1):
        if benchmark_mode and device.type == "cuda":
            torch.cuda.synchronize(device)
        benchmark_step_start = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        aggregate: dict[str, float] = {}
        micro_samples = 0
        for _ in range(accumulation):
            batch, train_iterator = _next_batch(train_loader, sampler, train_iterator)
            clean, labels = _move_batch(batch, device, config.data.class_conditional)
            clean = _augment_training_images(
                clean,
                config.data.random_horizontal_flip_prob,
            )
            micro_samples += clean.shape[0]
            noise = torch.randn_like(clean)
            timesteps = schedule.sample_timesteps(clean.shape[0], device=device)
            noisy = schedule.add_noise(clean, noise, timesteps)
            with autocast_context(device, config.runtime.precision):
                output = model(noisy, timesteps, class_labels=labels)
                ema_teacher_scale = consistency_weight_scale(
                    step,
                    start_step=config.loss.ema_teacher_consistency_start_step,
                    warmup_steps=config.loss.ema_teacher_consistency_warmup_steps,
                )
                ema_teacher_loss = None
                if (
                    config.loss.ema_teacher_consistency_weight > 0.0
                    and ema_teacher_scale > 0.0
                ):
                    ema_teacher_loss = ema_teacher_consistency_loss(
                        base_model,
                        ema_state=ema.shadow,
                        student_epsilon=output.epsilon,
                        noisy_images=noisy,
                        timesteps=timesteps,
                        class_labels=labels,
                        batch_fraction=(
                            config.loss.ema_teacher_consistency_batch_fraction
                        ),
                    )
                class_ranking_scale = consistency_weight_scale(
                    step,
                    start_step=config.loss.class_conditioning_ranking_start_step,
                    warmup_steps=config.loss.class_conditioning_ranking_warmup_steps,
                )
                class_ranking = None
                if (
                    config.loss.class_conditioning_ranking_weight > 0.0
                    and class_ranking_scale > 0.0
                ):
                    if labels is None:
                        raise RuntimeError(
                            "class-conditioning ranking requires class labels"
                        )
                    class_ranking = class_conditioning_ranking_loss(
                        base_model,
                        noisy_images=noisy,
                        noise=noise,
                        timesteps=timesteps,
                        class_labels=labels,
                        num_classes=config.model.num_classes,
                        batch_fraction=(
                            config.loss.class_conditioning_ranking_batch_fraction
                        ),
                        margin=config.loss.class_conditioning_ranking_margin,
                        wrong_label_offset=(
                            config.loss.class_conditioning_ranking_wrong_label_offset
                        ),
                        min_timestep=(
                            config.loss.class_conditioning_ranking_min_timestep
                        ),
                    )
                rollout_scale = rollout_consistency_weight_scale(
                    step,
                    start_step=config.loss.rollout_consistency_start_step,
                    warmup_steps=config.loss.rollout_consistency_warmup_steps,
                )
                rollout_consistency = None
                if config.loss.rollout_consistency_weight > 0.0 and rollout_scale > 0.0:
                    rollout_consistency = rollout_consistency_loss(
                        model,
                        first_epsilon=output.epsilon,
                        schedule=schedule,
                        noisy_images=noisy,
                        clean_images=clean,
                        timesteps=timesteps,
                        class_labels=labels,
                        timestep_delta=config.loss.rollout_consistency_timestep_delta,
                        unroll_steps=config.loss.rollout_consistency_unroll_steps,
                        batch_fraction=config.loss.rollout_consistency_batch_fraction,
                        clip_x0=config.loss.rollout_consistency_clip_x0,
                        mode=config.loss.rollout_consistency_mode,
                    )
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
                    rollout_consistency=rollout_consistency,
                    rollout_consistency_scale=rollout_scale,
                    ema_teacher_consistency=ema_teacher_loss,
                    ema_teacher_consistency_scale=ema_teacher_scale,
                    class_conditioning_ranking=class_ranking,
                    class_conditioning_ranking_scale=class_ranking_scale,
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
        if benchmark_mode:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            benchmark_durations.append(time.perf_counter() - benchmark_step_start)
        completed_step = step
        step_elapsed_seconds = time.time() - training_start

        last_metrics = {
            "step": step,
            **aggregate,
            "grad_norm": float(grad_norm.detach().float().item()),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "ema_decay": ema._effective_decay(),
            "samples_seen": step * micro_samples,
            "elapsed_seconds": step_elapsed_seconds,
            "cumulative_elapsed_seconds": (
                cumulative_elapsed_before_segment + step_elapsed_seconds
            ),
        }
        if not stop.requested and (
            step % config.runtime.evaluation_interval == 0
            or step == config.runtime.steps
        ):
            try:
                eval_batch = next(eval_iterator)
            except StopIteration:
                eval_iterator = iter(eval_loader)
                eval_batch = next(eval_iterator)
            validation_mse, validation_num_images, validation_noise_seed = _evaluate_batch(
                model,
                schedule,
                eval_batch,
                config,
                device,
            )
            last_metrics.update(
                {
                    "validation_epsilon_mse": validation_mse,
                    "validation_event_index": completed_validation_events,
                    "validation_batch_index": completed_validation_events % len(eval_loader),
                    "validation_num_images": validation_num_images,
                    "validation_noise_seed": validation_noise_seed,
                }
            )
            completed_validation_events += 1

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

        should_checkpoint = not benchmark_mode and (
            step % config.runtime.checkpoint_interval == 0
            or step == loop_end
            or stop.requested
        )
        if should_checkpoint:
            segment_elapsed_seconds = time.time() - training_start
            segment_peak_vram_bytes = (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            )
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
                extra_state={
                    "sampler": sampler.state_dict(),
                    "git": git_provenance,
                    "runtime_environment": runtime_environment,
                    "runtime_environment_sha256": runtime_environment_sha,
                    "dataset_provenance": dataset_provenance,
                    "training_authorization": training_authorization,
                    "resume_revision_transition": resume_revision_transition,
                    "cumulative_elapsed_seconds": (
                        cumulative_elapsed_before_segment + segment_elapsed_seconds
                    ),
                    "cumulative_peak_vram_bytes": max(
                        cumulative_peak_vram_before_segment,
                        segment_peak_vram_bytes,
                    ),
                },
            )
            prune_checkpoints(
                output_dir,
                config.runtime.keep_last_checkpoints,
                protected_steps=config.runtime.protected_checkpoint_steps,
            )
        if stop.requested:
            break

    segment_elapsed_seconds = time.time() - training_start
    segment_peak_vram_bytes = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
    if benchmark_mode:
        measured = benchmark_durations[args.benchmark_warmup_steps :]
        mean_seconds = statistics.fmean(measured)
        sorted_measured = sorted(measured)
        p95_index = max(0, (95 * len(sorted_measured) + 99) // 100 - 1)
        effective_batch_size = config.data.batch_size * config.optimization.gradient_accumulation_steps
        benchmark_report = {
            "schema_version": 1,
            "status": "completed",
            "role": "training_runtime_selection_only",
            "config": config_to_dict(config),
            "config_path": str(Path(args.config).resolve()),
            "git": manifest["git"],
            "runtime_environment": manifest["runtime_environment"],
            "runtime_environment_sha256": manifest[
                "runtime_environment_sha256"
            ],
            "dataset_provenance": manifest["dataset_provenance"],
            "device": manifest["device"],
            "device_name": manifest["device_name"],
            "parameter_count": manifest["parameter_count"],
            "benchmark_steps": args.benchmark_steps,
            "warmup_steps": args.benchmark_warmup_steps,
            "measured_steps": len(measured),
            "micro_batch_size": config.data.batch_size,
            "gradient_accumulation_steps": config.optimization.gradient_accumulation_steps,
            "effective_batch_size": effective_batch_size,
            "mean_optimizer_step_seconds": mean_seconds,
            "median_optimizer_step_seconds": statistics.median(measured),
            "p95_optimizer_step_seconds": sorted_measured[p95_index],
            "images_per_second": effective_batch_size / mean_seconds,
            "peak_vram_bytes": segment_peak_vram_bytes,
            "device_total_memory_bytes": (
                torch.cuda.get_device_properties(device).total_memory
                if device.type == "cuda"
                else 0
            ),
            "durations_seconds": benchmark_durations,
            "measured_durations_seconds": measured,
            "last_metrics": last_metrics,
            "checkpoint_written": False,
        }
        write_json_report(args.benchmark_output, benchmark_report)
        print(f"wrote {args.benchmark_output}")
        return
    report = {
        **manifest,
        "completed_steps": completed_step,
        "target_steps": config.runtime.steps,
        "training_complete": completed_step == config.runtime.steps,
        "stop_requested": stop.requested,
        "stop_signal": stop.signal_number,
        "final_metrics": last_metrics,
        "segment_elapsed_seconds": segment_elapsed_seconds,
        "elapsed_seconds": cumulative_elapsed_before_segment + segment_elapsed_seconds,
        "peak_vram_bytes": max(
            cumulative_peak_vram_before_segment,
            segment_peak_vram_bytes,
        ),
        "latest_checkpoint": json.loads((output_dir / "latest.json").read_text(encoding="utf-8")),
    }
    write_json_report(output_dir / "training_report.json", report)
    print(f"wrote {output_dir / 'training_report.json'}")


if __name__ == "__main__":
    main()
