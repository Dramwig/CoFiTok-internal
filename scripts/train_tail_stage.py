from __future__ import annotations

import argparse
import time
from collections.abc import Iterator
from pathlib import Path

import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.utils import make_grid, save_image

from cofitok.configs import config_to_dict, load_config
from cofitok.data import build_dataloader
from cofitok.diagnostics import run_synthesis_diagnostics
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokOutput, CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.training import compute_losses
from cofitok.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune late CoFiTok tokens on residual epsilon.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Warm-start checkpoint path.")
    parser.add_argument("--replay-checkpoint", default="", help="Optional teacher checkpoint for tail replay.")
    parser.add_argument("--output-dir", required=True, help="Directory for report, checkpoint, and prefix grid.")
    parser.add_argument("--tail-start", type=int, required=True, help="0-based first token index to train.")
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
    ratios = [energy / total for energy in energies] if total > 0 else [0.0 for _ in energies]
    half = len(ratios) // 2
    return {
        "component_energy": energies,
        "component_energy_ratio": ratios,
        "tail_energy_ratio": sum(ratios[1:]) if len(ratios) > 1 else 0.0,
        "late_half_energy_ratio": sum(ratios[half:]) if ratios else 0.0,
        "active_tail_tokens": sum(1 for ratio in ratios[1:] if ratio >= 0.01),
    }


def _set_tail_trainable(model: CoFiTokTiny, tail_start: int) -> list[str]:
    if not 0 <= tail_start < model.config.token_count:
        raise ValueError(f"tail_start must be in [0, {model.config.token_count - 1}], got {tail_start}")
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    trainable_names: list[str] = []
    for index in range(tail_start, model.config.token_count):
        for parameter in model.predictor.token_heads[index].parameters():
            parameter.requires_grad_(True)
        trainable_names.append(f"predictor.token_heads.{index}")
        for parameter in model.synthesis.synthesizers[index].parameters():
            parameter.requires_grad_(True)
        trainable_names.append(f"synthesis.synthesizers.{index}")

    for index in range(tail_start, len(model.predictor.feedback)):
        for parameter in model.predictor.feedback[index].parameters():
            parameter.requires_grad_(True)
        trainable_names.append(f"predictor.feedback.{index}")
    return trainable_names


def _validate_tail_dropout(config, tail_start: int) -> None:
    probability = config.loss.tail_early_dropout_prob
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"tail_early_dropout_prob must be in [0, 1], got {probability}")
    start = config.loss.tail_early_dropout_start
    if start < 0:
        raise ValueError(f"tail_early_dropout_start must be non-negative, got {start}")
    if config.loss.tail_early_dropout_weight > 0.0 and start >= tail_start:
        raise ValueError(
            "tail_early_dropout_start must be smaller than tail_start when "
            "tail_early_dropout_weight is enabled"
        )
    late_prefix_start = config.loss.tail_late_prefix_start
    if config.loss.tail_late_prefix_weight > 0.0 and late_prefix_start < 0:
        raise ValueError(f"tail_late_prefix_start must be non-negative, got {late_prefix_start}")
    late_monotonic_start = config.loss.tail_late_monotonic_start
    if config.loss.tail_late_monotonic_weight > 0.0 and late_monotonic_start < 0:
        raise ValueError(
            f"tail_late_monotonic_start must be non-negative, got {late_monotonic_start}"
        )


def _build_replay_model(
    config,
    checkpoint_path: str,
    device: torch.device,
) -> CoFiTokTiny | None:
    if not checkpoint_path:
        if config.loss.tail_replay_weight > 0.0:
            raise ValueError("tail_replay_weight requires --replay-checkpoint")
        return None
    replay_model = CoFiTokTiny(config.model).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    replay_model.load_state_dict(checkpoint["model"])
    replay_model.eval()
    for parameter in replay_model.parameters():
        parameter.requires_grad_(False)
    return replay_model


def _forward_with_early_token_dropout(
    model: CoFiTokTiny,
    noisy_images: torch.Tensor,
    timesteps: torch.Tensor,
    tail_start: int,
    dropout_start: int,
    dropout_prob: float,
) -> CoFiTokOutput:
    hidden = model.predictor.input_proj(noisy_images)
    time_embedding = model.predictor.time_embed(timesteps).view(timesteps.shape[0], -1, 1, 1)
    hidden = hidden + time_embedding
    for block in model.predictor.blocks:
        hidden = block(hidden)

    tokens = []
    keep_masks = []
    state = hidden
    batch_size = noisy_images.shape[0]
    for index, head in enumerate(model.predictor.token_heads):
        token = head(state)
        tokens.append(token)
        if index < tail_start and index >= dropout_start and dropout_prob > 0.0:
            keep_mask = (
                torch.rand(batch_size, 1, 1, 1, device=token.device, dtype=token.dtype)
                >= dropout_prob
            ).to(dtype=token.dtype)
        else:
            keep_mask = torch.ones(batch_size, 1, 1, 1, device=token.device, dtype=token.dtype)
        keep_masks.append(keep_mask)
        if index < len(model.predictor.feedback):
            feedback = model.predictor.feedback[index](token)
            if index < tail_start:
                feedback = feedback * keep_mask
            state = state + feedback

    components = model.synthesis(tokens)
    masked_components = [
        component * keep_masks[index] if index < tail_start else component
        for index, component in enumerate(components)
    ]
    prefix_epsilons = []
    running = torch.zeros_like(masked_components[0])
    for component in masked_components:
        running = running + component
        prefix_epsilons.append(running)
    return CoFiTokOutput(
        tokens=tokens,
        components=masked_components,
        prefix_epsilons=prefix_epsilons,
        epsilon=prefix_epsilons[-1],
    )


def _late_prefix_guard_loss(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    start_index: int,
) -> torch.Tensor:
    if start_index >= len(output.prefix_epsilons):
        return output.epsilon.new_zeros(())
    losses = []
    for prefix_epsilon in output.prefix_epsilons[start_index:]:
        pred_x0 = schedule.predict_x0_from_epsilon(noisy_images, prefix_epsilon, timesteps)
        losses.append(torch.nn.functional.mse_loss(pred_x0, clean_images))
    return torch.stack(losses).mean()


def _endpoint_guard_loss(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    pred_x0 = schedule.predict_x0_from_epsilon(noisy_images, output.epsilon, timesteps)
    return torch.nn.functional.mse_loss(pred_x0, clean_images)


def _late_monotonic_guard_loss(
    output: CoFiTokOutput,
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    clean_images: torch.Tensor,
    timesteps: torch.Tensor,
    start_index: int,
    margin: float,
) -> torch.Tensor:
    prefix_epsilons = output.prefix_epsilons[start_index:]
    if len(prefix_epsilons) < 2:
        return output.epsilon.new_zeros(())
    errors = []
    for prefix_epsilon in prefix_epsilons:
        pred_x0 = schedule.predict_x0_from_epsilon(noisy_images, prefix_epsilon, timesteps)
        errors.append(torch.nn.functional.mse_loss(pred_x0, clean_images))
    penalties = [
        torch.relu(current - previous + margin)
        for previous, current in zip(errors[:-1], errors[1:])
    ]
    return torch.stack(penalties).mean()


def _tail_stage_step(
    config,
    model: CoFiTokTiny,
    schedule: DiffusionSchedule,
    optimizer: torch.optim.Optimizer,
    clean_images: torch.Tensor,
    tail_start: int,
    grad_clip_norm: float,
    replay_model: CoFiTokTiny | None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    noise = torch.randn_like(clean_images)
    timesteps = schedule.sample_timesteps(clean_images.shape[0], device=clean_images.device)
    noisy_images = schedule.add_noise(clean_images, noise, timesteps)
    output = model(noisy_images, timesteps)
    early_prefix = torch.stack(output.components[:tail_start]).sum(dim=0).detach()
    tail_prediction = torch.stack(output.components[tail_start:]).sum(dim=0)
    target_tail = noise - early_prefix
    tail_loss = torch.nn.functional.mse_loss(tail_prediction, target_tail)
    epsilon_loss = torch.nn.functional.mse_loss(output.epsilon, noise)
    early_dropout_loss = output.epsilon.new_zeros(())
    if config.loss.tail_early_dropout_weight > 0.0:
        dropout_output = _forward_with_early_token_dropout(
            model,
            noisy_images,
            timesteps,
            tail_start=tail_start,
            dropout_start=config.loss.tail_early_dropout_start,
            dropout_prob=config.loss.tail_early_dropout_prob,
        )
        dropout_early_prefix = torch.stack(dropout_output.components[:tail_start]).sum(dim=0).detach()
        dropout_tail_prediction = torch.stack(dropout_output.components[tail_start:]).sum(dim=0)
        dropout_target_tail = noise - dropout_early_prefix
        early_dropout_loss = torch.nn.functional.mse_loss(
            dropout_tail_prediction,
            dropout_target_tail,
        )
    tail_replay_loss = output.epsilon.new_zeros(())
    if config.loss.tail_replay_weight > 0.0:
        if replay_model is None:
            raise ValueError("tail_replay_weight requires a replay model")
        with torch.no_grad():
            replay_output = replay_model(noisy_images, timesteps)
        replay_losses = [
            torch.nn.functional.mse_loss(component, replay_component)
            for component, replay_component in zip(
                output.components[tail_start:],
                replay_output.components[tail_start:],
            )
        ]
        tail_replay_loss = torch.stack(replay_losses).mean()
    late_prefix_loss = output.epsilon.new_zeros(())
    if config.loss.tail_late_prefix_weight > 0.0:
        late_prefix_loss = _late_prefix_guard_loss(
            output,
            schedule,
            noisy_images,
            clean_images,
            timesteps,
            start_index=config.loss.tail_late_prefix_start,
        )
    endpoint_loss = output.epsilon.new_zeros(())
    if config.loss.tail_endpoint_weight > 0.0:
        endpoint_loss = _endpoint_guard_loss(
            output,
            schedule,
            noisy_images,
            clean_images,
            timesteps,
        )
    late_monotonic_loss = output.epsilon.new_zeros(())
    if config.loss.tail_late_monotonic_weight > 0.0:
        late_monotonic_loss = _late_monotonic_guard_loss(
            output,
            schedule,
            noisy_images,
            clean_images,
            timesteps,
            start_index=config.loss.tail_late_monotonic_start,
            margin=config.loss.tail_late_monotonic_margin,
        )
    prefix_loss = output.epsilon.new_zeros(())
    monotonic_loss = output.epsilon.new_zeros(())
    if config.loss.prefix_weight > 0.0 or config.loss.monotonic_weight > 0.0:
        regular_losses = compute_losses(
            config.loss,
            output,
            schedule,
            noisy_images,
            clean_images,
            noise,
            timesteps,
            zero_components=model.synthesis.zero_components_like(output.tokens),
        )
        prefix_loss = regular_losses.prefix
        monotonic_loss = regular_losses.monotonic
    total = (
        tail_loss
        + config.loss.tail_early_dropout_weight * early_dropout_loss
        + config.loss.tail_replay_weight * tail_replay_loss
        + config.loss.tail_late_prefix_weight * late_prefix_loss
        + config.loss.tail_endpoint_weight * endpoint_loss
        + config.loss.tail_late_monotonic_weight * late_monotonic_loss
        + config.loss.prefix_weight * prefix_loss
        + config.loss.monotonic_weight * monotonic_loss
    )
    total.backward()
    grad_norm = clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        grad_clip_norm,
    )
    optimizer.step()
    return {
        "total": float(total.detach().cpu().item()),
        "tail": float(tail_loss.detach().cpu().item()),
        "early_dropout": float(early_dropout_loss.detach().cpu().item()),
        "tail_replay": float(tail_replay_loss.detach().cpu().item()),
        "late_prefix": float(late_prefix_loss.detach().cpu().item()),
        "endpoint": float(endpoint_loss.detach().cpu().item()),
        "late_monotonic": float(late_monotonic_loss.detach().cpu().item()),
        "epsilon": float(epsilon_loss.detach().cpu().item()),
        "prefix": float(prefix_loss.detach().cpu().item()),
        "monotonic": float(monotonic_loss.detach().cpu().item()),
        "grad_norm": float(grad_norm.detach().cpu().item()),
    }


@torch.no_grad()
def _write_prefix_grid(
    path: Path,
    model: CoFiTokTiny,
    schedule: DiffusionSchedule,
    clean_images: torch.Tensor,
    visualization_count: int,
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
    prefix_images = [
        schedule.predict_x0_from_epsilon(noisy_images, prefix_epsilon, timesteps)
        for prefix_epsilon in output.prefix_epsilons
    ]
    rows = [images, noisy_images, *prefix_images]
    grid_items = torch.cat([_denormalize(row).cpu() for row in rows], dim=0)
    grid = make_grid(grid_items, nrow=images.shape[0], padding=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)

    prefix_mse = [
        torch.nn.functional.mse_loss(prefix_image, images).detach().cpu().item()
        for prefix_image in prefix_images
    ]
    return {
        "grid_path": str(path),
        "rows": ["clean", "noisy", *[f"prefix_{index + 1}" for index in range(len(prefix_images))]],
        "timestep": int(timesteps[0].detach().cpu().item()),
        "prefix_mse_to_clean": prefix_mse,
        **_component_energy_summary(output.components),
        "diagnostics": run_synthesis_diagnostics(model, output.tokens),
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.runtime.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(config.runtime.device)

    model = CoFiTokTiny(config.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    _validate_tail_dropout(config, args.tail_start)
    replay_model = _build_replay_model(config, args.replay_checkpoint, device)
    trainable_names = _set_tail_trainable(model, args.tail_start)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=config.optimization.learning_rate,
        betas=tuple(config.optimization.betas),
        weight_decay=config.optimization.weight_decay,
    )
    schedule = DiffusionSchedule(config.diffusion, device=device)
    train_loader = build_dataloader(config.data, split="train")
    eval_loader = build_dataloader(config.data, split="val")
    train_batches = _cycle(train_loader)

    start = time.time()
    history = []
    latest_losses = {}
    for step in range(1, config.runtime.steps + 1):
        clean_images = _batch_images(next(train_batches), device)
        latest_losses = _tail_stage_step(
            config,
            model,
            schedule,
            optimizer,
            clean_images,
            tail_start=args.tail_start,
            grad_clip_norm=config.optimization.grad_clip_norm,
            replay_model=replay_model,
        )
        if step == 1 or step % config.optimization.log_interval == 0 or step == config.runtime.steps:
            entry = {"step": step, **latest_losses}
            history.append(entry)
            print(
                f"step {step:04d} "
                f"total={latest_losses['total']:.4f} "
                f"tail={latest_losses['tail']:.4f} "
                f"early_dropout={latest_losses['early_dropout']:.4f} "
                f"tail_replay={latest_losses['tail_replay']:.4f} "
                f"late_prefix={latest_losses['late_prefix']:.4f} "
                f"endpoint={latest_losses['endpoint']:.4f} "
                f"late_mono={latest_losses['late_monotonic']:.4f} "
                f"epsilon={latest_losses['epsilon']:.4f} "
                f"prefix={latest_losses['prefix']:.4f}"
            )

    eval_images = _batch_images(next(iter(eval_loader)), device)
    prefix_summary = _write_prefix_grid(
        output_dir / "prefix_final.png",
        model,
        schedule,
        eval_images,
        visualization_count=config.optimization.visualization_count,
    )
    checkpoint_path = output_dir / "checkpoint_final.pt"
    torch.save(
        {
            "config": config_to_dict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": config.runtime.steps,
            "tail_start": args.tail_start,
            "history": history,
            "warm_start_checkpoint": args.checkpoint,
            "replay_checkpoint": args.replay_checkpoint,
            "trainable_names": trainable_names,
        },
        checkpoint_path,
    )
    report = {
        "config": config_to_dict(config),
        "runtime": {
            "requested_device": config.runtime.device,
            "actual_device": str(device),
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "warm_start_checkpoint": args.checkpoint,
        "replay_checkpoint": args.replay_checkpoint,
        "tail_start": args.tail_start,
        "trainable_names": trainable_names,
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
