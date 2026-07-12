from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from cofitok.configs import config_to_dict, load_config
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample images from a CoFiTok checkpoint.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pt.")
    parser.add_argument("--output-dir", required=True, help="Directory for sample grids and report.")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--eta", type=float, default=0.0, help="DDIM stochasticity; eta=0 is deterministic.")
    parser.add_argument("--seed", type=int, default=-1, help="Sampling seed; default uses config runtime seed.")
    parser.add_argument("--prefix-budgets", default="", help="Comma-separated budgets; default is 1,2,4,8,K.")
    parser.add_argument("--clip-x0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-images", action="store_true", help="Also save individual PNG samples per prefix budget.")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def denormalize(images: torch.Tensor) -> torch.Tensor:
    return (images.clamp(-1.0, 1.0) + 1.0) * 0.5


def save_individual_samples(samples: torch.Tensor, output_dir: Path, prefix_budget: int) -> list[str]:
    image_dir = output_dir / f"samples_prefix_{prefix_budget}"
    image_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, image in enumerate(denormalize(samples)):
        path = image_dir / f"sample_{index:05d}.png"
        save_image(image, path)
        paths.append(str(path))
    return paths


def select_sampling_timesteps(num_train_timesteps: int, sample_steps: int) -> list[int]:
    if sample_steps < 1:
        raise ValueError(f"sample_steps must be >= 1, got {sample_steps}")
    steps = min(sample_steps, num_train_timesteps)
    values = torch.linspace(0, num_train_timesteps - 1, steps=steps).round().to(dtype=torch.long).unique()
    timesteps = sorted((int(value.item()) for value in values), reverse=True)
    if timesteps[-1] != 0:
        timesteps.append(0)
    return timesteps


def parse_prefix_budgets(raw: str, token_count: int) -> list[int]:
    if raw.strip():
        budgets = [int(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        budgets = [1, 2, 4, 8, token_count]
    budgets = sorted({budget for budget in budgets if 1 <= budget <= token_count})
    if token_count not in budgets:
        budgets.append(token_count)
    return budgets


def _prefix_epsilon(output: object, budget: int) -> torch.Tensor:
    prefix_epsilons = getattr(output, "prefix_epsilons")
    if not 1 <= budget <= len(prefix_epsilons):
        raise ValueError(f"prefix budget must be in [1, {len(prefix_epsilons)}], got {budget}")
    return prefix_epsilons[budget - 1]


@torch.no_grad()
def ddim_sample(
    model: CoFiTokTiny,
    schedule: DiffusionSchedule,
    shape: tuple[int, int, int, int],
    sample_steps: int,
    prefix_budget: int,
    eta: float,
    clip_x0: bool,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    model.eval()
    image = torch.randn(shape, device=device, generator=generator)
    timesteps = select_sampling_timesteps(schedule.num_train_timesteps, sample_steps)
    for index, timestep in enumerate(timesteps):
        previous_timestep = timesteps[index + 1] if index + 1 < len(timesteps) else -1
        t = torch.full((shape[0],), timestep, dtype=torch.long, device=device)
        output = model(image, t)
        epsilon = _prefix_epsilon(output, prefix_budget)
        predicted_x0 = schedule.predict_x0_from_epsilon(image, epsilon, t)
        if clip_x0:
            predicted_x0 = predicted_x0.clamp(-1.0, 1.0)
        if previous_timestep < 0:
            image = predicted_x0
            continue

        alpha_t = schedule.alphas_cumprod[timestep]
        alpha_prev = schedule.alphas_cumprod[previous_timestep]
        sigma = eta * torch.sqrt((1.0 - alpha_prev) / (1.0 - alpha_t).clamp_min(1e-12))
        sigma = sigma * torch.sqrt((1.0 - alpha_t / alpha_prev).clamp_min(0.0))
        direction_scale = torch.sqrt((1.0 - alpha_prev - sigma.pow(2)).clamp_min(0.0))
        image = torch.sqrt(alpha_prev) * predicted_x0 + direction_scale * epsilon
        if eta > 0.0:
            image = image + sigma * torch.randn(image.shape, device=device, generator=generator)
    return image


@torch.no_grad()
def sample_budget_grid(
    model: CoFiTokTiny,
    schedule: DiffusionSchedule,
    batch_shape: tuple[int, int, int, int],
    total_samples: int,
    sample_steps: int,
    prefix_budget: int,
    eta: float,
    clip_x0: bool,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    samples = []
    remaining = total_samples
    batch_size = batch_shape[0]
    generator = torch.Generator(device=device).manual_seed(seed)
    while remaining > 0:
        current = min(batch_size, remaining)
        shape = (current, batch_shape[1], batch_shape[2], batch_shape[3])
        samples.append(
            ddim_sample(
                model=model,
                schedule=schedule,
                shape=shape,
                sample_steps=sample_steps,
                prefix_budget=prefix_budget,
                eta=eta,
                clip_x0=clip_x0,
                device=device,
                generator=generator,
            ).cpu()
        )
        remaining -= current
    return torch.cat(samples, dim=0)


def main() -> None:
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    config = load_config(args.config)
    seed = config.runtime.seed if args.seed < 0 else args.seed
    seed_everything(seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(config.runtime.device)

    model = CoFiTokTiny(config.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    schedule = DiffusionSchedule(config.diffusion, device=device)
    budgets = parse_prefix_budgets(args.prefix_budgets, config.model.token_count)
    sample_timesteps = select_sampling_timesteps(schedule.num_train_timesteps, args.sample_steps)

    start = time.time()
    artifacts: dict[str, str] = {}
    for budget in budgets:
        samples = sample_budget_grid(
            model=model,
            schedule=schedule,
            batch_shape=(min(args.batch_size, args.num_samples), config.model.image_channels, config.model.image_size, config.model.image_size),
            total_samples=args.num_samples,
            sample_steps=args.sample_steps,
            prefix_budget=budget,
            eta=args.eta,
            clip_x0=args.clip_x0,
            device=device,
            seed=seed,
        )
        grid = make_grid(denormalize(samples), nrow=min(args.batch_size, args.num_samples), padding=2)
        path = output_dir / f"samples_prefix_{budget}.png"
        save_image(grid, path)
        artifacts[f"samples_prefix_{budget}"] = str(path)
        if args.save_images:
            image_paths = save_individual_samples(samples, output_dir, budget)
            artifacts[f"samples_prefix_{budget}_dir"] = str(output_dir / f"samples_prefix_{budget}")
            artifacts[f"samples_prefix_{budget}_count"] = len(image_paths)

    report = {
        "config": config_to_dict(config),
        "checkpoint": args.checkpoint,
        "sampling": {
            "num_samples": args.num_samples,
            "batch_size": args.batch_size,
            "sample_steps": args.sample_steps,
            "actual_timestep_count": len(sample_timesteps),
            "first_timestep": sample_timesteps[0],
            "last_timestep": sample_timesteps[-1],
            "prefix_budgets": budgets,
            "eta": args.eta,
            "clip_x0": args.clip_x0,
            "save_images": args.save_images,
            "seed": seed,
        },
        "runtime": {
            "requested_device": config.runtime.device,
            "actual_device": str(device),
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "artifacts": artifacts,
    }
    report_path = output_dir / "sample_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")
    for path in artifacts.values():
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
