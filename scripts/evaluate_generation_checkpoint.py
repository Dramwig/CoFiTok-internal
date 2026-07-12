from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from cofitok.configs import config_from_dict, config_to_dict
from cofitok.data import build_dataloader
from cofitok.diffusion import DiffusionSchedule
from cofitok.metrics import normalized_curve_auc
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.training import ExponentialMovingAverage
from cofitok.training.runtime import autocast_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a production generation checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=1024)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--random-orders", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    return parser.parse_args()


def component_orders(token_count: int, random_orders: int, seed: int) -> dict[str, list[int]]:
    if token_count < 1:
        raise ValueError("token_count must be positive")
    if random_orders < 0:
        raise ValueError("random_orders must be non-negative")
    orders = {
        "ordered": list(range(token_count)),
        "reverse": list(reversed(range(token_count))),
    }
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for index in range(random_orders):
        orders[f"random_{index:02d}"] = torch.randperm(token_count, generator=generator).tolist()
    unique: dict[tuple[int, ...], str] = {}
    deduplicated = {}
    for name, order in orders.items():
        key = tuple(order)
        if key in unique:
            continue
        unique[key] = name
        deduplicated[name] = order
    return deduplicated


def prefix_tensors(components: list[torch.Tensor], order: list[int]) -> list[torch.Tensor]:
    running = torch.zeros_like(components[0])
    prefixes = []
    for index in order:
        running = running + components[index]
        prefixes.append(running)
    return prefixes


def spatial_prefix_targets(clean: torch.Tensor, count: int) -> list[torch.Tensor]:
    targets = []
    max_scale = min(clean.shape[-2:])
    for index in range(count):
        if index == count - 1:
            targets.append(clean)
            continue
        scale = min(2 ** (count - index - 1), max_scale)
        pooled = torch.nn.functional.avg_pool2d(
            clean,
            kernel_size=scale,
            stride=scale,
            ceil_mode=True,
        )
        targets.append(
            torch.nn.functional.interpolate(
                pooled,
                size=clean.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        )
    return targets


def denoise_path_targets(
    schedule: DiffusionSchedule,
    noisy: torch.Tensor,
    clean: torch.Tensor,
    timesteps: torch.Tensor,
    count: int,
    progress_power: float,
) -> list[torch.Tensor]:
    start = schedule.predict_x0_from_epsilon(noisy, torch.zeros_like(noisy), timesteps)
    targets = []
    for index, spatial in enumerate(spatial_prefix_targets(clean, count)):
        progress = ((index + 1) / count) ** max(progress_power, 1e-6)
        targets.append(torch.lerp(start, spatial, progress))
    return targets


def _batch(batch: object, device: torch.device, class_conditional: bool):
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
        raise ValueError("class-conditional evaluation batch is missing labels")
    return images, labels.to(device=device, dtype=torch.long, non_blocking=True)


def _psnr(mse: float) -> float:
    return 10.0 * math.log10(4.0 / max(mse, 1e-12))


@torch.no_grad()
def evaluate(
    *,
    model: CoFiTokTiny,
    schedule: DiffusionSchedule,
    loader: torch.utils.data.DataLoader,
    num_images: int,
    timestep: int,
    orders: dict[str, list[int]],
    progress_power: float,
    class_conditional: bool,
    device: torch.device,
    precision: str,
    seed: int,
) -> dict[str, object]:
    model.eval()
    token_count = model.config.token_count
    clean_sums = {name: [0.0] * token_count for name in orders}
    path_sums = {name: [0.0] * token_count for name in orders}
    component_energy_sums = [0.0] * token_count
    random_energy_sums = [0.0] * token_count
    shuffled_endpoint_sum = 0.0
    zero_max_abs = 0.0
    count = 0
    generator = torch.Generator(device=device).manual_seed(seed)

    for batch in loader:
        if count >= num_images:
            break
        clean, labels = _batch(batch, device, class_conditional)
        clean = clean[: num_images - count]
        if labels is not None:
            labels = labels[: clean.shape[0]]
        batch_size = clean.shape[0]
        timesteps = torch.full((batch_size,), timestep, device=device, dtype=torch.long)
        noise = torch.randn(clean.shape, device=device, generator=generator)
        noisy = schedule.add_noise(clean, noise, timesteps)
        with autocast_context(device, precision):
            output = model(noisy, timesteps, class_labels=labels)
        path_targets = denoise_path_targets(
            schedule,
            noisy,
            clean,
            timesteps,
            token_count,
            progress_power,
        )
        for index, component in enumerate(output.components):
            component_energy_sums[index] += float(component.float().square().mean()) * batch_size

        random_tokens = [
            torch.randn(token.shape, device=device, generator=generator, dtype=token.dtype)
            for token in output.tokens
        ]
        random_components = model.synthesis(random_tokens)
        for index, component in enumerate(random_components):
            random_energy_sums[index] += float(component.float().square().mean()) * batch_size

        zero_components = model.synthesis.zero_components_like(output.tokens)
        zero_max_abs = max(
            zero_max_abs,
            max(float(component.float().abs().max()) for component in zero_components),
        )

        shuffled_tokens = [torch.roll(token, shifts=index + 1, dims=0) for index, token in enumerate(output.tokens)]
        shuffled_epsilon = torch.stack(model.synthesis(shuffled_tokens)).sum(dim=0)
        shuffled_x0 = schedule.predict_x0_from_epsilon(noisy, shuffled_epsilon, timesteps)
        shuffled_endpoint_sum += float(
            torch.nn.functional.mse_loss(shuffled_x0.float(), clean.float())
        ) * batch_size

        for name, order in orders.items():
            for prefix_index, epsilon in enumerate(prefix_tensors(output.components, order)):
                predicted_x0 = schedule.predict_x0_from_epsilon(noisy, epsilon, timesteps)
                clean_sums[name][prefix_index] += float(
                    torch.nn.functional.mse_loss(predicted_x0.float(), clean.float())
                ) * batch_size
                path_sums[name][prefix_index] += float(
                    torch.nn.functional.mse_loss(
                        predicted_x0.float(),
                        path_targets[prefix_index].float(),
                    )
                ) * batch_size
        count += batch_size

    if count == 0:
        raise RuntimeError("evaluation loader yielded no images")
    order_metrics = {}
    for name, order in orders.items():
        clean_curve = [value / count for value in clean_sums[name]]
        path_curve = [value / count for value in path_sums[name]]
        order_metrics[name] = {
            "order": order,
            "prefix_clean_mse": clean_curve,
            "prefix_clean_psnr": [_psnr(value) for value in clean_curve],
            "prefix_clean_mse_auc": normalized_curve_auc(clean_curve),
            "prefix_path_mse": path_curve,
            "prefix_path_mse_auc": normalized_curve_auc(path_curve),
            "endpoint_clean_mse": clean_curve[-1],
            "endpoint_clean_psnr": _psnr(clean_curve[-1]),
        }
    ranked = sorted(order_metrics, key=lambda name: order_metrics[name]["prefix_path_mse_auc"])
    ordered_endpoint = order_metrics["ordered"]["endpoint_clean_mse"]
    return {
        "evaluated_images": count,
        "timestep": timestep,
        "orders": order_metrics,
        "ordered_rank_by_path_auc": ranked.index("ordered") + 1,
        "order_count": len(ranked),
        "ranked_orders_by_path_auc": ranked,
        "component_energy": [value / count for value in component_energy_sums],
        "random_token_component_energy": [value / count for value in random_energy_sums],
        "zero_token_max_abs": zero_max_abs,
        "shuffled_endpoint_clean_mse": shuffled_endpoint_sum / count,
        "shuffled_to_ordered_endpoint_ratio": (shuffled_endpoint_sum / count)
        / max(ordered_endpoint, 1e-12),
    }


def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("num-images must be positive")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = config_from_dict(checkpoint["config"])
    if not 0 <= args.timestep < config.diffusion.num_train_timesteps:
        raise ValueError("timestep is outside the diffusion schedule")
    device = torch.device(config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA checkpoint evaluation requested but CUDA is unavailable")
    model = CoFiTokTiny(config.model).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.weights == "ema":
        ema = ExponentialMovingAverage(
            model,
            decay=config.optimization.ema_decay,
            warmup_steps=config.optimization.ema_warmup_steps,
        )
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    schedule = DiffusionSchedule(config.diffusion, device=device)
    loader = build_dataloader(config.data, split="val", drop_last=False)
    orders = component_orders(config.model.token_count, args.random_orders, args.seed)
    start = time.time()
    metrics = evaluate(
        model=model,
        schedule=schedule,
        loader=loader,
        num_images=args.num_images,
        timestep=args.timestep,
        orders=orders,
        progress_power=config.loss.denoise_path_progress_power,
        class_conditional=config.data.class_conditional,
        device=device,
        precision=args.precision,
        seed=args.seed,
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "checkpoint": checkpoint_path.resolve().as_posix(),
        "checkpoint_step": int(checkpoint["step"]),
        "weights": args.weights,
        "precision": args.precision,
        "config": config_to_dict(config),
        "metrics": metrics,
        "runtime": {
            "elapsed_seconds": time.time() - start,
            "device": str(device),
            "torch_version": torch.__version__,
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "checkpoint_evaluation_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()

