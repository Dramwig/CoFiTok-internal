from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cofitok.configs import config_to_dict, load_config
from cofitok.data import build_dataloader
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.training.losses import _denoise_path_prefix_targets
from cofitok.utils.seed import seed_everything


def _resolve_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _batch_images(batch: object, device: torch.device) -> torch.Tensor:
    images = batch[0] if isinstance(batch, (list, tuple)) else batch
    return images.to(device=device, dtype=torch.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively evaluate component order for a small-K CoFiTok checkpoint."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "validation", "test"])
    parser.add_argument("--max-batches", type=int, default=256)
    parser.add_argument("--max-images", type=int, default=1024)
    parser.add_argument("--timestep", type=int, default=500)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser.parse_args()


def component_orders(token_count: int) -> list[tuple[int, ...]]:
    if token_count < 2:
        raise ValueError("order evaluation requires at least two components")
    if token_count > 7:
        raise ValueError("exhaustive order evaluation is limited to token_count <= 7")
    return list(itertools.permutations(range(token_count)))


def _curve_auc_per_image(values: list[torch.Tensor]) -> torch.Tensor:
    stacked = torch.stack(values, dim=0)
    if stacked.shape[0] == 1:
        return stacked[0]
    return (0.5 * (stacked[:-1] + stacked[1:])).sum(dim=0) / (stacked.shape[0] - 1)


def _per_image_mse(candidate: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(candidate, reference, reduction="none").flatten(1).mean(dim=1)


def bootstrap_mean_ci(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
    confidence: float = 0.95,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("bootstrap values must be a one-dimensional array with at least two entries")
    if repetitions < 100:
        raise ValueError("bootstrap repetitions must be at least 100")
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    chunk_size = 512
    for start in range(0, repetitions, chunk_size):
        stop = min(start + chunk_size, repetitions)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    low, high = np.quantile(means, [alpha, 1.0 - alpha])
    return {
        "mean": float(values.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "confidence": confidence,
        "repetitions": repetitions,
    }


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.max_batches < 1 or args.max_images < 1:
        raise ValueError("max batches and max images must be positive")

    config = load_config(args.config)
    token_count = config.model.token_count
    orders = component_orders(token_count)
    ordered = tuple(range(token_count))
    reverse = tuple(reversed(ordered))
    ordered_index = orders.index(ordered)
    reverse_index = orders.index(reverse)

    seed_everything(config.runtime.seed)
    device = _resolve_device(config.runtime.device)
    model = CoFiTokTiny(config.model).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    schedule = DiffusionSchedule(config.diffusion, device=device)
    if not 0 <= args.timestep < schedule.num_train_timesteps:
        raise ValueError("timestep is outside the diffusion schedule")
    loader = build_dataloader(config.data, split=args.split, drop_last=False)

    auc_sums = torch.zeros(len(orders), dtype=torch.float64)
    ordered_values: list[torch.Tensor] = []
    nonidentity_mean_values: list[torch.Tensor] = []
    reverse_values: list[torch.Tensor] = []
    endpoint_values: list[torch.Tensor] = []
    image_count = 0
    batch_count = 0
    start = time.time()

    for batch in loader:
        if batch_count >= args.max_batches or image_count >= args.max_images:
            break
        clean_images = _batch_images(batch, device)[: args.max_images - image_count]
        timesteps = torch.full(
            (clean_images.shape[0],), args.timestep, dtype=torch.long, device=device
        )
        noise = torch.randn_like(clean_images)
        noisy_images = schedule.add_noise(clean_images, noise, timesteps)
        output = model(noisy_images, timesteps)
        targets = _denoise_path_prefix_targets(
            schedule,
            noisy_images,
            clean_images,
            timesteps,
            count=token_count,
            progress_power=config.loss.denoise_path_progress_power,
        )

        order_aucs = []
        order_endpoints = []
        for order in orders:
            running = torch.zeros_like(output.components[0])
            path_errors = []
            for position, component_index in enumerate(order):
                running = running + output.components[component_index]
                prediction = schedule.predict_x0_from_epsilon(noisy_images, running, timesteps)
                path_errors.append(_per_image_mse(prediction, targets[position]))
            order_aucs.append(_curve_auc_per_image(path_errors))
            order_endpoints.append(_per_image_mse(running, output.epsilon))

        auc_matrix = torch.stack(order_aucs, dim=0)
        endpoint_matrix = torch.stack(order_endpoints, dim=0)
        nonidentity_indices = [index for index in range(len(orders)) if index != ordered_index]
        auc_sums += auc_matrix.sum(dim=1).to(dtype=torch.float64, device="cpu")
        ordered_values.append(auc_matrix[ordered_index].cpu())
        nonidentity_mean_values.append(auc_matrix[nonidentity_indices].mean(dim=0).cpu())
        reverse_values.append(auc_matrix[reverse_index].cpu())
        endpoint_values.append(endpoint_matrix.cpu())
        image_count += int(clean_images.shape[0])
        batch_count += 1

    if image_count == 0:
        raise RuntimeError("No images were evaluated")

    permutation_means = (auc_sums / image_count).tolist()
    ordered_per_image = torch.cat(ordered_values).numpy()
    nonidentity_per_image = torch.cat(nonidentity_mean_values).numpy()
    reverse_per_image = torch.cat(reverse_values).numpy()
    endpoint_matrix = torch.cat(endpoint_values, dim=1)
    nonidentity_delta = nonidentity_per_image - ordered_per_image
    reverse_delta = reverse_per_image - ordered_per_image
    ranked = sorted(range(len(orders)), key=lambda index: permutation_means[index])

    permutation_rows = [
        {
            "order": list(order),
            "denoise_path_mse_auc": float(permutation_means[index]),
            "rank": ranked.index(index) + 1,
        }
        for index, order in enumerate(orders)
    ]
    return {
        "schema_version": 1,
        "config": config_to_dict(config),
        "checkpoint": args.checkpoint,
        "evaluation": {
            "split": args.split,
            "image_count": image_count,
            "batch_count": batch_count,
            "fixed_timestep": args.timestep,
            "token_count": token_count,
            "permutation_count": len(orders),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "bootstrap_seed": args.bootstrap_seed,
        },
        "summary": {
            "ordered_path_auc": float(permutation_means[ordered_index]),
            "reverse_path_auc": float(permutation_means[reverse_index]),
            "nonidentity_mean_path_auc": float(
                sum(value for index, value in enumerate(permutation_means) if index != ordered_index)
                / (len(orders) - 1)
            ),
            "ordered_rank": ranked.index(ordered_index) + 1,
            "ordered_rank_fraction": (ranked.index(ordered_index) + 1) / len(orders),
            "fraction_nonidentity_worse": float(
                sum(
                    value > permutation_means[ordered_index]
                    for index, value in enumerate(permutation_means)
                    if index != ordered_index
                )
                / (len(orders) - 1)
            ),
            "nonidentity_minus_ordered_paired_bootstrap": bootstrap_mean_ci(
                nonidentity_delta,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed,
            ),
            "reverse_minus_ordered_paired_bootstrap": bootstrap_mean_ci(
                reverse_delta,
                repetitions=args.bootstrap_repetitions,
                seed=args.bootstrap_seed + 1,
            ),
            "endpoint_epsilon_sum_mse_max": float(endpoint_matrix.max().item()),
        },
        "permutations": permutation_rows,
        "runtime": {
            "actual_device": str(device),
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = evaluate(args)
    report_path = output_dir / "order_permutation_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
