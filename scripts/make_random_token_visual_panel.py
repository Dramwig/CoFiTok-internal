from __future__ import annotations

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image

from cofitok.configs import ExperimentConfig, _build_config, config_to_dict, load_config
from cofitok.data import build_dataloader
from cofitok.diagnostics import run_synthesis_diagnostics, shuffle_tokens_across_batch
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.utils.seed import seed_everything
from train_short import _batch_images, _denormalize, _resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render appendix visuals for random-token synthesis diagnostics.")
    parser.add_argument("--checkpoint", required=True, help="Path to a CoFiTok checkpoint_final.pt.")
    parser.add_argument("--output-dir", required=True, help="Directory for PNG panel and JSON report.")
    parser.add_argument("--config", default="", help="Optional config JSON. Defaults to checkpoint['config'].")
    parser.add_argument("--split", default="val", choices=["train", "val", "validation", "test"])
    parser.add_argument("--num-images", type=int, default=4, help="Number of columns/images in the panel.")
    parser.add_argument("--seed", type=int, default=20260710)
    parser.add_argument("--timestep", type=int, default=-1, help="Fixed timestep. Defaults to midpoint.")
    parser.add_argument("--prefix-budgets", default="", help="Comma-separated budgets. Defaults to 1,2,4,8,K.")
    return parser.parse_args()


def _load_config_from_checkpoint(checkpoint: dict[str, Any], config_path: str) -> ExperimentConfig:
    if config_path:
        return load_config(config_path)
    raw_config = checkpoint.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("Checkpoint does not contain a config dict; pass --config explicitly.")
    return _build_config(raw_config)


def _prefix_epsilons_from_components(components: list[torch.Tensor]) -> list[torch.Tensor]:
    prefix_epsilons = []
    running = torch.zeros_like(components[0])
    for component in components:
        running = running + component
        prefix_epsilons.append(running)
    return prefix_epsilons


def _prefix_budgets(raw: str, token_count: int) -> list[int]:
    if raw.strip():
        budgets = [int(value.strip()) for value in raw.split(",") if value.strip()]
    else:
        budgets = [1, 2, 4, 8, token_count]
    return sorted({budget for budget in budgets if 1 <= budget <= token_count})


def _predict_prefix_images(
    schedule: DiffusionSchedule,
    noisy_images: torch.Tensor,
    prefix_epsilons: list[torch.Tensor],
    timesteps: torch.Tensor,
    budgets: list[int],
) -> dict[int, torch.Tensor]:
    return {
        budget: schedule.predict_x0_from_epsilon(noisy_images, prefix_epsilons[budget - 1], timesteps)
        for budget in budgets
    }


def _mse_by_budget(prefix_images: dict[int, torch.Tensor], clean_images: torch.Tensor) -> dict[str, float]:
    return {
        str(budget): float(F.mse_loss(image, clean_images).detach().cpu().item())
        for budget, image in prefix_images.items()
    }


def _row_grid(row: torch.Tensor, num_images: int) -> Image.Image:
    grid = make_grid(_denormalize(row).cpu(), nrow=num_images, padding=2)
    return to_pil_image(grid)


def _save_labeled_panel(rows: list[tuple[str, torch.Tensor]], path: Path, num_images: int) -> None:
    row_images = [_row_grid(row, num_images) for _, row in rows]
    label_width = 184
    row_gap = 8
    title_height = 36
    width = label_width + max(image.width for image in row_images)
    height = title_height + sum(image.height for image in row_images) + row_gap * (len(row_images) - 1)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "Random-token synthesis diagnostic", fill=(20, 20, 20))
    y = title_height
    for (label, _), image in zip(rows, row_images):
        draw.text((12, y + 8), label, fill=(20, 20, 20))
        canvas.paste(image.convert("RGB"), (label_width, y))
        y += image.height + row_gap
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def _save_unlabeled_grid(rows: list[tuple[str, torch.Tensor]], path: Path, num_images: int) -> None:
    grid_items = torch.cat([_denormalize(row).cpu() for _, row in rows], dim=0)
    grid = make_grid(grid_items, nrow=num_images, padding=2)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.num_images < 1:
        raise ValueError("--num-images must be >= 1")

    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = _load_config_from_checkpoint(checkpoint, args.config)
    seed_everything(args.seed)
    config = replace(
        config,
        data=replace(config.data, batch_size=max(args.num_images, config.data.batch_size)),
        runtime=replace(config.runtime, seed=args.seed),
    )

    device = _resolve_device(config.runtime.device)
    model = CoFiTokTiny(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    schedule = DiffusionSchedule(config.diffusion, device=device)
    timestep = args.timestep if args.timestep >= 0 else schedule.num_train_timesteps // 2
    if not 0 <= timestep < schedule.num_train_timesteps:
        raise ValueError(f"--timestep must be in [0, {schedule.num_train_timesteps}), got {timestep}")
    budgets = _prefix_budgets(args.prefix_budgets, config.model.token_count)
    if not budgets:
        raise ValueError("No valid prefix budgets were selected.")

    start = time.time()
    eval_loader = build_dataloader(config.data, split=args.split, drop_last=False)
    clean_images = _batch_images(next(iter(eval_loader)), device)[: args.num_images]
    timesteps = torch.full((clean_images.shape[0],), timestep, dtype=torch.long, device=device)
    noise = torch.randn_like(clean_images)
    noisy_images = schedule.add_noise(clean_images, noise, timesteps)

    output = model(noisy_images, timesteps)
    ordered_prefix_images = _predict_prefix_images(
        schedule,
        noisy_images,
        _prefix_epsilons_from_components(output.components),
        timesteps,
        budgets,
    )

    random_tokens = [torch.randn_like(token) for token in output.tokens]
    random_components = model.synthesis(random_tokens)
    random_prefix_images = _predict_prefix_images(
        schedule,
        noisy_images,
        _prefix_epsilons_from_components(random_components),
        timesteps,
        budgets,
    )

    zero_components = model.synthesis.zero_components_like(output.tokens)
    zero_prefix_images = _predict_prefix_images(
        schedule,
        noisy_images,
        _prefix_epsilons_from_components(zero_components),
        timesteps,
        [config.model.token_count],
    )

    shuffled_components = model.synthesis(shuffle_tokens_across_batch(output.tokens))
    shuffled_prefix_images = _predict_prefix_images(
        schedule,
        noisy_images,
        _prefix_epsilons_from_components(shuffled_components),
        timesteps,
        [config.model.token_count],
    )

    rows: list[tuple[str, torch.Tensor]] = [
        ("clean x0", clean_images),
        ("noisy xt", noisy_images),
    ]
    rows.extend((f"ordered m={budget}", ordered_prefix_images[budget]) for budget in budgets)
    rows.extend((f"random z m={budget}", random_prefix_images[budget]) for budget in budgets)
    rows.append((f"zero z m={config.model.token_count}", zero_prefix_images[config.model.token_count]))
    rows.append((f"shuffled z m={config.model.token_count}", shuffled_prefix_images[config.model.token_count]))

    panel_path = output_dir / "random_token_prefix_panel.png"
    grid_path = output_dir / "random_token_prefix_grid.png"
    _save_labeled_panel(rows, panel_path, clean_images.shape[0])
    _save_unlabeled_grid(rows, grid_path, clean_images.shape[0])

    diagnostics = run_synthesis_diagnostics(model, output.tokens)
    report = {
        "checkpoint": str(checkpoint_path),
        "config": config_to_dict(config),
        "evaluation": {
            "split": args.split,
            "seed": args.seed,
            "num_images": int(clean_images.shape[0]),
            "fixed_timestep": int(timestep),
            "prefix_budgets": budgets,
            "random_token_distribution": "iid torch.randn_like(model_tokens)",
            "image_range": "[-1, 1]",
            "note": "Diagnostic visualization only; not a generation-quality metric.",
        },
        "rows": [label for label, _ in rows],
        "metrics": {
            "ordered_prefix_mse_to_clean": _mse_by_budget(ordered_prefix_images, clean_images),
            "random_token_prefix_mse_to_clean": _mse_by_budget(random_prefix_images, clean_images),
            "zero_token_final_mse_to_clean": _mse_by_budget(zero_prefix_images, clean_images),
            "shuffled_token_final_mse_to_clean": _mse_by_budget(shuffled_prefix_images, clean_images),
        },
        "diagnostics": diagnostics,
        "artifacts": {
            "labeled_panel": str(panel_path),
            "unlabeled_grid": str(grid_path),
        },
        "runtime": {
            "requested_device": config.runtime.device,
            "actual_device": str(device),
            "elapsed_seconds": time.time() - start,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    report_path = output_dir / "random_token_diagnostic_report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")
    print(f"wrote {panel_path}")
    print(f"wrote {grid_path}")


if __name__ == "__main__":
    main()
