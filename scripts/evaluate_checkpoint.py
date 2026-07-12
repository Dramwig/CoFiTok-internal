from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cofitok.configs import config_to_dict, load_config
from cofitok.data import build_dataloader
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.utils.seed import seed_everything
from train_short import _batch_images, _resolve_device, _write_prefix_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved CoFiTok checkpoint.")
    parser.add_argument("--config", required=True, help="Path to the JSON config used for the checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_final.pt.")
    parser.add_argument("--output-dir", required=True, help="Directory for report and prefix grids.")
    parser.add_argument(
        "--component-order",
        choices=["ordered", "reverse", "random"],
        default="ordered",
        help="Order used when accumulating components into prefixes.",
    )
    parser.add_argument("--random-order-seed", type=int, default=0, help="Seed for random component order.")
    return parser.parse_args()


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
    schedule = DiffusionSchedule(config.diffusion, device=device)
    eval_loader = build_dataloader(config.data, split="val")

    start = time.time()
    eval_images = _batch_images(next(iter(eval_loader)), device)
    prefix_summary = _write_prefix_grid(
        output_dir / "prefix_final.png",
        model,
        schedule,
        eval_images,
        visualization_count=config.optimization.visualization_count,
        denoise_path_progress_power=config.loss.denoise_path_progress_power,
        component_order=args.component_order,
        random_order_seed=args.random_order_seed,
    )
    elapsed = time.time() - start

    report = {
        "config": config_to_dict(config),
        "checkpoint": args.checkpoint,
        "runtime": {
            "requested_device": config.runtime.device,
            "actual_device": str(device),
            "elapsed_seconds": elapsed,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "prefix_summary": prefix_summary,
        "artifacts": {
            "prefix_grid": str(output_dir / "prefix_final.png"),
            "shuffled_prefix_grid": prefix_summary["shuffled_grid_path"],
        },
    }
    report_path = output_dir / "report.json"
    write_json_report(report_path, report)
    print(f"wrote {report_path}")
    print(f"wrote {output_dir / 'prefix_final.png'}")
    print(f"wrote {prefix_summary['shuffled_grid_path']}")


if __name__ == "__main__":
    main()
