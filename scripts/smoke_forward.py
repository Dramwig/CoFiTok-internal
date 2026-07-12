from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from cofitok.configs import config_to_dict, load_config
from cofitok.data import build_dataloader
from cofitok.diagnostics import run_synthesis_diagnostics
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report
from cofitok.training import run_smoke_step
from cofitok.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal CoFiTok forward/loss smoke test.")
    parser.add_argument("--config", required=True, help="Path to JSON experiment config.")
    parser.add_argument("--output", required=True, help="Path to write JSON report.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(config.runtime.seed)

    device_name = config.runtime.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    dataloader = build_dataloader(config.data, split="train")
    batch = next(iter(dataloader))
    if isinstance(batch, (list, tuple)):
        clean_images = batch[0]
    else:
        clean_images = batch
    clean_images = clean_images.to(device=device, dtype=torch.float32)

    model = CoFiTokTiny(config.model).to(device)
    schedule = DiffusionSchedule(config.diffusion, device=device)

    start = time.time()
    smoke = run_smoke_step(config, model, schedule, clean_images)
    elapsed = time.time() - start

    with torch.no_grad():
        noise = torch.randn_like(clean_images)
        timesteps = schedule.sample_timesteps(clean_images.shape[0], device=device)
        noisy_images = schedule.add_noise(clean_images, noise, timesteps)
        output = model(noisy_images, timesteps)
        diagnostics = run_synthesis_diagnostics(model, output.tokens)

    report = {
        "config": config_to_dict(config),
        "runtime": {
            "requested_device": config.runtime.device,
            "actual_device": str(device),
            "elapsed_seconds": elapsed,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "losses": smoke.losses.as_dict(),
        "output_shapes": smoke.output_shapes,
        "component_energy": smoke.component_energy,
        "diagnostics": diagnostics,
    }
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
