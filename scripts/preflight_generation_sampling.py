from __future__ import annotations

import argparse
import gc
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from cofitok.diffusion import predict_epsilon
from cofitok.generation import load_generation_model
from cofitok.reporting import write_json_report
from cofitok.training.runtime import autocast_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real generation forward before creating a formal sample set."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--prefix-budget", type=int, default=0, help="Default: all tokens.")
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--cfg-batch-mode", choices=["batched", "sequential"], default="batched")
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup-forwards", type=int, default=0)
    parser.add_argument("--measured-forwards", type=int, default=1)
    return parser.parse_args()


def _cuda_memory(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def run_sampling_preflight(
    checkpoint: str | Path,
    *,
    batch_size: int,
    prefix_budget: int = 0,
    guidance_scale: float = 1.5,
    guidance_rescale: float = 0.0,
    cfg_batch_mode: str = "batched",
    weights: str = "ema",
    precision: str = "bf16",
    warmup_forwards: int = 0,
    measured_forwards: int = 1,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch-size must be positive")
    if guidance_scale < 0.0:
        raise ValueError("guidance-scale must be non-negative")
    if cfg_batch_mode not in {"batched", "sequential"}:
        raise ValueError("cfg-batch-mode must be batched or sequential")
    if precision not in {"fp32", "bf16", "fp16"}:
        raise ValueError("precision must be fp32, bf16, or fp16")
    if warmup_forwards < 0 or measured_forwards < 1:
        raise ValueError("sampling preflight forward counts are invalid")

    loaded = load_generation_model(checkpoint, weights=weights)
    model = loaded.model
    config = loaded.config
    device = loaded.device
    budget = prefix_budget or config.model.token_count
    if not 1 <= budget <= config.model.token_count:
        raise ValueError("prefix-budget is outside the model token range")

    class_labels = None
    if config.model.num_classes > 0:
        class_labels = torch.arange(batch_size, device=device) % config.model.num_classes
    images = torch.zeros(
        batch_size,
        config.model.image_channels,
        config.model.image_size,
        config.model.image_size,
        device=device,
    )
    timesteps = torch.full(
        (batch_size,),
        config.diffusion.num_train_timesteps - 1,
        dtype=torch.long,
        device=device,
    )
    uses_cfg = class_labels is not None and guidance_scale != 1.0
    effective_batch_size = batch_size * 2 if uses_cfg and cfg_batch_mode == "batched" else batch_size
    forward_passes = 2 if uses_cfg and cfg_batch_mode == "sequential" else 1

    report = {
        "schema_version": 1,
        "status": "running",
        "checkpoint": str(loaded.checkpoint_path),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_integrity_manifest": str(loaded.checkpoint_integrity_manifest),
        "checkpoint_step": loaded.checkpoint_step,
        "weights": loaded.weights,
        "requested_weights": weights,
        "artifact_type": loaded.artifact_type,
        "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
        "device": str(device),
        "torch_version": torch.__version__,
        "request": {
            "batch_size": batch_size,
            "effective_model_batch_size": effective_batch_size,
            "forward_passes": forward_passes,
            "image_shape": [
                config.model.image_channels,
                config.model.image_size,
                config.model.image_size,
            ],
            "prefix_budget": budget,
            "token_count": config.model.token_count,
            "precision": precision,
            "guidance_scale": guidance_scale,
            "guidance_rescale": guidance_rescale,
            "cfg_batch_mode": cfg_batch_mode,
            "class_conditional": class_labels is not None,
            "timestep": config.diffusion.num_train_timesteps - 1,
            "warmup_forwards": warmup_forwards,
            "measured_forwards": measured_forwards,
        },
    }
    try:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        with torch.inference_mode(), autocast_context(device, precision):
            for _ in range(warmup_forwards):
                epsilon = predict_epsilon(
                    model,
                    images,
                    timesteps,
                    prefix_budget=budget,
                    class_labels=class_labels,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    cfg_batch_mode=cfg_batch_mode,
                )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        baseline_memory = _cuda_memory(device)
        durations = []
        for _ in range(measured_forwards):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode(), autocast_context(device, precision):
                epsilon = predict_epsilon(
                    model,
                    images,
                    timesteps,
                    prefix_budget=budget,
                    class_labels=class_labels,
                    guidance_scale=guidance_scale,
                    guidance_rescale=guidance_rescale,
                    cfg_batch_mode=cfg_batch_mode,
                )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            durations.append(time.perf_counter() - started)
        output_finite = bool(torch.isfinite(epsilon).all().item())
        elapsed_seconds = sum(durations)
        mean_forward_seconds = statistics.fmean(durations)
        sorted_durations = sorted(durations)
        p95_index = max(0, (95 * len(sorted_durations) + 99) // 100 - 1)
        peak_memory = _cuda_memory(device)
        if not output_finite:
            raise RuntimeError("sampling preflight produced non-finite epsilon values")
        if tuple(epsilon.shape) != tuple(images.shape):
            raise RuntimeError("sampling preflight epsilon shape does not match the image batch")
        report.update(
            status="passed",
            result={
                "elapsed_seconds": elapsed_seconds,
                "mean_forward_seconds": mean_forward_seconds,
                "median_forward_seconds": statistics.median(durations),
                "p95_forward_seconds": sorted_durations[p95_index],
                "durations_seconds": durations,
                "output_images_per_second": batch_size / mean_forward_seconds,
                "effective_model_images_per_second": (
                    effective_batch_size / mean_forward_seconds
                ),
                "output_shape": list(epsilon.shape),
                "output_dtype": str(epsilon.dtype),
                "output_finite": output_finite,
                "cuda_memory_before_forward": baseline_memory,
                "cuda_memory_after_forward": peak_memory,
                "device_total_memory_bytes": (
                    torch.cuda.get_device_properties(device).total_memory
                    if device.type == "cuda"
                    else 0
                ),
            },
        )
    except Exception as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            cuda_memory_at_failure=_cuda_memory(device),
        )
    finally:
        del images, timesteps, class_labels, model, loaded
        if "epsilon" in locals():
            del epsilon
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    try:
        report = run_sampling_preflight(
            args.checkpoint,
            batch_size=args.batch_size,
            prefix_budget=args.prefix_budget,
            guidance_scale=args.guidance_scale,
            guidance_rescale=args.guidance_rescale,
            cfg_batch_mode=args.cfg_batch_mode,
            weights=args.weights,
            precision=args.precision,
            warmup_forwards=args.warmup_forwards,
            measured_forwards=args.measured_forwards,
        )
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "failed",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "request": {
                "batch_size": args.batch_size,
                "prefix_budget": args.prefix_budget,
                "precision": args.precision,
                "guidance_scale": args.guidance_scale,
                "guidance_rescale": args.guidance_rescale,
                "cfg_batch_mode": args.cfg_batch_mode,
                "weights": args.weights,
                "warmup_forwards": args.warmup_forwards,
                "measured_forwards": args.measured_forwards,
            },
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json_report(output, report)
        raise
    write_json_report(output, report)
    if report["status"] != "passed":
        raise SystemExit(f"sampling preflight failed: {report['error_type']}: {report['error']}")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
