from __future__ import annotations

import math
from typing import Any


def training_cost_summary(report: dict[str, Any]) -> dict[str, Any]:
    config = report["config"]
    target_steps = int(report["target_steps"])
    micro_batch = int(config["data"]["batch_size"])
    accumulation = int(config["optimization"]["gradient_accumulation_steps"])
    effective_batch = micro_batch * accumulation
    expected_samples = target_steps * effective_batch
    samples_seen = int(report.get("final_metrics", {}).get("samples_seen", -1))
    elapsed_seconds = float(report.get("elapsed_seconds", math.nan))
    peak_vram_bytes = int(report.get("peak_vram_bytes", -1))
    cuda_run = config["runtime"].get("device") == "cuda"
    valid = (
        samples_seen == expected_samples
        and math.isfinite(elapsed_seconds)
        and elapsed_seconds > 0.0
        and peak_vram_bytes >= 0
        and (not cuda_run or peak_vram_bytes > 0)
    )
    return {
        "valid": valid,
        "target_steps": target_steps,
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch,
        "expected_samples_seen": expected_samples,
        "samples_seen": samples_seen,
        "elapsed_seconds": elapsed_seconds,
        "images_per_second": (
            samples_seen / elapsed_seconds
            if samples_seen >= 0 and math.isfinite(elapsed_seconds) and elapsed_seconds > 0.0
            else None
        ),
        "peak_vram_bytes": peak_vram_bytes,
    }
