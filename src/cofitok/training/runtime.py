from __future__ import annotations

import math
from contextlib import nullcontext

import torch


def autocast_context(device: torch.device, precision: str):
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    if precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    if precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    raise ValueError(f"Unknown precision: {precision}")


def build_grad_scaler(device: torch.device, precision: str) -> torch.amp.GradScaler | None:
    if device.type == "cuda" and precision == "fp16":
        return torch.amp.GradScaler("cuda")
    return None


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_steps: int,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be in [0, total_steps)")
    base_learning_rate = max(group["lr"] for group in optimizer.param_groups)
    if not 0.0 <= min_learning_rate <= base_learning_rate:
        raise ValueError("min_learning_rate must be in [0, base_learning_rate]")
    minimum_ratio = min_learning_rate / base_learning_rate

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(step, 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)

