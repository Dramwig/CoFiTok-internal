from __future__ import annotations

import torch
from torch import nn


def group_norm_groups(channels: int, maximum: int = 8) -> int:
    if channels < 1:
        raise ValueError(f"channels must be positive, got {channels}")
    return next(groups for groups in range(min(maximum, channels), 0, -1) if channels % groups == 0)


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(num_groups=group_norm_groups(out_channels), num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class TimestepEmbedder(nn.Module):
    def __init__(self, channels: int, max_period: int = 10000) -> None:
        super().__init__()
        self.channels = channels
        self.max_period = max_period
        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, channels),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.channels // 2
        frequencies = torch.exp(
            -torch.log(torch.tensor(float(self.max_period), device=timesteps.device))
            * torch.arange(half, device=timesteps.device, dtype=torch.float32)
            / max(half, 1)
        )
        args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
        if embedding.shape[1] < self.channels:
            embedding = torch.nn.functional.pad(embedding, (0, self.channels - embedding.shape[1]))
        return self.proj(embedding)
