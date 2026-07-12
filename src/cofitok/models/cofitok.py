from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from cofitok.configs import ModelConfig
from cofitok.models.predictors import build_token_predictor
from cofitok.models.synthesis import build_synthesis_bank


@dataclass
class CoFiTokOutput:
    tokens: list[torch.Tensor]
    components: list[torch.Tensor]
    prefix_epsilons: list[torch.Tensor]
    epsilon: torch.Tensor


class CoFiTokTiny(nn.Module):
    """Minimal ordered denoising-token model for MVP smoke experiments."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.predictor = build_token_predictor(
            predictor_type=config.predictor_type,
            image_channels=config.image_channels,
            token_count=config.token_count,
            token_channels=config.token_channels,
            base_channels=config.base_channels,
            depth=config.predictor_depth,
            use_feedback=config.predictor_use_feedback,
            multiscale_levels=config.predictor_multiscale_levels,
        )
        self.synthesis = build_synthesis_bank(
            synthesis_mode=config.synthesis_mode,
            token_count=config.token_count,
            token_channels=config.token_channels,
            image_channels=config.image_channels,
            kernel_size=config.synthesis_kernel_size,
            gamma_mode=config.gamma_mode,
            token_strides=config.synthesis_token_strides,
            active_token_channels=config.synthesis_active_token_channels,
            deep_hidden_channels=config.deep_synthesis_hidden_channels,
            deep_depth=config.deep_synthesis_depth,
        )

    def forward(self, noisy_images: torch.Tensor, timesteps: torch.Tensor) -> CoFiTokOutput:
        tokens = self.predictor(noisy_images, timesteps)
        components = self.synthesis(tokens)
        prefix_epsilons = []
        running = torch.zeros_like(components[0])
        for component in components:
            running = running + component
            prefix_epsilons.append(running)
        return CoFiTokOutput(
            tokens=tokens,
            components=components,
            prefix_epsilons=prefix_epsilons,
            epsilon=prefix_epsilons[-1],
        )
