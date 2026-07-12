from __future__ import annotations

import torch
from torch import nn

from cofitok.models.blocks import ConvNormAct, TimestepEmbedder, group_norm_groups
from cofitok.models.scalable_unet import ScalableUNetTokenPredictor


class TinyTokenPredictor(nn.Module):
    """Predicts ordered dense token fields from a noisy image and previous tokens.

    The predictor is intentionally allowed to be expressive. The restricted
    information bottleneck is enforced by the synthesis bank, not by this module.
    """

    def __init__(
        self,
        image_channels: int,
        token_count: int,
        token_channels: int,
        base_channels: int,
        depth: int,
        use_feedback: bool = True,
    ) -> None:
        super().__init__()
        self.token_count = token_count
        self.token_channels = token_channels
        self.use_feedback = use_feedback
        self.input_proj = ConvNormAct(image_channels, base_channels)
        self.time_embed = TimestepEmbedder(base_channels)
        blocks = []
        for _ in range(depth):
            blocks.append(ConvNormAct(base_channels, base_channels))
        self.blocks = nn.ModuleList(blocks)
        self.token_heads = nn.ModuleList(
            [nn.Conv2d(base_channels, token_channels, kernel_size=1) for _ in range(token_count)]
        )
        if use_feedback:
            self.feedback = nn.ModuleList(
                [nn.Conv2d(token_channels, base_channels, kernel_size=1) for _ in range(token_count - 1)]
            )
        else:
            self.feedback = nn.ModuleList()

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        force_unconditional: bool = False,
    ) -> list[torch.Tensor]:
        del force_unconditional
        if class_labels is not None:
            raise ValueError("TinyTokenPredictor does not support class conditioning")
        hidden = self.input_proj(noisy_images)
        time_embedding = self.time_embed(timesteps).view(timesteps.shape[0], -1, 1, 1)
        hidden = hidden + time_embedding
        for block in self.blocks:
            hidden = block(hidden)

        tokens = []
        state = hidden
        for index, head in enumerate(self.token_heads):
            token = head(state)
            tokens.append(token)
            if self.use_feedback and index < len(self.feedback):
                state = state + self.feedback[index](token)
        return tokens


class MultiScaleTokenPredictor(nn.Module):
    """Small U-Net-like next-token predictor.

    This strengthens only `T_k`: it can use the noisy image, timestep, and
    previous token feedback. The restricted `S_k` contract remains enforced by
    the synthesis bank.
    """

    def __init__(
        self,
        image_channels: int,
        token_count: int,
        token_channels: int,
        base_channels: int,
        depth: int,
        use_feedback: bool = True,
        levels: int = 2,
    ) -> None:
        super().__init__()
        if levels < 1:
            raise ValueError(f"levels must be >= 1, got {levels}")
        self.token_count = token_count
        self.token_channels = token_channels
        self.use_feedback = use_feedback
        self.levels = levels
        self.input_proj = ConvNormAct(image_channels, base_channels)
        self.time_embed = TimestepEmbedder(base_channels)

        down_blocks = []
        channels = base_channels
        for _ in range(levels):
            next_channels = channels * 2
            down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, next_channels, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(num_groups=group_norm_groups(next_channels), num_channels=next_channels),
                    nn.SiLU(),
                    ConvNormAct(next_channels, next_channels),
                )
            )
            channels = next_channels
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bottleneck = nn.ModuleList([ConvNormAct(channels, channels) for _ in range(depth)])

        up_blocks = []
        for _ in range(levels):
            skip_channels = channels // 2
            up_blocks.append(ConvNormAct(channels + skip_channels, skip_channels))
            channels = skip_channels
        self.up_blocks = nn.ModuleList(up_blocks)
        self.output_blocks = nn.ModuleList([ConvNormAct(base_channels, base_channels) for _ in range(max(depth, 1))])
        self.token_heads = nn.ModuleList(
            [nn.Conv2d(base_channels, token_channels, kernel_size=1) for _ in range(token_count)]
        )
        if use_feedback:
            self.feedback = nn.ModuleList(
                [nn.Conv2d(token_channels, base_channels, kernel_size=1) for _ in range(token_count - 1)]
            )
        else:
            self.feedback = nn.ModuleList()

    def _encode(self, noisy_images: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        hidden = self.input_proj(noisy_images)
        time_embedding = self.time_embed(timesteps).view(timesteps.shape[0], -1, 1, 1)
        hidden = hidden + time_embedding
        skips = [hidden]
        state = hidden
        for down_block in self.down_blocks:
            state = down_block(state)
            skips.append(state)
        for block in self.bottleneck:
            state = block(state)
        for up_block, skip in zip(self.up_blocks, reversed(skips[:-1])):
            state = torch.nn.functional.interpolate(
                state,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            state = up_block(torch.cat([state, skip], dim=1))
        for block in self.output_blocks:
            state = block(state)
        return state

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        force_unconditional: bool = False,
    ) -> list[torch.Tensor]:
        del force_unconditional
        if class_labels is not None:
            raise ValueError("MultiScaleTokenPredictor does not support class conditioning")
        state = self._encode(noisy_images, timesteps)
        tokens = []
        for index, head in enumerate(self.token_heads):
            token = head(state)
            tokens.append(token)
            if self.use_feedback and index < len(self.feedback):
                state = state + self.feedback[index](token)
        return tokens


def build_token_predictor(
    predictor_type: str,
    image_channels: int,
    token_count: int,
    token_channels: int,
    base_channels: int,
    depth: int,
    use_feedback: bool = True,
    multiscale_levels: int = 2,
    image_size: int = 32,
    channel_multipliers: list[int] | None = None,
    num_res_blocks: int = 2,
    attention_resolutions: list[int] | None = None,
    num_heads: int = 4,
    dropout: float = 0.0,
    gradient_checkpointing: bool = False,
    num_classes: int = 0,
    class_dropout_prob: float = 0.0,
) -> nn.Module:
    if predictor_type in {"tiny", "tiny_conv"}:
        return TinyTokenPredictor(
            image_channels=image_channels,
            token_count=token_count,
            token_channels=token_channels,
            base_channels=base_channels,
            depth=depth,
            use_feedback=use_feedback,
        )
    if predictor_type in {"multiscale", "multiscale_unet", "small_unet"}:
        return MultiScaleTokenPredictor(
            image_channels=image_channels,
            token_count=token_count,
            token_channels=token_channels,
            base_channels=base_channels,
            depth=depth,
            use_feedback=use_feedback,
            levels=multiscale_levels,
        )
    if predictor_type in {"scalable_unet", "adm_unet", "generation_unet"}:
        return ScalableUNetTokenPredictor(
            image_channels=image_channels,
            image_size=image_size,
            token_count=token_count,
            token_channels=token_channels,
            base_channels=base_channels,
            channel_multipliers=channel_multipliers or [1, 2, 4, 4],
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions or [],
            num_heads=num_heads,
            dropout=dropout,
            use_feedback=use_feedback,
            gradient_checkpointing=gradient_checkpointing,
            num_classes=num_classes,
            class_dropout_prob=class_dropout_prob,
        )
    raise ValueError(f"Unknown predictor_type: {predictor_type}")
