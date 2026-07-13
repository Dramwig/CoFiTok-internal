from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from cofitok.models.blocks import TimestepEmbedder, group_norm_groups


class ConditionedResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_norm = nn.GroupNorm(group_norm_groups(in_channels, maximum=32), in_channels)
        self.in_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.embedding = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embedding_channels, 2 * out_channels),
        )
        self.out_norm = nn.GroupNorm(group_norm_groups(out_channels, maximum=32), out_channels)
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, inputs: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.in_conv(F.silu(self.in_norm(inputs)))
        scale, shift = self.embedding(embedding).chunk(2, dim=1)
        hidden = self.out_norm(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return self.skip(inputs) + self.out(hidden)


class SpatialSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError(f"channels ({channels}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.norm = nn.GroupNorm(group_norm_groups(channels, maximum=32), channels)
        self.qkv = nn.Conv2d(channels, 3 * channels, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = inputs.shape
        qkv = self.qkv(self.norm(inputs)).reshape(
            batch,
            3,
            self.num_heads,
            self.head_dim,
            height * width,
        )
        query, key, value = qkv.unbind(dim=1)
        query = query.transpose(-2, -1)
        key = key.transpose(-2, -1)
        value = value.transpose(-2, -1)
        attended = F.scaled_dot_product_attention(query, key, value)
        attended = attended.transpose(-2, -1).reshape(batch, channels, height, width)
        return inputs + self.proj(attended)


class ConditionedStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        embedding_channels: int,
        num_res_blocks: int,
        dropout: float,
        use_attention: bool,
        num_heads: int,
    ) -> None:
        super().__init__()
        blocks = []
        current_channels = in_channels
        for _ in range(num_res_blocks):
            blocks.append(
                ConditionedResBlock(
                    current_channels,
                    out_channels,
                    embedding_channels,
                    dropout,
                )
            )
            current_channels = out_channels
        self.blocks = nn.ModuleList(blocks)
        self.attention = SpatialSelfAttention(out_channels, num_heads) if use_attention else nn.Identity()

    def forward(self, inputs: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        hidden = inputs
        for block in self.blocks:
            hidden = block(hidden, embedding)
        return self.attention(hidden)


class ScalableUNetTokenPredictor(nn.Module):
    """Class-conditional ADM-style U-Net for the expressive CoFiTok predictor T_k.

    Class and timestep information terminate inside this predictor. Returned token
    fields are the only values passed to the restricted synthesis operators S_k.
    """

    def __init__(
        self,
        image_channels: int,
        image_size: int,
        token_count: int,
        token_channels: int,
        base_channels: int,
        channel_multipliers: Sequence[int],
        num_res_blocks: int,
        attention_resolutions: Sequence[int],
        num_heads: int,
        dropout: float,
        use_feedback: bool,
        gradient_checkpointing: bool,
        num_classes: int,
        class_dropout_prob: float,
    ) -> None:
        super().__init__()
        if not channel_multipliers:
            raise ValueError("channel_multipliers must not be empty")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be positive")
        if not 0.0 <= class_dropout_prob <= 1.0:
            raise ValueError("class_dropout_prob must be in [0, 1]")
        self.token_count = token_count
        self.token_channels = token_channels
        self.use_feedback = use_feedback
        self.gradient_checkpointing = gradient_checkpointing
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.null_class = num_classes

        embedding_channels = 4 * base_channels
        self.input_proj = nn.Conv2d(image_channels, base_channels, kernel_size=3, padding=1)
        self.time_embed = TimestepEmbedder(embedding_channels)
        self.class_embed = (
            nn.Embedding(num_classes + 1, embedding_channels) if num_classes > 0 else None
        )

        attention_set = set(attention_resolutions)
        encoder = []
        downsample = []
        skip_channels = []
        current_channels = base_channels
        resolution = image_size
        for index, multiplier in enumerate(channel_multipliers):
            out_channels = base_channels * multiplier
            encoder.append(
                ConditionedStage(
                    current_channels,
                    out_channels,
                    embedding_channels,
                    num_res_blocks,
                    dropout,
                    resolution in attention_set,
                    num_heads,
                )
            )
            skip_channels.append(out_channels)
            current_channels = out_channels
            if index < len(channel_multipliers) - 1:
                downsample.append(
                    nn.Conv2d(current_channels, current_channels, kernel_size=3, stride=2, padding=1)
                )
                resolution = math.ceil(resolution / 2)
        self.encoder = nn.ModuleList(encoder)
        self.downsample = nn.ModuleList(downsample)

        self.middle = nn.ModuleList(
            [
                ConditionedResBlock(current_channels, current_channels, embedding_channels, dropout),
                SpatialSelfAttention(current_channels, num_heads),
                ConditionedResBlock(current_channels, current_channels, embedding_channels, dropout),
            ]
        )

        decoder = []
        decoder_resolutions = []
        for level in reversed(range(len(channel_multipliers))):
            resolution = math.ceil(image_size / (2**level))
            out_channels = base_channels * channel_multipliers[level]
            decoder.append(
                ConditionedStage(
                    current_channels + skip_channels[level],
                    out_channels,
                    embedding_channels,
                    num_res_blocks,
                    dropout,
                    resolution in attention_set,
                    num_heads,
                )
            )
            decoder_resolutions.append(resolution)
            current_channels = out_channels
        self.decoder = nn.ModuleList(decoder)
        self.decoder_resolutions = decoder_resolutions
        self.output_norm = nn.GroupNorm(group_norm_groups(current_channels, maximum=32), current_channels)
        self.token_heads = nn.ModuleList(
            [nn.Conv2d(current_channels, token_channels, kernel_size=1) for _ in range(token_count)]
        )
        for head in self.token_heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.feedback = (
            nn.ModuleList(
                [nn.Conv2d(token_channels, current_channels, kernel_size=1) for _ in range(token_count - 1)]
            )
            if use_feedback
            else nn.ModuleList()
        )

    def _run_conditioned(
        self,
        module: nn.Module,
        hidden: torch.Tensor,
        embedding: torch.Tensor,
    ) -> torch.Tensor:
        if self.gradient_checkpointing and self.training:
            return checkpoint(module, hidden, embedding, use_reentrant=False)
        return module(hidden, embedding)

    def _conditioning(
        self,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None,
        force_unconditional: bool,
    ) -> torch.Tensor:
        embedding = self.time_embed(timesteps)
        if self.class_embed is None:
            if class_labels is not None:
                raise ValueError("class_labels were provided to an unconditional predictor")
            return embedding
        if class_labels is None:
            if not force_unconditional:
                raise ValueError("class_labels are required for a class-conditional predictor")
            labels = torch.full_like(timesteps, self.null_class)
        else:
            labels = class_labels.to(device=timesteps.device, dtype=torch.long)
            if labels.shape != timesteps.shape:
                raise ValueError("class_labels must have shape [batch]")
            if force_unconditional:
                labels = torch.full_like(labels, self.null_class)
            elif self.training and self.class_dropout_prob > 0.0:
                drop = torch.rand(labels.shape, device=labels.device) < self.class_dropout_prob
                labels = torch.where(drop, self.null_class, labels)
        return embedding + self.class_embed(labels)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        class_labels: torch.Tensor | None = None,
        force_unconditional: bool = False,
    ) -> list[torch.Tensor]:
        embedding = self._conditioning(timesteps, class_labels, force_unconditional)
        hidden = self.input_proj(noisy_images)
        skips = []
        for index, stage in enumerate(self.encoder):
            hidden = self._run_conditioned(stage, hidden, embedding)
            skips.append(hidden)
            if index < len(self.downsample):
                hidden = self.downsample[index](hidden)

        hidden = self._run_conditioned(self.middle[0], hidden, embedding)
        hidden = self.middle[1](hidden)
        hidden = self._run_conditioned(self.middle[2], hidden, embedding)

        for stage, skip in zip(self.decoder, reversed(skips)):
            if hidden.shape[-2:] != skip.shape[-2:]:
                hidden = F.interpolate(hidden, size=skip.shape[-2:], mode="nearest")
            hidden = self._run_conditioned(stage, torch.cat([hidden, skip], dim=1), embedding)

        state = F.silu(self.output_norm(hidden))
        tokens = []
        for index, head in enumerate(self.token_heads):
            token = head(state)
            tokens.append(token)
            if self.use_feedback and index < len(self.feedback):
                state = state + self.feedback[index](token)
        return tokens
