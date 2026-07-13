from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def _validate_token_layout(
    token_channels: int,
    token_stride: int,
    active_token_channels: int | None,
) -> int:
    if token_stride < 1:
        raise ValueError(f"token_stride must be >= 1, got {token_stride}")
    if active_token_channels is None:
        active_token_channels = token_channels
    if not 1 <= active_token_channels <= token_channels:
        raise ValueError(
            f"active_token_channels must be in [1, {token_channels}], got {active_token_channels}"
        )
    return active_token_channels


def _make_token_channel_mask(token_channels: int, active_token_channels: int) -> torch.Tensor:
    mask = torch.zeros(1, token_channels, 1, 1)
    mask[:, :active_token_channels] = 1.0
    return mask


def _normalize_per_token_values(
    values: list[int] | None,
    token_count: int,
    default: int,
    name: str,
) -> list[int]:
    if values is None or not values:
        values = [default for _ in range(token_count)]
    if len(values) != token_count:
        raise ValueError(f"Expected {token_count} {name}, got {len(values)}")
    return values


def _make_gamma(gamma_mode: str) -> nn.Parameter | torch.Tensor:
    if gamma_mode == "learned_scalar":
        return nn.Parameter(torch.ones(1))
    if gamma_mode == "fixed_one":
        return torch.ones(1)
    raise ValueError(f"Unknown gamma_mode: {gamma_mode}")


class RestrictedSynthesis(nn.Module):
    """Maps one denoising token to one dense noise component.

    This module is deliberately condition-free: it receives only the current
    token. All convolutions are bias-free, so zero tokens map exactly to zero.
    """

    def __init__(
        self,
        token_channels: int,
        image_channels: int,
        kernel_size: int = 3,
        gamma_mode: str = "learned_scalar",
        token_stride: int = 1,
        active_token_channels: int | None = None,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        active_token_channels = _validate_token_layout(token_channels, token_stride, active_token_channels)
        self.token_stride = token_stride
        self.active_token_channels = active_token_channels
        self.output_size = output_size
        self.register_buffer(
            "token_channel_mask",
            _make_token_channel_mask(token_channels, active_token_channels),
            persistent=False,
        )
        padding = kernel_size // 2
        self.proj = nn.Conv2d(token_channels, image_channels, kernel_size=1, bias=False)
        self.local = nn.Conv2d(
            image_channels,
            image_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=1,
            bias=False,
        )
        gamma = _make_gamma(gamma_mode)
        if isinstance(gamma, nn.Parameter):
            self.gamma = gamma
        else:
            self.register_buffer("gamma", gamma, persistent=False)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        token = token * self.token_channel_mask
        if self.token_stride > 1:
            original_size = token.shape[-2:]
            token = F.avg_pool2d(token, kernel_size=self.token_stride, stride=self.token_stride, ceil_mode=True)
            token = F.interpolate(token, size=original_size, mode="bilinear", align_corners=False)
        projected = self.proj(token)
        if self.output_size is not None and projected.shape[-2:] != (self.output_size, self.output_size):
            projected = F.interpolate(
                projected,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.gamma.view(1, 1, 1, 1) * self.local(projected)


class DeepSynthesis(nn.Module):
    """Ablation-only stronger synthesis operator.

    It still receives only the current token, but it intentionally uses biased
    nonlinear convolutions and therefore is not a valid default CoFiTok `S_k`.
    """

    def __init__(
        self,
        token_channels: int,
        image_channels: int,
        kernel_size: int = 3,
        gamma_mode: str = "learned_scalar",
        token_stride: int = 1,
        active_token_channels: int | None = None,
        hidden_channels: int = 0,
        depth: int = 3,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"deep synthesis depth must be >= 2, got {depth}")
        active_token_channels = _validate_token_layout(token_channels, token_stride, active_token_channels)
        self.token_stride = token_stride
        self.active_token_channels = active_token_channels
        self.output_size = output_size
        self.register_buffer(
            "token_channel_mask",
            _make_token_channel_mask(token_channels, active_token_channels),
            persistent=False,
        )
        hidden_channels = hidden_channels or max(token_channels, image_channels * 8)
        padding = kernel_size // 2
        layers: list[nn.Module] = [
            nn.Conv2d(token_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.SiLU(),
        ]
        for _ in range(depth - 2):
            layers.extend(
                [
                    nn.Conv2d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding, bias=True),
                    nn.SiLU(),
                ]
            )
        layers.append(
            nn.Conv2d(hidden_channels, image_channels, kernel_size=kernel_size, padding=padding, bias=True)
        )
        self.net = nn.Sequential(*layers)
        gamma = _make_gamma(gamma_mode)
        if isinstance(gamma, nn.Parameter):
            self.gamma = gamma
        else:
            self.register_buffer("gamma", gamma, persistent=False)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        token = token * self.token_channel_mask
        if self.token_stride > 1:
            original_size = token.shape[-2:]
            token = F.avg_pool2d(token, kernel_size=self.token_stride, stride=self.token_stride, ceil_mode=True)
            token = F.interpolate(token, size=original_size, mode="bilinear", align_corners=False)
        component = self.net(token)
        if self.output_size is not None and component.shape[-2:] != (self.output_size, self.output_size):
            component = F.interpolate(
                component,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return self.gamma.view(1, 1, 1, 1) * component


class RestrictedSynthesisBank(nn.Module):
    def __init__(
        self,
        token_count: int,
        token_channels: int,
        image_channels: int,
        kernel_size: int,
        gamma_mode: str,
        token_strides: list[int] | None = None,
        active_token_channels: list[int] | None = None,
        token_channel_schedule: list[int] | None = None,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        token_strides = _normalize_per_token_values(token_strides, token_count, 1, "token strides")
        active_token_channels = _normalize_per_token_values(
            active_token_channels or token_channel_schedule,
            token_count,
            token_channels,
            "active channel values",
        )
        token_channels_per_token = _normalize_per_token_values(
            token_channel_schedule,
            token_count,
            token_channels,
            "token channel values",
        )
        if token_channel_schedule and active_token_channels != token_channels_per_token:
            raise ValueError(
                "active_token_channels must be omitted or equal token_channel_schedule "
                "for variable-channel tokens"
            )
        self.synthesizers = nn.ModuleList(
            [
                RestrictedSynthesis(
                    token_channels=channels,
                    image_channels=image_channels,
                    kernel_size=kernel_size,
                    gamma_mode=gamma_mode,
                    token_stride=token_stride,
                    active_token_channels=active_channels,
                    output_size=output_size,
                )
                for token_stride, active_channels, channels in zip(
                    token_strides,
                    active_token_channels,
                    token_channels_per_token,
                )
            ]
        )

    def forward(self, tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(tokens) != len(self.synthesizers):
            raise ValueError(f"Expected {len(self.synthesizers)} tokens, got {len(tokens)}")
        return [synthesizer(token) for synthesizer, token in zip(self.synthesizers, tokens)]

    def zero_components_like(self, tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        return [synthesizer(torch.zeros_like(token)) for synthesizer, token in zip(self.synthesizers, tokens)]


class DeepSynthesisBank(nn.Module):
    def __init__(
        self,
        token_count: int,
        token_channels: int,
        image_channels: int,
        kernel_size: int,
        gamma_mode: str,
        token_strides: list[int] | None = None,
        active_token_channels: list[int] | None = None,
        hidden_channels: int = 0,
        depth: int = 3,
        token_channel_schedule: list[int] | None = None,
        output_size: int | None = None,
    ) -> None:
        super().__init__()
        token_strides = _normalize_per_token_values(token_strides, token_count, 1, "token strides")
        active_token_channels = _normalize_per_token_values(
            active_token_channels or token_channel_schedule,
            token_count,
            token_channels,
            "active channel values",
        )
        token_channels_per_token = _normalize_per_token_values(
            token_channel_schedule,
            token_count,
            token_channels,
            "token channel values",
        )
        if token_channel_schedule and active_token_channels != token_channels_per_token:
            raise ValueError(
                "active_token_channels must be omitted or equal token_channel_schedule "
                "for variable-channel tokens"
            )
        self.synthesizers = nn.ModuleList(
            [
                DeepSynthesis(
                    token_channels=channels,
                    image_channels=image_channels,
                    kernel_size=kernel_size,
                    gamma_mode=gamma_mode,
                    token_stride=token_stride,
                    active_token_channels=active_channels,
                    hidden_channels=hidden_channels,
                    depth=depth,
                    output_size=output_size,
                )
                for token_stride, active_channels, channels in zip(
                    token_strides,
                    active_token_channels,
                    token_channels_per_token,
                )
            ]
        )

    def forward(self, tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(tokens) != len(self.synthesizers):
            raise ValueError(f"Expected {len(self.synthesizers)} tokens, got {len(tokens)}")
        return [synthesizer(token) for synthesizer, token in zip(self.synthesizers, tokens)]

    def zero_components_like(self, tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        return [synthesizer(torch.zeros_like(token)) for synthesizer, token in zip(self.synthesizers, tokens)]


class DenseIdentitySynthesisBank(nn.Module):
    """Internal baseline adapter for a direct full-resolution epsilon head.

    The predictor emits one 3-channel dense field. This adapter only preserves
    the common output interface; it is not a compressed token synthesis operator.
    """

    def __init__(self, token_count: int, token_channels: int, image_channels: int) -> None:
        super().__init__()
        if token_count != 1:
            raise ValueError(f"dense identity baseline requires token_count=1, got {token_count}")
        if token_channels != image_channels:
            raise ValueError(
                "dense identity baseline requires token_channels == image_channels, "
                f"got {token_channels} and {image_channels}"
            )

    def forward(self, tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(tokens) != 1:
            raise ValueError(f"dense identity baseline expects one dense field, got {len(tokens)}")
        return tokens

    def zero_components_like(self, tokens: list[torch.Tensor]) -> list[torch.Tensor]:
        if len(tokens) != 1:
            raise ValueError(f"dense identity baseline expects one dense field, got {len(tokens)}")
        return [torch.zeros_like(tokens[0])]


def build_synthesis_bank(
    synthesis_mode: str,
    token_count: int,
    token_channels: int,
    image_channels: int,
    kernel_size: int,
    gamma_mode: str,
    token_strides: list[int] | None = None,
    active_token_channels: list[int] | None = None,
    deep_hidden_channels: int = 0,
    deep_depth: int = 3,
    token_channel_schedule: list[int] | None = None,
    output_size: int | None = None,
) -> nn.Module:
    if synthesis_mode in {"dense", "dense_identity", "monolithic_dense"}:
        return DenseIdentitySynthesisBank(
            token_count=token_count,
            token_channels=token_channels,
            image_channels=image_channels,
        )
    if synthesis_mode == "restricted":
        return RestrictedSynthesisBank(
            token_count=token_count,
            token_channels=token_channels,
            image_channels=image_channels,
            kernel_size=kernel_size,
            gamma_mode=gamma_mode,
            token_strides=token_strides,
            active_token_channels=active_token_channels,
            token_channel_schedule=token_channel_schedule,
            output_size=output_size,
        )
    if synthesis_mode in {"deep", "deep_decoder", "unrestricted"}:
        return DeepSynthesisBank(
            token_count=token_count,
            token_channels=token_channels,
            image_channels=image_channels,
            kernel_size=kernel_size,
            gamma_mode=gamma_mode,
            token_strides=token_strides,
            active_token_channels=active_token_channels,
            hidden_channels=deep_hidden_channels,
            depth=deep_depth,
            token_channel_schedule=token_channel_schedule,
            output_size=output_size,
        )
    raise ValueError(f"Unknown synthesis_mode: {synthesis_mode}")
