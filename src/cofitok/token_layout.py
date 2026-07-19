from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class TokenLayout:
    image_channels: int
    image_size: int
    channels: tuple[int, ...]
    spatial_strides: tuple[int, ...]
    spatial_sizes: tuple[int, ...]
    scalar_counts: tuple[int, ...]
    dense_scalar_count: int

    @property
    def total_scalar_count(self) -> int:
        return sum(self.scalar_counts)

    @property
    def per_token_compression(self) -> tuple[float, ...]:
        return tuple(self.dense_scalar_count / count for count in self.scalar_counts)

    @property
    def aggregate_compression(self) -> float:
        return self.dense_scalar_count / self.total_scalar_count

    @property
    def full_resolution_channels(self) -> int:
        return sum(
            channel
            for channel, stride in zip(self.channels, self.spatial_strides)
            if stride == 1
        )


def resolve_token_layout(
    *,
    image_size: int,
    image_channels: int,
    token_count: int,
    token_channels: int,
    token_channel_schedule: Sequence[int] | None = None,
    token_spatial_strides: Sequence[int] | None = None,
) -> TokenLayout:
    if image_size < 1 or image_channels < 1 or token_count < 1 or token_channels < 1:
        raise ValueError("image size/channels and token count/channels must be positive")

    channels = tuple(token_channel_schedule or [token_channels] * token_count)
    strides = tuple(token_spatial_strides or [1] * token_count)
    if len(channels) != token_count:
        raise ValueError(f"Expected {token_count} token channel values, got {len(channels)}")
    if len(strides) != token_count:
        raise ValueError(f"Expected {token_count} token spatial strides, got {len(strides)}")
    if any(channel < 1 or channel > token_channels for channel in channels):
        raise ValueError(
            f"token channel schedule values must be in [1, {token_channels}], got {list(channels)}"
        )
    if any(stride < 1 or image_size % stride != 0 for stride in strides):
        raise ValueError(
            f"token spatial strides must be positive divisors of image_size={image_size}, "
            f"got {list(strides)}"
        )

    spatial_sizes = tuple(image_size // stride for stride in strides)
    scalar_counts = tuple(
        channel * spatial_size * spatial_size
        for channel, spatial_size in zip(channels, spatial_sizes)
    )
    return TokenLayout(
        image_channels=image_channels,
        image_size=image_size,
        channels=channels,
        spatial_strides=strides,
        spatial_sizes=spatial_sizes,
        scalar_counts=scalar_counts,
        dense_scalar_count=image_channels * image_size * image_size,
    )


def compressed_token_layout_issues(layout: TokenLayout) -> list[str]:
    issues = []
    for index, scalar_count in enumerate(layout.scalar_counts, start=1):
        if scalar_count >= layout.dense_scalar_count:
            issues.append(
                f"token {index} has {scalar_count} scalars, which is not smaller than "
                f"the dense field ({layout.dense_scalar_count})"
            )
    if any(
        later < earlier
        for earlier, later in zip(layout.scalar_counts, layout.scalar_counts[1:])
    ):
        issues.append(
            f"token scalar capacities must be coarse-to-fine nondecreasing, got "
            f"{list(layout.scalar_counts)}"
        )
    if layout.full_resolution_channels < layout.image_channels:
        issues.append(
            "full-resolution token channels must span the dense output channels: "
            f"got {layout.full_resolution_channels}, need at least {layout.image_channels}"
        )
    return issues


def token_layout_summary(layout: TokenLayout) -> dict[str, object]:
    return {
        "channels": list(layout.channels),
        "spatial_strides": list(layout.spatial_strides),
        "spatial_sizes": list(layout.spatial_sizes),
        "scalar_counts": list(layout.scalar_counts),
        "dense_scalar_count": layout.dense_scalar_count,
        "total_scalar_count": layout.total_scalar_count,
        "per_token_compression": list(layout.per_token_compression),
        "aggregate_compression": layout.aggregate_compression,
        "aggregate_token_to_dense_ratio": (
            layout.total_scalar_count / layout.dense_scalar_count
        ),
        "full_resolution_channels": layout.full_resolution_channels,
        "required_full_resolution_channels": layout.image_channels,
        "issues": compressed_token_layout_issues(layout),
    }
