from __future__ import annotations

from collections.abc import Iterator, Mapping, Sized

import torch
from torch.utils.data import Sampler


class StatefulRandomSampler(Sampler[int]):
    """Random sampler whose consumed position can be checkpointed safely.

    DataLoader workers may prefetch indices ahead of the training loop. The
    sampler therefore tracks issued and consumed positions separately and only
    serializes the consumed position.
    """

    def __init__(self, data_source: Sized, seed: int) -> None:
        self.data_source = data_source
        self.generator = torch.Generator().manual_seed(seed)
        self.epoch = 0
        self.order = torch.randperm(len(data_source), generator=self.generator).tolist()
        self.issued_position = 0
        self.consumed_position = 0

    def __len__(self) -> int:
        return len(self.order) - self.consumed_position

    def __iter__(self) -> Iterator[int]:
        while self.issued_position < len(self.order):
            index = self.order[self.issued_position]
            self.issued_position += 1
            yield index

    def mark_consumed(self, count: int) -> None:
        if count < 0:
            raise ValueError("consumed count must be non-negative")
        self.consumed_position = min(self.consumed_position + count, len(self.order))

    def start_next_epoch(self) -> None:
        self.epoch += 1
        self.order = torch.randperm(len(self.data_source), generator=self.generator).tolist()
        self.issued_position = 0
        self.consumed_position = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "order": torch.tensor(self.order, dtype=torch.int32),
            "position": self.consumed_position,
            "generator_state": self.generator.get_state(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        order = state["order"]
        if not isinstance(order, torch.Tensor) or order.numel() != len(self.data_source):
            raise ValueError("sampler checkpoint does not match dataset length")
        position = int(state["position"])
        if not 0 <= position <= len(order):
            raise ValueError("sampler checkpoint position is invalid")
        self.epoch = int(state["epoch"])
        self.order = order.to(dtype=torch.int64, device="cpu").tolist()
        self.consumed_position = position
        self.issued_position = position
        generator_state = state["generator_state"]
        if not isinstance(generator_state, torch.Tensor):
            raise TypeError("sampler generator state must be a tensor")
        self.generator.set_state(generator_state)
