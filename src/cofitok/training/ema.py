from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999, warmup_steps: int = 0) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        if warmup_steps < 0:
            raise ValueError("EMA warmup_steps must be non-negative")
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    def _effective_decay(self) -> float:
        if self.warmup_steps == 0:
            return self.decay
        ramp = min(1.0, self.num_updates / self.warmup_steps)
        return self.decay * ramp

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        decay = self._effective_decay()
        current = model.state_dict()
        if current.keys() != self.shadow.keys():
            raise ValueError("EMA model state structure changed after initialization")
        for name, value in current.items():
            shadow = self.shadow[name]
            if torch.is_floating_point(shadow):
                shadow.lerp_(value.detach(), 1.0 - decay)
            else:
                shadow.copy_(value.detach())

    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.shadow, strict=True)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "warmup_steps": self.warmup_steps,
            "num_updates": self.num_updates,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.decay = float(state["decay"])
        self.warmup_steps = int(state["warmup_steps"])
        self.num_updates = int(state["num_updates"])
        loaded_shadow = state["shadow"]
        if not isinstance(loaded_shadow, Mapping):
            raise TypeError("EMA shadow state must be a mapping")
        if loaded_shadow.keys() != self.shadow.keys():
            raise ValueError("EMA checkpoint does not match model state structure")
        self.shadow = {
            str(name): value.detach().clone()
            for name, value in loaded_shadow.items()
            if isinstance(value, torch.Tensor)
        }

