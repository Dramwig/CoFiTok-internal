from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from cofitok.diffusion import DiffusionSchedule, ddim_sample, select_sampling_timesteps
from cofitok.generation.runtime import LoadedGenerationModel, load_generation_model
from cofitok.training.runtime import autocast_context


@dataclass(frozen=True)
class GenerationRequest:
    seeds: tuple[int, ...]
    class_labels: tuple[int, ...] | None = None
    sample_steps: int = 250
    prefix_budget: int = 0
    guidance_scale: float = 1.5
    guidance_rescale: float = 0.0
    cfg_batch_mode: str = "batched"
    eta: float = 0.0
    clip_x0: bool = True
    precision: str = "bf16"

    def __post_init__(self) -> None:
        if not self.seeds:
            raise ValueError("generation request requires at least one seed")
        if any(
            not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed >= 2**63
            for seed in self.seeds
        ):
            raise ValueError("generation seeds must be in [0, 2^63)")
        if self.class_labels is not None and len(self.class_labels) != len(self.seeds):
            raise ValueError("class labels must match the seed count")
        if self.class_labels is not None and any(
            not isinstance(label, int) or isinstance(label, bool)
            for label in self.class_labels
        ):
            raise ValueError("class labels must be integers")
        if self.sample_steps < 1:
            raise ValueError("sample_steps must be positive")
        if self.prefix_budget < 0:
            raise ValueError("prefix_budget must be non-negative")
        if (
            not math.isfinite(self.guidance_scale)
            or not math.isfinite(self.guidance_rescale)
            or self.guidance_scale < 0.0
            or not 0.0 <= self.guidance_rescale <= 1.0
        ):
            raise ValueError("guidance settings are invalid")
        if self.cfg_batch_mode not in {"batched", "sequential"}:
            raise ValueError("cfg_batch_mode must be batched or sequential")
        if not math.isfinite(self.eta) or self.eta < 0.0:
            raise ValueError("eta must be non-negative")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16, or fp16")


@dataclass(frozen=True)
class GenerationResult:
    images: torch.Tensor
    metadata: dict[str, Any]


class GenerationSession:
    """Reusable, provenance-carrying inference session for one checkpoint."""

    def __init__(self, loaded: LoadedGenerationModel) -> None:
        self.loaded = loaded
        self.schedule = DiffusionSchedule(loaded.config.diffusion, device=loaded.device)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        weights: str = "ema",
    ) -> GenerationSession:
        return cls(load_generation_model(checkpoint, weights=weights))

    @property
    def device(self) -> torch.device:
        return self.loaded.device

    @property
    def token_count(self) -> int:
        return self.loaded.config.model.token_count

    @property
    def num_classes(self) -> int:
        return self.loaded.config.model.num_classes

    def _validated_labels(self, request: GenerationRequest) -> torch.Tensor | None:
        labels = request.class_labels
        if self.num_classes > 0:
            if labels is None:
                raise ValueError("class-conditional checkpoint requires class labels")
            if any(label < 0 or label >= self.num_classes for label in labels):
                raise ValueError("class label is outside the checkpoint class range")
            return torch.tensor(labels, device=self.device, dtype=torch.long)
        if labels is not None:
            raise ValueError("unconditional checkpoint does not accept class labels")
        if request.guidance_scale != 1.0:
            raise ValueError("unconditional checkpoint requires guidance_scale=1")
        return None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        budget = request.prefix_budget or self.token_count
        if not 1 <= budget <= self.token_count:
            raise ValueError("prefix_budget is outside the checkpoint token range")
        labels = self._validated_labels(request)
        generators = [
            torch.Generator(device=self.device).manual_seed(seed)
            for seed in request.seeds
        ]
        config = self.loaded.config
        shape = (
            len(request.seeds),
            config.model.image_channels,
            config.model.image_size,
            config.model.image_size,
        )
        with torch.inference_mode(), autocast_context(self.device, request.precision):
            images = ddim_sample(
                self.loaded.model,
                self.schedule,
                shape,
                sample_steps=request.sample_steps,
                prefix_budget=budget,
                eta=request.eta,
                clip_x0=request.clip_x0,
                device=self.device,
                sample_generators=generators,
                class_labels=labels,
                guidance_scale=request.guidance_scale,
                guidance_rescale=request.guidance_rescale,
                cfg_batch_mode=request.cfg_batch_mode,
            )
        if tuple(images.shape) != shape or not bool(torch.isfinite(images).all().item()):
            raise RuntimeError("generation session produced invalid samples")
        metadata = {
            "schema_version": 1,
            "checkpoint": self.loaded.checkpoint_path.as_posix(),
            "checkpoint_sha256": self.loaded.checkpoint_sha256,
            "checkpoint_integrity_manifest": (
                self.loaded.checkpoint_integrity_manifest.as_posix()
            ),
            "checkpoint_step": self.loaded.checkpoint_step,
            "weights": self.loaded.weights,
            "artifact_type": self.loaded.artifact_type,
            "source_checkpoint_sha256": self.loaded.source_checkpoint_sha256,
            "source_runtime_environment_sha256": (
                self.loaded.source_runtime_environment_sha256
            ),
            "source_git": self.loaded.source_git_provenance,
            "training_authorization": self.loaded.training_authorization,
            "release_authorization": self.loaded.release_authorization,
            "device": str(self.device),
            "request": {
                "seeds": list(request.seeds),
                "class_labels": list(request.class_labels) if request.class_labels else None,
                "sample_steps": request.sample_steps,
                "actual_timesteps": select_sampling_timesteps(
                    self.schedule.num_train_timesteps,
                    request.sample_steps,
                ),
                "prefix_budget": budget,
                "guidance_scale": request.guidance_scale,
                "guidance_rescale": request.guidance_rescale,
                "cfg_batch_mode": request.cfg_batch_mode,
                "eta": request.eta,
                "clip_x0": request.clip_x0,
                "precision": request.precision,
                "image_shape": list(shape[1:]),
            },
        }
        return GenerationResult(images=images.detach().cpu(), metadata=metadata)
