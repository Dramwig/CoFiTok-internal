from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DataConfig:
    dataset: str = "random"
    root: str = "/root/autodl-tmp/CoFiTok/datasets"
    image_size: int = 32
    channels: int = 3
    batch_size: int = 4
    num_workers: int = 0
    class_conditional: bool = False
    persistent_workers: bool = False
    prefetch_factor: int = 2
    random_horizontal_flip_prob: float = 0.0


@dataclass(frozen=True)
class DiffusionConfig:
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    prediction_target: str = "epsilon"
    schedule_type: str = "linear"
    cosine_endpoint_fraction: float = 1.0


@dataclass(frozen=True)
class ModelConfig:
    image_channels: int = 3
    image_size: int = 32
    token_count: int = 4
    token_channels: int = 32
    base_channels: int = 32
    predictor_type: str = "tiny_conv"
    predictor_depth: int = 3
    predictor_use_feedback: bool = True
    predictor_multiscale_levels: int = 2
    synthesis_mode: str = "restricted"
    synthesis_kernel_size: int = 3
    gamma_mode: str = "learned_scalar"
    token_channel_schedule: list[int] = field(default_factory=list)
    token_spatial_strides: list[int] = field(default_factory=list)
    synthesis_token_strides: list[int] = field(default_factory=list)
    synthesis_active_token_channels: list[int] = field(default_factory=list)
    deep_synthesis_hidden_channels: int = 0
    deep_synthesis_depth: int = 3
    predictor_channel_multipliers: list[int] = field(default_factory=lambda: [1, 2, 4, 4])
    predictor_num_res_blocks: int = 2
    predictor_attention_resolutions: list[int] = field(default_factory=lambda: [32, 16])
    predictor_num_heads: int = 4
    predictor_dropout: float = 0.0
    predictor_gradient_checkpointing: bool = False
    num_classes: int = 0
    class_dropout_prob: float = 0.0


@dataclass(frozen=True)
class LossConfig:
    epsilon_weight: float = 1.0
    prefix_weight: float = 0.25
    monotonic_weight: float = 0.05
    monotonic_margin: float = 0.0
    zero_token_weight: float = 0.01
    energy_budget_weight: float = 0.0
    energy_target: list[float] = field(default_factory=list)
    energy_budget_scope: str = "batch"
    residual_component_weight: float = 0.0
    component_decorrelation_weight: float = 0.0
    component_decorrelation_start_step: int = 0
    component_decorrelation_warmup_steps: int = 0
    tail_floor_weight: float = 0.0
    tail_floor_min_ratio: float = 0.0
    sampled_prefix_weight: float = 0.0
    sampled_component_weight: float = 0.0
    group_residual_weight: float = 0.0
    group_residual_size: int = 0
    epsilon_band_prefix_weight: float = 0.0
    epsilon_band_component_weight: float = 0.0
    denoise_path_prefix_weight: float = 0.0
    denoise_path_component_weight: float = 0.0
    denoise_path_energy_weight: float = 0.0
    denoise_path_energy_mode: str = "mse"
    denoise_path_energy_capacity_weight: float = 0.0
    denoise_path_energy_capacity_power: float = 0.5
    denoise_path_progress_power: float = 1.0
    denoise_path_progress_mode: str = "power"
    low_snr_high_frequency_weight: float = 0.0
    low_snr_high_frequency_power: float = 1.0
    rollout_consistency_weight: float = 0.0
    rollout_consistency_start_step: int = 0
    rollout_consistency_warmup_steps: int = 0
    rollout_consistency_timestep_delta: int = 10
    rollout_consistency_unroll_steps: int = 1
    rollout_consistency_batch_fraction: float = 0.25
    rollout_consistency_clip_x0: bool = True
    rollout_consistency_mode: str = "epsilon"
    ema_teacher_consistency_weight: float = 0.0
    ema_teacher_consistency_start_step: int = 0
    ema_teacher_consistency_warmup_steps: int = 0
    ema_teacher_consistency_batch_fraction: float = 0.0625
    tail_early_dropout_weight: float = 0.0
    tail_early_dropout_prob: float = 0.0
    tail_early_dropout_start: int = 1
    tail_replay_weight: float = 0.0
    tail_late_prefix_weight: float = 0.0
    tail_late_prefix_start: int = 0
    tail_endpoint_weight: float = 0.0
    tail_late_monotonic_weight: float = 0.0
    tail_late_monotonic_start: int = 0
    tail_late_monotonic_margin: float = 0.0


@dataclass(frozen=True)
class RuntimeConfig:
    seed: int = 13
    device: str = "cpu"
    steps: int = 1
    precision: str = "fp32"
    compile_model: bool = False
    allow_tf32: bool = True
    cudnn_benchmark: bool = False
    checkpoint_interval: int = 1000
    evaluation_interval: int = 1000
    keep_last_checkpoints: int = 3
    protected_checkpoint_steps: list[int] = field(default_factory=list)


@dataclass(frozen=True)
class OptimizationConfig:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    grad_clip_norm: float = 1.0
    log_interval: int = 10
    visualization_count: int = 8
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 0
    min_learning_rate: float = 0.0
    ema_decay: float = 0.9999
    ema_warmup_steps: int = 2000


@dataclass(frozen=True)
class ExperimentConfig:
    name: str = "smoke_random_cpu"
    data: DataConfig = field(default_factory=DataConfig)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)


def _merge_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def _dataclass_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return {
            field_name: _dataclass_to_dict(getattr(obj, field_name))
            for field_name in obj.__dataclass_fields__
        }
    return obj


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    default = _dataclass_to_dict(ExperimentConfig())
    merged = _merge_dict(default, raw)
    return ExperimentConfig(
        name=merged["name"],
        data=DataConfig(**merged["data"]),
        diffusion=DiffusionConfig(**merged["diffusion"]),
        model=ModelConfig(**merged["model"]),
        loss=LossConfig(**merged["loss"]),
        runtime=RuntimeConfig(**merged["runtime"]),
        optimization=OptimizationConfig(**merged["optimization"]),
    )


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return _build_config(raw)


def config_from_dict(raw: dict[str, Any]) -> ExperimentConfig:
    return _build_config(raw)


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return _dataclass_to_dict(config)
