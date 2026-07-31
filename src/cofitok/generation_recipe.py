from __future__ import annotations

from typing import Any

from cofitok.generation_pair import generation_pair_contract
from cofitok.token_layout import resolve_token_layout, token_layout_summary


GENERATION_TRAINING_RECIPE_SCHEMA = "cofitok_generation_training_recipe_v4"
RECIPE_STAGES = {
    "legacy_scaling",
    "scaling",
    "full",
    "stability_scaling",
    "stability_full",
}
ALLOWED_RUNTIME_BATCHES = {
    (1, 64),
    (2, 32),
    (4, 16),
    (8, 8),
    (16, 4),
    (32, 2),
    (64, 1),
}


def _normalized(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_normalized(item) for item in value]
    if isinstance(value, list):
        return [_normalized(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalized(item) for key, item in value.items()}
    return value


def _path_value(config: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = config
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return _normalized(value)


def infer_generation_training_stage(
    cofitok_config: dict[str, Any],
    dense_config: dict[str, Any],
) -> str:
    stability_recipe = any(
        float(config.get("loss", {}).get(field, 0.0)) > 0.0
        for config in (cofitok_config, dense_config)
        for field in (
            "rollout_consistency_weight",
            "ema_teacher_consistency_weight",
        )
    )
    identities = {
        (
            str(config.get("data", {}).get("dataset", "")),
            int(config.get("runtime", {}).get("steps", -1)),
        )
        for config in (cofitok_config, dense_config)
    }
    if identities == {("imagenet_256_10pct", 50_000)}:
        schedule = _path_value(cofitok_config, "model.token_channel_schedule", [])
        strides = _path_value(cofitok_config, "model.token_spatial_strides", [])
        if not schedule or not strides:
            return "legacy_scaling"
        return "stability_scaling" if stability_recipe else "scaling"
    if identities == {("imagenet_256", 300_000)}:
        return "stability_full" if stability_recipe else "full"
    raise ValueError(f"cannot infer formal generation training stage: {sorted(identities)}")


def _expected_shared(stage: str) -> dict[str, Any]:
    full = stage in {"full", "stability_full"}
    stability = stage in {"stability_scaling", "stability_full"}
    expected = {
        "data.dataset": "imagenet_256" if full else "imagenet_256_10pct",
        "data.image_size": 256,
        "data.channels": 3,
        "data.class_conditional": True,
        "data.random_horizontal_flip_prob": 0.0 if stage == "legacy_scaling" else 0.5,
        "diffusion.num_train_timesteps": 1_000,
        "diffusion.schedule_type": "cosine",
        "diffusion.prediction_target": "epsilon",
        "model.image_channels": 3,
        "model.image_size": 256,
        "model.base_channels": 256 if stage == "stability_full" else 128,
        "model.predictor_type": "scalable_unet",
        "model.predictor_channel_multipliers": [1, 2, 3, 4],
        "model.predictor_num_res_blocks": 2,
        "model.predictor_attention_resolutions": [32, 16],
        "model.predictor_num_heads": 8,
        "model.predictor_dropout": 0.0,
        "model.predictor_gradient_checkpointing": True,
        "model.num_classes": 1_000,
        "model.class_dropout_prob": 0.1,
        "runtime.seed": 2_027,
        "runtime.device": "cuda",
        "runtime.steps": 300_000 if full else 50_000,
        "runtime.precision": "bf16",
        "runtime.allow_tf32": True,
        "runtime.checkpoint_interval": 5_000,
        "runtime.evaluation_interval": 2_000 if full else 1_000,
        "runtime.keep_last_checkpoints": 3,
        "runtime.protected_checkpoint_steps": (
            [50_000, 100_000, 200_000, 300_000] if full else []
        ),
        "optimization.learning_rate": 1e-4,
        "optimization.min_learning_rate": 5e-6 if full else 1e-5,
        "optimization.warmup_steps": 5_000 if full else 1_000,
        "optimization.weight_decay": 0.01,
        "optimization.betas": [0.9, 0.99],
        "optimization.grad_clip_norm": 1.0,
        "optimization.ema_decay": 0.9999,
        "optimization.ema_warmup_steps": 2_000,
    }
    if stability:
        horizon = 300_000 if full else 50_000
        expected.update(
            {
                "loss.rollout_consistency_weight": 0.1,
                "loss.rollout_consistency_start_step": 0,
                "loss.rollout_consistency_warmup_steps": horizon // 5,
                "loss.rollout_consistency_timestep_delta": 10,
                "loss.rollout_consistency_unroll_steps": 2,
                "loss.rollout_consistency_batch_fraction": 0.125,
                "loss.rollout_consistency_clip_x0": True,
                "loss.rollout_consistency_mode": "clipped_x0",
                "loss.ema_teacher_consistency_weight": 0.25,
                "loss.ema_teacher_consistency_start_step": 3 * horizon // 5,
                "loss.ema_teacher_consistency_warmup_steps": horizon // 5,
                "loss.ema_teacher_consistency_batch_fraction": 0.0625,
            }
        )
    return expected


def _expected_method(method: str, stage: str) -> dict[str, Any]:
    if method == "cofitok":
        if stage in {"stability_scaling", "stability_full"}:
            return {
                "model.token_count": 8,
                "model.token_channels": 8,
                "model.token_channel_schedule": [4, 4, 8, 8, 8, 1, 1, 1],
                "model.token_spatial_strides": [16, 16, 8, 8, 4, 1, 1, 1],
                "model.predictor_use_feedback": True,
                "model.synthesis_mode": "fixed_basis",
                "model.synthesis_kernel_size": 1,
                "model.gamma_mode": "fixed_one",
                "model.synthesis_active_token_channels": [],
                "model.synthesis_token_strides": [],
                "loss.epsilon_weight": 1.0,
                "loss.denoise_path_prefix_weight": 0.05,
                "loss.denoise_path_component_weight": 0.1,
                "loss.denoise_path_energy_weight": 0.15,
                "loss.denoise_path_energy_mode": "hellinger_stable",
                "loss.denoise_path_energy_capacity_weight": 0.75,
                "loss.denoise_path_energy_capacity_power": 0.5,
                "loss.denoise_path_progress_power": 1.0,
                "loss.denoise_path_progress_mode": "token_capacity",
                "loss.low_snr_high_frequency_weight": 0.5,
                "loss.low_snr_high_frequency_power": 0.5,
            }
        if stage != "legacy_scaling":
            return {
                "model.token_count": 8,
                "model.token_channels": 8,
                "model.token_channel_schedule": [4, 4, 8, 8, 8, 8, 1, 2],
                "model.token_spatial_strides": [16, 16, 8, 8, 4, 4, 1, 1],
                "model.predictor_use_feedback": True,
                "model.synthesis_mode": "fixed_basis",
                "model.synthesis_kernel_size": 1,
                "model.gamma_mode": "fixed_one",
                "model.synthesis_active_token_channels": [],
                "model.synthesis_token_strides": [],
                "loss.epsilon_weight": 1.0,
                "loss.denoise_path_prefix_weight": 0.05,
                "loss.denoise_path_component_weight": 0.1,
                "loss.denoise_path_energy_weight": 0.1,
                "loss.denoise_path_energy_mode": "hellinger",
                "loss.denoise_path_progress_power": 1.0,
                "loss.denoise_path_progress_mode": "token_capacity",
            }
        return {
            "model.token_count": 8,
            "model.token_channels": 64,
            "model.predictor_use_feedback": True,
            "model.synthesis_mode": "restricted",
            "model.synthesis_kernel_size": 3,
            "model.gamma_mode": "learned_scalar",
            "model.synthesis_active_token_channels": [8, 16, 24, 32, 40, 48, 56, 64],
            "loss.epsilon_weight": 1.0,
            "loss.denoise_path_prefix_weight": 0.05,
            "loss.denoise_path_component_weight": 0.1,
            "loss.denoise_path_progress_power": 1.5,
        }
    dense_expected = {
        "model.token_count": 1,
        "model.token_channels": 3,
        "model.predictor_use_feedback": False,
        "model.synthesis_mode": "dense_identity",
        "loss.epsilon_weight": 1.0,
    }
    if stage != "legacy_scaling":
        dense_expected.update(
            {
                "model.token_channel_schedule": [],
                "model.token_spatial_strides": [],
            }
        )
    return dense_expected


def _unexpected_auxiliary_losses(
    config: dict[str, Any],
    *,
    allowed: set[str],
) -> list[str]:
    loss = config.get("loss", {})
    return sorted(
        key
        for key, value in loss.items()
        if key.endswith("_weight")
        and key not in allowed
        and isinstance(value, (int, float))
        and float(value) != 0.0
    )


def generation_training_recipe_contract(
    cofitok_config: dict[str, Any],
    dense_config: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    if stage not in RECIPE_STAGES:
        raise ValueError(f"unsupported generation training recipe stage: {stage}")

    pair_contract = generation_pair_contract(cofitok_config, dense_config)
    issues = [f"pair_contract: {issue}" for issue in pair_contract["issues"]]
    expected_shared = _expected_shared(stage)
    observed = {"cofitok": {}, "dense_identity": {}}
    for method, config in (
        ("cofitok", cofitok_config),
        ("dense_identity", dense_config),
    ):
        for path, expected in expected_shared.items():
            legacy_defaults = {
                "data.random_horizontal_flip_prob": 0.0,
                "runtime.protected_checkpoint_steps": [],
            }
            default = legacy_defaults.get(path)
            actual = _path_value(config, path, default)
            observed[method][path] = actual
            if actual != expected:
                issues.append(f"{method}.{path}: expected {expected!r}, got {actual!r}")
        for path, expected in _expected_method(method, stage).items():
            actual = _path_value(config, path)
            observed[method][path] = actual
            if actual != expected:
                issues.append(f"{method}.{path}: expected {expected!r}, got {actual!r}")

    stability = stage in {"stability_scaling", "stability_full"}
    shared_stability_losses = (
        {"rollout_consistency_weight", "ema_teacher_consistency_weight"}
        if stability
        else set()
    )
    cofitok_allowed_losses = {
        "epsilon_weight",
        "denoise_path_prefix_weight",
        "denoise_path_component_weight",
        "denoise_path_energy_weight",
        *shared_stability_losses,
    }
    dense_allowed_losses = {"epsilon_weight", *shared_stability_losses}
    if stability:
        cofitok_allowed_losses.update(
            {
                "denoise_path_energy_capacity_weight",
                "low_snr_high_frequency_weight",
            }
        )
    for method, config, allowed in (
        ("cofitok", cofitok_config, cofitok_allowed_losses),
        ("dense_identity", dense_config, dense_allowed_losses),
    ):
        unexpected = _unexpected_auxiliary_losses(config, allowed=allowed)
        if unexpected:
            issues.append(f"{method} has non-recipe auxiliary losses: {', '.join(unexpected)}")

    effective_batches = {}
    for method, config in (
        ("cofitok", cofitok_config),
        ("dense_identity", dense_config),
    ):
        micro_batch = int(_path_value(config, "data.batch_size", 0))
        accumulation = int(
            _path_value(config, "optimization.gradient_accumulation_steps", 0)
        )
        effective_batch = micro_batch * accumulation
        effective_batches[method] = {
            "micro_batch_size": micro_batch,
            "gradient_accumulation_steps": accumulation,
            "effective_batch_size": effective_batch,
        }
        if (micro_batch, accumulation) not in ALLOWED_RUNTIME_BATCHES:
            issues.append(
                f"{method}.runtime_batch: expected one of "
                f"{sorted(ALLOWED_RUNTIME_BATCHES)!r}, got "
                f"{(micro_batch, accumulation)!r} (effective {effective_batch})"
            )

    token_layout = None
    if stage != "legacy_scaling":
        model = cofitok_config.get("model", {})
        layout = resolve_token_layout(
            image_size=int(model.get("image_size", 0)),
            image_channels=int(model.get("image_channels", 0)),
            token_count=int(model.get("token_count", 0)),
            token_channels=int(model.get("token_channels", 0)),
            token_channel_schedule=model.get("token_channel_schedule", []),
            token_spatial_strides=model.get("token_spatial_strides", []),
        )
        token_layout = token_layout_summary(layout)
        issues.extend(f"compressed_token_layout: {issue}" for issue in token_layout["issues"])

    return {
        "schema": GENERATION_TRAINING_RECIPE_SCHEMA,
        "stage": stage,
        "valid": not issues,
        "issues": issues,
        "expected_shared": expected_shared,
        "allowed_runtime_batches": [list(pair) for pair in sorted(ALLOWED_RUNTIME_BATCHES)],
        "effective_batches": effective_batches,
        "observed": observed,
        "pair_contract": pair_contract,
        "token_layout": token_layout,
    }
