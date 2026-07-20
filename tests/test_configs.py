import json
from pathlib import Path

from cofitok.configs import load_config


def test_load_config_with_optimization_defaults(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"name": "tiny"}), encoding="utf-8")

    config = load_config(path)

    assert config.name == "tiny"
    assert config.optimization.learning_rate == 2e-4
    assert config.optimization.log_interval == 10
    assert config.model.predictor_type == "tiny_conv"
    assert config.model.predictor_use_feedback is True
    assert config.model.predictor_multiscale_levels == 2
    assert config.model.synthesis_mode == "restricted"
    assert config.model.deep_synthesis_hidden_channels == 0
    assert config.model.deep_synthesis_depth == 3
    assert config.runtime.cudnn_benchmark is False
    assert config.loss.energy_budget_weight == 0.0
    assert config.loss.energy_target == []
    assert config.loss.energy_budget_scope == "batch"
    assert config.loss.residual_component_weight == 0.0
    assert config.loss.component_decorrelation_weight == 0.0
    assert config.loss.component_decorrelation_start_step == 0
    assert config.loss.component_decorrelation_warmup_steps == 0
    assert config.loss.tail_floor_weight == 0.0
    assert config.loss.tail_floor_min_ratio == 0.0
    assert config.loss.sampled_prefix_weight == 0.0
    assert config.loss.sampled_component_weight == 0.0
    assert config.loss.group_residual_weight == 0.0
    assert config.loss.group_residual_size == 0
    assert config.loss.epsilon_band_prefix_weight == 0.0
    assert config.loss.epsilon_band_component_weight == 0.0
    assert config.loss.denoise_path_prefix_weight == 0.0
    assert config.loss.denoise_path_component_weight == 0.0
    assert config.loss.denoise_path_energy_weight == 0.0
    assert config.loss.denoise_path_progress_power == 1.0
    assert config.loss.tail_early_dropout_weight == 0.0
    assert config.loss.tail_early_dropout_prob == 0.0
    assert config.loss.tail_early_dropout_start == 1
    assert config.loss.tail_replay_weight == 0.0
    assert config.loss.tail_late_prefix_weight == 0.0
    assert config.loss.tail_late_prefix_start == 0
    assert config.loss.tail_endpoint_weight == 0.0
    assert config.loss.tail_late_monotonic_weight == 0.0
    assert config.loss.tail_late_monotonic_start == 0
    assert config.loss.tail_late_monotonic_margin == 0.0


def test_multiscale_epsilononly_controls_match_20k_backbone_settings() -> None:
    root = Path(__file__).resolve().parents[1]
    tiny = load_config(root / "configs/train_tiny_imagenet_k8_epsilononly_p150eval_multiscale_20k_cuda.json")
    hf = load_config(root / "configs/train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_multiscale_20k_cuda.json")

    for config in (tiny, hf):
        assert config.model.predictor_type == "multiscale_unet"
        assert config.model.predictor_depth == 2
        assert config.model.predictor_multiscale_levels == 2
        assert config.model.synthesis_mode == "restricted"
        assert config.loss.epsilon_weight == 1.0
        assert config.loss.prefix_weight == 0.0
        assert config.loss.monotonic_weight == 0.0
        assert config.loss.denoise_path_prefix_weight == 0.0
        assert config.loss.denoise_path_component_weight == 0.0
        assert config.loss.zero_token_weight == 0.01
        assert config.runtime.steps == 20000
        assert config.runtime.seed == 151
        assert config.data.batch_size == 24


def test_load_config_with_optimization_overrides(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "optimization": {
                    "learning_rate": 1e-3,
                    "log_interval": 3,
                    "visualization_count": 2,
                },
                "model": {
                    "predictor_type": "multiscale_unet",
                    "predictor_use_feedback": False,
                    "predictor_multiscale_levels": 3,
                    "synthesis_mode": "deep_decoder",
                    "deep_synthesis_hidden_channels": 24,
                    "deep_synthesis_depth": 4,
                },
                "loss": {
                    "energy_budget_weight": 2.0,
                    "energy_target": [0.5, 0.25, 0.15, 0.1],
                    "energy_budget_scope": "sample",
                    "residual_component_weight": 0.1,
                    "component_decorrelation_weight": 0.2,
                    "component_decorrelation_start_step": 100,
                    "component_decorrelation_warmup_steps": 200,
                    "tail_floor_weight": 3.0,
                    "tail_floor_min_ratio": 0.02,
                    "sampled_prefix_weight": 0.4,
                    "sampled_component_weight": 1.0,
                    "group_residual_weight": 0.5,
                    "group_residual_size": 2,
                    "epsilon_band_prefix_weight": 0.25,
                    "epsilon_band_component_weight": 0.75,
                    "denoise_path_prefix_weight": 0.2,
                    "denoise_path_component_weight": 0.6,
                    "denoise_path_energy_weight": 0.4,
                    "denoise_path_progress_power": 1.5,
                    "tail_early_dropout_weight": 0.25,
                    "tail_early_dropout_prob": 0.5,
                    "tail_early_dropout_start": 1,
                    "tail_replay_weight": 0.5,
                    "tail_late_prefix_weight": 0.03,
                    "tail_late_prefix_start": 4,
                    "tail_endpoint_weight": 0.1,
                    "tail_late_monotonic_weight": 0.05,
                    "tail_late_monotonic_start": 4,
                    "tail_late_monotonic_margin": 0.01,
                }
            }
        ),
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.optimization.learning_rate == 1e-3
    assert config.optimization.log_interval == 3
    assert config.optimization.visualization_count == 2
    assert config.model.predictor_type == "multiscale_unet"
    assert config.model.predictor_use_feedback is False
    assert config.model.predictor_multiscale_levels == 3
    assert config.model.synthesis_mode == "deep_decoder"
    assert config.model.deep_synthesis_hidden_channels == 24
    assert config.model.deep_synthesis_depth == 4
    assert config.loss.energy_budget_weight == 2.0
    assert config.loss.energy_target == [0.5, 0.25, 0.15, 0.1]
    assert config.loss.energy_budget_scope == "sample"
    assert config.loss.residual_component_weight == 0.1
    assert config.loss.component_decorrelation_weight == 0.2
    assert config.loss.component_decorrelation_start_step == 100
    assert config.loss.component_decorrelation_warmup_steps == 200
    assert config.loss.tail_floor_weight == 3.0
    assert config.loss.tail_floor_min_ratio == 0.02
    assert config.loss.sampled_prefix_weight == 0.4
    assert config.loss.sampled_component_weight == 1.0
    assert config.loss.group_residual_weight == 0.5
    assert config.loss.group_residual_size == 2
    assert config.loss.epsilon_band_prefix_weight == 0.25
    assert config.loss.epsilon_band_component_weight == 0.75
    assert config.loss.denoise_path_prefix_weight == 0.2
    assert config.loss.denoise_path_component_weight == 0.6
    assert config.loss.denoise_path_energy_weight == 0.4
    assert config.loss.denoise_path_progress_power == 1.5
    assert config.loss.tail_early_dropout_weight == 0.25
    assert config.loss.tail_early_dropout_prob == 0.5
    assert config.loss.tail_early_dropout_start == 1
    assert config.loss.tail_replay_weight == 0.5
    assert config.loss.tail_late_prefix_weight == 0.03
    assert config.loss.tail_late_prefix_start == 4
    assert config.loss.tail_endpoint_weight == 0.1
    assert config.loss.tail_late_monotonic_weight == 0.05
    assert config.loss.tail_late_monotonic_start == 4
    assert config.loss.tail_late_monotonic_margin == 0.01
