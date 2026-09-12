import dataclasses
import json

import pytest
import torch

from cofitok.configs import config_from_dict
from cofitok.generation.frozen_loss_gradients import gradient_statistics, measure_selected_example, parameter_group
from cofitok.models import CoFiTokTiny
from cofitok.diffusion import DiffusionSchedule


@pytest.fixture
def probe():
    torch.set_num_threads(1)
    config = config_from_dict({
        "data": {"image_size": 8, "batch_size": 64, "class_conditional": True},
        "diffusion": {"schedule_type": "cosine", "num_train_timesteps": 32},
        "model": {"image_size": 8, "base_channels": 8, "predictor_type": "scalable_unet",
                  "predictor_channel_multipliers": [1], "predictor_num_res_blocks": 1,
                  "predictor_attention_resolutions": [], "predictor_num_heads": 1,
                  "predictor_gradient_checkpointing": True, "num_classes": 3,
                  "class_dropout_prob": 0.4, "synthesis_mode": "dense_identity",
                  "gamma_mode": "fixed_one", "synthesis_kernel_size": 1,
                  "token_count": 1, "token_channels": 3, "predictor_use_feedback": False},
        "loss": {"prefix_weight": 0, "monotonic_weight": 0, "zero_token_weight": 0,
                 "rollout_consistency_weight": 0.1, "rollout_consistency_timestep_delta": 2,
                 "rollout_consistency_unroll_steps": 2, "rollout_consistency_mode": "clipped_x0",
                 "rollout_consistency_batch_fraction": 0.125,
                 "ema_teacher_consistency_weight": 0.25,
                 "ema_teacher_consistency_start_step": 10, "ema_teacher_consistency_warmup_steps": 10},
    })
    with torch.random.fork_rng():
        torch.manual_seed(991)
        model = CoFiTokTiny(config.model).eval()
        # Untrained zero heads give zero conditioning gradients. Populate only
        # the tiny test model so every group has a measurable reference gradient.
        for name, param in model.named_parameters():
            if name.endswith("weight") and not torch.count_nonzero(param):
                torch.nn.init.normal_(param, std=0.01)
        ema = {k: v.detach().clone() * 0.97 for k, v in model.state_dict().items()}
        clean = torch.randn(1, 3, 8, 8).clamp(-1, 1)
    return dict(model=model, ema_state=ema, config=config, clean=clean, label=1,
                timestep=24, checkpoint_step=100, noise_seed=12, dropout_seed=15)


def test_nonupdating_probe_is_reproducible_and_restores_everything(probe):
    model = probe["model"]
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    rng = torch.random.get_rng_state().clone()
    first = measure_selected_example(**probe)
    second = measure_selected_example(**probe)
    assert first == second
    torch.testing.assert_close(torch.random.get_rng_state(), rng)
    assert not model.training
    assert all(p.grad is None for p in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], atol=0, rtol=0)
    assert not model._forward_hooks
    assert not model.predictor.class_embed._forward_pre_hooks
    assert [r["phase"] for r in first["conditioning_routes"]] == ["main", "ema_teacher", "rollout", "rollout"]
    assert first["conditioning_routes"][1]["effective_labels"] == [1]
    assert len(first["rollout_steps"]) == 2
    assert first["normalization"]["original_teacher_selected_count"] == 4
    assert first["normalization"]["original_rollout_selected_count_if_all_timesteps_valid"] == 8
    assert first["normalization"]["is_original_minibatch_gradient"] is False
    assert first["terms"]["epsilon"]["gradient"]["all"]["norm"] > 0
    residual = first["gradient_sum_rounding_residual"]["all"]
    assert residual["norm_ratio_to_total"] < 1e-5
    json.dumps(first, allow_nan=False)


def test_teacher_before_activation_is_not_attributed(probe):
    probe["checkpoint_step"] = 10
    report = measure_selected_example(**probe)
    assert report["terms"]["ema_teacher"]["weighted_loss"] == 0
    assert report["terms"]["ema_teacher"]["gradient"]["all"]["norm"] == 0
    assert "ema_teacher" not in [r["phase"] for r in report["conditioning_routes"]]


def test_invalid_timestep_rollout_records_empty_not_fabricated_zero_rows(probe):
    probe["timestep"] = 1
    report = measure_selected_example(**probe)
    assert report["rollout_steps"] == []
    assert report["terms"]["rollout"]["gradient"]["all"]["norm"] == 0


def test_gradient_statistics_zero_unused_and_undefined_cosine():
    names = ["predictor.class_embed.weight", "predictor.token_heads.0.weight"]
    stats = gradient_statistics(names, [None, torch.zeros(2)], [torch.ones(2), torch.ones(2)])
    assert stats["class_embedding"]["state"] == "unused"
    assert stats["output_heads"]["state"] == "zero"
    assert stats["all"]["cosine_to_epsilon"] is None
    aligned = gradient_statistics(names, [torch.ones(2), torch.ones(2)], [torch.ones(2), torch.ones(2)])
    assert aligned["all"]["cosine_to_epsilon"] == pytest.approx(1)
    opposed = gradient_statistics(names, [-torch.ones(2), -torch.ones(2)], [torch.ones(2), torch.ones(2)])
    assert opposed["all"]["cosine_to_epsilon"] == pytest.approx(-1)


def test_nonfinite_and_unknown_parameter_fail_closed():
    with pytest.raises(ValueError, match="nonfinite"):
        gradient_statistics(["predictor.class_embed.weight"], [torch.tensor(float("nan"))], [None])
    with pytest.raises(ValueError, match="outside predictor"):
        parameter_group("synthesis.decoder.weight")


def test_original_100k_default_schedule_is_byte_equivalent(probe):
    config = dataclasses.replace(probe["config"].diffusion, num_train_timesteps=1000)
    values = torch.linspace(0, 1000, 1001)
    cumulative = torch.cos(((values / 1000 + 0.008) / 1.008) * torch.pi * 0.5).pow(2)
    cumulative = cumulative / cumulative[0]
    old_betas = (1 - cumulative[1:] / cumulative[:-1]).clamp(1e-4, 0.999)
    actual = DiffusionSchedule(config, "cpu")
    assert torch.equal(actual.betas, old_betas)
    assert torch.equal(actual.alphas_cumprod, torch.cumprod(1 - old_betas, dim=0))


@pytest.mark.parametrize("field,value", [("label", True), ("timestep", 32), ("checkpoint_step", -1), ("noise_seed", -2)])
def test_invalid_probe_arguments_rejected(probe, field, value):
    probe[field] = value
    with pytest.raises(ValueError, match="invalid"):
        measure_selected_example(**probe)


def test_rejects_original_batch_claim_on_multi_image_call(probe):
    probe["clean"] = probe["clean"].expand(2, -1, -1, -1)
    with pytest.raises(ValueError, match="exactly one"):
        measure_selected_example(**probe)


def test_bf16_cannot_be_claimed_on_cpu(probe):
    config = probe["config"]
    probe["config"] = dataclasses.replace(config, runtime=dataclasses.replace(config.runtime, precision="bf16"))
    with pytest.raises(ValueError, match="silently fall back"):
        measure_selected_example(**probe)


def test_fixed_basis_auxiliary_terms_are_included(probe):
    config = probe["config"]
    model_config = dataclasses.replace(config.model, synthesis_mode="fixed_basis", token_count=3,
        token_channels=1, token_channel_schedule=[1, 1, 1], token_spatial_strides=[1, 1, 1],
        predictor_use_feedback=True)
    loss = dataclasses.replace(config.loss, denoise_path_prefix_weight=0.05,
        denoise_path_component_weight=0.1, denoise_path_energy_weight=0.15,
        denoise_path_energy_mode="hellinger_stable", low_snr_high_frequency_weight=0.5)
    probe["config"] = dataclasses.replace(config, model=model_config, loss=loss)
    with torch.random.fork_rng():
        torch.manual_seed(55)
        model = CoFiTokTiny(model_config).eval()
        for head in model.predictor.token_heads:
            torch.nn.init.normal_(head.weight, std=0.01)
    probe["model"] = model
    probe["ema_state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    result = measure_selected_example(**probe)
    assert result["terms"]["other_auxiliary"]["weighted_loss"] > 0
    assert result["terms"]["other_auxiliary"]["gradient"]["all"]["norm"] > 0
    assert result["weighted_components"]["low_snr_high_frequency"]["effective_weight"] == 0.5


def test_rejects_config_drift_and_restores_on_hook_exception(probe):
    original = probe["config"]
    probe["config"] = dataclasses.replace(original, model=dataclasses.replace(original.model, class_dropout_prob=0.9))
    with pytest.raises(ValueError, match="config differs"):
        measure_selected_example(**probe)
    probe["config"] = original
    model = probe["model"]
    def failure(*args):
        raise ValueError("test failure")
    handle = model.predictor.class_embed.register_forward_pre_hook(failure)
    try:
        with pytest.raises(ValueError, match="test failure"):
            measure_selected_example(**probe)
        assert not model.training
        assert not model._forward_hooks
        assert len(model.predictor.class_embed._forward_pre_hooks) == 1
    finally:
        handle.remove()
