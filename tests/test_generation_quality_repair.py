from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from cofitok.configs import DiffusionConfig, LossConfig
from cofitok.diffusion import DiffusionSchedule, ddim_sample, select_sampling_timesteps
from cofitok.diffusion.sampling import constrain_predicted_x0
from cofitok.generation import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from cofitok.generation.protocol import sampling_protocol_contract
from cofitok.generation.session import GenerationRequest
from cofitok.models.cofitok import CoFiTokOutput
from cofitok.training.losses import compute_losses


class _ZeroModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=0)
        self.first_input: torch.Tensor | None = None

    def forward(self, images, timesteps, class_labels=None, force_unconditional=False):
        del timesteps, class_labels, force_unconditional
        if self.first_input is None:
            self.first_input = images.detach().clone()
        zeros = torch.zeros_like(images)
        return CoFiTokOutput(
            tokens=[zeros],
            components=[zeros],
            prefix_epsilons=[zeros],
            epsilon=zeros,
        )


def _single_output(epsilon: torch.Tensor) -> CoFiTokOutput:
    return CoFiTokOutput(
        tokens=[torch.zeros_like(epsilon)],
        components=[epsilon],
        prefix_epsilons=[epsilon],
        epsilon=epsilon,
    )


def test_min_snr_weights_match_epsilon_reference() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=16, schedule_type="cosine"),
        device="cpu",
    )
    timesteps = torch.tensor([0, 7, 15])
    snr = schedule.snr(timesteps)
    expected = torch.minimum(snr, torch.full_like(snr, 5.0)) / snr

    torch.testing.assert_close(
        schedule.min_snr_loss_weights(timesteps, 5.0),
        expected,
    )
    torch.testing.assert_close(
        schedule.min_snr_loss_weights(timesteps, 0.0),
        torch.ones(3),
    )


def test_min_snr_loss_is_per_sample_and_reports_raw_mse() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=16, schedule_type="cosine"),
        device="cpu",
    )
    clean = torch.zeros(2, 1, 2, 2)
    noise = torch.zeros_like(clean)
    prediction = torch.stack([torch.ones(1, 2, 2), torch.full((1, 2, 2), 2.0)])
    timesteps = torch.tensor([0, 15])
    noisy = schedule.add_noise(clean, noise, timesteps)
    weights = schedule.min_snr_loss_weights(timesteps, 5.0)

    losses = compute_losses(
        LossConfig(
            min_snr_gamma=5.0,
            prefix_weight=0.0,
            monotonic_weight=0.0,
            zero_token_weight=0.0,
        ),
        _single_output(prediction),
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.epsilon_unweighted.item() == pytest.approx(2.5)
    assert losses.epsilon.item() == pytest.approx(
        float((torch.tensor([1.0, 4.0]) * weights).mean())
    )
    assert losses.min_snr_weight_mean.item() == pytest.approx(float(weights.mean()))


def test_sampling_timesteps_support_auditable_nonterminal_start() -> None:
    assert select_sampling_timesteps(
        1000,
        5,
        start_timestep=899,
    ) == [899, 674, 450, 225, 0]
    with pytest.raises(ValueError, match="start_timestep"):
        select_sampling_timesteps(1000, 5, start_timestep=1000)
    with pytest.raises(ValueError, match="at least two"):
        select_sampling_timesteps(1000, 1, start_timestep=899)


def test_generation_request_rejects_non_integer_start_timestep() -> None:
    with pytest.raises(ValueError, match="start_timestep"):
        GenerationRequest(seeds=(0,), start_timestep=899.0)  # type: ignore[arg-type]


def test_initial_noise_can_be_scaled_by_selected_schedule_sigma() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=8, schedule_type="cosine"),
        device="cpu",
    )
    unit_model = _ZeroModel()
    sigma_model = _ZeroModel()
    common = {
        "schedule": schedule,
        "shape": (2, 1, 2, 2),
        "sample_steps": 3,
        "prefix_budget": 1,
        "eta": 0.0,
        "clip_x0": False,
        "device": torch.device("cpu"),
        "start_timestep": 6,
    }
    ddim_sample(
        unit_model,
        generator=torch.Generator().manual_seed(31),
        **common,
    )
    ddim_sample(
        sigma_model,
        generator=torch.Generator().manual_seed(31),
        scale_initial_noise_by_sigma=True,
        **common,
    )

    assert unit_model.first_input is not None
    assert sigma_model.first_input is not None
    torch.testing.assert_close(
        sigma_model.first_input,
        unit_model.first_input * schedule.sqrt_one_minus_alphas_cumprod[6],
    )


def test_dynamic_threshold_is_per_sample_and_bounded() -> None:
    predicted = torch.tensor(
        [
            [[[0.0, 1.0], [2.0, 100.0]]],
            [[[0.0, 0.5], [0.75, 1.0]]],
        ]
    )

    constrained = constrain_predicted_x0(
        predicted,
        clip_x0=True,
        dynamic_threshold_percentile=0.75,
    )

    assert constrained.abs().max().item() <= 1.0
    assert constrained[0, 0, 0, 1].item() < 1.0
    torch.testing.assert_close(constrained[1], predicted[1])


def test_repair_sampling_protocol_is_explicit_and_non_formal() -> None:
    timesteps = select_sampling_timesteps(1000, 100, start_timestep=899)
    sampling = {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": INFERENCE_API,
        "sampler": "ddim",
        "num_train_timesteps": 1000,
        "sample_steps": 100,
        "actual_timesteps": timesteps,
        "start_timestep": 899,
        "requested_start_timestep": 899,
        "scale_initial_noise_by_sigma": True,
        "initial_noise_scale": "schedule_sigma",
        "clip_x0": True,
        "x0_constraint": "dynamic_threshold",
        "dynamic_threshold_percentile": 0.995,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "eta": 0.0,
        "precision": "bf16",
    }

    assert sampling_protocol_contract(sampling)["valid"] is True
    formal = sampling_protocol_contract(
        {
            **sampling,
            "num_samples": 10_000,
            "start_index": 0,
            "batch_size": 32,
            "cfg_batch_mode": "batched",
            "seed": 0,
            "class_schedule": "balanced_modulo",
            "random_stream": {
                "prefix_budgets_share_stream": True,
                "batch_size_invariant": True,
                "resume_index_invariant": True,
            },
        },
        stage="scaling",
        expected_num_train_timesteps=1000,
    )
    assert formal["valid"] is False
    assert "formal_start_timestep" in formal["issues"]
    assert "formal_x0_constraint" in formal["issues"]
