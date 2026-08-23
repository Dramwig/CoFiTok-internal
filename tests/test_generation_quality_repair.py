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


class _RecordingZeroModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(num_classes=0)
        self.inputs: list[torch.Tensor] = []

    def forward(self, images, timesteps, class_labels=None, force_unconditional=False):
        del timesteps, class_labels, force_unconditional
        self.inputs.append(images.detach().clone())
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


def test_epsilon_can_be_reconstructed_after_x0_constraint() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=8, schedule_type="cosine"),
        device="cpu",
    )
    clean = torch.randn(2, 1, 2, 2)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([2, 6])
    noisy = schedule.add_noise(clean, noise, timesteps)

    torch.testing.assert_close(
        schedule.predict_epsilon_from_x0(noisy, clean, timesteps),
        noise,
    )


def test_ddim_recomputed_epsilon_drives_the_next_direction_update() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=8, schedule_type="cosine"),
        device="cpu",
    )
    legacy_model = _RecordingZeroModel()
    consistent_model = _RecordingZeroModel()
    common = {
        "schedule": schedule,
        "shape": (1, 1, 2, 2),
        "sample_steps": 2,
        "prefix_budget": 1,
        "eta": 0.0,
        "clip_x0": True,
        "device": torch.device("cpu"),
        "start_timestep": 7,
    }

    ddim_sample(
        legacy_model,
        generator=torch.Generator().manual_seed(17),
        **common,
    )
    ddim_sample(
        consistent_model,
        generator=torch.Generator().manual_seed(17),
        recompute_epsilon_after_x0_constraint=True,
        **common,
    )

    assert len(legacy_model.inputs) == 2
    assert len(consistent_model.inputs) == 2
    initial = legacy_model.inputs[0]
    torch.testing.assert_close(initial, consistent_model.inputs[0])
    timestep = torch.tensor([7])
    constrained_x0 = schedule.predict_x0_from_epsilon(
        initial,
        torch.zeros_like(initial),
        timestep,
    ).clamp(-1.0, 1.0)
    consistent_epsilon = schedule.predict_epsilon_from_x0(
        initial,
        constrained_x0,
        timestep,
    )
    alpha_previous = schedule.alphas_cumprod[0]
    legacy_next = torch.sqrt(alpha_previous) * constrained_x0
    consistent_next = legacy_next + torch.sqrt(
        1.0 - alpha_previous
    ) * consistent_epsilon

    torch.testing.assert_close(legacy_model.inputs[1], legacy_next)
    torch.testing.assert_close(consistent_model.inputs[1], consistent_next)
    assert not torch.allclose(legacy_model.inputs[1], consistent_model.inputs[1])


def test_epsilon_recomputation_requires_x0_constraint() -> None:
    with pytest.raises(ValueError, match="epsilon recomputation"):
        GenerationRequest(
            seeds=(0,),
            clip_x0=False,
            recompute_epsilon_after_x0_constraint=True,
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
        "recompute_epsilon_after_x0_constraint": True,
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
    assert "formal_recompute_epsilon_after_x0_constraint" in formal["issues"]
