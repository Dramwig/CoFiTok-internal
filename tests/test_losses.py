import pytest
import torch

from cofitok.configs import DiffusionConfig, LossConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokOutput
from cofitok.training.conditioning import ClassConditioningRankingResult
from cofitok.training.losses import (
    _component_energy_distribution_loss,
    _low_snr_high_frequency_loss,
    compute_losses,
    denoise_path_schedule,
)


def _fake_output() -> CoFiTokOutput:
    components = [
        torch.ones(2, 3, 4, 4),
        torch.ones(2, 3, 4, 4) * 0.5,
    ]
    return CoFiTokOutput(
        tokens=[torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 4, 4)],
        components=components,
        prefix_epsilons=[components[0], components[0] + components[1]],
        epsilon=components[0] + components[1],
    )


def _fake_output_with_count(count: int, shape: tuple[int, int] = (32, 32)) -> CoFiTokOutput:
    components = [
        torch.randn(2, 3, shape[0], shape[1]) * (0.1 + index * 0.01)
        for index in range(count)
    ]
    prefix_epsilons = []
    running = torch.zeros_like(components[0])
    for component in components:
        running = running + component
        prefix_epsilons.append(running)
    return CoFiTokOutput(
        tokens=[torch.zeros(2, 1, shape[0], shape[1]) for _ in range(count)],
        components=components,
        prefix_epsilons=prefix_epsilons,
        epsilon=prefix_epsilons[-1],
    )


def test_token_capacity_denoise_path_schedule_uses_actual_token_layout() -> None:
    clean = torch.randn(2, 3, 16, 16)
    tokens = [
        torch.zeros(2, 1, 4, 4),
        torch.zeros(2, 1, 16, 16),
    ]

    spatial_targets, progress = denoise_path_schedule(
        clean,
        tokens,
        progress_power=99.0,
        progress_mode="token_capacity",
    )

    assert progress == [0.2, 1.0]
    assert spatial_targets[0].shape == clean.shape
    assert not torch.equal(spatial_targets[0], clean)
    assert torch.equal(spatial_targets[-1], clean)


def test_energy_budget_loss_is_reported_when_target_is_set() -> None:
    output = _fake_output()
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 4, 4)
    noise = torch.zeros_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(energy_budget_weight=1.0, energy_target=[0.5, 0.5]),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.energy_budget.item() > 0
    assert "energy_budget" in losses.as_dict()


def test_ema_teacher_consistency_is_scaled_in_total() -> None:
    output = _fake_output()
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 4, 4)
    noise = torch.zeros_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(
            epsilon_weight=0.0,
            prefix_weight=0.0,
            monotonic_weight=0.0,
            zero_token_weight=0.0,
            ema_teacher_consistency_weight=2.0,
        ),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
        ema_teacher_consistency=torch.tensor(4.0),
        ema_teacher_consistency_scale=0.5,
    )

    assert losses.total.item() == pytest.approx(4.0)
    assert losses.ema_teacher_consistency.item() == pytest.approx(4.0)
    assert losses.ema_teacher_consistency_scale.item() == pytest.approx(0.5)


def test_class_conditioning_ranking_is_scaled_and_reported_in_total() -> None:
    output = _fake_output()
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 4, 4)
    noise = output.epsilon.detach().clone()
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)
    ranking = ClassConditioningRankingResult(
        loss=torch.tensor(2.0),
        correct_mse=torch.tensor(0.2),
        wrong_mse=torch.tensor(0.3),
        null_mse=torch.tensor(0.4),
        correct_better_wrong_fraction=torch.tensor(0.75),
        correct_better_null_fraction=torch.tensor(0.5),
        selected_fraction=torch.tensor(0.125),
    )

    losses = compute_losses(
        LossConfig(
            epsilon_weight=0.0,
            prefix_weight=0.0,
            monotonic_weight=0.0,
            zero_token_weight=0.0,
            class_conditioning_ranking_weight=3.0,
        ),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
        class_conditioning_ranking=ranking,
        class_conditioning_ranking_scale=0.5,
    )

    assert losses.total.item() == pytest.approx(3.0)
    assert losses.class_conditioning_ranking.item() == pytest.approx(2.0)
    assert losses.class_conditioning_ranking_scale.item() == pytest.approx(0.5)
    assert losses.class_conditioning_correct_mse.item() == pytest.approx(0.2)
    assert losses.class_conditioning_ranking_selected_fraction.item() == pytest.approx(
        0.125
    )
    assert "class_conditioning_correct_better_null_fraction" in losses.as_dict()


def test_consistency_scales_remain_float32_for_bfloat16_outputs() -> None:
    output = _fake_output()
    output = CoFiTokOutput(
        tokens=[tensor.bfloat16() for tensor in output.tokens],
        components=[tensor.bfloat16() for tensor in output.components],
        prefix_epsilons=[
            tensor.bfloat16() for tensor in output.prefix_epsilons
        ],
        epsilon=output.epsilon.bfloat16(),
    )
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=10),
        device="cpu",
    )
    clean = torch.zeros(2, 3, 4, 4, dtype=torch.bfloat16)
    noise = torch.zeros_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(
            epsilon_weight=0.0,
            prefix_weight=0.0,
            monotonic_weight=0.0,
            zero_token_weight=0.0,
            rollout_consistency_weight=1.0,
            ema_teacher_consistency_weight=1.0,
        ),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
        rollout_consistency=torch.tensor(3.0),
        rollout_consistency_scale=1.0 / 12.0,
        ema_teacher_consistency=torch.tensor(4.0),
        ema_teacher_consistency_scale=1.0 / 6.0,
    )

    assert losses.rollout_consistency_scale.dtype == torch.float32
    assert losses.ema_teacher_consistency_scale.dtype == torch.float32
    assert losses.rollout_consistency_scale.item() == pytest.approx(1.0 / 12.0)
    assert losses.ema_teacher_consistency_scale.item() == pytest.approx(1.0 / 6.0)
    assert losses.total.item() == pytest.approx(11.0 / 12.0)


def test_sample_energy_budget_cannot_be_satisfied_by_batch_complementarity() -> None:
    first = torch.stack([torch.ones(3, 4, 4), torch.zeros(3, 4, 4)])
    second = torch.stack([torch.zeros(3, 4, 4), torch.ones(3, 4, 4)])
    output = CoFiTokOutput(
        tokens=[torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 4, 4)],
        components=[first, second],
        prefix_epsilons=[first, first + second],
        epsilon=first + second,
    )
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 4, 4)
    noise = torch.zeros_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    batch_loss = compute_losses(
        LossConfig(
            epsilon_weight=0.0,
            energy_budget_weight=1.0,
            energy_target=[1.0, 1.0],
            energy_budget_scope="batch",
        ),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )
    sample_loss = compute_losses(
        LossConfig(
            epsilon_weight=0.0,
            energy_budget_weight=1.0,
            energy_target=[1.0, 1.0],
            energy_budget_scope="sample",
        ),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert batch_loss.energy_budget.item() == 0.0
    assert sample_loss.energy_budget.item() == 0.25


def test_residual_component_loss_is_reported_when_enabled() -> None:
    output = _fake_output()
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 4, 4)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(residual_component_weight=0.1),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.residual_component.item() > 0
    assert "residual_component" in losses.as_dict()


def test_component_decorrelation_loss_is_reported_when_enabled() -> None:
    output = _fake_output()
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 4, 4)
    noise = torch.zeros_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(component_decorrelation_weight=0.1),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.component_decorrelation.item() > 0
    assert "component_decorrelation" in losses.as_dict()


def test_tail_floor_loss_is_reported_when_enabled() -> None:
    output = _fake_output()
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 4, 4)
    noise = torch.zeros_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(tail_floor_weight=1.0, tail_floor_min_ratio=0.3),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.tail_floor.item() > 0
    assert "tail_floor" in losses.as_dict()


def test_sampled_progressive_losses_are_reported_when_enabled() -> None:
    torch.manual_seed(7)
    output = _fake_output()
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 4, 4)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(sampled_prefix_weight=0.2, sampled_component_weight=0.5),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.sampled_prefix.item() > 0
    assert losses.sampled_component.item() > 0
    assert losses.sampled_budget.item() in {1.0, 2.0}
    assert "sampled_prefix" in losses.as_dict()
    assert "sampled_component" in losses.as_dict()


def test_prefix_losses_support_k8_on_cifar_sized_images() -> None:
    output = _fake_output_with_count(8, shape=(32, 32))
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 32, 32)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert torch.isfinite(losses.prefix)
    assert torch.isfinite(losses.monotonic)


def test_group_residual_loss_is_reported_when_enabled() -> None:
    output = _fake_output_with_count(8, shape=(32, 32))
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 32, 32)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(group_residual_weight=0.5, group_residual_size=2),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.group_residual.item() > 0
    assert "group_residual" in losses.as_dict()


def test_epsilon_band_losses_are_reported_when_enabled() -> None:
    output = _fake_output_with_count(4, shape=(16, 16))
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 16, 16)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(epsilon_band_prefix_weight=0.5, epsilon_band_component_weight=0.5),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.epsilon_band_prefix.item() > 0
    assert losses.epsilon_band_component.item() > 0
    assert "epsilon_band_prefix" in losses.as_dict()
    assert "epsilon_band_component" in losses.as_dict()


def test_denoise_path_losses_are_reported_when_enabled() -> None:
    output = _fake_output_with_count(4, shape=(16, 16))
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.zeros(2, 3, 16, 16)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(
            denoise_path_prefix_weight=0.5,
            denoise_path_component_weight=0.5,
            denoise_path_energy_weight=0.5,
        ),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.denoise_path_prefix.item() > 0
    assert losses.denoise_path_component.item() > 0
    assert losses.denoise_path_energy.item() > 0
    assert "denoise_path_prefix" in losses.as_dict()
    assert "denoise_path_component" in losses.as_dict()
    assert "denoise_path_energy" in losses.as_dict()


def test_denoise_path_energy_loss_backpropagates_to_every_component() -> None:
    components = [
        torch.randn(2, 3, 16, 16, requires_grad=True)
        for _ in range(4)
    ]
    prefixes = []
    running = torch.zeros_like(components[0])
    for component in components:
        running = running + component
        prefixes.append(running)
    output = CoFiTokOutput(
        tokens=[torch.zeros(2, 1, 16, 16) for _ in components],
        components=components,
        prefix_epsilons=prefixes,
        epsilon=prefixes[-1],
    )
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    clean = torch.randn(2, 3, 16, 16)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([3, 7])
    noisy = schedule.add_noise(clean, noise, timesteps)

    losses = compute_losses(
        LossConfig(
            epsilon_weight=0.0,
            prefix_weight=0.0,
            monotonic_weight=0.0,
            zero_token_weight=0.0,
            denoise_path_energy_weight=1.0,
        ),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )
    losses.total.backward()

    assert losses.denoise_path_energy.item() > 0
    assert all(component.grad is not None for component in components)
    assert all(torch.isfinite(component.grad).all() for component in components)


def test_component_energy_distribution_modes_match_and_penalize_collapse() -> None:
    target = [
        torch.full((2, 1, 2, 2), 0.2),
        torch.full((2, 1, 2, 2), 0.4),
        torch.full((2, 1, 2, 2), 0.8),
    ]
    matched = [component.clone() for component in target]
    collapsed = [
        torch.full((2, 1, 2, 2), 1e-3),
        torch.full((2, 1, 2, 2), 1e-3),
        torch.full((2, 1, 2, 2), 1.0),
    ]

    assert _component_energy_distribution_loss(matched, target, "mse").item() == 0.0
    assert _component_energy_distribution_loss(matched, target, "hellinger").item() == 0.0
    mse = _component_energy_distribution_loss(collapsed, target, "mse")
    hellinger = _component_energy_distribution_loss(collapsed, target, "hellinger")

    assert torch.isfinite(mse)
    assert torch.isfinite(hellinger)
    assert hellinger.item() > mse.item()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_denoise_path_hellinger_energy_backpropagates_finite_gradients(
    dtype: torch.dtype,
) -> None:
    components = [
        torch.full((2, 3, 8, 8), value, dtype=dtype, requires_grad=True)
        for value in (1e-3, 0.1, 1.0)
    ]
    targets = [
        torch.full((2, 3, 8, 8), value, dtype=dtype)
        for value in (0.2, 0.4, 0.8)
    ]

    loss = _component_energy_distribution_loss(components, targets, "hellinger")
    loss.backward()

    assert loss.item() > 0.0
    assert all(component.grad is not None for component in components)
    assert all(torch.isfinite(component.grad).all() for component in components)


def test_stable_hellinger_bounds_near_zero_component_gradients() -> None:
    components = [
        torch.full((2, 1, 4, 4), value, requires_grad=True)
        for value in (1e-8, 0.1, 1.0)
    ]
    targets = [
        torch.full((2, 1, 4, 4), value)
        for value in (0.3, 0.3, 0.4)
    ]

    loss = _component_energy_distribution_loss(
        components,
        targets,
        "hellinger_stable",
    )
    loss.backward()

    assert loss.item() > 0.0
    assert all(component.grad is not None for component in components)
    assert all(torch.isfinite(component.grad).all() for component in components)
    assert max(float(component.grad.abs().max()) for component in components) < 10.0


def test_unknown_denoise_path_energy_mode_is_rejected() -> None:
    components = [torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2)]

    with pytest.raises(ValueError, match="Unknown denoise path energy mode"):
        _component_energy_distribution_loss(components, components, "unknown")


def test_component_energy_capacity_prior_changes_collapsed_target() -> None:
    components = [
        torch.full((2, 1, 4, 4), value, requires_grad=True)
        for value in (1.0, 1.0, 4.0)
    ]
    targets = [component.detach().clone() for component in components]
    tokens = [
        torch.zeros(2, 1, 1, 1),
        torch.zeros(2, 1, 2, 2),
        torch.zeros(2, 1, 4, 4),
    ]

    path_loss = _component_energy_distribution_loss(
        components,
        targets,
        "hellinger",
    )
    capacity_loss = _component_energy_distribution_loss(
        components,
        targets,
        "hellinger",
        tokens=tokens,
        output_channels=1,
        capacity_weight=1.0,
        capacity_power=0.5,
    )

    assert path_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert capacity_loss.item() > 0.0
    capacity_loss.backward()
    assert all(component.grad is not None for component in components)


def test_low_snr_high_frequency_loss_emphasizes_noisy_timesteps() -> None:
    schedule = DiffusionSchedule(
        DiffusionConfig(num_train_timesteps=16, schedule_type="cosine"),
        device="cpu",
    )
    rows = torch.arange(8).view(8, 1)
    columns = torch.arange(8).view(1, 8)
    checkerboard = ((rows + columns) % 2).float().mul(2.0).sub(1.0)
    checkerboard = checkerboard.view(1, 1, 8, 8)
    output = CoFiTokOutput(
        tokens=[torch.zeros_like(checkerboard)],
        components=[torch.zeros_like(checkerboard)],
        prefix_epsilons=[torch.zeros_like(checkerboard)],
        epsilon=torch.zeros_like(checkerboard),
    )
    early = _low_snr_high_frequency_loss(
        output,
        checkerboard,
        schedule,
        torch.tensor([1]),
        power=1.0,
    )
    late = _low_snr_high_frequency_loss(
        output,
        checkerboard,
        schedule,
        torch.tensor([15]),
        power=1.0,
    )

    assert late.item() > early.item()
