import torch

from cofitok.configs import DiffusionConfig, LossConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokOutput
from cofitok.training.losses import compute_losses


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
        LossConfig(denoise_path_prefix_weight=0.5, denoise_path_component_weight=0.5),
        output,
        schedule,
        noisy,
        clean,
        noise,
        timesteps,
    )

    assert losses.denoise_path_prefix.item() > 0
    assert losses.denoise_path_component.item() > 0
    assert "denoise_path_prefix" in losses.as_dict()
    assert "denoise_path_component" in losses.as_dict()
