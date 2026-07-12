import torch

from cofitok.configs import DiffusionConfig, ModelConfig
from cofitok.diffusion import DiffusionSchedule
from cofitok.models import CoFiTokTiny


def test_cofitok_tiny_forward_shapes() -> None:
    model_config = ModelConfig(
        image_channels=3,
        image_size=32,
        token_count=4,
        token_channels=8,
        base_channels=16,
        predictor_depth=1,
    )
    model = CoFiTokTiny(model_config)
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([1, 2])
    output = model(images, timesteps)
    assert len(output.tokens) == 4
    assert len(output.components) == 4
    assert len(output.prefix_epsilons) == 4
    assert output.epsilon.shape == images.shape


def test_cofitok_tiny_supports_simultaneous_token_prediction() -> None:
    model_config = ModelConfig(
        image_channels=3,
        image_size=32,
        token_count=4,
        token_channels=8,
        base_channels=16,
        predictor_depth=1,
        predictor_use_feedback=False,
    )
    model = CoFiTokTiny(model_config)
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([1, 2])
    output = model(images, timesteps)

    assert model.predictor.use_feedback is False
    assert len(model.predictor.feedback) == 0
    assert len(output.tokens) == 4
    assert output.epsilon.shape == images.shape


def test_cofitok_tiny_supports_multiscale_predictor() -> None:
    model_config = ModelConfig(
        image_channels=3,
        image_size=32,
        token_count=4,
        token_channels=8,
        base_channels=16,
        predictor_type="multiscale_unet",
        predictor_depth=1,
        predictor_multiscale_levels=2,
    )
    model = CoFiTokTiny(model_config)
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([1, 2])
    output = model(images, timesteps)

    assert len(output.tokens) == 4
    assert all(token.shape == (2, 8, 32, 32) for token in output.tokens)
    assert output.epsilon.shape == images.shape


def test_cofitok_tiny_supports_deep_synthesis_ablation() -> None:
    model_config = ModelConfig(
        image_channels=3,
        image_size=32,
        token_count=4,
        token_channels=8,
        base_channels=16,
        predictor_depth=1,
        synthesis_mode="deep_decoder",
        deep_synthesis_hidden_channels=12,
        deep_synthesis_depth=3,
    )
    model = CoFiTokTiny(model_config)
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([1, 2])
    output = model(images, timesteps)

    assert len(output.components) == 4
    assert output.epsilon.shape == images.shape


def test_model_supports_monolithic_dense_epsilon_baseline() -> None:
    model_config = ModelConfig(
        image_channels=3,
        image_size=32,
        token_count=1,
        token_channels=3,
        base_channels=16,
        predictor_depth=1,
        predictor_use_feedback=False,
        synthesis_mode="dense_identity",
    )
    model = CoFiTokTiny(model_config)
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([1, 2])
    output = model(images, timesteps)

    assert len(output.components) == 1
    assert output.tokens[0] is output.components[0]
    assert output.epsilon.shape == images.shape


def test_diffusion_schedule_roundtrip_shape() -> None:
    schedule = DiffusionSchedule(DiffusionConfig(num_train_timesteps=10), device="cpu")
    images = torch.randn(2, 3, 32, 32)
    noise = torch.randn_like(images)
    timesteps = torch.tensor([1, 2])
    noisy = schedule.add_noise(images, noise, timesteps)
    x0 = schedule.predict_x0_from_epsilon(noisy, noise, timesteps)
    assert x0.shape == images.shape
    assert torch.isfinite(x0).all()
