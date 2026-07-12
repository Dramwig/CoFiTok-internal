import pytest
import torch

from cofitok.models.predictors import MultiScaleTokenPredictor, TinyTokenPredictor, build_token_predictor


def test_build_token_predictor_selects_tiny_default() -> None:
    predictor = build_token_predictor(
        predictor_type="tiny_conv",
        image_channels=3,
        token_count=2,
        token_channels=4,
        base_channels=8,
        depth=1,
    )

    assert isinstance(predictor, TinyTokenPredictor)


def test_multiscale_token_predictor_outputs_full_resolution_tokens() -> None:
    predictor = MultiScaleTokenPredictor(
        image_channels=3,
        token_count=3,
        token_channels=4,
        base_channels=8,
        depth=1,
        levels=2,
    )
    images = torch.randn(2, 3, 32, 32)
    timesteps = torch.tensor([1, 2])

    tokens = predictor(images, timesteps)

    assert len(tokens) == 3
    assert all(token.shape == (2, 4, 32, 32) for token in tokens)


def test_build_token_predictor_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unknown predictor_type"):
        build_token_predictor(
            predictor_type="mystery",
            image_channels=3,
            token_count=2,
            token_channels=4,
            base_channels=8,
            depth=1,
        )
