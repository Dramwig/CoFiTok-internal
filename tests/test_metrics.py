import math

import torch

from cofitok.metrics import (
    energy_distribution_metrics,
    frechet_distance_from_features,
    lowres_image_features,
    normalized_curve_auc,
    psnr_from_mse,
)


def test_normalized_curve_auc_uses_equal_spaced_trapezoids() -> None:
    assert normalized_curve_auc([]) == 0.0
    assert normalized_curve_auc([3.0]) == 3.0
    assert normalized_curve_auc([1.0, 3.0, 5.0]) == 3.0


def test_energy_distribution_metrics_separate_collapse_from_uniform() -> None:
    collapsed = energy_distribution_metrics([1.0, 0.0, 0.0, 0.0])
    uniform = energy_distribution_metrics([0.25, 0.25, 0.25, 0.25])

    assert collapsed["energy_entropy"] == 0.0
    assert collapsed["energy_entropy_normalized"] == 0.0
    assert collapsed["energy_effective_token_count"] == 1.0
    assert math.isclose(uniform["energy_entropy_normalized"], 1.0)
    assert math.isclose(uniform["energy_effective_token_count"], 4.0)


def test_psnr_from_mse_uses_project_image_range() -> None:
    assert math.isinf(psnr_from_mse(0.0))
    assert math.isclose(psnr_from_mse(1.0), 20.0 * math.log10(2.0))


def test_lowres_features_are_deterministic_and_include_color_stats() -> None:
    images = torch.zeros(2, 3, 16, 16)
    features = lowres_image_features(images, feature_size=4)

    assert features.shape == (2, 3 * 4 * 4 + 6)
    assert torch.equal(features, lowres_image_features(images, feature_size=4))


def test_frechet_distance_is_zero_for_identical_features() -> None:
    features = torch.tensor([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]])

    assert frechet_distance_from_features(features, features) < 1e-8
