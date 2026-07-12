from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F


def normalized_curve_auc(values: Sequence[float]) -> float:
    """Return equal-spaced trapezoid AUC on the scale of the input values."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    total = 0.0
    for left, right in zip(values[:-1], values[1:]):
        total += 0.5 * (float(left) + float(right))
    return total / (len(values) - 1)


def energy_distribution_metrics(ratios: Sequence[float]) -> dict[str, float]:
    total = sum(max(float(ratio), 0.0) for ratio in ratios)
    if total <= 0.0:
        return {
            "energy_entropy": 0.0,
            "energy_entropy_normalized": 0.0,
            "energy_effective_token_count": 0.0,
        }
    probabilities = [max(float(ratio), 0.0) / total for ratio in ratios]
    entropy = -sum(probability * math.log(probability) for probability in probabilities if probability > 0.0)
    if len(probabilities) > 1:
        normalized = entropy / math.log(len(probabilities))
    else:
        normalized = 0.0
    return {
        "energy_entropy": entropy,
        "energy_entropy_normalized": normalized,
        "energy_effective_token_count": math.exp(entropy),
    }


def psnr_from_mse(mse: float, max_value: float = 2.0) -> float:
    if mse <= 0.0:
        return math.inf
    return 20.0 * math.log10(max_value) - 10.0 * math.log10(float(mse))


def lowres_image_features(images: torch.Tensor, feature_size: int = 8) -> torch.Tensor:
    """Extract deterministic no-download image features for quick rFID proxies.

    Inputs are expected in the project image range [-1, 1]. Features are not a
    substitute for Inception/FID; they are a lightweight relative signal for
    small validation sweeps.
    """

    if feature_size < 1:
        raise ValueError(f"feature_size must be >= 1, got {feature_size}")
    images = (images.detach().clamp(-1.0, 1.0) + 1.0) * 0.5
    pooled = F.adaptive_avg_pool2d(images, output_size=(feature_size, feature_size))
    channel_mean = images.mean(dim=(2, 3))
    channel_std = images.std(dim=(2, 3), unbiased=False)
    return torch.cat([pooled.flatten(1), channel_mean, channel_std], dim=1).to(dtype=torch.float64)


def feature_mean_and_covariance(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    features = features.to(dtype=torch.float64)
    if features.ndim != 2:
        raise ValueError(f"Expected [N, D] feature matrix, got shape {list(features.shape)}")
    if features.shape[0] == 0:
        raise ValueError("Cannot compute feature statistics for an empty feature matrix")
    mean = features.mean(dim=0)
    if features.shape[0] == 1:
        covariance = torch.zeros(
            features.shape[1],
            features.shape[1],
            dtype=features.dtype,
            device=features.device,
        )
    else:
        centered = features - mean
        covariance = centered.T.matmul(centered) / (features.shape[0] - 1)
    return mean, covariance


def _matrix_sqrt_psd(matrix: torch.Tensor) -> torch.Tensor:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = torch.linalg.eigh(symmetric)
    eigenvalues = eigenvalues.clamp_min(0.0).sqrt()
    return (eigenvectors * eigenvalues.unsqueeze(0)).matmul(eigenvectors.T)


def frechet_distance_from_features(
    reference_features: torch.Tensor,
    candidate_features: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute a Fréchet distance between two feature distributions."""

    reference_mean, reference_cov = feature_mean_and_covariance(reference_features)
    candidate_mean, candidate_cov = feature_mean_and_covariance(candidate_features)
    identity = torch.eye(reference_cov.shape[0], dtype=reference_cov.dtype, device=reference_cov.device)
    reference_cov = reference_cov + eps * identity
    candidate_cov = candidate_cov + eps * identity
    reference_sqrt = _matrix_sqrt_psd(reference_cov)
    middle = reference_sqrt.matmul(candidate_cov).matmul(reference_sqrt)
    covmean = _matrix_sqrt_psd(middle)
    mean_delta = reference_mean - candidate_mean
    distance = mean_delta.dot(mean_delta) + torch.trace(reference_cov + candidate_cov - 2.0 * covmean)
    return float(distance.clamp_min(0.0).detach().cpu().item())
