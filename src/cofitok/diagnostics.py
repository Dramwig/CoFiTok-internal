from __future__ import annotations

import torch
import torch.nn.functional as F

from cofitok.models import CoFiTokTiny


def shuffle_tokens_across_batch(tokens: list[torch.Tensor]) -> list[torch.Tensor]:
    shuffled_tokens = []
    for token in tokens:
        if token.shape[0] > 1:
            shuffled_tokens.append(token.flip(0))
        else:
            shuffled_tokens.append(token)
    return shuffled_tokens


@torch.no_grad()
def run_synthesis_diagnostics(model: CoFiTokTiny, tokens: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    components = model.synthesis(tokens)
    zero_components = model.synthesis.zero_components_like(tokens)
    random_tokens = [torch.randn_like(token) for token in tokens]
    random_components = model.synthesis(random_tokens)
    shuffled_components = model.synthesis(shuffle_tokens_across_batch(tokens))

    zero_energy = torch.stack([component.pow(2).mean() for component in zero_components]).mean()
    random_energy = torch.stack([component.pow(2).mean() for component in random_components]).mean()
    original_energy = torch.stack([component.pow(2).mean() for component in components]).mean()
    shuffle_delta = torch.stack(
        [F.mse_loss(shuffled, original) for shuffled, original in zip(shuffled_components, components)]
    ).mean()
    normalizer = original_energy.clamp_min(1e-12)

    return {
        "zero_token_component_energy": zero_energy,
        "zero_token_component_energy_ratio": zero_energy / normalizer,
        "random_token_component_energy": random_energy,
        "random_token_component_energy_ratio": random_energy / normalizer,
        "original_component_energy": original_energy,
        "shuffled_component_mse": shuffle_delta,
        "shuffled_component_relative_mse": shuffle_delta / normalizer,
    }
