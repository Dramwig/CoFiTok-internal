import torch

from cofitok.configs import ModelConfig
from cofitok.diagnostics import run_synthesis_diagnostics, shuffle_tokens_across_batch
from cofitok.models import CoFiTokTiny


def test_shuffle_tokens_across_batch_flips_each_token_batch() -> None:
    tokens = [torch.arange(2 * 3 * 1 * 1).view(2, 3, 1, 1)]

    shuffled = shuffle_tokens_across_batch(tokens)

    assert torch.equal(shuffled[0][0], tokens[0][1])
    assert torch.equal(shuffled[0][1], tokens[0][0])


def test_synthesis_diagnostics_report_normalized_shuffle_metrics() -> None:
    torch.manual_seed(11)
    model = CoFiTokTiny(
        ModelConfig(
            image_channels=3,
            image_size=8,
            token_count=2,
            token_channels=4,
            base_channels=8,
        )
    )
    tokens = [torch.randn(2, 4, 8, 8) for _ in range(2)]

    diagnostics = run_synthesis_diagnostics(model, tokens)

    assert diagnostics["zero_token_component_energy"].item() == 0.0
    assert diagnostics["zero_token_component_energy_ratio"].item() == 0.0
    assert diagnostics["random_token_component_energy_ratio"].item() >= 0.0
    assert diagnostics["shuffled_component_relative_mse"].item() >= 0.0
