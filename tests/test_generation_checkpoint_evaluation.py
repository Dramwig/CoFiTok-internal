from __future__ import annotations

import torch

from scripts.evaluate_generation_checkpoint import (
    component_orders,
    prefix_tensors,
    spatial_prefix_targets,
)


def test_component_orders_are_deterministic_and_deduplicated() -> None:
    first = component_orders(token_count=4, random_orders=16, seed=7)
    second = component_orders(token_count=4, random_orders=16, seed=7)

    assert first == second
    assert first["ordered"] == [0, 1, 2, 3]
    assert first["reverse"] == [3, 2, 1, 0]
    assert len({tuple(order) for order in first.values()}) == len(first)


def test_prefix_tensors_respect_requested_order() -> None:
    components = [torch.full((1, 1, 1, 1), float(value)) for value in (1, 2, 4)]

    prefixes = prefix_tensors(components, [2, 0, 1])

    assert [float(value) for value in prefixes] == [4.0, 5.0, 7.0]


def test_spatial_prefix_targets_end_at_clean_image() -> None:
    clean = torch.randn(2, 3, 16, 16)

    targets = spatial_prefix_targets(clean, count=4)

    assert len(targets) == 4
    assert torch.equal(targets[-1], clean)
    assert targets[0].shape == clean.shape
