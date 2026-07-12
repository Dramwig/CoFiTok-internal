import torch
from torch import nn

from cofitok.models.synthesis import DenseIdentitySynthesisBank, DeepSynthesis, DeepSynthesisBank, RestrictedSynthesis, RestrictedSynthesisBank, build_synthesis_bank


def test_restricted_synthesis_zero_token_maps_to_zero() -> None:
    module = RestrictedSynthesis(token_channels=8, image_channels=3)
    token = torch.zeros(2, 8, 16, 16)
    component = module(token)
    assert torch.count_nonzero(component) == 0


def test_restricted_synthesis_has_no_bias_terms() -> None:
    module = RestrictedSynthesis(token_channels=8, image_channels=3)
    assert module.proj.bias is None
    assert module.local.bias is None


def test_synthesis_bank_preserves_component_shapes() -> None:
    bank = RestrictedSynthesisBank(
        token_count=4,
        token_channels=8,
        image_channels=3,
        kernel_size=3,
        gamma_mode="learned_scalar",
    )
    tokens = [torch.randn(2, 8, 16, 16) for _ in range(4)]
    components = bank(tokens)
    assert len(components) == 4
    assert all(component.shape == (2, 3, 16, 16) for component in components)


def test_synthesis_bank_supports_token_strides() -> None:
    bank = RestrictedSynthesisBank(
        token_count=4,
        token_channels=8,
        image_channels=3,
        kernel_size=3,
        gamma_mode="learned_scalar",
        token_strides=[4, 2, 1, 1],
    )
    tokens = [torch.randn(2, 8, 17, 19) for _ in range(4)]
    components = bank(tokens)

    assert [module.token_stride for module in bank.synthesizers] == [4, 2, 1, 1]
    assert all(component.shape == (2, 3, 17, 19) for component in components)


def test_restricted_synthesis_masks_inactive_token_channels() -> None:
    module = RestrictedSynthesis(token_channels=4, image_channels=3, active_token_channels=2)
    with torch.no_grad():
        module.proj.weight.fill_(1.0)
        module.local.weight.zero_()
        module.local.weight[:, :, 1, 1] = 1.0
        module.gamma.fill_(1.0)
    active = torch.zeros(1, 4, 3, 3)
    inactive = torch.zeros(1, 4, 3, 3)
    active[:, :2] = 1.0
    inactive[:, 2:] = 1.0

    active_component = module(active)
    inactive_component = module(inactive)

    assert torch.count_nonzero(active_component) > 0
    assert torch.count_nonzero(inactive_component) == 0


def test_synthesis_bank_supports_active_token_channels() -> None:
    bank = RestrictedSynthesisBank(
        token_count=4,
        token_channels=16,
        image_channels=3,
        kernel_size=3,
        gamma_mode="learned_scalar",
        active_token_channels=[4, 8, 16, 16],
    )

    assert [module.active_token_channels for module in bank.synthesizers] == [4, 8, 16, 16]


def test_deep_synthesis_ablation_has_bias_and_nonlinearity() -> None:
    module = DeepSynthesis(token_channels=8, image_channels=3, hidden_channels=12, depth=3)

    convs = [layer for layer in module.net if isinstance(layer, nn.Conv2d)]
    activations = [layer for layer in module.net if isinstance(layer, nn.SiLU)]

    assert len(convs) == 3
    assert all(conv.bias is not None for conv in convs)
    assert len(activations) == 2


def test_deep_synthesis_ablation_can_violate_zero_token_constraint() -> None:
    module = DeepSynthesis(token_channels=8, image_channels=3, hidden_channels=12, depth=2)
    with torch.no_grad():
        for layer in module.net:
            if isinstance(layer, nn.Conv2d):
                layer.weight.zero_()
                layer.bias.zero_()
        final_conv = [layer for layer in module.net if isinstance(layer, nn.Conv2d)][-1]
        final_conv.bias.fill_(0.25)
        module.gamma.fill_(1.0)

    component = module(torch.zeros(2, 8, 16, 16))

    assert torch.count_nonzero(component) > 0


def test_deep_synthesis_bank_preserves_component_shapes() -> None:
    bank = DeepSynthesisBank(
        token_count=4,
        token_channels=8,
        image_channels=3,
        kernel_size=3,
        gamma_mode="learned_scalar",
        hidden_channels=12,
        depth=3,
    )
    tokens = [torch.randn(2, 8, 16, 16) for _ in range(4)]
    components = bank(tokens)

    assert len(components) == 4
    assert all(component.shape == (2, 3, 16, 16) for component in components)


def test_build_synthesis_bank_selects_restricted_default() -> None:
    bank = build_synthesis_bank(
        synthesis_mode="restricted",
        token_count=4,
        token_channels=8,
        image_channels=3,
        kernel_size=3,
        gamma_mode="learned_scalar",
    )

    assert isinstance(bank, RestrictedSynthesisBank)


def test_build_synthesis_bank_selects_deep_ablation() -> None:
    bank = build_synthesis_bank(
        synthesis_mode="deep_decoder",
        token_count=4,
        token_channels=8,
        image_channels=3,
        kernel_size=3,
        gamma_mode="learned_scalar",
        deep_hidden_channels=12,
        deep_depth=3,
    )

    assert isinstance(bank, DeepSynthesisBank)


def test_dense_identity_bank_is_a_direct_full_resolution_head() -> None:
    bank = build_synthesis_bank(
        synthesis_mode="dense_identity",
        token_count=1,
        token_channels=3,
        image_channels=3,
        kernel_size=3,
        gamma_mode="fixed_one",
    )
    dense = torch.randn(2, 3, 8, 8)

    assert isinstance(bank, DenseIdentitySynthesisBank)
    assert bank([dense])[0] is dense
    assert torch.count_nonzero(bank.zero_components_like([dense])[0]) == 0
