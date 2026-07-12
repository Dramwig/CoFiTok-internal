import random

import numpy as np
import torch

from cofitok.configs import ModelConfig, load_config
from cofitok.data.sampler import StatefulRandomSampler
from cofitok.models import CoFiTokTiny, ScalableUNetTokenPredictor
from cofitok.training.checkpointing import load_training_checkpoint, save_training_checkpoint
from cofitok.training.ema import ExponentialMovingAverage
from cofitok.training.runtime import build_warmup_cosine_scheduler


def _small_model_config() -> ModelConfig:
    return ModelConfig(
        image_channels=3,
        image_size=16,
        token_count=4,
        token_channels=8,
        base_channels=16,
        predictor_type="scalable_unet",
        predictor_channel_multipliers=[1, 2],
        predictor_num_res_blocks=1,
        predictor_attention_resolutions=[8],
        predictor_num_heads=2,
        num_classes=10,
        class_dropout_prob=0.0,
        synthesis_active_token_channels=[2, 4, 6, 8],
    )


def test_scalable_predictor_keeps_conditioning_inside_tk() -> None:
    model = CoFiTokTiny(_small_model_config()).eval()
    with torch.no_grad():
        model.predictor.encoder[0].blocks[0].out[-1].weight.normal_(std=0.01)
    images = torch.randn(2, 3, 16, 16)
    timesteps = torch.tensor([2, 3])
    labels = torch.tensor([1, 2])

    output = model(images, timesteps, class_labels=labels)
    unconditional = model(images, timesteps, class_labels=labels, force_unconditional=True)

    assert isinstance(model.predictor, ScalableUNetTokenPredictor)
    assert len(output.tokens) == 4
    assert output.epsilon.shape == images.shape
    assert not torch.equal(output.tokens[0], unconditional.tokens[0])
    zero_components = model.synthesis.zero_components_like(output.tokens)
    assert all(torch.count_nonzero(component) == 0 for component in zero_components)


def test_generation_configs_form_a_matched_backbone_pair() -> None:
    cofitok = load_config("configs/generation/imagenet256_10pct_cofitok_k8_50k.json")
    dense = load_config("configs/generation/imagenet256_10pct_dense_50k.json")

    assert cofitok.data == dense.data
    assert cofitok.diffusion == dense.diffusion
    assert cofitok.runtime == dense.runtime
    assert cofitok.optimization == dense.optimization
    assert cofitok.model.predictor_type == dense.model.predictor_type == "scalable_unet"
    assert cofitok.model.base_channels == dense.model.base_channels
    assert cofitok.model.predictor_channel_multipliers == dense.model.predictor_channel_multipliers
    assert cofitok.model.synthesis_mode == "restricted"
    assert dense.model.synthesis_mode == "dense_identity"


def test_stateful_sampler_restores_consumed_not_prefetched_position() -> None:
    dataset = list(range(20))
    sampler = StatefulRandomSampler(dataset, seed=7)
    iterator = iter(sampler)
    issued = [next(iterator) for _ in range(8)]
    sampler.mark_consumed(3)
    state = sampler.state_dict()

    resumed = StatefulRandomSampler(dataset, seed=999)
    resumed.load_state_dict(state)

    assert list(iter(resumed))[:5] == issued[3:8]


def test_checkpoint_roundtrip_restores_all_training_and_rng_state(tmp_path) -> None:
    model = CoFiTokTiny(_small_model_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_steps=10,
        warmup_steps=1,
        min_learning_rate=1e-4,
    )
    ema = ExponentialMovingAverage(model, decay=0.99, warmup_steps=1)
    images = torch.randn(2, 3, 16, 16)
    output = model(images, torch.tensor([1, 2]), class_labels=torch.tensor([3, 4]))
    output.epsilon.square().mean().backward()
    optimizer.step()
    scheduler.step()
    ema.update(model)
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    path = tmp_path / "checkpoint_step_00000001.pt"
    save_training_checkpoint(
        path,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=None,
        step=1,
        config={"name": "test"},
        extra_state={"marker": 9},
    )
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))

    restored_model = CoFiTokTiny(_small_model_config())
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scheduler = build_warmup_cosine_scheduler(
        restored_optimizer,
        total_steps=10,
        warmup_steps=1,
        min_learning_rate=1e-4,
    )
    restored_ema = ExponentialMovingAverage(restored_model, decay=0.5)
    checkpoint = load_training_checkpoint(
        path,
        model=restored_model,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))

    assert checkpoint["step"] == 1
    assert checkpoint["extra_state"]["marker"] == 9
    assert actual == expected
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_ema.num_updates == ema.num_updates
