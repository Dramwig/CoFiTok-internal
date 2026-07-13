import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from cofitok.configs import ModelConfig, load_config
from cofitok.data.sampler import StatefulRandomSampler
from cofitok.models import CoFiTokTiny, ScalableUNetTokenPredictor
from cofitok.training.checkpointing import (
    backfill_training_checkpoint_integrity,
    checkpoint_integrity_path,
    load_training_checkpoint,
    prune_checkpoints,
    resolve_latest_checkpoint,
    save_training_checkpoint,
    verify_training_checkpoint,
)
from cofitok.training.ema import ExponentialMovingAverage
from cofitok.training.runtime import build_warmup_cosine_scheduler
from scripts.validate_generation_configs import validate_pair


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
        model.predictor.token_heads[0].weight.normal_(std=0.01)
    images = torch.randn(2, 3, 16, 16)
    timesteps = torch.tensor([2, 3])
    labels = torch.tensor([1, 2])

    output = model(images, timesteps, class_labels=labels)
    unconditional = model(images, timesteps, class_labels=labels, force_unconditional=True)
    explicit_unconditional = model(
        images,
        timesteps,
        class_labels=torch.full_like(labels, model.config.num_classes),
    )

    assert isinstance(model.predictor, ScalableUNetTokenPredictor)
    assert len(output.tokens) == 4
    assert output.epsilon.shape == images.shape
    assert not torch.equal(output.tokens[0], unconditional.tokens[0])
    torch.testing.assert_close(
        explicit_unconditional.epsilon,
        unconditional.epsilon,
        rtol=0.0,
        atol=0.0,
    )
    zero_components = model.synthesis.zero_components_like(output.tokens)
    assert all(torch.count_nonzero(component) == 0 for component in zero_components)


@pytest.mark.parametrize("synthesis_mode", ["restricted", "dense_identity"])
def test_scalable_generation_heads_start_at_zero_and_receive_gradient(
    synthesis_mode: str,
) -> None:
    config = _small_model_config()
    if synthesis_mode == "dense_identity":
        config = replace(
            config,
            token_count=1,
            token_channels=3,
            predictor_use_feedback=False,
            synthesis_mode="dense_identity",
            synthesis_active_token_channels=[],
        )
    model = CoFiTokTiny(config)
    images = torch.randn(2, 3, 16, 16)
    timesteps = torch.tensor([2, 3])
    labels = torch.tensor([1, 2])
    target = torch.randn_like(images)

    initial = model(images, timesteps, class_labels=labels)
    assert torch.count_nonzero(initial.epsilon) == 0
    assert all(torch.count_nonzero(head.weight) == 0 for head in model.predictor.token_heads)
    assert all(torch.count_nonzero(head.bias) == 0 for head in model.predictor.token_heads)

    torch.nn.functional.mse_loss(initial.epsilon, target).backward()
    head_gradient = sum(
        float(head.weight.grad.abs().sum())
        for head in model.predictor.token_heads
        if head.weight.grad is not None
    )
    assert head_gradient > 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.step()
    learned = model(images, timesteps, class_labels=labels)
    assert torch.count_nonzero(learned.epsilon) > 0


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


def test_full_generation_configs_keep_matched_runtime_and_checkpoint_cadence() -> None:
    cofitok = load_config("configs/generation/imagenet256_cofitok_k8_300k.json")
    dense = load_config("configs/generation/imagenet256_dense_300k.json")

    assert cofitok.data == dense.data
    assert cofitok.diffusion == dense.diffusion
    assert cofitok.runtime == dense.runtime
    assert cofitok.optimization == dense.optimization
    assert cofitok.data.random_horizontal_flip_prob == 0.5
    assert cofitok.runtime.steps == 300_000
    assert cofitok.runtime.checkpoint_interval == 5_000
    assert cofitok.runtime.keep_last_checkpoints == 3
    assert cofitok.runtime.protected_checkpoint_steps == [50_000, 100_000, 200_000, 300_000]
    assert dense.runtime.protected_checkpoint_steps == cofitok.runtime.protected_checkpoint_steps

    runbook = Path("artifacts/runbooks/generation_full_matched_300k_after_gate.sh").read_text(
        encoding="utf-8"
    )
    assert runbook.count("--required-checkpoint-steps 50000,100000,200000,300000") == 2
    assert "cofitok_training_audit.json" in runbook
    assert "dense_training_audit.json" in runbook
    assert "select_generation_training_runtime.py" in runbook
    assert "validate_generation_configs.py" in runbook
    assert "--stage full" in runbook
    assert "config_recipe.json" in runbook
    assert "--candidates 16x4,32x2,64x1" in runbook
    assert "SELECTED_MICRO_BATCH * SELECTED_ACCUMULATION != 64" in runbook
    assert '--micro-batch-size "$SELECTED_MICRO_BATCH"' in runbook
    assert '--gradient-accumulation-steps "$SELECTED_ACCUMULATION"' in runbook
    assert "paired_milestone_complete" in runbook
    assert "paired milestone %s already complete; skipping" in runbook
    assert 'checkpoint_tag.integrity.json' in runbook
    assert "passed milestone %s without protected checkpoint" in runbook
    assert "monitor_generation_pair.py" in runbook
    assert "generation_full_matched_300k_monitor.json" in runbook
    assert "--checkpoint-interval 5000" in runbook
    assert runbook.count("snapshot_full_monitor") == 4
    assert "monitor_report_passes" in runbook
    assert "published pass and exited" in runbook

    preflight = validate_pair(cofitok, dense, max_parameter_gap=0.02)
    assert preflight["status"] == "pass"
    assert preflight["matched_runtime"]["steps"] == 300_000
    assert preflight["matched_runtime"]["protected_checkpoint_steps"] == [
        50_000,
        100_000,
        200_000,
        300_000,
    ]
    assert preflight["matched_optimization"]["gradient_accumulation_steps"] == 4


def test_formal_sampling_runbooks_select_one_shared_batch() -> None:
    for path, sample_count in (
        ("artifacts/runbooks/generation_10pct_posteval_2026-07-12.sh", "10000"),
        ("artifacts/runbooks/generation_full_posteval_50k.sh", "50000"),
    ):
        runbook = Path(path).read_text(encoding="utf-8")
        assert "select_generation_sampling_batch.py" in runbook
        assert "--candidates 16,32,64,128" in runbook
        assert "--max-memory-fraction 0.90" in runbook
        assert runbook.count("--sampling-output-dir") == 2
        assert "git diff --quiet" in runbook
        assert "git diff --cached --quiet" in runbook
        assert runbook.count('--batch-size "$SAMPLING_BATCH"') == 2
        assert runbook.count(f"--num-samples {sample_count}") == 2
        assert "build_generation_visual_audit.py" in runbook
        assert "--prefix-indices 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15" in runbook


def test_inference_export_runbook_verifies_and_smoke_tests_both_methods() -> None:
    runbook = Path(
        "artifacts/runbooks/generation_export_inference_artifacts.sh"
    ).read_text(encoding="utf-8")

    assert runbook.count("export_generation_inference_artifact.py") == 2
    assert runbook.count("preflight_generation_sampling.py") == 2
    assert runbook.count("infer_generation.py") == 2
    assert "cofitok_k8_ema_inference.pt" in runbook
    assert "dense_identity_ema_inference.pt" in runbook
    assert "--prefix-budgets 1,8" in runbook
    assert "--sample-steps 10" in runbook


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
    invalid_path = tmp_path / "invalid_checkpoint.pt"
    with pytest.raises(ValueError, match="must contain exactly"):
        save_training_checkpoint(
            invalid_path,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            step=1,
            config={"name": "test"},
            extra_state={"git": {"revision": "a" * 40}},
        )
    assert not invalid_path.exists()
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
        extra_state={
            "marker": 9,
            "git": {
                "revision": "a" * 40,
                "branch": "scale/generative-system",
                "dirty": False,
            },
        },
    )
    integrity = verify_training_checkpoint(path)
    latest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
    assert integrity["checkpoint"] == path.name
    assert integrity["checkpoint_bytes"] == path.stat().st_size
    assert len(integrity["checkpoint_sha256"]) == 64
    assert latest["checkpoint_sha256"] == integrity["checkpoint_sha256"]
    assert latest["integrity_manifest"] == checkpoint_integrity_path(path).name
    assert latest["git_revision"] == "a" * 40
    assert resolve_latest_checkpoint(tmp_path) == path
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
    pristine_parameters = [
        parameter.detach().clone() for parameter in restored_model.parameters()
    ]
    with pytest.raises(ValueError, match=r"config\.name"):
        load_training_checkpoint(
            path,
            model=restored_model,
            expected_config={"name": "changed"},
            restore_rng=False,
        )
    with pytest.raises(ValueError, match="differs from expected revision"):
        load_training_checkpoint(
            path,
            model=restored_model,
            expected_config={"name": "test"},
            expected_git_provenance={
                "revision": "b" * 40,
                "branch": "scale/generative-system",
                "dirty": False,
            },
            restore_rng=False,
        )
    for pristine, current in zip(
        pristine_parameters, restored_model.parameters(), strict=True
    ):
        assert torch.equal(pristine, current)

    checkpoint = load_training_checkpoint(
        path,
        model=restored_model,
        ema=restored_ema,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_config={"name": "test"},
    )
    actual = (random.random(), float(np.random.rand()), float(torch.rand(())))

    assert checkpoint["step"] == 1
    assert checkpoint["extra_state"]["marker"] == 9
    assert actual == expected
    for original, restored in zip(model.parameters(), restored_model.parameters()):
        assert torch.equal(original, restored)
    assert restored_scheduler.state_dict() == scheduler.state_dict()
    assert restored_ema.num_updates == ema.num_updates

    latest["step"] = 2
    (tmp_path / "latest.json").write_text(json.dumps(latest), encoding="utf-8")
    with pytest.raises(ValueError, match="step does not match"):
        resolve_latest_checkpoint(tmp_path)

    checkpoint_bytes = bytearray(path.read_bytes())
    checkpoint_bytes[len(checkpoint_bytes) // 2] ^= 1
    path.write_bytes(checkpoint_bytes)
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_training_checkpoint(path)


def test_prune_checkpoints_removes_matching_integrity_manifests(tmp_path) -> None:
    old = tmp_path / "checkpoint_step_00000001.pt"
    latest = tmp_path / "checkpoint_step_00000002.pt"
    for path in (old, latest):
        path.write_bytes(b"checkpoint")
        checkpoint_integrity_path(path).write_text("{}\n", encoding="utf-8")

    removed = prune_checkpoints(tmp_path, keep_last=1)

    assert removed == [old]
    assert not old.exists()
    assert not checkpoint_integrity_path(old).exists()
    assert latest.exists()
    assert checkpoint_integrity_path(latest).exists()


def test_prune_checkpoints_keeps_protected_milestones_and_recent_recovery_points(
    tmp_path,
) -> None:
    paths = []
    for step in (50_000, 90_000, 95_000, 100_000, 105_000):
        path = tmp_path / f"checkpoint_step_{step:08d}.pt"
        path.write_bytes(b"checkpoint")
        checkpoint_integrity_path(path).write_text("{}\n", encoding="utf-8")
        paths.append(path)

    removed = prune_checkpoints(
        tmp_path,
        keep_last=2,
        protected_steps=[50_000, 100_000],
    )

    assert removed == paths[1:3]
    assert {path.name for path in tmp_path.glob("checkpoint_step_*.pt")} == {
        "checkpoint_step_00050000.pt",
        "checkpoint_step_00100000.pt",
        "checkpoint_step_00105000.pt",
    }
    assert not checkpoint_integrity_path(paths[1]).exists()
    assert not checkpoint_integrity_path(paths[2]).exists()


def test_legacy_checkpoint_integrity_backfill_preserves_checkpoint_bytes(tmp_path) -> None:
    path = tmp_path / "checkpoint_step_00000017.pt"
    torch.save(
        {
            "format_version": 1,
            "step": 17,
            "config": {"name": "legacy"},
            "model": {},
            "ema": {},
            "optimizer": {},
            "scheduler": {},
            "rng_state": {},
            "extra_state": {"sampler": {"position": 3}},
        },
        path,
    )
    before = path.read_bytes()
    (tmp_path / "latest.json").write_text(
        json.dumps({"checkpoint": path.name, "step": 17}),
        encoding="utf-8",
    )

    integrity = backfill_training_checkpoint_integrity(path, update_latest=True)

    assert path.read_bytes() == before
    assert integrity["step"] == 17
    assert len(integrity["checkpoint_sha256"]) == 64
    assert verify_training_checkpoint(path) == integrity
    assert resolve_latest_checkpoint(tmp_path) == path


def test_legacy_checkpoint_backfill_requires_sampler_state(tmp_path) -> None:
    path = tmp_path / "checkpoint_step_00000017.pt"
    torch.save(
        {
            "format_version": 1,
            "step": 17,
            "config": {},
            "model": {},
            "ema": {},
            "optimizer": {},
            "scheduler": {},
            "rng_state": {},
            "extra_state": {},
        },
        path,
    )

    with pytest.raises(ValueError, match="sampler state"):
        backfill_training_checkpoint_integrity(path)
    assert not checkpoint_integrity_path(path).exists()
