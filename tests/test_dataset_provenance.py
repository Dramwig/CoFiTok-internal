from __future__ import annotations

import copy
import os
from types import SimpleNamespace

import pytest
import torch

from cofitok.data.provenance import (
    FORMAL_GENERATION_DATASETS,
    FormalDatasetProvenanceSpec,
    capture_dataset_provenance,
    dataset_provenance_identity_sha256,
    validate_dataset_provenance,
)
from cofitok.reporting import file_sha256
from cofitok.training import ExponentialMovingAverage
from cofitok.training.checkpointing import (
    load_training_checkpoint,
    save_training_checkpoint,
    verify_training_checkpoint,
)


def _fixture_provenance(tmp_path, monkeypatch) -> tuple[dict, SimpleNamespace]:
    dataset_root = tmp_path / "fixture_imagenet"
    manifest = dataset_root / "metadata" / "image_manifest.jsonl"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        '{"path":"extracted/train/a.jpg","split":"train"}\n'
        '{"path":"extracted/val/b.jpg","split":"val"}\n',
        encoding="utf-8",
    )
    monkeypatch.setitem(
        FORMAL_GENERATION_DATASETS,
        "fixture_imagenet",
        FormalDatasetProvenanceSpec(
            dataset="fixture_imagenet",
            manifest_sha256=file_sha256(manifest),
            manifest_bytes=manifest.stat().st_size,
            train_images=1,
            val_images=1,
        ),
    )
    config = SimpleNamespace(dataset="fixture_imagenet", root=str(tmp_path))
    return capture_dataset_provenance(config, train_images=1, val_images=1), config


def test_capture_and_validate_formal_dataset_provenance(tmp_path, monkeypatch) -> None:
    report, _ = _fixture_provenance(tmp_path, monkeypatch)

    evidence = validate_dataset_provenance(
        report,
        expected_dataset="fixture_imagenet",
    )

    assert report["status"] == "pass"
    assert evidence["identity_sha256"] == report["identity_sha256"]
    assert evidence["splits"] == {"train": 1, "val": 1}


def test_capture_rejects_manifest_or_split_drift(tmp_path, monkeypatch) -> None:
    report, config = _fixture_provenance(tmp_path, monkeypatch)
    assert report["status"] == "pass"
    manifest = tmp_path / "fixture_imagenet" / "metadata" / "image_manifest.jsonl"
    manifest.write_text(manifest.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    changed = capture_dataset_provenance(config, train_images=2, val_images=1)

    assert changed["status"] == "fail"
    assert any("manifest" in issue for issue in changed["issues"])
    assert any("train images" in issue for issue in changed["issues"])


def test_validator_rejects_forged_dataset_identity(tmp_path, monkeypatch) -> None:
    report, _ = _fixture_provenance(tmp_path, monkeypatch)
    report["dataset_root"] = str(tmp_path / "another-copy")

    with pytest.raises(ValueError, match="identity SHA256"):
        validate_dataset_provenance(report, expected_dataset="fixture_imagenet")


def test_checkpoint_binds_dataset_before_deserialization(tmp_path, monkeypatch) -> None:
    provenance, _ = _fixture_provenance(tmp_path, monkeypatch)
    model = torch.nn.Linear(2, 2)
    ema = ExponentialMovingAverage(model, warmup_steps=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "checkpoint_step_00000001.pt"
    config = {"data": {"dataset": "fixture_imagenet"}}
    save_training_checkpoint(
        checkpoint,
        model=model,
        ema=ema,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        step=1,
        config=config,
        extra_state={"dataset_provenance": provenance},
    )
    integrity = verify_training_checkpoint(checkpoint)
    assert integrity["dataset_identity_sha256"] == provenance["identity_sha256"]

    drifted = copy.deepcopy(provenance)
    drifted["dataset_root"] = str(tmp_path / "another-copy")
    drifted["identity_sha256"] = dataset_provenance_identity_sha256(drifted)
    real_torch_load = torch.load

    def fail_if_deserialized(*args, **kwargs):
        raise AssertionError("checkpoint was deserialized before dataset validation")

    monkeypatch.setattr(torch, "load", fail_if_deserialized)
    with pytest.raises(ValueError, match="dataset identity differs"):
        load_training_checkpoint(
            checkpoint,
            model=torch.nn.Linear(2, 2),
            expected_config=config,
            expected_dataset_provenance=drifted,
        )
    monkeypatch.setattr(torch, "load", real_torch_load)


@pytest.mark.skipif(
    os.name == "nt",
    reason="Unix symlink creation requires elevated privileges on Windows",
)
def test_capture_rejects_symlinked_dataset_root_or_manifest_parent(
    tmp_path, monkeypatch
) -> None:
    report, config = _fixture_provenance(tmp_path, monkeypatch)
    assert report["status"] == "pass"

    dataset_root = tmp_path / "fixture_imagenet"
    alias_config = SimpleNamespace(dataset="fixture_imagenet", root=str(tmp_path))
    # The normal dataset name remains real; put the alias in an upstream root
    # so the path itself contains a symlinked parent component.
    upstream_alias = tmp_path / "upstream-alias"
    upstream_alias.symlink_to(tmp_path, target_is_directory=True)
    alias_config.root = str(upstream_alias)
    with pytest.raises(ValueError, match="must not contain a symlink"):
        capture_dataset_provenance(alias_config, train_images=1, val_images=1)

    manifest = dataset_root / "metadata" / "image_manifest.jsonl"
    external = tmp_path / "external-manifest.jsonl"
    external.write_bytes(manifest.read_bytes())
    manifest.unlink()
    manifest.symlink_to(external)
    with pytest.raises(ValueError, match="must not contain a symlink"):
        capture_dataset_provenance(config, train_images=1, val_images=1)
