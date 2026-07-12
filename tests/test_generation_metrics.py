from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from PIL import Image

from scripts.evaluate_generation_metrics import (
    calculate_metrics,
    find_images,
    validate_sampling_provenance,
)


def test_find_images_is_recursive_and_filters_extensions(tmp_path) -> None:
    nested = tmp_path / "class_a"
    nested.mkdir()
    Image.new("RGB", (4, 4)).save(nested / "a.png")
    Image.new("RGB", (4, 4)).save(tmp_path / "b.jpg")
    (nested / "ignore.txt").write_text("x", encoding="utf-8")

    assert [path.name for path in find_images(tmp_path)] == ["b.jpg", "a.png"]


def test_calculate_metrics_uses_generated_as_precision_input(monkeypatch, tmp_path) -> None:
    calls = []

    def fake_calculate_metrics(**kwargs):
        calls.append(kwargs)
        return {
            "frechet_inception_distance": 12.5,
            "inception_score_mean": 3.0,
            "precision": 0.6,
            "recall": 0.4,
        }

    monkeypatch.setitem(
        sys.modules,
        "torch_fidelity",
        SimpleNamespace(__version__="0.4.0", calculate_metrics=fake_calculate_metrics),
    )
    generated = tmp_path / "generated"
    real = tmp_path / "real"
    generated.mkdir()
    real.mkdir()

    metrics, version = calculate_metrics(
        real_dir=real,
        generated_dir=generated,
        batch_size=16,
        prc_batch_size=100,
        seed=7,
        cuda=False,
        cache_root="cache",
        real_cache_name="real-v1",
        prc=True,
    )

    assert version == "0.4.0"
    assert metrics["frechet_inception_distance"] == 12.5
    assert calls[0]["input1"] == generated.as_posix()
    assert calls[0]["input2"] == real.as_posix()
    assert calls[0]["isc"] is True
    assert calls[0]["fid"] is True
    assert calls[0]["prc"] is True
    assert calls[0]["samples_find_deep"] is True
    assert calls[0]["input2_cache_name"] == "real-v1"


def test_validate_sampling_provenance_requires_exact_numbered_set(tmp_path) -> None:
    generated = tmp_path / "samples" / "prefix_8"
    generated.mkdir(parents=True)
    for index in range(2):
        Image.new("RGB", (4, 4)).save(generated / f"{index:06d}.png")
    report_path = generated.parent / "sampling_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint": "/checkpoints/model.pt",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_step": 50_000,
                "weights": "ema",
                "sampling": {"start_index": 0, "num_samples": 2, "image_shape": [3, 4, 4]},
                "output_dirs": {"8": generated.resolve().as_posix()},
            }
        ),
        encoding="utf-8",
    )

    provenance = validate_sampling_provenance(report_path, generated, find_images(generated))

    assert provenance["selected_prefix_budget"] == 8
    assert provenance["checkpoint_step"] == 50_000
    assert provenance["image_shape"] == [3, 4, 4]


def test_validate_sampling_provenance_rejects_stale_extra_sample(tmp_path) -> None:
    generated = tmp_path / "samples" / "prefix_8"
    generated.mkdir(parents=True)
    for index in range(3):
        Image.new("RGB", (4, 4)).save(generated / f"{index:06d}.png")
    report_path = generated.parent / "sampling_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint": "/checkpoints/model.pt",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_step": 50_000,
                "weights": "ema",
                "sampling": {"start_index": 0, "num_samples": 2, "image_shape": [3, 4, 4]},
                "output_dirs": {"8": generated.resolve().as_posix()},
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_sampling_provenance(report_path, generated, find_images(generated))
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("stale extra sample was accepted")


def test_validate_sampling_provenance_rejects_corrupt_png(tmp_path) -> None:
    generated = tmp_path / "samples" / "prefix_8"
    generated.mkdir(parents=True)
    (generated / "000000.png").write_bytes(b"corrupt")
    report_path = generated.parent / "sampling_report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "checkpoint": "/checkpoints/model.pt",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_step": 50_000,
                "weights": "ema",
                "sampling": {"start_index": 0, "num_samples": 1, "image_shape": [3, 4, 4]},
                "output_dirs": {"8": generated.resolve().as_posix()},
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_sampling_provenance(report_path, generated, find_images(generated))
    except ValueError as error:
        assert "integrity failed" in str(error)
    else:
        raise AssertionError("corrupt PNG was accepted")
