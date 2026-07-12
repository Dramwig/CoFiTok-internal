from __future__ import annotations

import sys
from types import SimpleNamespace

from PIL import Image

from scripts.evaluate_generation_metrics import calculate_metrics, find_images


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
