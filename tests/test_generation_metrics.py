from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from cofitok.environment import runtime_environment_sha256
from cofitok.image_integrity import sample_set_sha256
from cofitok.reporting import file_sha256
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
    sample_sha256 = sample_set_sha256(find_images(generated))
    sampling = {
        "start_index": 0,
        "num_samples": 2,
        "image_shape": [3, 4, 4],
        "prefix_budgets": [8],
    }
    sample_sets = {"8": {"count": 2, "sha256": sample_sha256}}
    runtime_environment = {"schema_version": 1, "device": {"type": "cpu"}}
    runtime_environment_sha = runtime_environment_sha256(runtime_environment)
    manifest_path = generated.parent / "sampling_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "runtime_environment": runtime_environment,
                "runtime_environment_sha256": runtime_environment_sha,
            }
        ),
        encoding="utf-8",
    )
    manifest_sha256 = file_sha256(manifest_path)
    progress_path = generated.parent / "sampling_progress.json"
    progress_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "sampling_manifest_sha256": manifest_sha256,
                "completed_samples": 2,
                "prefix_budgets": [8],
                "sample_sets": sample_sets,
                "invocation": 1,
                "cumulative_elapsed_seconds": 1.0,
            }
        ),
        encoding="utf-8",
    )
    report_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "git": {
                    "revision": "a" * 40,
                    "branch": "scale/generative-system",
                    "tracked_dirty": False,
                },
                "runtime_environment": runtime_environment,
                "runtime_environment_sha256": runtime_environment_sha,
                "checkpoint": "/checkpoints/model.pt",
                "checkpoint_sha256": "a" * 64,
                "checkpoint_integrity_manifest": "/checkpoints/model.pt.integrity.json",
                "checkpoint_step": 50_000,
                "weights": "ema",
                "sampling": sampling,
                "sampling_manifest_sha256": manifest_sha256,
                "sampling_progress": progress_path.resolve().as_posix(),
                "output_dirs": {"8": generated.resolve().as_posix()},
                "sample_sets": sample_sets,
            }
        ),
        encoding="utf-8",
    )

    provenance = validate_sampling_provenance(report_path, generated, find_images(generated))

    assert provenance["selected_prefix_budget"] == 8
    assert provenance["git"]["tracked_dirty"] is False
    assert provenance["checkpoint_step"] == 50_000
    assert provenance["checkpoint_integrity_manifest"].endswith("model.pt.integrity.json")
    assert provenance["image_shape"] == [3, 4, 4]
    assert provenance["sample_set_sha256"] == sample_sha256
    assert provenance["sampling_progress"]["status"] == "completed"
    assert provenance["runtime_environment_sha256"] == runtime_environment_sha

    sampling_report = json.loads(report_path.read_text(encoding="utf-8"))
    sampling_report["runtime_environment_sha256"] = "f" * 64
    report_path.write_text(json.dumps(sampling_report), encoding="utf-8")
    with pytest.raises(ValueError, match="runtime environment SHA256 differs"):
        validate_sampling_provenance(report_path, generated, find_images(generated))
    sampling_report["runtime_environment_sha256"] = runtime_environment_sha
    report_path.write_text(json.dumps(sampling_report), encoding="utf-8")

    sampling_manifest_text = manifest_path.read_text(encoding="utf-8")
    sampling_manifest = json.loads(sampling_manifest_text)
    sampling_manifest["runtime_environment"]["device"]["type"] = "cuda"
    manifest_path.write_text(json.dumps(sampling_manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest runtime environment differs"):
        validate_sampling_provenance(report_path, generated, find_images(generated))
    manifest_path.write_text(sampling_manifest_text, encoding="utf-8")

    sampling_report = json.loads(report_path.read_text(encoding="utf-8"))
    sampling_git = sampling_report.pop("git")
    report_path.write_text(json.dumps(sampling_report), encoding="utf-8")
    with pytest.raises(ValueError, match="Git revision"):
        validate_sampling_provenance(report_path, generated, find_images(generated))
    sampling_report["git"] = sampling_git
    report_path.write_text(json.dumps(sampling_report), encoding="utf-8")

    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["status"] = "running"
    progress_path.write_text(json.dumps(progress), encoding="utf-8")
    try:
        validate_sampling_provenance(report_path, generated, find_images(generated))
    except ValueError as error:
        assert "progress is not completed" in str(error)
    else:
        raise AssertionError("incomplete sampling progress was accepted")


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
                "sample_sets": {"8": {"count": 2, "sha256": "b" * 64}},
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
                "sample_sets": {"8": {"count": 1, "sha256": "b" * 64}},
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


def test_validate_sampling_provenance_rejects_sample_set_digest_mismatch(tmp_path) -> None:
    generated = tmp_path / "samples" / "prefix_8"
    generated.mkdir(parents=True)
    Image.new("RGB", (4, 4)).save(generated / "000000.png")
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
                "sample_sets": {"8": {"count": 1, "sha256": "b" * 64}},
            }
        ),
        encoding="utf-8",
    )

    try:
        validate_sampling_provenance(report_path, generated, find_images(generated))
    except ValueError as error:
        assert "sample-set SHA256" in str(error)
    else:
        raise AssertionError("mismatched sample-set digest was accepted")
