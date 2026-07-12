from pathlib import Path

import pytest

from scripts.make_next_validation_queue import NEXT_VALIDATION_RUNS, render_queue, write_queue


def test_render_queue_contains_core_validation_steps() -> None:
    script = render_queue(sample_count=128, sample_steps=20, real_count=256)

    assert "train_tiny_imagenet_k8_denoisepath_p150_light_20k_seed2_cuda.json" in script
    assert "train_tiny_imagenet_k8_denoisepath_p150_light_multiscale_10k_cuda.json" in script
    assert "train_tiny_imagenet_k8_denoisepath_p150_light_nopathprefix_10k_cuda.json" in script
    assert "train_imagenet_1k_64x64_hf_k8_denoisepath_p150_light_cleanmono_10k_cuda.json" in script
    assert "train_imagenet_1k_64x64_hf_k8_epsilononly_p150eval_20k_seed2_cuda.json" in script
    assert "--enable-inception-fid" in script
    assert 'QUALITY_BATCHES="${QUALITY_BATCHES:-64}"' in script
    assert "scripts/inspect_next_validation_queue.py" in script
    assert 'write_queue_status "start"' in script
    assert 'write_queue_status "final"' in script
    assert "scripts/evaluate_checkpoint.py" in script
    assert "scripts/evaluate_generated_samples.py" in script
    assert "scripts/validate_mvp_evidence.py" in script
    assert "scripts/validate_publication_readiness.py" in script
    assert "REQUIRE_PUBLICATION_READY" in script
    assert "scripts/validate_idea_requirements.py" in script
    assert "idea_requirements_2026-07-08.json" in script
    assert "scripts/validate_dataset_conditions.py" in script
    assert "scripts/validate_synthesis_contract.py" in script
    assert "$PYTHON\" -m pytest -q" in script


def test_write_queue_writes_script_and_manifest(tmp_path: Path) -> None:
    output = tmp_path / "next_validation.sh"

    manifest = write_queue(output, sample_count=64, sample_steps=10, real_count=128)

    assert output.exists()
    assert Path(manifest["manifest"]).exists()
    assert manifest["train_count"] == len(NEXT_VALIDATION_RUNS)
    assert manifest["quality_batches"] == 64
    assert manifest["order_eval_count"] == 24
    assert manifest["sampling_count"] == 6
    assert manifest["generated_quality_count"] == 6


def test_render_queue_rejects_nonpositive_counts() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        render_queue(sample_count=0)
