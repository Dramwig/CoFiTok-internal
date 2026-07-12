from __future__ import annotations

import pytest

from scripts.select_generation_sampling_batch import (
    parse_candidates,
    select_sampling_batch,
)


def _method(
    batch_size: int,
    throughput: float,
    *,
    peak_fraction: float = 0.5,
    prefix_budget: int = 8,
    checkpoint_step: int = 300_000,
) -> dict:
    total = 100_000
    return {
        "status": "passed",
        "checkpoint_step": checkpoint_step,
        "weights": "ema",
        "request": {
            "batch_size": batch_size,
            "effective_model_batch_size": batch_size * 2,
            "forward_passes": 1,
            "image_shape": [3, 256, 256],
            "prefix_budget": prefix_budget,
            "token_count": prefix_budget,
            "precision": "bf16",
            "guidance_scale": 1.5,
            "guidance_rescale": 0.0,
            "cfg_batch_mode": "batched",
            "class_conditional": True,
            "timestep": 999,
            "warmup_forwards": 2,
            "measured_forwards": 5,
        },
        "result": {
            "output_images_per_second": throughput,
            "cuda_memory_after_forward": {
                "peak_allocated_bytes": int(total * peak_fraction)
            },
            "device_total_memory_bytes": total,
        },
    }


def _candidate(batch_size: int, cofitok: dict, dense: dict) -> dict:
    return {
        "batch_size": batch_size,
        "methods": {"cofitok": cofitok, "dense_identity": dense},
    }


def test_sampling_candidate_parser_requires_baseline() -> None:
    assert parse_candidates("16,32,64,128", baseline_batch_size=32) == [16, 32, 64, 128]

    with pytest.raises(ValueError, match="invalid"):
        parse_candidates("16,nope,32", baseline_batch_size=32)
    with pytest.raises(ValueError, match="baseline"):
        parse_candidates("16,64", baseline_batch_size=32)


def test_sampling_selector_maximizes_worst_method_throughput() -> None:
    report = select_sampling_batch(
        [
            _candidate(16, _method(16, 30.0), _method(16, 32.0, prefix_budget=1)),
            _candidate(32, _method(32, 50.0), _method(32, 48.0, prefix_budget=1)),
            _candidate(64, _method(64, 72.0), _method(64, 70.0, prefix_budget=1)),
        ],
        baseline_batch_size=32,
        max_memory_fraction=0.9,
    )

    assert report["selected"]["batch_size"] == 64
    assert report["selected"]["selection_score_images_per_second"] == 70.0
    assert report["selected"]["estimated_speedup_over_baseline"] == pytest.approx(
        70.0 / 48.0
    )


def test_sampling_selector_rejects_oom_and_memory_pressure() -> None:
    failed = {"status": "failed", "error": "CUDA out of memory"}
    report = select_sampling_batch(
        [
            _candidate(32, _method(32, 50.0), _method(32, 48.0, prefix_budget=1)),
            _candidate(
                64,
                _method(64, 72.0, peak_fraction=0.95),
                _method(64, 70.0, prefix_budget=1),
            ),
            _candidate(128, failed, _method(128, 80.0, prefix_budget=1)),
        ],
        baseline_batch_size=32,
        max_memory_fraction=0.9,
    )

    assert report["selected"]["batch_size"] == 32
    assert "cofitok_memory_headroom" in report["candidates"][1]["ineligible_reasons"]
    assert "cofitok_preflight_incomplete" in report["candidates"][2]["ineligible_reasons"]


def test_sampling_selector_rejects_mismatched_protocol_or_checkpoint() -> None:
    dense = _method(32, 48.0, prefix_budget=1, checkpoint_step=299_999)
    dense["request"]["guidance_scale"] = 2.0

    with pytest.raises(ValueError, match="no shared"):
        select_sampling_batch(
            [_candidate(32, _method(32, 50.0), dense)],
            baseline_batch_size=32,
            max_memory_fraction=0.9,
        )
