from __future__ import annotations

import pytest

from cofitok.environment import runtime_environment_sha256
from scripts.select_generation_training_runtime import (
    _benchmark_matches,
    parse_candidates,
    select_runtime_candidate,
)


def _method(seconds: float, *, peak_fraction: float = 0.5) -> dict:
    total = 100_000
    environment = {
        "schema_version": 1,
        "device": {"type": "cuda", "name": "GPU"},
    }
    return {
        "status": "completed",
        "runtime_environment": environment,
        "runtime_environment_sha256": runtime_environment_sha256(environment),
        "effective_batch_size": 64,
        "mean_optimizer_step_seconds": seconds,
        "images_per_second": 64 / seconds,
        "peak_vram_bytes": int(total * peak_fraction),
        "device_total_memory_bytes": total,
    }


def _candidate(micro: int, accumulation: int, cofitok: dict, dense: dict) -> dict:
    return {
        "micro_batch_size": micro,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro * accumulation,
        "methods": {"cofitok": cofitok, "dense_identity": dense},
    }


def test_parse_candidates_preserves_effective_batch_and_baseline() -> None:
    assert parse_candidates("16x4,32x2,64x1", expected_effective_batch=64) == [
        (16, 4),
        (32, 2),
        (64, 1),
    ]

    with pytest.raises(ValueError, match="changes effective batch"):
        parse_candidates("16x4,32x1", expected_effective_batch=64)
    with pytest.raises(ValueError, match="16x4 baseline"):
        parse_candidates("32x2,64x1", expected_effective_batch=64)


def test_selector_minimizes_worst_method_runtime() -> None:
    report = select_runtime_candidate(
        [
            _candidate(16, 4, _method(2.5), _method(2.4)),
            _candidate(32, 2, _method(1.9), _method(2.0)),
            _candidate(64, 1, _method(1.8), _method(2.2)),
        ],
        expected_effective_batch=64,
        max_memory_fraction=0.9,
    )

    assert report["selected"]["micro_batch_size"] == 32
    assert report["selected"]["gradient_accumulation_steps"] == 2
    assert report["selected"]["selection_score_seconds"] == 2.0
    assert report["selected"]["estimated_speedup_over_16x4"] == 1.25
    assert len(report["runtime_environment_sha256"]) == 64


def test_selector_rejects_oom_or_insufficient_memory_headroom() -> None:
    failed = {"status": "failed", "failure_type": "cuda_oom"}
    report = select_runtime_candidate(
        [
            _candidate(16, 4, _method(2.5), _method(2.4)),
            _candidate(32, 2, _method(1.9, peak_fraction=0.95), _method(2.0)),
            _candidate(64, 1, failed, _method(1.7)),
        ],
        expected_effective_batch=64,
        max_memory_fraction=0.9,
    )

    assert report["selected"]["micro_batch_size"] == 16
    assert "cofitok_memory_headroom" in report["candidates"][1]["ineligible_reasons"]
    assert "cofitok_benchmark_incomplete" in report["candidates"][2]["ineligible_reasons"]


def test_selector_fails_closed_when_conservative_baseline_fails() -> None:
    with pytest.raises(ValueError, match="16x4 runtime baseline"):
        select_runtime_candidate(
            [
                _candidate(
                    16,
                    4,
                    {"status": "failed", "failure_type": "process_error"},
                    _method(2.4),
                ),
                _candidate(32, 2, _method(2.0), _method(2.0)),
            ],
            expected_effective_batch=64,
            max_memory_fraction=0.9,
        )


def test_selector_rejects_invalid_or_drifted_runtime_environment() -> None:
    invalid = _method(2.0)
    invalid["runtime_environment"]["device"]["name"] = "changed"

    with pytest.raises(ValueError, match="environment provenance failed"):
        select_runtime_candidate(
            [
                _candidate(16, 4, _method(2.5), _method(2.4)),
                _candidate(32, 2, _method(2.0), invalid),
            ],
            expected_effective_batch=64,
            max_memory_fraction=0.9,
        )

    changed_cofitok = _method(2.0)
    changed_dense = _method(2.0)
    for report in (changed_cofitok, changed_dense):
        report["runtime_environment"]["device"]["name"] = "another GPU"
        report["runtime_environment_sha256"] = runtime_environment_sha256(
            report["runtime_environment"]
        )
    with pytest.raises(ValueError, match="changed across benchmark candidates"):
        select_runtime_candidate(
            [
                _candidate(16, 4, _method(2.5), _method(2.4)),
                _candidate(32, 2, changed_cofitok, changed_dense),
            ],
            expected_effective_batch=64,
            max_memory_fraction=0.9,
        )

    changed_dense = _method(1.8)
    changed_dense["runtime_environment"]["device"]["name"] = "third GPU"
    changed_dense["runtime_environment_sha256"] = runtime_environment_sha256(
        changed_dense["runtime_environment"]
    )
    with pytest.raises(ValueError, match="changed across benchmark candidates"):
        select_runtime_candidate(
            [
                _candidate(16, 4, _method(2.5), _method(2.4)),
                _candidate(
                    64,
                    1,
                    {"status": "failed", "failure_type": "cuda_oom"},
                    changed_dense,
                ),
            ],
            expected_effective_batch=64,
            max_memory_fraction=0.9,
        )


def test_benchmark_cache_requires_clean_git_and_valid_environment() -> None:
    expected_config = {"name": "benchmark"}
    report = _method(2.0)
    report.update(
        config=expected_config,
        git={"revision": "a" * 40, "dirty": False},
        benchmark_steps=8,
        warmup_steps=2,
        checkpoint_written=False,
    )

    assert _benchmark_matches(
        report,
        expected_config=expected_config,
        expected_revision="a" * 40,
        benchmark_steps=8,
        warmup_steps=2,
    )
    report["git"]["dirty"] = True
    assert not _benchmark_matches(
        report,
        expected_config=expected_config,
        expected_revision="a" * 40,
        benchmark_steps=8,
        warmup_steps=2,
    )
    report["git"]["dirty"] = False
    report["runtime_environment_sha256"] = "0" * 64
    assert not _benchmark_matches(
        report,
        expected_config=expected_config,
        expected_revision="a" * 40,
        benchmark_steps=8,
        warmup_steps=2,
    )
