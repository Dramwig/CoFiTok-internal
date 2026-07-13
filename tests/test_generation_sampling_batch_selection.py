from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from cofitok.environment import runtime_environment_sha256
from scripts.select_generation_sampling_batch import (
    _preflight_matches,
    _sampling_selection_lock,
    parse_candidates,
    reuse_sampling_selection_after_sampling_start,
    select_sampling_batch,
    validate_frozen_sampling_selection,
)


REVISION = "a" * 40


def _method(
    batch_size: int,
    throughput: float,
    *,
    peak_fraction: float = 0.5,
    prefix_budget: int = 8,
    checkpoint_step: int = 300_000,
) -> dict:
    total = 100_000
    environment = {"schema_version": 1, "device": {"type": "cuda", "name": "GPU"}}
    return {
        "status": "passed",
        "checkpoint_step": checkpoint_step,
        "weights": "ema",
        "runtime_environment": environment,
        "runtime_environment_sha256": runtime_environment_sha256(environment),
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


def _frozen_selection(tmp_path: Path) -> tuple[dict, dict, list[Path]]:
    identities = {
        "cofitok": {
            "path": "/checkpoints/cofitok.pt",
            "sha256": "b" * 64,
            "step": 300_000,
            "integrity_manifest": "/checkpoints/cofitok.pt.integrity.json",
        },
        "dense_identity": {
            "path": "/checkpoints/dense.pt",
            "sha256": "c" * 64,
            "step": 300_000,
            "integrity_manifest": "/checkpoints/dense.pt.integrity.json",
        },
    }
    rows = []
    for batch_size, throughput in ((16, 30.0), (32, 48.0), (64, 70.0), (128, 65.0)):
        methods = {}
        for method, prefix_budget in (("cofitok", 8), ("dense_identity", 1)):
            report = _method(batch_size, throughput, prefix_budget=prefix_budget)
            report.update(
                git={
                    "revision": REVISION,
                    "branch": "scale/generative-system",
                    "tracked_dirty": False,
                },
                checkpoint=identities[method]["path"],
                checkpoint_sha256=identities[method]["sha256"],
                checkpoint_integrity_manifest=identities[method][
                    "integrity_manifest"
                ],
            )
            methods[method] = report
        rows.append(
            _candidate(batch_size, methods["cofitok"], methods["dense_identity"])
        )
    selection = select_sampling_batch(
        rows,
        baseline_batch_size=32,
        max_memory_fraction=0.9,
    )
    output_dirs = [(tmp_path / "cofitok").resolve(), (tmp_path / "dense").resolve()]
    lock = _sampling_selection_lock(
        output_dirs=output_dirs,
        candidates=[16, 32, 64, 128],
        baseline_batch_size=32,
        max_memory_fraction=0.9,
        cofitok_prefix_budget=8,
        dense_prefix_budget=1,
        guidance_scale=1.5,
        guidance_rescale=0.0,
        cfg_batch_mode="batched",
        weights="ema",
        precision="bf16",
        warmup_forwards=2,
        measured_forwards=5,
        git={
            "revision": REVISION,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        checkpoints=identities,
        benchmark_root=(tmp_path / "benchmarks").resolve(),
    )
    selection.update(
        git_revision=REVISION,
        checkpoints=identities,
        benchmark_root=lock["benchmark_root"],
        selection_lock=lock,
    )
    return selection, lock, output_dirs


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
    assert len(report["runtime_environment_sha256"]) == 64


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


def test_sampling_selector_rejects_mismatched_runtime_environment() -> None:
    dense = _method(32, 48.0, prefix_budget=1)
    dense["runtime_environment"]["device"]["name"] = "another GPU"
    dense["runtime_environment_sha256"] = runtime_environment_sha256(
        dense["runtime_environment"]
    )

    with pytest.raises(ValueError, match="no shared"):
        select_sampling_batch(
            [_candidate(32, _method(32, 50.0), dense)],
            baseline_batch_size=32,
            max_memory_fraction=0.9,
        )


def test_sampling_selector_rejects_environment_changes_across_candidates() -> None:
    changed_cofitok = _method(64, 70.0)
    changed_dense = _method(64, 68.0, prefix_budget=1)
    for report in (changed_cofitok, changed_dense):
        report["runtime_environment"]["device"]["name"] = "another GPU"
        report["runtime_environment_sha256"] = runtime_environment_sha256(
            report["runtime_environment"]
        )

    with pytest.raises(ValueError, match="changed across sampling candidates"):
        select_sampling_batch(
            [
                _candidate(
                    32,
                    _method(32, 50.0),
                    _method(32, 48.0, prefix_budget=1),
                ),
                _candidate(64, changed_cofitok, changed_dense),
            ],
            baseline_batch_size=32,
            max_memory_fraction=0.9,
        )


def test_sampling_preflight_cache_requires_exact_git_revision() -> None:
    report = _method(32, 50.0)
    report.update(
        git={
            "revision": "a" * 40,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        checkpoint="/checkpoints/model.pt",
        checkpoint_sha256="b" * 64,
        checkpoint_integrity_manifest="/checkpoints/model.pt.integrity.json",
    )
    expected = {
        "checkpoint_identity": {
            "path": "/checkpoints/model.pt",
            "sha256": "b" * 64,
            "step": 300_000,
            "integrity_manifest": "/checkpoints/model.pt.integrity.json",
        },
        "expected_revision": "a" * 40,
        "batch_size": 32,
        "prefix_budget": 8,
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "weights": "ema",
        "precision": "bf16",
        "warmup_forwards": 2,
        "measured_forwards": 5,
    }

    assert _preflight_matches(report, **expected)
    expected["expected_revision"] = "c" * 40
    assert not _preflight_matches(report, **expected)

    expected["expected_revision"] = "a" * 40
    report["git"]["tracked_dirty"] = True
    assert not _preflight_matches(report, **expected)

    report["git"]["tracked_dirty"] = False
    report["checkpoint_integrity_manifest"] = "/checkpoints/other.integrity.json"
    assert not _preflight_matches(report, **expected)


def test_sampling_selection_is_read_only_after_partial_samples_exist(
    tmp_path: Path,
) -> None:
    selection, lock, output_dirs = _frozen_selection(tmp_path)
    selection_path = tmp_path / "sampling_runtime_selection.json"
    selection_path.write_text(json.dumps(selection), encoding="utf-8")
    for output_dir in output_dirs:
        output_dir.mkdir()

    assert reuse_sampling_selection_after_sampling_start(
        selection_path=selection_path,
        output_dirs=output_dirs,
        expected_lock=lock,
    ) is None

    (output_dirs[0] / "sampling_manifest.json").write_text("{}\n", encoding="ascii")
    original = selection_path.read_bytes()
    assert reuse_sampling_selection_after_sampling_start(
        selection_path=selection_path,
        output_dirs=output_dirs,
        expected_lock=lock,
    ) == 64
    assert selection_path.read_bytes() == original


def test_sampling_state_without_selection_fails_before_preflight(tmp_path: Path) -> None:
    _, lock, output_dirs = _frozen_selection(tmp_path)
    output_dirs[0].mkdir()
    (output_dirs[0] / "sampling_progress.json").write_text("{}\n", encoding="ascii")

    with pytest.raises(FileNotFoundError, match="without a frozen batch selection"):
        reuse_sampling_selection_after_sampling_start(
            selection_path=tmp_path / "missing.json",
            output_dirs=output_dirs,
            expected_lock=lock,
        )


@pytest.mark.parametrize("drift", ["lock", "candidate", "checkpoint"])
def test_frozen_sampling_selection_rejects_resume_drift(
    tmp_path: Path,
    drift: str,
) -> None:
    selection, lock, _ = _frozen_selection(tmp_path)
    if drift == "lock":
        lock = copy.deepcopy(lock)
        lock["protocol"]["measured_forwards"] = 6
    elif drift == "candidate":
        selection["candidates"][0]["batch_size"] = 8
    else:
        selection["checkpoints"]["cofitok"]["sha256"] = "d" * 64

    with pytest.raises(ValueError, match="frozen"):
        validate_frozen_sampling_selection(selection, expected_lock=lock)
