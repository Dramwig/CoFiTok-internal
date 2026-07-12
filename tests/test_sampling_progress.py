from __future__ import annotations

import json

import pytest

from cofitok.sampling_progress import (
    build_sampling_progress,
    load_sampling_progress_state,
    write_sampling_progress,
)


def _state(tmp_path):
    return load_sampling_progress_state(
        tmp_path / "sampling_progress.json",
        sampling_manifest_sha256="a" * 64,
        total_samples=100,
        start_index=0,
        prefix_budgets=[8],
    )


def test_progress_is_atomic_and_accumulates_across_resume(tmp_path) -> None:
    path = tmp_path / "sampling_progress.json"
    first = _state(tmp_path)
    write_sampling_progress(
        path,
        build_sampling_progress(
            first,
            status="running",
            completed_samples=40,
            invocation_elapsed_seconds=10.0,
        ),
    )

    resumed = _state(tmp_path)
    payload = build_sampling_progress(
        resumed,
        status="running",
        completed_samples=60,
        invocation_elapsed_seconds=5.0,
    )

    assert resumed["invocation"] == 2
    assert payload["completed_samples"] == 60
    assert payload["cumulative_elapsed_seconds"] == 15.0
    assert payload["samples_per_second"] == 4.0
    assert payload["eta_seconds"] == 10.0


def test_progress_rejects_another_sampling_manifest(tmp_path) -> None:
    path = tmp_path / "sampling_progress.json"
    payload = build_sampling_progress(
        _state(tmp_path),
        status="running",
        completed_samples=1,
        invocation_elapsed_seconds=1.0,
    )
    write_sampling_progress(path, payload)

    with pytest.raises(ValueError, match="another manifest"):
        load_sampling_progress_state(
            path,
            sampling_manifest_sha256="b" * 64,
            total_samples=100,
            start_index=0,
            prefix_budgets=[8],
        )


def test_completed_progress_requires_all_samples_and_digests(tmp_path) -> None:
    state = _state(tmp_path)
    with pytest.raises(ValueError, match="all samples and digests"):
        build_sampling_progress(
            state,
            status="completed",
            completed_samples=99,
            invocation_elapsed_seconds=1.0,
            sample_sets={"8": {"count": 99, "sha256": "c" * 64}},
        )


def test_failed_progress_records_error(tmp_path) -> None:
    payload = build_sampling_progress(
        _state(tmp_path),
        status="failed",
        completed_samples=32,
        invocation_elapsed_seconds=2.0,
        error=RuntimeError("CUDA out of memory"),
    )

    assert payload["status"] == "failed"
    assert payload["error_type"] == "RuntimeError"
    assert payload["error"] == "CUDA out of memory"


def test_progress_file_is_valid_json_without_temporary_residue(tmp_path) -> None:
    path = tmp_path / "sampling_progress.json"
    payload = build_sampling_progress(
        _state(tmp_path),
        status="running",
        completed_samples=10,
        invocation_elapsed_seconds=1.0,
    )
    write_sampling_progress(path, payload)

    assert json.loads(path.read_text(encoding="utf-8"))["completed_samples"] == 10
    assert not list(tmp_path.glob("*.tmp"))
