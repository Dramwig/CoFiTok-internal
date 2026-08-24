from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from cofitok.environment import runtime_environment_sha256
from cofitok.generation import (
    EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY,
    build_epsilon_stability_artifact_report,
    compute_epsilon_stability_artifact_metrics,
    numbered_epsilon_stability_pngs,
    validate_epsilon_stability_artifact_report,
)


def _write_rgb(path: Path, value: int) -> None:
    pixels = np.full((256, 256, 3), value, dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


def test_artifact_metrics_and_numbered_set_are_exact(tmp_path: Path) -> None:
    _write_rgb(tmp_path / "000000.png", 0)
    images = numbered_epsilon_stability_pngs(
        tmp_path,
        sample_count=1,
        start_index=0,
    )

    assert compute_epsilon_stability_artifact_metrics(images) == {
        "channel_saturation_fraction": 1.0,
        "total_variation": 0.0,
        "median_filter_residual_fraction": 0.0,
    }

    _write_rgb(tmp_path / "000001.png", 128)
    with pytest.raises(ValueError, match="exact numbered sample set"):
        numbered_epsilon_stability_pngs(
            tmp_path,
            sample_count=1,
            start_index=0,
        )


def test_artifact_report_round_trips_with_non_authorizing_boundary() -> None:
    git = {
        "revision": "a" * 40,
        "branch": "analysis/epsilon-stability-test",
        "tracked_dirty": False,
    }
    environment = {"schema_version": 1, "device": {"type": "cpu"}}
    environment_sha256 = runtime_environment_sha256(environment)
    metrics = {
        "channel_saturation_fraction": 0.1,
        "total_variation": 0.2,
        "median_filter_residual_fraction": 0.3,
    }
    report = build_epsilon_stability_artifact_report(
        source_kind="generated",
        git=git,
        runtime_environment=environment,
        runtime_environment_sha256_value=environment_sha256,
        sample_count=1_000,
        sample_set_sha256_value="b" * 64,
        source_binding={"sampling_report": {"sha256": "c" * 64}},
        metrics=metrics,
    )

    normalized = validate_epsilon_stability_artifact_report(
        report,
        expected_source_kind="generated",
        expected_git=git,
        expected_runtime_environment_sha256=environment_sha256,
    )
    assert normalized["metrics"] == metrics
    assert normalized["authorization_boundary"] == (
        EPSILON_STABILITY_NON_AUTHORIZING_BOUNDARY
    )
    assert all(
        value is False
        for value in report["authorization_boundary"].values()
    )

    report["authorization_boundary"]["training_launch_allowed"] = True
    with pytest.raises(ValueError, match="non-authorizing boundary"):
        validate_epsilon_stability_artifact_report(
            report,
            expected_source_kind="generated",
            expected_git=git,
            expected_runtime_environment_sha256=environment_sha256,
        )
