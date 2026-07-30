from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from cofitok.generation.stability_scaling import (
    build_stability_scaling_decision,
)
from cofitok.reporting import file_sha256, write_json_report
from scripts import validate_generation_stability_scaling_decision as validator


REVISION = "a" * 40


def _qualification(*, seed: int, images: int) -> dict:
    gates = {
        name: {"passed": True}
        for name in (
            "tail_two_energy",
            "single_token_energy",
            "ordered_rank",
            "endpoint_regression",
            "validation_regression",
            "predicted_x0_high_frequency",
            "reconstruction_regression",
            "zero_token",
            "shuffle_mismatch",
        )
    }
    return {
        "schema_version": 2,
        "status": "pass",
        "protocol": {
            "weights": "model",
            "checkpoint_step": 5_000,
            "checkpoint_evaluated_images": 256,
            "checkpoint_timestep": 500,
            "high_frequency_timesteps": [595, 394, 192, 91],
            "rollout": {
                "num_images": images,
                "batch_size": 8 if images >= 64 else 2,
                "sample_steps": 100,
                "guidance_scale": 1.5,
                "seed": seed,
            },
        },
        "identity": {
            "git_revision": REVISION,
            "cofitok_checkpoint_sha256": "b" * 64,
            "dense_checkpoint_sha256": "c" * 64,
        },
        "metrics": {
            "tail_two_energy_ratio": 0.57,
            "max_single_token_energy_ratio": 0.29,
            "endpoint_ratio": 1.01,
            "validation_ratio": 1.01,
            "peak_predicted_x0_high_frequency_ratio": 1.1,
            "reconstruction_ratio": 1.03,
            "cofitok_reconstruction_amplification": 1.08,
            "dense_reconstruction_amplification": 1.07,
        },
        "gates": gates,
        "pair_contract": {"valid": True},
    }


def _source(path: Path) -> dict:
    return {
        "path": path.resolve().as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _decision(tmp_path: Path) -> Path:
    screening_path = tmp_path / "screening.json"
    robust_paths = [tmp_path / "robust-2029.json", tmp_path / "robust-2039.json"]
    write_json_report(screening_path, _qualification(seed=2029, images=8))
    for path, seed in zip(robust_paths, (2029, 2039), strict=True):
        write_json_report(path, _qualification(seed=seed, images=64))
    decision = build_stability_scaling_decision(
        screening_report=json.loads(screening_path.read_text(encoding="utf-8")),
        robust_reports=[
            json.loads(path.read_text(encoding="utf-8"))
            for path in robust_paths
        ],
    )
    decision["sources"] = {
        "screening_report": _source(screening_path),
        "robust_reports": [_source(path) for path in robust_paths],
    }
    decision_path = tmp_path / "decision.json"
    write_json_report(decision_path, decision)
    return decision_path


def _argv(decision: Path, output: Path) -> list[str]:
    return [
        "validate_generation_stability_scaling_decision.py",
        "--decision",
        str(decision),
        "--expected-decision-sha256",
        file_sha256(decision),
        "--expected-source-revision",
        REVISION,
        "--output",
        str(output),
    ]


def test_validator_rebuilds_and_rehashes_the_stability_decision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = _decision(tmp_path)
    output = tmp_path / "validation.json"
    monkeypatch.setattr(sys, "argv", _argv(decision, output))

    validator.main()

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "pass"
    assert report["source_revision"] == REVISION
    assert report["robust_seeds"] == [2029, 2039]
    assert len(report["verified_sources"]["robust_reports"]) == 2


def test_validator_rejects_a_rewritten_decision_even_with_its_new_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = _decision(tmp_path)
    payload = json.loads(decision.read_text(encoding="utf-8"))
    payload["robust_rows"][0]["tail_two_energy_ratio"] = 0.1
    write_json_report(decision, payload)
    output = tmp_path / "validation.json"
    monkeypatch.setattr(sys, "argv", _argv(decision, output))

    with pytest.raises(ValueError, match="does not match its rehashed"):
        validator.main()
