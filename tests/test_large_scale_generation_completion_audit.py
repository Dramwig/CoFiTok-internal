from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

from scripts.audit_large_scale_generation_completion import (
    MILESTONE_STEPS,
    build_completion_audit,
)


TEN_REVISION = "a" * 40
FULL_REVISION = "b" * 40
ROOT = Path(__file__).resolve().parents[1]


def _training(
    *, steps: int, dataset: str, revision: str, parameters: int, checkpoint_sha: str
) -> dict:
    return {
        "training_complete": True,
        "completed_steps": steps,
        "target_steps": steps,
        "parameter_count": parameters,
        "git": {
            "dirty": False,
            "revision": revision,
            "branch": "scale/generative-system",
        },
        "latest_checkpoint": {
            "checkpoint": f"checkpoint_step_{steps:08d}.pt",
            "step": steps,
            "checkpoint_sha256": checkpoint_sha,
            "integrity_manifest": f"checkpoint_step_{steps:08d}.pt.integrity.json",
        },
        "config": {
            "data": {"dataset": dataset, "batch_size": 16},
            "diffusion": {"schedule_type": "cosine"},
            "runtime": {"precision": "bf16"},
            "optimization": {"gradient_accumulation_steps": 4},
            "model": {"token_count": 8 if parameters > 100_000 else 1},
        },
    }


def _gate(stage: str) -> dict:
    decision = (
        "promote_to_full_imagenet256"
        if stage == "scaling"
        else "large_scale_generation_ready"
    )
    gates = [{"name": "all_evidence", "passed": True}]
    if stage == "full":
        gates.append(
            {
                "name": "matched_sampling_provenance",
                "passed": True,
                "evidence": {
                    "cofitok_checkpoint_sha256": "a" * 64,
                    "dense_checkpoint_sha256": "b" * 64,
                    "cofitok_sample_set_sha256": "A" * 64,
                    "dense_sample_set_sha256": "B" * 64,
                },
            }
        )
    return {
        "stage": stage,
        "status": "pass",
        "decision": decision,
        "gates": gates,
    }


def _training_audit() -> dict:
    return {
        "status": "complete",
        "issues": [],
        "last_step": 300_000,
        "validation": {"logging_complete": True, "event_count": 150},
        "checkpoint": {
            "steps": list(MILESTONE_STEPS),
            "missing_required_steps": [],
        },
    }


def _runtime_selection() -> dict:
    return {
        "status": "selected",
        "git_revision": FULL_REVISION,
        "selected": {
            "micro_batch_size": 16,
            "gradient_accumulation_steps": 4,
            "effective_batch_size": 64,
            "estimated_speedup_over_16x4": 1.0,
        },
    }


def _milestone(step: int, alerts: list[str] | None = None) -> dict:
    row = {
        "checkpoint_step": step,
        "sample_count": 2_048,
        "fid": 20.0,
    }
    return {
        "status": "completed",
        "milestone_step": step,
        "expected_samples": 2_048,
        "methods": {"cofitok": row, "dense_identity": dict(row)},
        "quality_alerts": alerts or [],
    }


def _generation(seed: str) -> dict:
    return {
        "status": "completed",
        "counts": {"generated_image_count": 50_000},
        "sample_provenance": {
            "checkpoint_step": 300_000,
            "weights": "ema",
            "checkpoint_sha256": seed * 64,
            "sample_set_sha256": seed.upper() * 64,
            "checkpoint_integrity_manifest": "/run/checkpoint_step_00300000.pt.integrity.json",
            "sampling_progress": {
                "status": "completed",
                "completed_samples": 50_000,
                "cumulative_elapsed_seconds": 10_000.0,
            },
            "sampling": {
                "batch_size": 64,
                "inference_api": {
                    "name": "cofitok.generation.GenerationSession",
                    "version": 1,
                },
                "random_stream": {"batch_size_invariant": True},
            },
        },
    }


def _comparison() -> dict:
    return {
        "status": "ready",
        "matched_training_rows": [
            {
                "method": "CoFiTok K=8",
                "sample_count": 50_000,
                "checkpoint_sha256": "a" * 64,
                "sample_set_sha256": "A" * 64,
            },
            {
                "method": "Dense identity",
                "sample_count": 50_000,
                "checkpoint_sha256": "b" * 64,
                "sample_set_sha256": "B" * 64,
            },
        ],
        "official_context_rows": [{}, {}, {}],
    }


def _sampling_runtime_selection() -> dict:
    return {
        "status": "selected",
        "git_revision": FULL_REVISION,
        "policy": {
            "shared_candidate_required": True,
            "batch_size_invariant_random_stream_required": True,
        },
        "selected": {
            "batch_size": 64,
            "estimated_speedup_over_baseline": 1.4,
        },
        "checkpoints": {
            "cofitok": {"sha256": "a" * 64, "step": 300_000},
            "dense_identity": {"sha256": "b" * 64, "step": 300_000},
        },
    }


def _visual_audit() -> dict:
    return {
        "status": "completed",
        "claim_policy": {"quantitative_metric": False},
        "indices": [0, 1],
        "prefix_indices": [0, 1],
        "prefix_budgets": [1, 2, 4, 8],
        "sources": {
            "cofitok": {
                "checkpoint_sha256": "a" * 64,
                "sample_set_sha256": "A" * 64,
            },
            "dense_identity": {
                "checkpoint_sha256": "b" * 64,
                "sample_set_sha256": "B" * 64,
            },
        },
        "statistics": {
            "cofitok": {"exact_duplicate_count": 0, "pixel_std": 0.2},
            "dense_identity": {"exact_duplicate_count": 0, "pixel_std": 0.3},
        },
        "panels": {
            "cofitok": {"sha256": "c" * 64, "image_count": 2},
            "dense_identity": {"sha256": "d" * 64, "image_count": 2},
            "cofitok_prefix_paths": {"sha256": "e" * 64, "image_count": 8},
        },
    }


def _kwargs() -> dict:
    return {
        "expected_10pct_revision": TEN_REVISION,
        "expected_full_revision": FULL_REVISION,
        "cofitok_10pct_training": _training(
            steps=50_000,
            dataset="imagenet_256_10pct",
            revision=TEN_REVISION,
            parameters=100_500,
            checkpoint_sha="c" * 64,
        ),
        "dense_10pct_training": _training(
            steps=50_000,
            dataset="imagenet_256_10pct",
            revision=TEN_REVISION,
            parameters=100_000,
            checkpoint_sha="d" * 64,
        ),
        "scaling_gate": _gate("scaling"),
        "cofitok_full_training": _training(
            steps=300_000,
            dataset="imagenet_256",
            revision=FULL_REVISION,
            parameters=100_500,
            checkpoint_sha="a" * 64,
        ),
        "dense_full_training": _training(
            steps=300_000,
            dataset="imagenet_256",
            revision=FULL_REVISION,
            parameters=100_000,
            checkpoint_sha="b" * 64,
        ),
        "cofitok_training_audit": _training_audit(),
        "dense_training_audit": _training_audit(),
        "runtime_selection": _runtime_selection(),
        "sampling_runtime_selection": _sampling_runtime_selection(),
        "visual_audit": _visual_audit(),
        "milestones": {step: _milestone(step) for step in MILESTONE_STEPS},
        "cofitok_generation": _generation("a"),
        "dense_generation": _generation("b"),
        "final_gate": _gate("full"),
        "comparison": _comparison(),
    }


def test_completion_audit_requires_every_large_scale_artifact() -> None:
    report = build_completion_audit(**_kwargs())

    assert report["status"] == "complete"
    assert report["complete"] is True
    assert report["failed_checks"] == []
    assert report["missing_checks"] == []
    assert len(report["checks"]) == 11


def test_completion_audit_reports_missing_work_as_in_progress() -> None:
    kwargs = _kwargs()
    kwargs["dense_full_training"] = None

    report = build_completion_audit(**kwargs)

    assert report["status"] == "in_progress"
    assert report["complete"] is False
    assert report["missing_checks"] == [
        "full_matched_training",
        "full_runtime_selection",
        "formal_50k_generation",
    ]


def test_completion_audit_rejects_failed_final_gate() -> None:
    kwargs = _kwargs()
    kwargs["final_gate"]["status"] = "fail"
    kwargs["final_gate"]["decision"] = "hold"

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["final_generation_gate"]


def test_completion_audit_rejects_incomplete_formal_sampling() -> None:
    kwargs = _kwargs()
    kwargs["cofitok_generation"]["sample_provenance"]["sampling_progress"][
        "completed_samples"
    ] = 49_999

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_50k_generation"]


def test_completion_audit_rejects_stale_gate_and_comparison_provenance() -> None:
    kwargs = _kwargs()
    gate_sampling = kwargs["final_gate"]["gates"][1]["evidence"]
    gate_sampling["cofitok_sample_set_sha256"] = "Z" * 64
    kwargs["comparison"]["matched_training_rows"][1]["checkpoint_sha256"] = "e" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == [
        "final_generation_gate",
        "final_comparison_report",
    ]


def test_completion_audit_rejects_training_that_ignores_selected_runtime() -> None:
    kwargs = _kwargs()
    kwargs["runtime_selection"]["selected"].update(
        micro_batch_size=32,
        gradient_accumulation_steps=2,
        estimated_speedup_over_16x4=1.2,
    )

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["full_runtime_selection"]


def test_completion_audit_rejects_sampling_that_ignores_selected_batch() -> None:
    kwargs = _kwargs()
    kwargs["dense_generation"]["sample_provenance"]["sampling"]["batch_size"] = 32

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_sampling_runtime_selection"]


def test_completion_audit_rejects_sampling_outside_stable_inference_api() -> None:
    kwargs = _kwargs()
    del kwargs["cofitok_generation"]["sample_provenance"]["sampling"]["inference_api"]

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["formal_50k_generation"]


def test_completion_audit_rejects_visual_audit_from_stale_sample_set() -> None:
    kwargs = _kwargs()
    kwargs["visual_audit"]["sources"]["cofitok"]["sample_set_sha256"] = "Z" * 64

    report = build_completion_audit(**kwargs)

    assert report["status"] == "failed"
    assert report["failed_checks"] == ["deterministic_visual_quality_audit"]


def test_completion_audit_preserves_nonblocking_milestone_alerts() -> None:
    kwargs = copy.deepcopy(_kwargs())
    kwargs["milestones"][50_000] = _milestone(
        50_000, ["cofitok_fid_more_than_25pct_above_dense"]
    )

    report = build_completion_audit(**kwargs)

    assert report["complete"] is True
    assert report["warnings"] == [
        "milestone_50000:cofitok_fid_more_than_25pct_above_dense"
    ]


def test_completion_audit_direct_cli_reports_in_progress(tmp_path) -> None:
    output_root = tmp_path / "empty_outputs"
    output_root.mkdir()
    output = tmp_path / "completion_audit.json"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/audit_large_scale_generation_completion.py"),
            "--project-root",
            str(ROOT),
            "--output-root",
            str(output_root),
            "--expected-full-revision",
            FULL_REVISION,
            "--output",
            str(output),
            "--allow-incomplete",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"status": "in_progress"' in result.stdout
    assert output.is_file()
