from __future__ import annotations

import pytest

from scripts.build_generation_gate_report import build_report


def _training(parameters: int, token_count: int) -> dict:
    checkpoint_sha256 = ("a" if token_count > 1 else "b") * 64
    return {
        "training_complete": True,
        "completed_steps": 50_000,
        "target_steps": 50_000,
        "parameter_count": parameters,
        "elapsed_seconds": 1_000.0,
        "peak_vram_bytes": 24 * 1024**3,
        "final_metrics": {"samples_seen": 3_200_000},
        "git": {
            "dirty": False,
            "revision": "a" * 40,
            "branch": "scale/generative-system",
        },
        "latest_checkpoint": {
            "checkpoint": "checkpoint_step_00050000.pt",
            "checkpoint_bytes": 1_000_000,
            "checkpoint_sha256": checkpoint_sha256,
            "integrity_manifest": "checkpoint_step_00050000.pt.integrity.json",
            "step": 50_000,
        },
        "config": {
            "data": {"dataset": "imagenet_256_10pct", "batch_size": 16},
            "diffusion": {"schedule": "cosine"},
            "runtime": {"steps": 50_000, "device": "cuda"},
            "optimization": {"batch": 64, "gradient_accumulation_steps": 4},
            "model": {
                "token_count": token_count,
                "token_channels": 64 if token_count > 1 else 3,
                "image_channels": 3,
                "image_size": 256,
                "base_channels": 128,
                "predictor_type": "scalable_unet",
                "predictor_use_feedback": token_count > 1,
                "num_classes": 1000,
                "class_dropout_prob": 0.1,
                "synthesis_mode": "restricted" if token_count > 1 else "dense_identity",
            },
            "loss": {
                "epsilon_weight": 1.0,
                "denoise_path_component_weight": 0.1 if token_count > 1 else 0.0,
            },
        },
    }


def _generation(fid: float, token_count: int, sha: str) -> dict:
    return {
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        "git": {
            "revision": "a" * 40,
            "branch": "scale/generative-system",
            "tracked_dirty": False,
        },
        "implementation": {"package": "torch_fidelity", "version": "0.4.0"},
        "paths": {
            "real_dir": "/datasets/imagenet_256/val",
            "generated_dir": f"/samples/prefix_{token_count}",
        },
        "counts": {"real_image_count": 50_000, "generated_image_count": 10_000},
        "parameters": {"batch_size": 64, "seed": 2027},
        "metrics": {
            "frechet_inception_distance": fid,
            "inception_score_mean": 18.0,
            "inception_score_std": 0.2,
            "precision": 0.6,
            "recall": 0.4,
        },
        "sample_provenance": {
            "git": {
                "revision": "a" * 40,
                "branch": "scale/generative-system",
                "tracked_dirty": False,
            },
            "checkpoint": "/checkpoint.pt",
            "checkpoint_sha256": sha,
            "checkpoint_integrity_manifest": "/checkpoints/checkpoint.pt.integrity.json",
            "checkpoint_step": 50_000,
            "weights": "ema",
            "sample_set_sha256": ("c" if token_count > 1 else "d") * 64,
            "selected_prefix_budget": token_count,
            "sampling_progress": {
                "report": "/samples/sampling_progress.json",
                "status": "completed",
                "invocation": 1,
                "completed_samples": 10_000,
                "cumulative_elapsed_seconds": 100.0,
            },
            "sampling": {
                "num_samples": 10_000,
                "start_index": 0,
                "batch_size": 32,
                "sample_steps": 100,
                "image_shape": [3, 256, 256],
                "class_schedule": "balanced_modulo",
                "prefix_budgets": [token_count],
                "guidance_scale": 1.5,
                "seed": 0,
                "random_stream": {
                    "prefix_budgets_share_stream": True,
                    "batch_size_invariant": True,
                    "resume_index_invariant": True,
                },
            },
        },
    }


def _checkpoint(endpoint: float, sha: str, rank: int = 1) -> dict:
    return {
        "checkpoint_sha256": sha,
        "checkpoint_integrity_manifest": "/checkpoints/checkpoint.pt.integrity.json",
        "checkpoint_step": 50_000,
        "metrics": {
            "orders": {"ordered": {"endpoint_clean_mse": endpoint}},
            "ordered_rank_by_path_auc": rank,
            "order_count": 18,
            "zero_token_max_abs": 0.0,
            "shuffled_to_ordered_endpoint_ratio": 1.5,
        }
    }


def test_generation_gate_passes_matched_quality_and_mechanism() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.102, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "promote_to_full_imagenet256"
    assert all(gate["passed"] for gate in report["gates"])


def test_generation_gate_holds_on_fid_regression() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(25.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    assert report["status"] == "fail"
    assert report["decision"] == "hold"
    assert not next(gate for gate in report["gates"] if gate["name"] == "fid_within_tolerance")[
        "passed"
    ]


def test_generation_gate_holds_on_unpaired_sampling_streams() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    dense["sample_provenance"]["sampling"]["random_stream"]["batch_size_invariant"] = False
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_requires_completed_sampling_progress() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    cofitok["sample_provenance"]["sampling_progress"]["status"] = "running"
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_rejects_dirty_sampling_code() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    cofitok["sample_provenance"]["git"]["tracked_dirty"] = True
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "matched_sampling_code_provenance"
    )
    assert gate["passed"] is False


def test_generation_gate_rejects_dirty_evaluator_code() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    cofitok["git"]["tracked_dirty"] = True
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "matched_evaluator_code_provenance"
    )
    assert gate["passed"] is False


def test_generation_gate_requires_positive_sampling_elapsed_time() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    cofitok["sample_provenance"]["sampling_progress"][
        "cumulative_elapsed_seconds"
    ] = 0.0
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_holds_on_mismatched_training_revision() -> None:
    dense_training = _training(100_000, 1)
    dense_training["git"]["revision"] = "b" * 40
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=dense_training,
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_training_revision")
    assert gate["passed"] is False


def test_generation_gate_holds_on_shared_backbone_drift() -> None:
    dense_training = _training(100_000, 1)
    dense_training["config"]["model"]["class_dropout_prob"] = 0.2
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=dense_training,
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "matched_training_protocol"
    )
    assert report["status"] == "fail"
    assert gate["passed"] is False


def test_generation_gate_requires_exact_sample_count_and_shared_real_set() -> None:
    dense = _generation(20.0, 1, "b" * 64)
    dense["counts"]["generated_image_count"] = 10_001
    dense["paths"]["real_dir"] = "/datasets/other/val"
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_generation_protocol")
    assert gate["passed"] is False


def test_generation_gate_requires_balanced_class_schedule() -> None:
    dense = _generation(20.0, 1, "b" * 64)
    dense["sample_provenance"]["sampling"]["class_schedule"] = None
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_holds_when_sampling_shape_is_unproven() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    del dense["sample_provenance"]["sampling"]["image_shape"]
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_holds_when_sample_set_digest_is_unproven() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    dense = _generation(20.0, 1, "b" * 64)
    del dense["sample_provenance"]["sample_set_sha256"]
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "matched_sampling_provenance")
    assert gate["passed"] is False


def test_generation_gate_holds_when_mechanism_eval_uses_another_checkpoint() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "c" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "checkpoint_evaluation_provenance"
    )
    assert gate["passed"] is False


def test_generation_gate_holds_when_mechanism_eval_uses_another_integrity_manifest() -> None:
    checkpoint = _checkpoint(0.1, "a" * 64)
    checkpoint["checkpoint_integrity_manifest"] = "/checkpoints/other.pt.integrity.json"
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=checkpoint,
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "checkpoint_evaluation_provenance"
    )
    assert gate["passed"] is False


def test_full_generation_gate_uses_ready_decision_and_absolute_fid() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(19.0, 8, "a" * 64),
        dense_generation=_generation(18.5, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    assert report["status"] == "pass"
    assert report["decision"] == "large_scale_generation_ready"
    assert report["stage"] == "full"


def test_full_generation_gate_holds_above_absolute_fid_limit() -> None:
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(20.5, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "absolute_fid_quality")
    assert report["status"] == "fail"
    assert report["decision"] == "hold"
    assert gate["passed"] is False


def test_full_generation_gate_requires_training_checkpoint_integrity() -> None:
    cofitok_training = _training(100_500, 8)
    del cofitok_training["latest_checkpoint"]["checkpoint_sha256"]
    report = build_report(
        cofitok_training=cofitok_training,
        dense_training=_training(100_000, 1),
        cofitok_generation=_generation(19.0, 8, "a" * 64),
        dense_generation=_generation(18.5, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "full_training_checkpoint_integrity"
    )
    assert report["status"] == "fail"
    assert gate["passed"] is False


def test_generation_gate_holds_when_quality_metrics_are_incomplete() -> None:
    cofitok = _generation(20.0, 8, "a" * 64)
    del cofitok["metrics"]["recall"]
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "generation_metrics_complete")
    assert gate["passed"] is False


def test_generation_gate_requires_complete_training_cost_accounting() -> None:
    dense_training = _training(100_000, 1)
    dense_training["final_metrics"]["samples_seen"] -= 1
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=dense_training,
        cofitok_generation=_generation(20.0, 8, "a" * 64),
        dense_generation=_generation(20.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(gate for gate in report["gates"] if gate["name"] == "training_cost_accounting")
    assert report["status"] == "fail"
    assert gate["passed"] is False


@pytest.mark.parametrize(
    ("metric", "value"),
    [
        ("frechet_inception_distance", -1.0),
        ("inception_score_mean", 0.0),
        ("inception_score_std", -0.1),
        ("precision", 1.1),
        ("recall", -0.1),
        ("recall", "not-a-number"),
    ],
)
def test_generation_gate_rejects_finite_but_invalid_metric_ranges(metric, value) -> None:
    cofitok = _generation(19.0, 8, "a" * 64)
    cofitok["metrics"][metric] = value
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=_generation(19.0, 1, "b" * 64),
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
    )

    gate = next(
        gate for gate in report["gates"] if gate["name"] == "distribution_metric_ranges"
    )
    assert report["status"] == "fail"
    assert gate["passed"] is False


@pytest.mark.parametrize(
    ("precision", "recall"),
    [(0.29, 0.40), (0.60, 0.29), (0.54, 0.40), (0.60, 0.34)],
)
def test_full_generation_gate_enforces_precision_recall_floor_and_retention(
    precision, recall
) -> None:
    cofitok = _generation(19.0, 8, "a" * 64)
    dense = _generation(19.0, 1, "b" * 64)
    cofitok["metrics"]["precision"] = precision
    cofitok["metrics"]["recall"] = recall
    dense["metrics"]["precision"] = 0.60
    dense["metrics"]["recall"] = 0.40
    report = build_report(
        cofitok_training=_training(100_500, 8),
        dense_training=_training(100_000, 1),
        cofitok_generation=cofitok,
        dense_generation=dense,
        cofitok_checkpoint=_checkpoint(0.1, "a" * 64),
        dense_checkpoint=_checkpoint(0.1, "b" * 64),
        min_samples=10_000,
        max_fid_regression=0.05,
        max_endpoint_regression=0.05,
        stage="full",
        max_absolute_fid=20.0,
    )

    gate = next(
        gate
        for gate in report["gates"]
        if gate["name"] == "full_precision_recall_quality"
    )
    assert report["status"] == "fail"
    assert gate["passed"] is False
