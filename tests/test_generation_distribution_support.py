from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import pytest

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import runtime_environment_sha256
from cofitok.generation import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from cofitok.generation.distribution_support import (
    build_stability_distribution_support_qualification,
)
from cofitok.generation_gate_sources import (
    GATE_SOURCE_SUFFIXES,
    build_generation_gate_source_reports,
)
from cofitok.reporting import write_json_report
import scripts.build_generation_stability_distribution_support as qualification_cli
from scripts.build_generation_stability_distribution_support import (
    build_bound_report,
)


BUILDER_GIT = {
    "revision": "f" * 40,
    "branch": "scale/generation-large-capacity",
    "tracked_dirty": False,
}


def _metrics_report(
    *,
    precision: float = 0.60,
    recall: float = 0.40,
    token_count: int = 8,
    weights: str = "ema",
    schema_version: int = 2,
) -> dict:
    checkpoint_sha = ("a" if token_count > 1 else "b") * 64
    sample_sha = ("c" if token_count > 1 else "d") * 64
    git = {
        "revision": "e" * 40,
        "branch": "scale/generation-stability-50k-posteval-v4",
        "tracked_dirty": False,
    }
    sampling = {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": copy.deepcopy(INFERENCE_API),
        "sampler": "ddim",
        "num_samples": 10_000,
        "start_index": 0,
        "batch_size": 32,
        "sample_steps": 100,
        "num_train_timesteps": 1000,
        "actual_timesteps": select_sampling_timesteps(1000, 100),
        "image_shape": [3, 256, 256],
        "class_schedule": "balanced_modulo",
        "prefix_budgets": [token_count],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "cfg_batch_mode": "batched",
        "eta": 0.0,
        "clip_x0": True,
        "seed": 0,
        "precision": "bf16",
        "random_stream": {
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": True,
            "resume_index_invariant": True,
        },
    }
    evaluator_environment = {
        "schema_version": 1,
        "device": {"type": "cuda", "name": "GPU"},
    }
    sampling_environment = {
        "schema_version": 1,
        "device": {"type": "cuda", "name": "GPU"},
    }
    report = {
        "schema_version": schema_version,
        "status": "completed",
        "protocol": "torch_fidelity_directory_metrics",
        "git": copy.deepcopy(git),
        "runtime_environment": evaluator_environment,
        "runtime_environment_sha256": runtime_environment_sha256(
            evaluator_environment
        ),
        "implementation": {"package": "torch_fidelity", "version": "0.4.0"},
        "paths": {
            "real_dir": "/datasets/imagenet_256/extracted/val",
            "generated_dir": f"/samples/prefix_{token_count}",
        },
        "counts": {
            "real_image_count": 50_000,
            "generated_image_count": 10_000,
        },
        "real_set": {
            "digest_schema": "cofitok_image_tree_sha256_v1",
            "sha256": "2" * 64,
            "root": "/datasets/imagenet_256/extracted/val",
            "image_count": 50_000,
        },
        "metrics": {
            "frechet_inception_distance": 19.0,
            "inception_score_mean": 18.0,
            "inception_score_std": 0.2,
            "precision": precision,
            "recall": recall,
        },
        "sample_provenance": {
            "checkpoint": "/checkpoints/checkpoint_step_00050000.pt",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_integrity_manifest": (
                "/checkpoints/checkpoint_step_00050000.pt.integrity.json"
            ),
            "checkpoint_step": 50_000,
            "weights": weights,
            "git": git,
            "runtime_environment": sampling_environment,
            "runtime_environment_sha256": runtime_environment_sha256(
                sampling_environment
            ),
            "selected_prefix_budget": token_count,
            "sample_set_sha256": sample_sha,
            "sampling_progress": {
                "status": "completed",
                "completed_samples": 10_000,
                "cumulative_elapsed_seconds": 100.0,
            },
            "sampling": sampling,
        },
    }
    if schema_version == 3:
        report["role"] = "generation_directory_metrics_report"
    return report


def _bound_gate(
    tmp_path: Path,
    *,
    cofitok: dict | None = None,
    dense: dict | None = None,
    gate_status: str = "pass",
) -> tuple[Path, dict[str, Path]]:
    cofitok = cofitok or _metrics_report(token_count=8)
    dense = dense or _metrics_report(token_count=1)
    profile_root = (
        Path("\\\\?\\" + str(tmp_path.resolve()))
        if os.name == "nt"
        else tmp_path
    )
    paths: dict[str, Path] = {}
    for index, (name, suffix) in enumerate(
        GATE_SOURCE_SUFFIXES["stability_scaling"].items()
    ):
        path = profile_root / suffix
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            cofitok
            if name == "cofitok_generation"
            else dense
            if name == "dense_generation"
            else {"name": name, "index": index}
        )
        write_json_report(path, payload)
        paths[name] = path
    gate = {
        "schema_version": 2,
        "stage": "scaling",
        "source_profile": "stability_scaling",
        "status": gate_status,
        "decision": (
            "promote_to_full_imagenet256" if gate_status == "pass" else "hold"
        ),
        "source_reports": build_generation_gate_source_reports(
            stage="scaling",
            profile="stability_scaling",
            paths=paths,
        ),
        "summary": {
            "cofitok_fid": cofitok["metrics"]["frechet_inception_distance"],
            "dense_fid": dense["metrics"]["frechet_inception_distance"],
            "cofitok_inception_score": cofitok["metrics"][
                "inception_score_mean"
            ],
            "dense_inception_score": dense["metrics"][
                "inception_score_mean"
            ],
            "cofitok_precision": cofitok["metrics"]["precision"],
            "dense_precision": dense["metrics"]["precision"],
            "cofitok_recall": cofitok["metrics"]["recall"],
            "dense_recall": dense["metrics"]["recall"],
        },
    }
    gate_path = tmp_path / "reports" / "promotion_gate.json"
    write_json_report(gate_path, gate)
    return gate_path, paths


def test_frozen_schema_v2_gate_gets_source_bound_distribution_support(
    tmp_path: Path,
) -> None:
    gate_path, _ = _bound_gate(tmp_path)

    report = build_bound_report(gate_path, builder_git=BUILDER_GIT)

    assert report["status"] == "pass"
    assert report["decision"] == "distribution_support_qualified"
    assert report["base_gate"]["schema_version"] == 2
    assert report["claim_boundary"] == {
        "supplemental_non_authorizing": True,
        "replaces_generation_gate": False,
        "replaces_rollout_stability_qualification": False,
        "full_training_launch_allowed": False,
        "interpretation": (
            "Matched 10K EMA precision/recall non-collapse qualification only"
        ),
    }
    assert all(check["passed"] for check in report["checks"])
    assert report["decision_boundary"] == {
        "base_gate_passed": True,
        "distribution_support_passed": True,
        "base_gate_and_distribution_support_passed": True,
        "scaling_authorization_evaluated": False,
        "full_training_launch_allowed": False,
    }
    assert set(report["sources"]) == {
        "promotion_gate",
        "cofitok_generation",
        "dense_generation",
    }


@pytest.mark.parametrize(
    ("precision", "recall", "dense_precision", "dense_recall", "failed_check"),
    (
        (0.09, 0.40, 0.60, 0.40, "minimum_precision"),
        (0.60, 0.09, 0.60, 0.40, "minimum_recall"),
        (0.54, 0.40, 0.60, 0.40, "matched_precision_retention"),
        (0.60, 0.34, 0.60, 0.40, "matched_recall_retention"),
    ),
)
def test_distribution_support_holds_on_collapse_or_dense_regression(
    tmp_path: Path,
    precision: float,
    recall: float,
    dense_precision: float,
    dense_recall: float,
    failed_check: str,
) -> None:
    gate_path, _ = _bound_gate(
        tmp_path,
        cofitok=_metrics_report(precision=precision, recall=recall, token_count=8),
        dense=_metrics_report(
            precision=dense_precision,
            recall=dense_recall,
            token_count=1,
        ),
    )

    report = build_bound_report(gate_path, builder_git=BUILDER_GIT)

    assert report["status"] == "fail"
    assert report["decision"] == "hold"
    failed = {check["name"] for check in report["checks"] if not check["passed"]}
    assert failed_check in failed
    assert report["claim_boundary"]["full_training_launch_allowed"] is False


def test_distribution_support_rejects_source_drift(tmp_path: Path) -> None:
    gate_path, paths = _bound_gate(tmp_path)
    paths["cofitok_generation"].write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed after binding"):
        build_bound_report(gate_path, builder_git=BUILDER_GIT)


def test_distribution_support_holds_nonformal_or_non_ema_metrics(
    tmp_path: Path,
) -> None:
    gate_path, _ = _bound_gate(
        tmp_path,
        cofitok=_metrics_report(token_count=8, weights="model"),
    )

    report = build_bound_report(gate_path, builder_git=BUILDER_GIT)

    contract = next(
        check for check in report["checks"] if check["name"] == "formal_matched_metrics"
    )
    assert report["status"] == "fail"
    assert contract["passed"] is False
    assert contract["evidence"]["cofitok"]["weights"] == "model"


def test_distribution_support_does_not_upgrade_a_failed_base_gate(
    tmp_path: Path,
) -> None:
    gate_path, _ = _bound_gate(tmp_path, gate_status="fail")

    report = build_bound_report(gate_path, builder_git=BUILDER_GIT)

    assert report["status"] == "pass"
    assert report["base_gate"]["status"] == "fail"
    assert report["base_gate"]["decision"] == "hold"
    assert report["claim_boundary"]["replaces_generation_gate"] is False
    assert report["claim_boundary"]["full_training_launch_allowed"] is False
    assert report["decision_boundary"] == {
        "base_gate_passed": False,
        "distribution_support_passed": True,
        "base_gate_and_distribution_support_passed": False,
        "scaling_authorization_evaluated": False,
        "full_training_launch_allowed": False,
    }


def test_distribution_support_accepts_current_schema_v3_metrics(
    tmp_path: Path,
) -> None:
    gate_path, _ = _bound_gate(
        tmp_path,
        cofitok=_metrics_report(token_count=8, schema_version=3),
        dense=_metrics_report(token_count=1, schema_version=3),
    )

    report = build_bound_report(gate_path, builder_git=BUILDER_GIT)

    assert report["status"] == "pass"
    formal = next(
        check for check in report["checks"] if check["name"] == "formal_matched_metrics"
    )
    assert formal["evidence"]["cofitok"]["schema_version"] == 3
    assert formal["evidence"]["dense_identity"]["schema_version"] == 3


def test_distribution_support_holds_gate_summary_metric_drift(
    tmp_path: Path,
) -> None:
    gate_path, _ = _bound_gate(tmp_path)
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    gate["summary"]["cofitok_precision"] = 0.99
    write_json_report(gate_path, gate)

    report = build_bound_report(gate_path, builder_git=BUILDER_GIT)

    binding = next(
        check
        for check in report["checks"]
        if check["name"] == "base_gate_metric_binding"
    )
    assert report["status"] == "fail"
    assert binding["passed"] is False


def test_distribution_support_holds_mismatched_sampling_protocol(
    tmp_path: Path,
) -> None:
    cofitok = _metrics_report(token_count=8)
    dense = _metrics_report(token_count=1)
    dense["sample_provenance"]["sampling"]["seed"] = 9
    gate_path, _ = _bound_gate(tmp_path, cofitok=cofitok, dense=dense)

    report = build_bound_report(gate_path, builder_git=BUILDER_GIT)

    formal = next(
        check for check in report["checks"] if check["name"] == "formal_matched_metrics"
    )
    assert report["status"] == "fail"
    assert formal["evidence"]["checks"]["matched_sampling_protocol"] is False


def test_distribution_support_holds_invalid_runtime_hash(tmp_path: Path) -> None:
    cofitok = _metrics_report(token_count=8)
    cofitok["runtime_environment_sha256"] = "0" * 64
    gate_path, _ = _bound_gate(tmp_path, cofitok=cofitok)

    report = build_bound_report(gate_path, builder_git=BUILDER_GIT)

    formal = next(
        check for check in report["checks"] if check["name"] == "formal_matched_metrics"
    )
    assert report["status"] == "fail"
    assert formal["evidence"]["cofitok"]["evaluator_runtime"]["valid"] is False


def test_distribution_support_cli_writes_once_and_replays_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_path, _ = _bound_gate(tmp_path)
    output = tmp_path / "supplemental" / "distribution_support.json"
    monkeypatch.setattr(
        qualification_cli,
        "git_provenance",
        lambda _root: copy.deepcopy(BUILDER_GIT),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_generation_stability_distribution_support.py",
            "--gate",
            str(gate_path),
            "--output",
            str(output),
            "--require-pass",
        ],
    )

    assert qualification_cli.main() == 0
    original = output.read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        qualification_cli.main()

    monkeypatch.setattr(sys, "argv", [*sys.argv, "--resume"])
    assert qualification_cli.main() == 0
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    ("keyword", "value"),
    (
        ("min_precision", 0.099),
        ("min_recall", 0.099),
        ("max_precision_regression", 0.051),
        ("max_recall_regression", 0.051),
    ),
)
def test_distribution_support_rejects_weaker_thresholds(
    keyword: str, value: float
) -> None:
    with pytest.raises(ValueError, match=keyword):
        build_stability_distribution_support_qualification(
            base_gate={
                "schema_version": 2,
                "stage": "scaling",
                "source_profile": "stability_scaling",
            },
            cofitok_generation=_metrics_report(token_count=8),
            dense_generation=_metrics_report(token_count=1),
            sources={
                "promotion_gate": {},
                "cofitok_generation": {},
                "dense_generation": {},
            },
            builder_git=BUILDER_GIT,
            **{keyword: value},
        )
