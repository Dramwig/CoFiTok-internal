from __future__ import annotations

import copy
from argparse import Namespace

import pytest

from cofitok.generation.quality_bridge import RESULT_AUTHORIZATION_BOUNDARY
from cofitok.generation.quality_bridge_followup import (
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_ROLE,
    FOLLOWUP_DECISION_SCHEMA_VERSION,
)
from cofitok.generation.terminal_distribution_support import (
    AUTHORIZATION_BOUNDARY,
    TERMINAL_DISTRIBUTION_SUPPORT_ROUTE,
    build_terminal_distribution_support_report,
    validate_terminal_distribution_support_report,
)
from cofitok.sample_diversity import IMAGE_STATISTIC_FIELDS
from scripts import diagnose_generation_terminal_distribution_support as diagnostic_cli


def _identity(name: str, character: str) -> dict[str, object]:
    return {
        "path": f"/evidence/{name}.json",
        "bytes": 100,
        "sha256": character * 64,
    }


def _sampling(budget: int) -> dict[str, object]:
    return {
        "protocol_schema": "cofitok_ddim_sampling_v1",
        "sampler": "ddim",
        "num_samples": 10_000,
        "sample_steps": 100,
        "start_index": 0,
        "class_schedule": "balanced_modulo",
        "prefix_budgets": [budget],
        "guidance_scale": 1.5,
        "guidance_rescale": 0.0,
        "image_shape": [3, 256, 256],
    }


def _method(method: str) -> dict[str, object]:
    character = "a" if method == "cofitok" else "b"
    budget = 8 if method == "cofitok" else 1
    return {
        "checkpoint_sha256": ("c" if method == "cofitok" else "d") * 64,
        "checkpoint_step": 100_000,
        "sample_set_sha256": character * 64,
        "sample_count": 10_000,
        "sampling": _sampling(budget),
        "sampling_report": _identity(f"{method}_sampling", "e" if method == "cofitok" else "f"),
        "sampling_manifest": _identity(f"{method}_manifest", "1" if method == "cofitok" else "2"),
        "sampling_progress": _identity(f"{method}_progress", "3" if method == "cofitok" else "4"),
        "real_set": {
            "root": "/datasets/imagenet_256/val",
            "image_count": 50_000,
            "sha256": "9" * 64,
            "digest_schema": "cofitok_image_tree_sha256_v1",
        },
        "fid": 120.0 if method == "cofitok" else 118.0,
        "precision": 0.4,
        "recall": 0.02,
    }


def _quality_bridge() -> tuple[dict[str, object], dict[str, object]]:
    identity = _identity("quality_bridge_result", "7")
    return (
        {
            "schema_version": 1,
            "status": "completed",
            "role": "stability_full_data_quality_bridge_result",
            "stage": "stability_quality_bridge",
            "source_reports": {
                "cofitok_generation": _identity("cofitok_metrics", "5"),
                "dense_generation": _identity("dense_metrics", "6"),
            },
            "terminal": {
                "methods": {
                    "cofitok": _method("cofitok"),
                    "dense_identity": _method("dense_identity"),
                }
            },
            "authorization_boundary": copy.deepcopy(RESULT_AUTHORIZATION_BOUNDARY),
        },
        identity,
    )


def _followup(result_identity: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "schema_version": FOLLOWUP_DECISION_SCHEMA_VERSION,
            "status": "completed",
            "role": FOLLOWUP_DECISION_ROLE,
            "source_reports": {"quality_bridge_result": result_identity},
            "recommended_next_stage": {
                "id": TERMINAL_DISTRIBUTION_SUPPORT_ROUTE,
                "category": "recipe_or_objective_intervention",
                "execution_ready": False,
                "gpu_execution_allowed": False,
                "full_300k_launch_allowed": False,
                "release_authorization_allowed": False,
            },
            "authorization_boundary": copy.deepcopy(FOLLOWUP_AUTHORIZATION_BOUNDARY),
        },
        _identity("followup", "8"),
    )


def _distribution(mean: float) -> dict[str, object]:
    return {
        "count": 10_000,
        "min": mean - 0.1,
        "p10": mean - 0.05,
        "p25": mean - 0.02,
        "median": mean,
        "mean": mean,
        "p75": mean + 0.02,
        "p90": mean + 0.05,
        "max": mean + 0.1,
    }


def _cohort(
    cohort_id: str,
    *,
    kind: str,
    digest: str,
    declared: bool,
    statistic_mean: float,
) -> dict[str, object]:
    duplicate = {
        "unique_count": 10_000,
        "duplicate_image_count": 0,
        "duplicate_image_fraction": 0.0,
        "duplicate_pair_count": 0,
    }
    thresholds = {
        "le_0": {"count": 0, "fraction": 0.0},
        "le_4": {"count": 10, "fraction": 0.001},
        "le_8": {"count": 100, "fraction": 0.01},
    }
    distance = {
        "count": 10_000,
        "min": 1,
        "p10": 10.0,
        "p25": 12.0,
        "median": 16.0,
        "mean": 16.0,
        "p75": 20.0,
        "p90": 22.0,
        "max": 30,
    }
    return {
        "cohort_id": cohort_id,
        "cohort_kind": kind,
        "sample_count": 10_000,
        "image_size": [256, 256],
        "cohort_digest": {
            "algorithm": "sha256",
            "framing": "identifier_utf8_nul_file_bytes_nul",
            "sha256": digest,
            "matches_declared_sample_set_sha256": True if declared else None,
        },
        "exact_duplicates": {
            "encoded_file": copy.deepcopy(duplicate),
            "decoded_rgb_pixels": copy.deepcopy(duplicate),
        },
        "dhash64": {
            "global_nearest_thresholds": copy.deepcopy(thresholds),
            "global_nearest_distance": copy.deepcopy(distance),
            "within_class": {
                "pair_thresholds": copy.deepcopy(thresholds),
                "nearest_distance": copy.deepcopy(distance),
            },
        },
        "image_statistics": {
            "metrics": {
                name: _distribution(statistic_mean) for name in IMAGE_STATISTIC_FIELDS
            }
        },
        "per_class": [{"class_index": index} for index in range(1_000)],
    }


def _inputs() -> dict[str, object]:
    result, result_identity = _quality_bridge()
    followup, followup_identity = _followup(result_identity)
    method_sources = {}
    cohorts = {}
    for method in ("cofitok", "dense_identity"):
        row = result["terminal"]["methods"][method]
        metrics_key = "cofitok_generation" if method == "cofitok" else "dense_generation"
        method_sources[method] = {
            "metrics_report": result["source_reports"][metrics_key],
            "sampling_report": row["sampling_report"],
            "sampling_manifest": row["sampling_manifest"],
            "sampling_progress": row["sampling_progress"],
            "generated_dir": f"/samples/{method}/prefix_{8 if method == 'cofitok' else 1}",
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_step": 100_000,
            "sample_set_sha256": row["sample_set_sha256"],
            "sampling": row["sampling"],
        }
        cohorts[method] = _cohort(
            method,
            kind="quality_bridge_terminal_generated",
            digest=row["sample_set_sha256"],
            declared=True,
            statistic_mean=0.4 if method == "cofitok" else 0.35,
        )
    cohorts["real_reference"] = _cohort(
        "real_reference",
        kind="balanced_real_reference",
        digest="0" * 64,
        declared=False,
        statistic_mean=0.3,
    )
    return {
        "followup_decision": followup,
        "followup_decision_identity": followup_identity,
        "quality_bridge_result": result,
        "quality_bridge_result_identity": result_identity,
        "method_sources": method_sources,
        "real_set": result["terminal"]["methods"]["cofitok"]["real_set"],
        "real_reference_selection": {
            "class_count": 1_000,
            "samples_per_class": 10,
            "selected_image_count": 10_000,
        },
        "cohorts": cohorts,
        "diagnostic_git": {
            "revision": "a" * 40,
            "branch": "analysis/generation-terminal-distribution-support-v1",
            "tracked_dirty": False,
        },
        "runtime": {"gpu_requested": False, "cuda_visible_devices": "-1"},
    }


def test_terminal_distribution_support_report_is_descriptive_and_non_authorizing() -> None:
    report = build_terminal_distribution_support_report(**_inputs())
    assert report["status"] == "completed"
    assert report["authorization_boundary"] == AUTHORIZATION_BOUNDARY
    assert report["authorization_boundary"]["recipe_probe_execution_allowed"] is False
    assert report["interpretation"]["causal_explanation_of_terminal_quality_established"] is False
    assert report["comparisons_to_real_reference"]["cofitok"]["image_statistics"][
        "grayscale_laplacian_rms"
    ]["mean_minus_real"] == pytest.approx(0.1)
    assert validate_terminal_distribution_support_report(report)["status"] == "verified"


def test_terminal_distribution_support_requires_exact_selected_route() -> None:
    inputs = _inputs()
    inputs["followup_decision"]["recommended_next_stage"]["id"] = (
        "prepare_matched_250m_capacity_qualification_probe"
    )
    with pytest.raises(ValueError, match="does not select"):
        build_terminal_distribution_support_report(**inputs)


def test_terminal_distribution_support_rejects_quality_bridge_source_drift() -> None:
    inputs = _inputs()
    inputs["quality_bridge_result_identity"] = _identity("other", "f")
    with pytest.raises(ValueError, match="binding differs"):
        build_terminal_distribution_support_report(**inputs)


def test_terminal_distribution_support_rejects_generated_sample_digest_drift() -> None:
    inputs = _inputs()
    inputs["cohorts"]["cofitok"]["cohort_digest"]["sha256"] = "f" * 64
    with pytest.raises(ValueError, match="sample digest differs"):
        build_terminal_distribution_support_report(**inputs)


def test_terminal_distribution_support_validator_rejects_authorization_upgrade() -> None:
    report = build_terminal_distribution_support_report(**_inputs())
    report["authorization_boundary"]["recipe_probe_execution_allowed"] = True
    with pytest.raises(ValueError, match="boundary differs"):
        validate_terminal_distribution_support_report(report)


def test_terminal_distribution_support_cli_requires_hidden_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES=-1"):
        diagnostic_cli.run(Namespace(nearest_chunk_size=8))


def test_terminal_distribution_support_cli_requires_zero_based_terminal_protocol() -> None:
    sampling = _sampling(8)
    assert diagnostic_cli._terminal_protocol(sampling, method="cofitok") == (256, 256)
    sampling["start_index"] = 10_000
    with pytest.raises(ValueError, match="protocol differs"):
        diagnostic_cli._terminal_protocol(sampling, method="cofitok")
