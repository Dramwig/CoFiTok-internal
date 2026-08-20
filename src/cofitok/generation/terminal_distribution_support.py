from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from pathlib import PurePosixPath
from typing import Any

from cofitok.generation.quality_bridge import RESULT_AUTHORIZATION_BOUNDARY
from cofitok.generation.quality_bridge_followup import (
    AUTHORIZATION_BOUNDARY as FOLLOWUP_AUTHORIZATION_BOUNDARY,
    FOLLOWUP_DECISION_ROLE,
    FOLLOWUP_DECISION_SCHEMA_VERSION,
    QUALITY_BRIDGE_RESULT_ROLE,
    QUALITY_BRIDGE_RESULT_STAGE,
)
from cofitok.sample_diversity import IMAGE_STATISTIC_FIELDS


TERMINAL_DISTRIBUTION_SUPPORT_SCHEMA_VERSION = 1
TERMINAL_DISTRIBUTION_SUPPORT_ROLE = (
    "stability_quality_bridge_terminal_distribution_support_diagnostic"
)
TERMINAL_DISTRIBUTION_SUPPORT_ROUTE = (
    "diagnose_terminal_distribution_support_then_recipe_probe"
)
METHODS = ("cofitok", "dense_identity")
EXPECTED_SAMPLE_COUNT = 10_000
EXPECTED_CLASS_COUNT = 1_000
EXPECTED_SAMPLES_PER_CLASS = 10

AUTHORIZATION_BOUNDARY = {
    "diagnostic_evidence_complete": True,
    "diagnostic_is_formal_quality_gate": False,
    "diagnostic_is_causal_attribution": False,
    "recommended_stage_execution_allowed": False,
    "recipe_probe_execution_allowed": False,
    "training_launch_allowed": False,
    "sampling_launch_allowed": False,
    "evaluation_launch_allowed": False,
    "full_training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_authorization_allowed": False,
    "release_authorization_allowed": False,
    "process_signals_allowed": False,
}


def _hex_digest(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length or value != value.lower():
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _identity(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} identity is missing")
    path = value.get("path")
    size = value.get("bytes")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or not PurePosixPath(path).is_absolute()
        or type(size) is not int
        or size < 1
        or not _hex_digest(digest, length=64)
    ):
        raise ValueError(f"{label} identity is malformed")
    return {"path": path, "bytes": size, "sha256": digest}


def _git(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} Git identity is missing")
    revision = value.get("revision")
    branch = value.get("branch")
    if (
        not _hex_digest(revision, length=40)
        or not isinstance(branch, str)
        or not branch
        or value.get("tracked_dirty") is not False
    ):
        raise ValueError(f"{label} Git identity must be exact and clean")
    return {
        "revision": revision,
        "branch": branch,
        "tracked_dirty": False,
    }


def validate_selected_followup_route(
    followup_decision: Mapping[str, Any],
) -> dict[str, Any]:
    recommendation = followup_decision.get("recommended_next_stage")
    if (
        int(followup_decision.get("schema_version", -1))
        != FOLLOWUP_DECISION_SCHEMA_VERSION
        or followup_decision.get("status") != "completed"
        or followup_decision.get("role") != FOLLOWUP_DECISION_ROLE
        or followup_decision.get("authorization_boundary")
        != FOLLOWUP_AUTHORIZATION_BOUNDARY
        or not isinstance(recommendation, Mapping)
        or recommendation.get("id") != TERMINAL_DISTRIBUTION_SUPPORT_ROUTE
        or recommendation.get("category") != "recipe_or_objective_intervention"
        or recommendation.get("execution_ready") is not False
        or recommendation.get("gpu_execution_allowed") is not False
        or recommendation.get("full_300k_launch_allowed") is not False
        or recommendation.get("release_authorization_allowed") is not False
    ):
        raise ValueError(
            "follow-up decision does not select terminal distribution support"
        )
    return copy.deepcopy(dict(recommendation))


def _validate_quality_bridge_result(
    result: Mapping[str, Any],
    *,
    result_identity: Mapping[str, Any],
    followup_decision: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    identity = _identity(result_identity, label="quality bridge result")
    sources = followup_decision.get("source_reports")
    if (
        int(result.get("schema_version", -1)) != 1
        or result.get("status") != "completed"
        or result.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or result.get("stage") != QUALITY_BRIDGE_RESULT_STAGE
        or result.get("authorization_boundary") != RESULT_AUTHORIZATION_BOUNDARY
        or not isinstance(sources, Mapping)
        or sources.get("quality_bridge_result") != identity
    ):
        raise ValueError("terminal distribution support quality-bridge binding differs")
    terminal = result.get("terminal")
    methods = terminal.get("methods") if isinstance(terminal, Mapping) else None
    if not isinstance(methods, Mapping) or set(methods) != set(METHODS):
        raise ValueError("quality bridge terminal method set differs")
    return identity, {method: methods[method] for method in METHODS}


def _finite(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _validate_cohort(
    value: object,
    *,
    cohort_id: str,
    cohort_kind: str,
    expected_sample_set_sha256: str | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{cohort_id} cohort is missing")
    cohort = copy.deepcopy(dict(value))
    digest = cohort.get("cohort_digest")
    duplicates = cohort.get("exact_duplicates")
    dhash = cohort.get("dhash64")
    statistics = cohort.get("image_statistics")
    metric_rows = statistics.get("metrics") if isinstance(statistics, Mapping) else None
    if (
        cohort.get("cohort_id") != cohort_id
        or cohort.get("cohort_kind") != cohort_kind
        or int(cohort.get("sample_count", -1)) != EXPECTED_SAMPLE_COUNT
        or cohort.get("image_size") != [256, 256]
        or not isinstance(digest, Mapping)
        or digest.get("algorithm") != "sha256"
        or digest.get("framing") != "identifier_utf8_nul_file_bytes_nul"
        or not _hex_digest(digest.get("sha256"), length=64)
        or not isinstance(duplicates, Mapping)
        or not isinstance(dhash, Mapping)
        or not isinstance(metric_rows, Mapping)
        or set(metric_rows) != set(IMAGE_STATISTIC_FIELDS)
        or len(cohort.get("per_class", [])) != EXPECTED_CLASS_COUNT
    ):
        raise ValueError(f"{cohort_id} cohort contract differs")
    if expected_sample_set_sha256 is None:
        if digest.get("matches_declared_sample_set_sha256") is not None:
            raise ValueError("real-reference cohort must not claim a generated digest")
    elif (
        digest.get("sha256") != expected_sample_set_sha256
        or digest.get("matches_declared_sample_set_sha256") is not True
    ):
        raise ValueError(f"{cohort_id} sample digest differs")
    for name in IMAGE_STATISTIC_FIELDS:
        distribution = metric_rows[name]
        if (
            not isinstance(distribution, Mapping)
            or int(distribution.get("count", -1)) != EXPECTED_SAMPLE_COUNT
        ):
            raise ValueError(f"{cohort_id} {name} distribution differs")
        for field in ("min", "p10", "p25", "median", "mean", "p75", "p90", "max"):
            _finite(distribution.get(field), label=f"{cohort_id} {name} {field}")
    return cohort


def _summary_row(
    cohort: Mapping[str, Any],
    *,
    terminal_method: Mapping[str, Any] | None,
) -> dict[str, Any]:
    thresholds = cohort["dhash64"]["global_nearest_thresholds"]
    within = cohort["dhash64"]["within_class"]
    return {
        "cohort_id": cohort["cohort_id"],
        "cohort_kind": cohort["cohort_kind"],
        "sample_count": cohort["sample_count"],
        "fid": terminal_method.get("fid") if terminal_method is not None else None,
        "precision": (
            terminal_method.get("precision") if terminal_method is not None else None
        ),
        "recall": (
            terminal_method.get("recall") if terminal_method is not None else None
        ),
        "encoded_duplicate_image_fraction": cohort["exact_duplicates"][
            "encoded_file"
        ]["duplicate_image_fraction"],
        "pixel_duplicate_image_fraction": cohort["exact_duplicates"][
            "decoded_rgb_pixels"
        ]["duplicate_image_fraction"],
        "global_nearest_dhash_le_4_fraction": thresholds["le_4"]["fraction"],
        "global_nearest_dhash_le_8_fraction": thresholds["le_8"]["fraction"],
        "global_nearest_dhash_median": cohort["dhash64"][
            "global_nearest_distance"
        ]["median"],
        "within_class_pair_dhash_le_4_fraction": within["pair_thresholds"][
            "le_4"
        ]["fraction"],
        "within_class_pair_dhash_le_8_fraction": within["pair_thresholds"][
            "le_8"
        ]["fraction"],
        "within_class_nearest_dhash_median": within["nearest_distance"][
            "median"
        ],
        "image_statistics_mean": {
            name: cohort["image_statistics"]["metrics"][name]["mean"]
            for name in IMAGE_STATISTIC_FIELDS
        },
    }


def _comparisons_to_real(
    generated: Mapping[str, Any],
    real: Mapping[str, Any],
) -> dict[str, Any]:
    generated_stats = generated["image_statistics"]["metrics"]
    real_stats = real["image_statistics"]["metrics"]
    statistics: dict[str, Any] = {}
    for name in IMAGE_STATISTIC_FIELDS:
        generated_mean = _finite(
            generated_stats[name]["mean"], label=f"generated {name} mean"
        )
        real_mean = _finite(real_stats[name]["mean"], label=f"real {name} mean")
        statistics[name] = {
            "generated_mean": generated_mean,
            "real_mean": real_mean,
            "mean_minus_real": generated_mean - real_mean,
            "mean_ratio_to_real": (
                generated_mean / real_mean if real_mean != 0.0 else None
            ),
            "generated_median": generated_stats[name]["median"],
            "real_median": real_stats[name]["median"],
            "median_minus_real": (
                generated_stats[name]["median"] - real_stats[name]["median"]
            ),
        }
    return {
        "pixel_duplicate_image_fraction_minus_real": (
            generated["exact_duplicates"]["decoded_rgb_pixels"]
            ["duplicate_image_fraction"]
            - real["exact_duplicates"]["decoded_rgb_pixels"]
            ["duplicate_image_fraction"]
        ),
        "global_nearest_dhash_le_4_fraction_minus_real": (
            generated["dhash64"]["global_nearest_thresholds"]["le_4"]["fraction"]
            - real["dhash64"]["global_nearest_thresholds"]["le_4"]["fraction"]
        ),
        "global_nearest_dhash_le_8_fraction_minus_real": (
            generated["dhash64"]["global_nearest_thresholds"]["le_8"]["fraction"]
            - real["dhash64"]["global_nearest_thresholds"]["le_8"]["fraction"]
        ),
        "global_nearest_dhash_median_minus_real": (
            generated["dhash64"]["global_nearest_distance"]["median"]
            - real["dhash64"]["global_nearest_distance"]["median"]
        ),
        "within_class_pair_dhash_le_4_fraction_minus_real": (
            generated["dhash64"]["within_class"]["pair_thresholds"]["le_4"]
            ["fraction"]
            - real["dhash64"]["within_class"]["pair_thresholds"]["le_4"]
            ["fraction"]
        ),
        "image_statistics": statistics,
    }


def build_terminal_distribution_support_report(
    *,
    followup_decision: Mapping[str, Any],
    followup_decision_identity: Mapping[str, Any],
    quality_bridge_result: Mapping[str, Any],
    quality_bridge_result_identity: Mapping[str, Any],
    method_sources: Mapping[str, Mapping[str, Any]],
    real_set: Mapping[str, Any],
    real_reference_selection: Mapping[str, Any],
    cohorts: Mapping[str, Mapping[str, Any]],
    diagnostic_git: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    route = validate_selected_followup_route(followup_decision)
    followup_identity = _identity(
        followup_decision_identity,
        label="quality bridge follow-up decision",
    )
    result_identity, terminal_methods = _validate_quality_bridge_result(
        quality_bridge_result,
        result_identity=quality_bridge_result_identity,
        followup_decision=followup_decision,
    )
    git = _git(diagnostic_git, label="terminal distribution support diagnostic")
    if set(method_sources) != set(METHODS) or set(cohorts) != {
        *METHODS,
        "real_reference",
    }:
        raise ValueError("terminal distribution support evidence set differs")

    normalized_sources: dict[str, Any] = {}
    normalized_cohorts: dict[str, Any] = {}
    for method in METHODS:
        terminal = terminal_methods[method]
        if not isinstance(terminal, Mapping):
            raise TypeError(f"quality bridge terminal {method} row is malformed")
        expected_digest = str(terminal.get("sample_set_sha256", ""))
        if not _hex_digest(expected_digest, length=64):
            raise ValueError(f"quality bridge terminal {method} digest is malformed")
        source = method_sources[method]
        metrics_key = "cofitok_generation" if method == "cofitok" else "dense_generation"
        expected_metrics = quality_bridge_result.get("source_reports", {}).get(
            metrics_key
        )
        normalized = {
            "metrics_report": _identity(
                source.get("metrics_report"), label=f"{method} metrics report"
            ),
            "sampling_report": _identity(
                source.get("sampling_report"), label=f"{method} sampling report"
            ),
            "sampling_manifest": _identity(
                source.get("sampling_manifest"), label=f"{method} sampling manifest"
            ),
            "sampling_progress": _identity(
                source.get("sampling_progress"), label=f"{method} sampling progress"
            ),
            "generated_dir": str(source.get("generated_dir", "")),
            "checkpoint_sha256": str(source.get("checkpoint_sha256", "")),
            "checkpoint_step": int(source.get("checkpoint_step", -1)),
            "sample_set_sha256": str(source.get("sample_set_sha256", "")),
            "sampling": copy.deepcopy(source.get("sampling")),
        }
        if (
            normalized["metrics_report"] != expected_metrics
            or normalized["sampling_report"] != terminal.get("sampling_report")
            or normalized["sampling_manifest"] != terminal.get("sampling_manifest")
            or normalized["sampling_progress"] != terminal.get("sampling_progress")
            or not PurePosixPath(normalized["generated_dir"]).is_absolute()
            or normalized["checkpoint_sha256"] != terminal.get("checkpoint_sha256")
            or normalized["checkpoint_step"] != 100_000
            or normalized["sample_set_sha256"] != expected_digest
            or normalized["sampling"] != terminal.get("sampling")
        ):
            raise ValueError(f"terminal {method} physical source binding differs")
        normalized_sources[method] = normalized
        normalized_cohorts[method] = _validate_cohort(
            cohorts[method],
            cohort_id=method,
            cohort_kind="quality_bridge_terminal_generated",
            expected_sample_set_sha256=expected_digest,
        )

    if not isinstance(real_set, Mapping) or real_set != terminal_methods["cofitok"].get(
        "real_set"
    ) or real_set != terminal_methods["dense_identity"].get("real_set"):
        raise ValueError("terminal distribution support real-set identity differs")
    real_cohort = _validate_cohort(
        cohorts["real_reference"],
        cohort_id="real_reference",
        cohort_kind="balanced_real_reference",
        expected_sample_set_sha256=None,
    )
    normalized_cohorts["real_reference"] = real_cohort

    summary_rows = [
        _summary_row(
            normalized_cohorts[method], terminal_method=terminal_methods[method]
        )
        for method in METHODS
    ]
    summary_rows.append(_summary_row(real_cohort, terminal_method=None))
    comparisons = {
        method: _comparisons_to_real(normalized_cohorts[method], real_cohort)
        for method in METHODS
    }
    exact_duplicates_absent = {
        method: normalized_cohorts[method]["exact_duplicates"][
            "decoded_rgb_pixels"
        ]["duplicate_image_count"]
        == 0
        for method in METHODS
    }
    report = {
        "schema_version": TERMINAL_DISTRIBUTION_SUPPORT_SCHEMA_VERSION,
        "role": TERMINAL_DISTRIBUTION_SUPPORT_ROLE,
        "status": "completed",
        "decision": "diagnostic_complete_non_authorizing",
        "git": git,
        "selected_route": route,
        "source_reports": {
            "quality_bridge_followup_decision": followup_identity,
            "quality_bridge_result": result_identity,
            "methods": normalized_sources,
        },
        "source_replay": {
            "followup_decision_rebuilt_byte_equivalent": True,
            "quality_bridge_result_rebuilt_byte_equivalent": True,
            "physical_checkpoint_sample_and_real_set_reverified": True,
            "terminal_sampling_reports_reopened_and_reverified": True,
            "terminal_metrics_reports_reopened_and_reverified": True,
            "generated_sample_set_sha256_recomputed": True,
        },
        "terminal_protocol": copy.deepcopy(terminal_methods["cofitok"]["sampling"]),
        "sources": {
            "real_set": copy.deepcopy(dict(real_set)),
            "real_reference_selection": copy.deepcopy(
                dict(real_reference_selection)
            ),
        },
        "methodology": {
            "exact_file_duplicate": "sha256_of_encoded_file_bytes",
            "exact_pixel_duplicate": (
                "sha256_of_RGB_mode_dimensions_and_decoded_RGB_bytes"
            ),
            "near_duplicate_heuristic": (
                "64-bit grayscale 9x8 horizontal difference hash after Lanczos resize"
            ),
            "image_statistics": {
                "normalization": "decoded_rgb_float_in_unit_interval",
                "fields": list(IMAGE_STATISTIC_FIELDS),
                "interpretation": (
                    "descriptive low-level frequency, contrast, and color statistics only"
                ),
            },
            "real_reference": copy.deepcopy(dict(real_reference_selection)),
            "threshold_policy": (
                "No post-hoc quality, diversity, frequency, or causal threshold is used."
            ),
        },
        "summary_rows": summary_rows,
        "comparisons_to_real_reference": comparisons,
        "interpretation": {
            "exact_decoded_pixel_duplicates_absent": exact_duplicates_absent,
            "literal_exact_duplication_observed_in_any_generated_method": not all(
                exact_duplicates_absent.values()
            ),
            "dhash_is_a_low_level_similarity_heuristic": True,
            "image_statistics_are_descriptive_not_perceptual_quality_metrics": True,
            "causal_explanation_of_terminal_quality_established": False,
            "recipe_or_objective_bottleneck_established": False,
            "training_exposure_saturation_established": False,
            "method_quality_superiority_established": False,
        },
        "cohorts": normalized_cohorts,
        "runtime": copy.deepcopy(dict(runtime)),
        "claim_policy": {
            "existing_terminal_fid_precision_recall_remain_authoritative": True,
            "diagnostic_can_replace_quality_bridge_result": False,
            "diagnostic_can_authorize_recipe_probe": False,
            "diagnostic_can_authorize_training_or_scaling": False,
            "cross_tier_numeric_ranking_allowed": False,
        },
        "authorization_boundary": copy.deepcopy(AUTHORIZATION_BOUNDARY),
    }
    validate_terminal_distribution_support_report(
        report,
        expected_followup_identity=followup_identity,
        expected_quality_bridge_identity=result_identity,
    )
    return report


def validate_terminal_distribution_support_report(
    report: Mapping[str, Any],
    *,
    expected_followup_identity: Mapping[str, Any] | None = None,
    expected_quality_bridge_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sources = report.get("source_reports")
    interpretation = report.get("interpretation")
    claim_policy = report.get("claim_policy")
    if (
        int(report.get("schema_version", -1))
        != TERMINAL_DISTRIBUTION_SUPPORT_SCHEMA_VERSION
        or report.get("role") != TERMINAL_DISTRIBUTION_SUPPORT_ROLE
        or report.get("status") != "completed"
        or report.get("decision") != "diagnostic_complete_non_authorizing"
        or report.get("authorization_boundary") != AUTHORIZATION_BOUNDARY
        or not isinstance(sources, Mapping)
        or not isinstance(interpretation, Mapping)
        or interpretation.get("causal_explanation_of_terminal_quality_established")
        is not False
        or interpretation.get("method_quality_superiority_established") is not False
        or not isinstance(claim_policy, Mapping)
        or claim_policy.get("diagnostic_can_authorize_recipe_probe") is not False
        or claim_policy.get("diagnostic_can_authorize_training_or_scaling") is not False
    ):
        raise ValueError("terminal distribution support report boundary differs")
    route = validate_selected_followup_route(
        {
            "schema_version": FOLLOWUP_DECISION_SCHEMA_VERSION,
            "status": "completed",
            "role": FOLLOWUP_DECISION_ROLE,
            "recommended_next_stage": report.get("selected_route"),
            "authorization_boundary": FOLLOWUP_AUTHORIZATION_BOUNDARY,
        }
    )
    followup = _identity(
        sources.get("quality_bridge_followup_decision"),
        label="terminal diagnostic follow-up decision",
    )
    quality_bridge = _identity(
        sources.get("quality_bridge_result"),
        label="terminal diagnostic quality bridge result",
    )
    if expected_followup_identity is not None and followup != dict(
        expected_followup_identity
    ):
        raise ValueError("terminal diagnostic follow-up identity differs")
    if expected_quality_bridge_identity is not None and quality_bridge != dict(
        expected_quality_bridge_identity
    ):
        raise ValueError("terminal diagnostic quality-bridge identity differs")
    cohorts = report.get("cohorts")
    if not isinstance(cohorts, Mapping) or set(cohorts) != {
        *METHODS,
        "real_reference",
    }:
        raise ValueError("terminal diagnostic cohort set differs")
    if len(report.get("summary_rows", [])) != 3 or set(
        report.get("comparisons_to_real_reference", {})
    ) != set(METHODS):
        raise ValueError("terminal diagnostic summary set differs")
    method_sources = sources.get("methods")
    if not isinstance(method_sources, Mapping) or set(method_sources) != set(METHODS):
        raise ValueError("terminal diagnostic method source set differs")
    for method in METHODS:
        source = method_sources[method]
        if not isinstance(source, Mapping):
            raise ValueError(f"terminal diagnostic {method} source differs")
        expected_digest = str(source.get("sample_set_sha256", ""))
        if not _hex_digest(expected_digest, length=64):
            raise ValueError(f"terminal diagnostic {method} sample digest is malformed")
        _validate_cohort(
            cohorts[method],
            cohort_id=method,
            cohort_kind="quality_bridge_terminal_generated",
            expected_sample_set_sha256=expected_digest,
        )
    _validate_cohort(
        cohorts["real_reference"],
        cohort_id="real_reference",
        cohort_kind="balanced_real_reference",
        expected_sample_set_sha256=None,
    )
    return {
        "status": "verified",
        "route": route["id"],
        "followup_decision_sha256": followup["sha256"],
        "quality_bridge_result_sha256": quality_bridge["sha256"],
        "sample_count_per_cohort": EXPECTED_SAMPLE_COUNT,
        "gpu_execution_allowed": False,
        "training_launch_allowed": False,
    }
