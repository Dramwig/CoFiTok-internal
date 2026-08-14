from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from cofitok.reporting import file_sha256, write_json_report
from cofitok.sample_diversity import (
    REAL_SELECTION_SALT,
    analyze_cohort,
    build_balanced_real_records,
    build_generated_records,
)


METHODS = ("cofitok", "dense_identity")
EXPECTED_CONFIRMATION_ROLE = "non_authorizing_matched_10000_sampling_confirmation"
REPORT_ROLE = "non_authorizing_matched_10000_sample_diversity_diagnostic"
REPORT_SCHEMA_VERSION = 1
CLAIM_BOUNDARY = {
    "diagnostic_non_authorizing": True,
    "dhash_is_formal_quality_metric": False,
    "replaces_fid_precision_recall": False,
    "replaces_frozen_promotion_gate": False,
    "training_launch_allowed": False,
    "full_300k_launch_allowed": False,
    "promotion_or_release_allowed": False,
}


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _source_matches(declared: Any, path: Path) -> bool:
    if not isinstance(declared, dict):
        return False
    actual = _source(path)
    return all(declared.get(key) == actual[key] for key in ("bytes", "sha256"))


def _git_identity(project: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(project), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "revision": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "tracked_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _validate_git(
    project: Path, *, expected_revision: str, expected_branch: str
) -> dict[str, Any]:
    identity = _git_identity(project)
    expected = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if identity != expected:
        raise ValueError(f"diagnostic Git identity mismatch: {identity} != {expected}")
    return identity


def _validate_confirmation(report: dict[str, Any]) -> dict[str, Any]:
    preflight = report.get("preflight")
    protocol = preflight.get("protocol") if isinstance(preflight, dict) else None
    rows = report.get("rows")
    if (
        report.get("schema_version") != 2
        or report.get("role") != EXPECTED_CONFIRMATION_ROLE
        or report.get("status") != "hold"
        or report.get("quality_confirmed") is not False
        or not isinstance(preflight, dict)
        or preflight.get("status") != "pass"
        or not isinstance(protocol, dict)
        or protocol.get("num_samples") != 10_000
        or protocol.get("class_count") != 1_000
        or protocol.get("class_schedule") != "balanced_modulo"
        or protocol.get("balanced_modulo_exact_coverage_required") is not True
        or protocol.get("start_index") != 10_000
        or not isinstance(rows, dict)
        or set(rows) != set(METHODS)
    ):
        raise ValueError("sampling-confirmation report contract mismatch")
    return protocol


def _load_method_sources(
    *,
    method: str,
    row: dict[str, Any],
    protocol: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    sampling_source = row.get("sampling_report")
    metrics_source = row.get("metrics_report")
    if not isinstance(sampling_source, dict) or not isinstance(metrics_source, dict):
        raise ValueError(f"{method} source identities are missing")
    sampling_path = Path(str(sampling_source.get("path")))
    metrics_path = Path(str(metrics_source.get("path")))
    if not _source_matches(sampling_source, sampling_path):
        raise ValueError(f"{method} sampling report identity changed")
    if not _source_matches(metrics_source, metrics_path):
        raise ValueError(f"{method} metrics report identity changed")
    sampling = _read_object(sampling_path)
    metrics = _read_object(metrics_path)
    prefix_budget = int(row["prefix_budget"])
    output_dirs = sampling.get("output_dirs")
    generated_dir = metrics.get("paths", {}).get("generated_dir")
    sampling_protocol = sampling.get("sampling")
    sample_set = sampling.get("sample_sets", {}).get(str(prefix_budget))
    if (
        sampling.get("status") != "completed"
        or sampling.get("checkpoint_sha256") != row.get("checkpoint_sha256")
        or sampling.get("checkpoint_step") != row.get("checkpoint_step")
        or not isinstance(output_dirs, dict)
        or set(output_dirs) != {str(prefix_budget)}
        or output_dirs[str(prefix_budget)] != generated_dir
        or not isinstance(sampling_protocol, dict)
        or sampling_protocol.get("num_samples") != protocol["num_samples"]
        or sampling_protocol.get("num_classes") != protocol["class_count"]
        or sampling_protocol.get("class_schedule") != protocol["class_schedule"]
        or sampling_protocol.get("start_index") != protocol["start_index"]
        or sampling_protocol.get("seed") != protocol["seed"]
        or not isinstance(sample_set, dict)
        or sample_set.get("count") != row.get("sample_count")
        or sample_set.get("sha256") != row.get("sample_set_sha256")
        or metrics.get("status") != "completed"
        or metrics.get("counts", {}).get("generated_image_count")
        != row.get("sample_count")
        or metrics.get("sample_provenance", {}).get("sample_set_sha256")
        != row.get("sample_set_sha256")
    ):
        raise ValueError(f"{method} sampling/metrics contract mismatch")
    return sampling, metrics, Path(generated_dir)


def _cohort_row(cohort: dict[str, Any], metrics: dict[str, Any] | None) -> dict[str, Any]:
    global_thresholds = cohort["dhash64"]["global_nearest_thresholds"]
    within = cohort["dhash64"]["within_class"]
    metric_values = metrics.get("metrics", {}) if metrics is not None else {}
    return {
        "cohort_id": cohort["cohort_id"],
        "cohort_kind": cohort["cohort_kind"],
        "sample_count": cohort["sample_count"],
        "fid": metric_values.get("frechet_inception_distance"),
        "precision": metric_values.get("precision"),
        "recall": metric_values.get("recall"),
        "encoded_duplicate_image_fraction": cohort["exact_duplicates"]["encoded_file"]
        ["duplicate_image_fraction"],
        "pixel_duplicate_image_fraction": cohort["exact_duplicates"]
        ["decoded_rgb_pixels"]["duplicate_image_fraction"],
        "global_nearest_dhash_le_0_fraction": global_thresholds["le_0"]["fraction"],
        "global_nearest_dhash_le_4_fraction": global_thresholds["le_4"]["fraction"],
        "global_nearest_dhash_le_8_fraction": global_thresholds["le_8"]["fraction"],
        "global_nearest_dhash_median": cohort["dhash64"]["global_nearest_distance"]
        ["median"],
        "within_class_pair_dhash_le_4_fraction": within["pair_thresholds"]["le_4"]
        ["fraction"],
        "within_class_pair_dhash_le_8_fraction": within["pair_thresholds"]["le_8"]
        ["fraction"],
        "within_class_nearest_dhash_median": within["nearest_distance"]["median"],
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    project = args.project.resolve()
    confirmation_path = args.confirmation_report.resolve()
    git = _validate_git(
        project,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )
    confirmation = _read_object(confirmation_path)
    protocol = _validate_confirmation(confirmation)
    samples_per_class = protocol["num_samples"] // protocol["class_count"]
    if samples_per_class * protocol["class_count"] != protocol["num_samples"]:
        raise ValueError("confirmation sample count is not class-balanced")

    cohorts: dict[str, dict[str, Any]] = {}
    metrics_by_method: dict[str, dict[str, Any]] = {}
    source_reports: dict[str, Any] = {}
    real_set: dict[str, Any] | None = None
    for method in METHODS:
        row = confirmation["rows"][method]
        sampling, metrics, generated_dir = _load_method_sources(
            method=method,
            row=row,
            protocol=protocol,
        )
        declared_real = metrics.get("real_set")
        if not isinstance(declared_real, dict):
            raise ValueError(f"{method} metrics report lacks real-set provenance")
        if real_set is None:
            real_set = declared_real
        elif declared_real != real_set:
            raise ValueError("method metrics reports use different real sets")
        records = build_generated_records(
            generated_dir,
            start_index=protocol["start_index"],
            sample_count=protocol["num_samples"],
            class_count=protocol["class_count"],
        )
        cohorts[method] = analyze_cohort(
            records,
            cohort_id=method,
            cohort_kind="generated_confirmation",
            expected_size=(256, 256),
            expected_class_count=protocol["class_count"],
            expected_samples_per_class=samples_per_class,
            expected_sample_set_sha256=row["sample_set_sha256"],
            nearest_chunk_size=args.nearest_chunk_size,
        )
        metrics_by_method[method] = metrics
        source_reports[method] = {
            "sampling_report": _source(Path(row["sampling_report"]["path"])),
            "metrics_report": _source(Path(row["metrics_report"]["path"])),
            "checkpoint_sha256": row["checkpoint_sha256"],
            "checkpoint_step": row["checkpoint_step"],
            "sample_set_sha256": row["sample_set_sha256"],
            "sampling_protocol": sampling["sampling"],
        }

    assert real_set is not None
    if (
        real_set.get("image_count") != 50_000
        or real_set.get("digest_schema") != "cofitok_image_tree_sha256_v1"
        or not isinstance(real_set.get("sha256"), str)
        or not isinstance(real_set.get("root"), str)
    ):
        raise ValueError("real-set provenance contract mismatch")
    real_records, real_selection = build_balanced_real_records(
        Path(real_set["root"]),
        class_count=protocol["class_count"],
        samples_per_class=samples_per_class,
        expected_image_count=real_set["image_count"],
        selection_salt=REAL_SELECTION_SALT,
    )
    cohorts["real_reference"] = analyze_cohort(
        real_records,
        cohort_id="real_reference",
        cohort_kind="balanced_real_reference",
        expected_size=(256, 256),
        expected_class_count=protocol["class_count"],
        expected_samples_per_class=samples_per_class,
        nearest_chunk_size=args.nearest_chunk_size,
    )

    summary_rows = [
        _cohort_row(cohorts[method], metrics_by_method[method]) for method in METHODS
    ]
    summary_rows.append(_cohort_row(cohorts["real_reference"], None))
    real_row = summary_rows[-1]
    comparisons: dict[str, Any] = {}
    for row in summary_rows[:-1]:
        comparisons[row["cohort_id"]] = {
            "pixel_duplicate_image_fraction_minus_real": (
                row["pixel_duplicate_image_fraction"]
                - real_row["pixel_duplicate_image_fraction"]
            ),
            "global_nearest_dhash_le_4_fraction_minus_real": (
                row["global_nearest_dhash_le_4_fraction"]
                - real_row["global_nearest_dhash_le_4_fraction"]
            ),
            "global_nearest_dhash_le_8_fraction_minus_real": (
                row["global_nearest_dhash_le_8_fraction"]
                - real_row["global_nearest_dhash_le_8_fraction"]
            ),
            "global_nearest_dhash_median_minus_real": (
                row["global_nearest_dhash_median"]
                - real_row["global_nearest_dhash_median"]
            ),
            "within_class_pair_dhash_le_4_fraction_minus_real": (
                row["within_class_pair_dhash_le_4_fraction"]
                - real_row["within_class_pair_dhash_le_4_fraction"]
            ),
            "within_class_nearest_dhash_median_minus_real": (
                row["within_class_nearest_dhash_median"]
                - real_row["within_class_nearest_dhash_median"]
            ),
        }

    exact_pixel_duplicates_absent = {
        method: cohorts[method]["exact_duplicates"]["decoded_rgb_pixels"]
        ["duplicate_image_count"]
        == 0
        for method in METHODS
    }
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "role": REPORT_ROLE,
        "status": "completed",
        "git": git,
        "claim_boundary": CLAIM_BOUNDARY,
        "question": (
            "Does the approximately 0.008 recall in the matched 10K confirmation "
            "coincide with literal or low-level near-duplicate sample collapse?"
        ),
        "confirmation": {
            "source": _source(confirmation_path),
            "decision": confirmation["decision"],
            "quality_confirmed": confirmation["quality_confirmed"],
            "protocol": protocol,
        },
        "sources": {
            "methods": source_reports,
            "real_set": real_set,
            "real_reference_selection": real_selection,
        },
        "methodology": {
            "exact_file_duplicate": "sha256_of_encoded_file_bytes",
            "exact_pixel_duplicate": "sha256_of_RGB_mode_dimensions_and_decoded_RGB_bytes",
            "near_duplicate_heuristic": (
                "64-bit grayscale 9x8 horizontal difference hash after Lanczos resize"
            ),
            "global_nearest": "exact Hamming nearest neighbor across each 10000-image cohort",
            "within_class": (
                "all 45 pairs and each image nearest neighbor among 10 samples per class"
            ),
            "real_reference": real_selection,
            "limitations": [
                "dHash detects low-frequency appearance similarity, not semantic mode coverage",
                "dHash collisions are candidates and are not exact duplicates",
                "the balanced real reference is a deterministic 10-of-50 path-hash sample per class",
                "this diagnostic cannot replace FID, precision, recall, or the frozen promotion gate",
            ],
        },
        "summary_rows": summary_rows,
        "comparisons_to_real_reference": comparisons,
        "interpretation": {
            "exact_decoded_pixel_duplicates_absent": exact_pixel_duplicates_absent,
            "literal_exact_duplication_observed_in_any_generated_method": not all(
                exact_pixel_duplicates_absent.values()
            ),
            "near_duplicate_interpretation_requires_real_reference_and_is_heuristic": True,
            "causal_explanation_of_low_recall_established": False,
        },
        "cohorts": cohorts,
        "runtime": {
            "elapsed_seconds": time.monotonic() - started,
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
            },
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pillow": importlib.metadata.version("pillow"),
            "nearest_chunk_size": args.nearest_chunk_size,
            "gpu_requested": False,
        },
    }
    write_json_report(args.output, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose exact and perceptual-hash diversity in a completed 10K confirmation."
    )
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--confirmation-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--nearest-chunk-size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report["summary_rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
