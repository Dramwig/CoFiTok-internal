from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.quality_bridge import (
    QUALITY_BRIDGE_RECIPE_STAGE,
    QUALITY_BRIDGE_RESULT_ROLE,
    QUALITY_BRIDGE_RESULT_SCHEMA_VERSION,
    QUALITY_BRIDGE_STEPS,
    QUALITY_BRIDGE_TERMINAL_SAMPLES,
    RESULT_AUTHORIZATION_BOUNDARY,
)
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from scripts.audit_generation_matched_uncertainty import (
    CLAIM_BOUNDARY,
    EXECUTION_MANIFEST_ROLE,
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    validate_matched_sampling,
)


ROLE = "quality_bridge_terminal_matched_uncertainty_manifest_builder"
DEFAULT_REAL_CACHE_NAME = "imagenet256_val_50k_torch_fidelity_v04"
DEFAULT_BATCH_SIZE = 64
DEFAULT_BLOCK_SIZE = 500
DEFAULT_REAL_FOLD_COUNT = 5
DEFAULT_BOOTSTRAP_REPETITIONS = 10_000
DEFAULT_SEED = 3031


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_revision(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} is not absolute")
    return reject_symlink_chain(path, name=label).resolve()


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _bound_source(
    sources: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    expected = sources.get(name)
    if not isinstance(expected, Mapping):
        raise ValueError(f"quality bridge source is missing: {name}")
    path = _absolute_path(expected.get("path"), label=f"quality bridge {name}")
    actual = file_identity(path)
    if actual != dict(expected):
        raise ValueError(f"quality bridge source identity differs: {name}")
    return actual


def _terminal_method(
    terminal: Mapping[str, Any],
    method: str,
    *,
    expected_prefix_budget: int,
) -> dict[str, Any]:
    methods = terminal.get("methods")
    if not isinstance(methods, Mapping):
        raise ValueError("quality bridge terminal methods are missing")
    row = methods.get(method)
    if not isinstance(row, Mapping):
        raise ValueError(f"quality bridge terminal method is missing: {method}")
    sampling = row.get("sampling")
    real_set = row.get("real_set")
    sampling_report = row.get("sampling_report")
    if not all(
        isinstance(value, Mapping)
        for value in (sampling, real_set, sampling_report)
    ):
        raise ValueError(f"quality bridge terminal method is incomplete: {method}")
    if (
        int(row.get("checkpoint_step", -1)) != QUALITY_BRIDGE_STEPS
        or int(row.get("sample_count", -1)) != QUALITY_BRIDGE_TERMINAL_SAMPLES
        or int(row.get("selected_prefix_budget", -1)) != expected_prefix_budget
        or row.get("weights") != "ema"
        or sampling.get("prefix_budgets") != [expected_prefix_budget]
        or not _is_sha256(row.get("checkpoint_sha256"))
        or not _is_sha256(row.get("sample_set_sha256"))
        or not _is_sha256(row.get("metrics_runtime_environment_sha256"))
        or real_set.get("digest_schema") != IMAGE_TREE_DIGEST_SCHEMA
        or not _is_sha256(real_set.get("sha256"))
        or int(real_set.get("image_count", -1)) < QUALITY_BRIDGE_TERMINAL_SAMPLES
    ):
        raise ValueError(f"quality bridge terminal identity differs: {method}")
    _finite(row.get("fid"), label=f"quality bridge {method} FID")
    return dict(row)


def _generation_paths(
    metrics_source: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Path]:
    report_path = _absolute_path(
        metrics_source.get("path"),
        label=f"{label} metrics report",
    )
    report = read_json_object(report_path, name=f"{label} generation metrics")
    paths = report.get("paths")
    if (
        int(report.get("schema_version", -1)) != 3
        or report.get("role") != "generation_directory_metrics_report"
        or report.get("protocol") != "torch_fidelity_directory_metrics"
        or report.get("status") != "completed"
        or not isinstance(paths, Mapping)
    ):
        raise ValueError(f"{label} generation metrics report contract differs")
    return {
        "generated_dir": _absolute_path(
            paths.get("generated_dir"),
            label=f"{label} generated directory",
        ),
        "real_dir": _absolute_path(
            paths.get("real_dir"),
            label=f"{label} real directory",
        ),
        "sampling_report": _absolute_path(
            paths.get("sampling_report"),
            label=f"{label} sampling report",
        ),
    }


def build_manifest(
    *,
    quality_result_path: Path,
    expected_quality_result_sha256: str,
    expected_revision: str,
    expected_branch: str,
    output_root: Path,
    real_cache_name: str = DEFAULT_REAL_CACHE_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    block_size: int = DEFAULT_BLOCK_SIZE,
    real_fold_count: int = DEFAULT_REAL_FOLD_COUNT,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int = DEFAULT_SEED,
    cpu: bool = False,
) -> dict[str, Any]:
    if (
        not _is_sha256(expected_quality_result_sha256)
        or not _is_git_revision(expected_revision)
        or not expected_branch
        or not real_cache_name
        or batch_size < 1
        or block_size < 2
        or real_fold_count < 1
        or bootstrap_repetitions < 100
        or seed < 0
    ):
        raise ValueError("quality bridge uncertainty manifest parameters are invalid")

    result_path = reject_symlink_chain(
        quality_result_path,
        name="quality bridge terminal result",
    ).resolve()
    result_identity = file_identity(result_path)
    if result_identity["sha256"] != expected_quality_result_sha256:
        raise ValueError("quality bridge terminal result SHA256 differs")
    result = read_json_object(result_path, name="quality bridge terminal result")
    terminal = result.get("terminal")
    sources = result.get("source_reports")
    quality_git = result.get("git")
    if (
        int(result.get("schema_version", -1))
        != QUALITY_BRIDGE_RESULT_SCHEMA_VERSION
        or result.get("role") != QUALITY_BRIDGE_RESULT_ROLE
        or result.get("status") != "completed"
        or result.get("stage") != QUALITY_BRIDGE_RECIPE_STAGE
        or result.get("authorization_boundary") != RESULT_AUTHORIZATION_BOUNDARY
        or quality_git
        != {
            "revision": expected_revision,
            "branch": expected_branch,
            "tracked_dirty": False,
        }
        or not isinstance(terminal, Mapping)
        or not isinstance(sources, Mapping)
    ):
        raise ValueError("quality bridge terminal result contract differs")

    cofitok = _terminal_method(
        terminal,
        "cofitok",
        expected_prefix_budget=8,
    )
    dense = _terminal_method(
        terminal,
        "dense_identity",
        expected_prefix_budget=1,
    )
    matched = validate_matched_sampling(
        {"sampling": cofitok["sampling"]},
        {"sampling": dense["sampling"]},
    )
    if matched["sample_count"] != QUALITY_BRIDGE_TERMINAL_SAMPLES:
        raise ValueError("quality bridge matched sample count differs")
    if (
        cofitok["real_set"] != dense["real_set"]
        or cofitok["metrics_runtime_environment_sha256"]
        != dense["metrics_runtime_environment_sha256"]
    ):
        raise ValueError("quality bridge terminal evaluator/real-set parity differs")

    metrics_sources = {
        "cofitok": _bound_source(sources, "cofitok_generation"),
        "dense_identity": _bound_source(sources, "dense_generation"),
    }
    generation_paths = {
        method: _generation_paths(
            metrics_sources[method],
            label=method,
        )
        for method in ("cofitok", "dense_identity")
    }
    real_dir = generation_paths["cofitok"]["real_dir"]
    if generation_paths["dense_identity"]["real_dir"] != real_dir:
        raise ValueError("quality bridge terminal real directories differ")

    sampling_sources: dict[str, dict[str, Any]] = {}
    for method, row in (("cofitok", cofitok), ("dense_identity", dense)):
        expected = row["sampling_report"]
        path = _absolute_path(
            expected.get("path"),
            label=f"{method} sampling report",
        )
        actual = file_identity(path)
        if actual != expected:
            raise ValueError(f"quality bridge {method} sampling report differs")
        if generation_paths[method]["sampling_report"] != path:
            raise ValueError(f"quality bridge {method} sampling path differs")
        sampling_sources[method] = actual

    output_root = reject_symlink_chain(
        output_root,
        name="quality bridge uncertainty output root",
    ).resolve()
    start_index = int(matched["start_index"])
    end_index = int(matched["end_index_exclusive"])
    stream_id = (
        "full_data_quality_bridge_terminal_100k_"
        f"{start_index:08d}_{end_index:08d}"
    )
    audit_output = output_root / "terminal_100k" / "matched_uncertainty.json"
    cache_root = output_root / "feature_cache"
    arguments = {
        "real_dir": real_dir.as_posix(),
        "cofitok_generated_dir": generation_paths["cofitok"][
            "generated_dir"
        ].as_posix(),
        "dense_generated_dir": generation_paths["dense_identity"][
            "generated_dir"
        ].as_posix(),
        "cofitok_sampling_report": sampling_sources["cofitok"]["path"],
        "dense_sampling_report": sampling_sources["dense_identity"]["path"],
        "cofitok_metrics_report": metrics_sources["cofitok"]["path"],
        "dense_metrics_report": metrics_sources["dense_identity"]["path"],
        "output": audit_output.as_posix(),
        "cache_root": cache_root.as_posix(),
        "real_cache_name": real_cache_name,
        "batch_size": batch_size,
        "min_samples": QUALITY_BRIDGE_TERMINAL_SAMPLES,
        "block_size": block_size,
        "real_fold_count": real_fold_count,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": seed,
        "cpu": cpu,
    }
    return {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
        "role": EXECUTION_MANIFEST_ROLE,
        "stream_id": stream_id,
        "claim_boundary": CLAIM_BOUNDARY,
        "arguments": arguments,
        "source_files": {
            "cofitok_sampling_report": sampling_sources["cofitok"],
            "dense_sampling_report": sampling_sources["dense_identity"],
            "cofitok_metrics_report": metrics_sources["cofitok"],
            "dense_metrics_report": metrics_sources["dense_identity"],
        },
        "expected": {
            "matched_sampling": {
                "start_index": start_index,
                "end_index_exclusive": end_index,
                "sample_count": int(matched["sample_count"]),
                "cofitok_prefix_budgets": matched["cofitok_prefix_budgets"],
                "dense_prefix_budgets": matched["dense_prefix_budgets"],
                "signature_sha256": _canonical_sha256(matched["signature"]),
            },
            "real_set": cofitok["real_set"],
            "sample_sets": {
                "cofitok": {
                    "sha256": cofitok["sample_set_sha256"],
                    "checkpoint_sha256": cofitok["checkpoint_sha256"],
                    "checkpoint_step": int(cofitok["checkpoint_step"]),
                },
                "dense_identity": {
                    "sha256": dense["sample_set_sha256"],
                    "checkpoint_sha256": dense["checkpoint_sha256"],
                    "checkpoint_step": int(dense["checkpoint_step"]),
                },
            },
            "fid_point_estimates": {
                "cofitok": float(cofitok["fid"]),
                "dense_identity": float(dense["fid"]),
            },
            "metrics_runtime_environment_sha256": cofitok[
                "metrics_runtime_environment_sha256"
            ],
        },
        "quality_bridge": {
            "result": result_identity,
            "git": dict(quality_git),
            "quality_screen": result.get("quality_screen"),
            "authorization_boundary": dict(RESULT_AUTHORIZATION_BOUNDARY),
        },
        "builder": {
            "role": ROLE,
            "source": file_identity(Path(__file__).resolve()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable matched-uncertainty execution manifest from a "
            "verified full-data 100K quality-bridge terminal result."
        )
    )
    parser.add_argument("--quality-result", type=Path, required=True)
    parser.add_argument("--expected-quality-result-sha256", required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--real-cache-name", default=DEFAULT_REAL_CACHE_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument(
        "--real-fold-count",
        type=int,
        default=DEFAULT_REAL_FOLD_COUNT,
    )
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="quality bridge uncertainty output root",
    ).resolve()
    manifest_output = reject_symlink_chain(
        args.manifest_output,
        name="quality bridge uncertainty manifest output",
    ).resolve()
    try:
        manifest_output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("uncertainty manifest output is outside output root") from error
    manifest = build_manifest(
        quality_result_path=args.quality_result,
        expected_quality_result_sha256=args.expected_quality_result_sha256,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        output_root=output_root,
        real_cache_name=args.real_cache_name,
        batch_size=args.batch_size,
        block_size=args.block_size,
        real_fold_count=args.real_fold_count,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
        cpu=args.cpu,
    )
    identity = prepare_manifest(
        manifest_output,
        manifest,
        resume=args.resume,
        overwrite=False,
    )
    print(json.dumps({"status": "pass", "manifest": identity}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
