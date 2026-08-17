from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from cofitok.generation.capacity_completion_result import (
    CAPACITY_COMPLETION_RESULT_BOUNDARY,
    CAPACITY_COMPLETION_RESULT_ROLE,
    CAPACITY_COMPLETION_RESULT_SCHEMA_VERSION,
    validate_capacity_completion_100k_result,
)
from cofitok.generation_gate_sources import verify_generation_gate_source_reports
from cofitok.image_integrity import IMAGE_TREE_DIGEST_SCHEMA
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)

try:
    import audit_generation_matched_uncertainty as uncertainty_audit
except ModuleNotFoundError:  # Imported as scripts.<module> by tests.
    from scripts import audit_generation_matched_uncertainty as uncertainty_audit


ROLE = "generation_capacity_matched_uncertainty_manifest_builder"
SOURCE_KINDS = {
    "capacity_completion_100k": {
        "checkpoint_step": 100_000,
        "sample_count": 10_000,
        "stream_prefix": "full_data_capacity_completion_terminal_100k",
        "output_name": "capacity_completion_100k",
        "default_real_fold_count": 5,
        "default_seed": 3041,
    },
    "capacity_full_300k": {
        "checkpoint_step": 300_000,
        "sample_count": 50_000,
        "stream_prefix": "capacity_full_terminal_300k",
        "output_name": "capacity_full_300k",
        "default_real_fold_count": 1,
        "default_seed": 3051,
    },
}
DEFAULT_REAL_CACHE_NAME = "imagenet256_val_50k_torch_fidelity_v04"
DEFAULT_BATCH_SIZE = 64
DEFAULT_BLOCK_SIZE = 500
DEFAULT_BOOTSTRAP_REPETITIONS = 10_000


def _is_hex(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not numeric") from error
    if not math.isfinite(number):
        raise ValueError(f"{label} is not finite")
    return number


def _absolute_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} is not a non-empty path")
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{label} is not absolute")
    return reject_symlink_chain(path, name=label).resolve()


def _expected_git(
    *,
    revision: str,
    tree: str,
    branch: str,
    label: str,
) -> dict[str, Any]:
    if not _is_hex(revision, length=40) or not _is_hex(tree, length=40) or not branch:
        raise ValueError(f"{label} Git expectation is invalid")
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
    }


def _bound_source(
    sources: Mapping[str, Any],
    name: str,
    *,
    label: str,
) -> dict[str, Any]:
    expected = sources.get(name)
    if not isinstance(expected, Mapping):
        raise ValueError(f"{label} source is missing: {name}")
    path = _absolute_path(expected.get("path"), label=f"{label} {name}")
    actual = file_identity(path)
    if actual != dict(expected):
        raise ValueError(f"{label} source identity differs: {name}")
    return actual


def _generation_paths(
    metrics_source: Mapping[str, Any],
    *,
    label: str,
) -> tuple[dict[str, Path], dict[str, Any]]:
    report_path = _absolute_path(
        metrics_source.get("path"),
        label=f"{label} generation metrics report",
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
    return (
        {
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
        },
        report,
    )


def _terminal_method(
    row: Mapping[str, Any],
    *,
    method: str,
    checkpoint_step: int,
    sample_count: int,
    expected_prefix_budget: int,
) -> dict[str, Any]:
    sampling = row.get("sampling")
    real_set = row.get("real_set")
    sampling_report = row.get("sampling_report")
    if not all(
        isinstance(value, Mapping)
        for value in (sampling, real_set, sampling_report)
    ):
        raise ValueError(f"capacity terminal method is incomplete: {method}")
    if (
        int(row.get("checkpoint_step", -1)) != checkpoint_step
        or int(row.get("sample_count", -1)) != sample_count
        or int(row.get("selected_prefix_budget", -1)) != expected_prefix_budget
        or row.get("weights") != "ema"
        or sampling.get("prefix_budgets") != [expected_prefix_budget]
        or int(sampling.get("num_samples", -1)) != sample_count
        or not _is_hex(row.get("checkpoint_sha256"), length=64)
        or not _is_hex(row.get("sample_set_sha256"), length=64)
        or not _is_hex(
            row.get("metrics_runtime_environment_sha256"), length=64
        )
        or real_set.get("digest_schema") != IMAGE_TREE_DIGEST_SCHEMA
        or not _is_hex(real_set.get("sha256"), length=64)
        or int(real_set.get("image_count", -1)) < sample_count
    ):
        raise ValueError(f"capacity terminal method identity differs: {method}")
    fid = _finite(row.get("fid"), label=f"capacity {method} FID")
    if fid < 0.0:
        raise ValueError(f"capacity {method} FID is negative")
    return dict(row)


def _terminal_method_from_metrics(
    report: Mapping[str, Any],
    *,
    method: str,
    expected_evaluation_git: Mapping[str, Any],
    checkpoint_step: int,
    sample_count: int,
    expected_prefix_budget: int,
) -> dict[str, Any]:
    provenance = report.get("sample_provenance")
    counts = report.get("counts")
    parameters = report.get("parameters")
    real_set = report.get("real_set")
    metrics = report.get("metrics")
    if not all(
        isinstance(value, Mapping)
        for value in (provenance, counts, parameters, real_set, metrics)
    ):
        raise ValueError(f"capacity-full {method} metrics provenance is incomplete")
    sampling = provenance.get("sampling")
    sampling_report = provenance.get("report_identity")
    if not isinstance(sampling, Mapping) or not isinstance(sampling_report, Mapping):
        raise ValueError(f"capacity-full {method} sampling provenance is missing")
    evaluation_git = {
        "revision": expected_evaluation_git["revision"],
        "branch": expected_evaluation_git["branch"],
        "tracked_dirty": False,
    }
    if (
        report.get("git") != evaluation_git
        or int(counts.get("generated_image_count", -1)) != sample_count
        or int(provenance.get("checkpoint_step", -1)) != checkpoint_step
        or provenance.get("weights") != "ema"
        or int(provenance.get("selected_prefix_budget", -1))
        != expected_prefix_budget
        or sampling.get("prefix_budgets") != [expected_prefix_budget]
        or int(sampling.get("num_samples", -1)) != sample_count
        or sampling.get("image_shape") != [3, 256, 256]
        or parameters.get("precision_recall_enabled") is not True
        or not _is_hex(provenance.get("checkpoint_sha256"), length=64)
        or not _is_hex(provenance.get("sample_set_sha256"), length=64)
        or not _is_hex(report.get("runtime_environment_sha256"), length=64)
        or real_set.get("digest_schema") != IMAGE_TREE_DIGEST_SCHEMA
        or not _is_hex(real_set.get("sha256"), length=64)
        or int(real_set.get("image_count", -1)) < sample_count
    ):
        raise ValueError(f"capacity-full {method} generation identity differs")
    fid = _finite(
        metrics.get("frechet_inception_distance"),
        label=f"capacity-full {method} FID",
    )
    if fid < 0.0:
        raise ValueError(f"capacity-full {method} FID is negative")
    return {
        "checkpoint_step": checkpoint_step,
        "sample_count": sample_count,
        "selected_prefix_budget": expected_prefix_budget,
        "weights": "ema",
        "sampling": copy.deepcopy(dict(sampling)),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "sample_set_sha256": provenance["sample_set_sha256"],
        "metrics_runtime_environment_sha256": report[
            "runtime_environment_sha256"
        ],
        "real_set": copy.deepcopy(dict(real_set)),
        "sampling_report": copy.deepcopy(dict(sampling_report)),
        "fid": fid,
    }


def _capacity_completion_source(
    *,
    anchor_path: Path,
    anchor_identity: Mapping[str, Any],
    expected_execution_git: Mapping[str, Any],
    expected_training_git: Mapping[str, Any],
    expected_result_git: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    result = read_json_object(anchor_path, name="capacity completion 100K result")
    validate_capacity_completion_100k_result(
        result,
        expected_execution_revision=str(expected_execution_git["revision"]),
        expected_execution_tree=str(expected_execution_git["tree"]),
        expected_execution_branch=str(expected_execution_git["branch"]),
        expected_training_revision=str(expected_training_git["revision"]),
        expected_training_tree=str(expected_training_git["tree"]),
        expected_training_branch=str(expected_training_git["branch"]),
        expected_result_revision=str(expected_result_git["revision"]),
        expected_result_tree=str(expected_result_git["tree"]),
        expected_result_branch=str(expected_result_git["branch"]),
    )
    if (
        int(result.get("schema_version", -1))
        != CAPACITY_COMPLETION_RESULT_SCHEMA_VERSION
        or result.get("role") != CAPACITY_COMPLETION_RESULT_ROLE
        or result.get("status") != "completed"
        or result.get("authorization_boundary")
        != CAPACITY_COMPLETION_RESULT_BOUNDARY
    ):
        raise ValueError("capacity completion 100K anchor contract differs")
    sources = result.get("source_reports")
    terminal = result.get("terminal")
    methods = terminal.get("methods") if isinstance(terminal, Mapping) else None
    if not isinstance(sources, Mapping) or not isinstance(methods, Mapping):
        raise ValueError("capacity completion 100K terminal evidence is missing")
    metrics_sources = {
        "cofitok": _bound_source(
            sources,
            "cofitok_generation",
            label="capacity completion 100K",
        ),
        "dense_identity": _bound_source(
            sources,
            "dense_generation",
            label="capacity completion 100K",
        ),
    }
    terminal_methods = {
        "cofitok": _terminal_method(
            methods.get("cofitok", {}),
            method="cofitok",
            checkpoint_step=100_000,
            sample_count=10_000,
            expected_prefix_budget=8,
        ),
        "dense_identity": _terminal_method(
            methods.get("dense_identity", {}),
            method="dense_identity",
            checkpoint_step=100_000,
            sample_count=10_000,
            expected_prefix_budget=1,
        ),
    }
    metadata = {
        "kind": "capacity_completion_100k",
        "anchor": dict(anchor_identity),
        "role": result["role"],
        "status": result["status"],
        "execution_git": dict(result["execution_git"]),
        "training_git": dict(result["training_git"]),
        "result_builder_git": dict(result["result_builder_git"]),
        "quality_screen": copy.deepcopy(result.get("quality_screen")),
        "decision_support": copy.deepcopy(result.get("decision_support")),
        "authorization_boundary": copy.deepcopy(
            result["authorization_boundary"]
        ),
    }
    return metadata, terminal_methods, metrics_sources


def _capacity_full_source(
    *,
    anchor_path: Path,
    anchor_identity: Mapping[str, Any],
    expected_training_git: Mapping[str, Any],
    expected_evaluation_git: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    gate = read_json_object(anchor_path, name="capacity-full final generation gate")
    verified = verify_generation_gate_source_reports(gate)
    provenance = gate.get("provenance_contract")
    rows = gate.get("gates")
    expected_provenance = {
        "training_revision": expected_training_git["revision"],
        "training_branch": expected_training_git["branch"],
        "evaluation_revision": expected_evaluation_git["revision"],
        "evaluation_branch": expected_evaluation_git["branch"],
    }
    if (
        gate.get("stage") != "full"
        or gate.get("source_profile") != "capacity_full"
        or verified.get("source_profile") != "capacity_full"
        or provenance != expected_provenance
        or not isinstance(rows, list)
        or not rows
        or any(not isinstance(row, Mapping) for row in rows)
        or any(type(row.get("passed")) is not bool for row in rows)
        or len({str(row.get("name", "")) for row in rows}) != len(rows)
    ):
        raise ValueError("capacity-full final gate contract differs")
    passed = all(bool(row["passed"]) for row in rows)
    expected_state = (
        ("pass", "large_scale_generation_ready")
        if passed
        else ("fail", "hold")
    )
    if (gate.get("status"), gate.get("decision")) != expected_state:
        raise ValueError("capacity-full final gate state differs from checks")
    verified_sources = verified.get("source_reports")
    if not isinstance(verified_sources, Mapping):
        raise ValueError("capacity-full final gate sources are missing")
    metrics_sources = {
        "cofitok": dict(verified_sources["cofitok_generation"]),
        "dense_identity": dict(verified_sources["dense_generation"]),
    }
    reports = {
        method: read_json_object(
            Path(metrics_sources[method]["path"]),
            name=f"capacity-full {method} generation metrics",
        )
        for method in ("cofitok", "dense_identity")
    }
    terminal_methods = {
        "cofitok": _terminal_method_from_metrics(
            reports["cofitok"],
            method="cofitok",
            expected_evaluation_git=expected_evaluation_git,
            checkpoint_step=300_000,
            sample_count=50_000,
            expected_prefix_budget=8,
        ),
        "dense_identity": _terminal_method_from_metrics(
            reports["dense_identity"],
            method="dense_identity",
            expected_evaluation_git=expected_evaluation_git,
            checkpoint_step=300_000,
            sample_count=50_000,
            expected_prefix_budget=1,
        ),
    }
    metadata = {
        "kind": "capacity_full_300k",
        "anchor": dict(anchor_identity),
        "role": "capacity_full_final_generation_gate",
        "status": gate["status"],
        "decision": gate["decision"],
        "training_git": dict(expected_training_git),
        "evaluation_git": dict(expected_evaluation_git),
        "provenance_contract": copy.deepcopy(expected_provenance),
        "failed_checks": [
            str(row.get("name", "")) for row in rows if row["passed"] is False
        ],
        "source_report_count": len(verified_sources),
        "diagnostic_report_count": len(
            verified.get("diagnostic_reports", {})
        ),
    }
    return metadata, terminal_methods, metrics_sources


def build_manifest(
    *,
    source_kind: str,
    source_anchor_path: Path,
    expected_source_anchor_sha256: str,
    expected_execution_revision: str = "",
    expected_execution_tree: str = "",
    expected_execution_branch: str = "",
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
    expected_result_revision: str = "",
    expected_result_tree: str = "",
    expected_result_branch: str = "",
    expected_evaluation_revision: str = "",
    expected_evaluation_tree: str = "",
    expected_evaluation_branch: str = "",
    output_root: Path,
    real_cache_name: str = DEFAULT_REAL_CACHE_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    block_size: int = DEFAULT_BLOCK_SIZE,
    real_fold_count: int | None = None,
    bootstrap_repetitions: int = DEFAULT_BOOTSTRAP_REPETITIONS,
    seed: int | None = None,
    cpu: bool = False,
) -> dict[str, Any]:
    spec = SOURCE_KINDS.get(source_kind)
    if spec is None:
        raise ValueError("capacity uncertainty source kind is invalid")
    if (
        not _is_hex(expected_source_anchor_sha256, length=64)
        or not real_cache_name
        or batch_size < 1
        or block_size < 2
        or bootstrap_repetitions < 100
    ):
        raise ValueError("capacity uncertainty manifest parameters are invalid")
    effective_real_folds = (
        int(spec["default_real_fold_count"])
        if real_fold_count is None
        else real_fold_count
    )
    effective_seed = int(spec["default_seed"]) if seed is None else seed
    if effective_real_folds < 1 or effective_seed < 0:
        raise ValueError("capacity uncertainty statistical parameters are invalid")

    training_git = _expected_git(
        revision=expected_training_revision,
        tree=expected_training_tree,
        branch=expected_training_branch,
        label="capacity training",
    )
    anchor_path = reject_symlink_chain(
        source_anchor_path,
        name="capacity uncertainty source anchor",
    ).resolve()
    anchor_identity = file_identity(anchor_path)
    if anchor_identity["sha256"] != expected_source_anchor_sha256:
        raise ValueError("capacity uncertainty source anchor SHA256 differs")

    if source_kind == "capacity_completion_100k":
        execution_git = _expected_git(
            revision=expected_execution_revision,
            tree=expected_execution_tree,
            branch=expected_execution_branch,
            label="capacity completion execution",
        )
        result_git = _expected_git(
            revision=expected_result_revision,
            tree=expected_result_tree,
            branch=expected_result_branch,
            label="capacity completion result",
        )
        source_metadata, methods, metrics_sources = _capacity_completion_source(
            anchor_path=anchor_path,
            anchor_identity=anchor_identity,
            expected_execution_git=execution_git,
            expected_training_git=training_git,
            expected_result_git=result_git,
        )
    else:
        evaluation_git = _expected_git(
            revision=expected_evaluation_revision,
            tree=expected_evaluation_tree,
            branch=expected_evaluation_branch,
            label="capacity-full evaluation",
        )
        source_metadata, methods, metrics_sources = _capacity_full_source(
            anchor_path=anchor_path,
            anchor_identity=anchor_identity,
            expected_training_git=training_git,
            expected_evaluation_git=evaluation_git,
        )

    cofitok = methods["cofitok"]
    dense = methods["dense_identity"]
    matched = uncertainty_audit.validate_matched_sampling(
        {"sampling": cofitok["sampling"]},
        {"sampling": dense["sampling"]},
    )
    sample_count = int(spec["sample_count"])
    checkpoint_step = int(spec["checkpoint_step"])
    if int(matched["sample_count"]) != sample_count:
        raise ValueError("capacity matched sample count differs")
    if (
        cofitok["real_set"] != dense["real_set"]
        or cofitok["metrics_runtime_environment_sha256"]
        != dense["metrics_runtime_environment_sha256"]
    ):
        raise ValueError("capacity terminal evaluator/real-set parity differs")
    real_image_count = int(cofitok["real_set"]["image_count"])
    if sample_count * effective_real_folds > real_image_count:
        raise ValueError("capacity uncertainty real-fold request exceeds real set")
    if sample_count % block_size != 0 or sample_count // block_size < 8:
        raise ValueError("capacity uncertainty block partition is invalid")

    generation_rows = {
        method: _generation_paths(metrics_sources[method], label=method)
        for method in ("cofitok", "dense_identity")
    }
    generation_paths = {
        method: generation_rows[method][0]
        for method in ("cofitok", "dense_identity")
    }
    real_dir = generation_paths["cofitok"]["real_dir"]
    if generation_paths["dense_identity"]["real_dir"] != real_dir:
        raise ValueError("capacity terminal real directories differ")

    sampling_sources: dict[str, dict[str, Any]] = {}
    for method, row in (("cofitok", cofitok), ("dense_identity", dense)):
        expected = row["sampling_report"]
        path = _absolute_path(
            expected.get("path"),
            label=f"capacity {method} sampling report",
        )
        actual = file_identity(path)
        if actual != expected:
            raise ValueError(f"capacity {method} sampling report differs")
        if generation_paths[method]["sampling_report"] != path:
            raise ValueError(f"capacity {method} sampling path differs")
        sampling_sources[method] = actual

    output_root = reject_symlink_chain(
        output_root,
        name="capacity uncertainty output root",
    ).resolve()
    start_index = int(matched["start_index"])
    end_index = int(matched["end_index_exclusive"])
    stream_id = (
        f"{spec['stream_prefix']}_{start_index:08d}_{end_index:08d}"
    )
    audit_output = (
        output_root / str(spec["output_name"]) / "matched_uncertainty.json"
    )
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
        "min_samples": sample_count,
        "block_size": block_size,
        "real_fold_count": effective_real_folds,
        "bootstrap_repetitions": bootstrap_repetitions,
        "seed": effective_seed,
        "cpu": cpu,
    }
    return {
        "schema_version": uncertainty_audit.EXECUTION_MANIFEST_SCHEMA_VERSION,
        "role": uncertainty_audit.EXECUTION_MANIFEST_ROLE,
        "stream_id": stream_id,
        "claim_boundary": uncertainty_audit.CLAIM_BOUNDARY,
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
                "sample_count": sample_count,
                "cofitok_prefix_budgets": matched["cofitok_prefix_budgets"],
                "dense_prefix_budgets": matched["dense_prefix_budgets"],
                "signature_sha256": uncertainty_audit._canonical_sha256(
                    matched["signature"]
                ),
            },
            "real_set": copy.deepcopy(cofitok["real_set"]),
            "sample_sets": {
                "cofitok": {
                    "sha256": cofitok["sample_set_sha256"],
                    "checkpoint_sha256": cofitok["checkpoint_sha256"],
                    "checkpoint_step": checkpoint_step,
                },
                "dense_identity": {
                    "sha256": dense["sample_set_sha256"],
                    "checkpoint_sha256": dense["checkpoint_sha256"],
                    "checkpoint_step": checkpoint_step,
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
        "capacity_source": source_metadata,
        "builder": {
            "role": ROLE,
            "source": file_identity(Path(__file__).resolve()),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a write-once matched-uncertainty execution manifest from "
            "a source-bound capacity 100K result or capacity-full 300K gate."
        )
    )
    parser.add_argument("--source-kind", choices=sorted(SOURCE_KINDS), required=True)
    parser.add_argument("--source-anchor", type=Path, required=True)
    parser.add_argument("--expected-source-anchor-sha256", required=True)
    parser.add_argument("--expected-execution-revision", default="")
    parser.add_argument("--expected-execution-tree", default="")
    parser.add_argument("--expected-execution-branch", default="")
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-result-revision", default="")
    parser.add_argument("--expected-result-tree", default="")
    parser.add_argument("--expected-result-branch", default="")
    parser.add_argument("--expected-evaluation-revision", default="")
    parser.add_argument("--expected-evaluation-tree", default="")
    parser.add_argument("--expected-evaluation-branch", default="")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument("--real-cache-name", default=DEFAULT_REAL_CACHE_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--real-fold-count", type=int)
    parser.add_argument(
        "--bootstrap-repetitions",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )
    parser.add_argument("--seed", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = reject_symlink_chain(
        args.output_root,
        name="capacity uncertainty output root",
    ).resolve()
    manifest_output = reject_symlink_chain(
        args.manifest_output,
        name="capacity uncertainty manifest output",
    ).resolve()
    try:
        manifest_output.relative_to(output_root)
    except ValueError as error:
        raise ValueError("capacity uncertainty manifest is outside output root") from error
    manifest = build_manifest(
        source_kind=args.source_kind,
        source_anchor_path=args.source_anchor,
        expected_source_anchor_sha256=args.expected_source_anchor_sha256,
        expected_execution_revision=args.expected_execution_revision,
        expected_execution_tree=args.expected_execution_tree,
        expected_execution_branch=args.expected_execution_branch,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        expected_result_revision=args.expected_result_revision,
        expected_result_tree=args.expected_result_tree,
        expected_result_branch=args.expected_result_branch,
        expected_evaluation_revision=args.expected_evaluation_revision,
        expected_evaluation_tree=args.expected_evaluation_tree,
        expected_evaluation_branch=args.expected_evaluation_branch,
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
