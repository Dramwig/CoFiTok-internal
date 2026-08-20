from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from cofitok.generation.terminal_distribution_support import (
    EXPECTED_CLASS_COUNT,
    EXPECTED_SAMPLE_COUNT,
    EXPECTED_SAMPLES_PER_CLASS,
    METHODS,
    build_terminal_distribution_support_report,
    validate_selected_followup_route,
)
from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance, write_json_report
from cofitok.sample_diversity import (
    REAL_SELECTION_SALT,
    analyze_cohort,
    build_balanced_real_records,
    build_generated_records,
)
from scripts.evaluate_generation_metrics import (
    find_images,
    validate_sampling_provenance,
)

try:
    from scripts.build_generation_quality_bridge_followup_decision import (
        replay_quality_bridge_result,
    )
    from scripts.build_generation_training_exposure_qualification import (
        replay_followup_decision,
    )
except ModuleNotFoundError:
    from build_generation_quality_bridge_followup_decision import (
        replay_quality_bridge_result,
    )
    from build_generation_training_exposure_qualification import (
        replay_followup_decision,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--project", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--followup-decision", type=Path, required=True)
    parser.add_argument("--expected-followup-decision-sha256", required=True)
    parser.add_argument("--expected-followup-revision", required=True)
    parser.add_argument("--expected-followup-branch", required=True)
    parser.add_argument("--quality-bridge-result", type=Path, required=True)
    parser.add_argument("--expected-quality-bridge-result-sha256", required=True)
    parser.add_argument("--expected-diagnostic-revision", required=True)
    parser.add_argument("--expected-diagnostic-branch", required=True)
    parser.add_argument("--nearest-chunk-size", type=int, default=128)


def _validate_git(
    project: Path,
    *,
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    observed = git_provenance(project)
    expected = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if observed != expected:
        raise ValueError(
            f"terminal distribution support checkout differs: {observed} != {expected}"
        )
    return observed


def _terminal_protocol(
    sampling: dict[str, Any],
    *,
    method: str,
) -> tuple[int, int]:
    expected_budget = 8 if method == "cofitok" else 1
    image_shape = sampling.get("image_shape")
    if (
        int(sampling.get("num_samples", -1)) != EXPECTED_SAMPLE_COUNT
        or sampling.get("class_schedule") != "balanced_modulo"
        or int(sampling.get("start_index", -1)) != 0
        or int(sampling.get("sample_steps", -1)) != 100
        or sampling.get("sampler") != "ddim"
        or sampling.get("prefix_budgets") != [expected_budget]
        or float(sampling.get("guidance_scale", -1.0)) != 1.5
        or float(sampling.get("guidance_rescale", -1.0)) != 0.0
        or image_shape != [3, 256, 256]
    ):
        raise ValueError(f"terminal {method} sampling protocol differs")
    return int(image_shape[2]), int(image_shape[1])


def _load_terminal_method(
    *,
    method: str,
    quality_bridge_result: dict[str, Any],
    nearest_chunk_size: int,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_name = "cofitok_generation" if method == "cofitok" else "dense_generation"
    declared_metrics = quality_bridge_result["source_reports"][source_name]
    metrics_identity = file_identity(str(declared_metrics.get("path", "")))
    if metrics_identity != declared_metrics:
        raise ValueError(f"terminal {method} metrics report identity changed")
    metrics = read_json_object(
        metrics_identity["path"],
        name=f"terminal {method} generation metrics",
    )
    terminal = quality_bridge_result["terminal"]["methods"][method]
    declared_sampling = terminal.get("sampling_report")
    if not isinstance(declared_sampling, dict):
        raise ValueError(f"terminal {method} sampling report identity is missing")
    sampling_identity = file_identity(str(declared_sampling.get("path", "")))
    if sampling_identity != declared_sampling:
        raise ValueError(f"terminal {method} sampling report identity changed")

    paths = metrics.get("paths")
    if not isinstance(paths, dict):
        raise ValueError(f"terminal {method} generation paths are missing")
    generated_dir = reject_symlink_chain(
        str(paths.get("generated_dir", "")),
        name=f"terminal {method} generated samples",
    ).resolve()
    if not generated_dir.is_dir():
        raise FileNotFoundError(generated_dir)
    generated_images = find_images(generated_dir)
    recomputed = validate_sampling_provenance(
        Path(sampling_identity["path"]),
        generated_dir,
        generated_images,
    )
    provenance = metrics.get("sample_provenance")
    real_set = metrics.get("real_set")
    if (
        metrics.get("status") != "completed"
        or recomputed != provenance
        or recomputed["report_identity"] != declared_sampling
        or recomputed["manifest_identity"] != terminal.get("sampling_manifest")
        or recomputed["sampling_progress"]["identity"]
        != terminal.get("sampling_progress")
        or recomputed["checkpoint_sha256"] != terminal.get("checkpoint_sha256")
        or int(recomputed["checkpoint_step"]) != 100_000
        or recomputed["sample_set_sha256"] != terminal.get("sample_set_sha256")
        or int(recomputed["sampling_progress"]["completed_samples"])
        != EXPECTED_SAMPLE_COUNT
        or metrics.get("counts", {}).get("generated_image_count")
        != EXPECTED_SAMPLE_COUNT
        or real_set != terminal.get("real_set")
    ):
        raise ValueError(f"terminal {method} physical sampling evidence differs")
    sampling = dict(recomputed["sampling"])
    width, height = _terminal_protocol(sampling, method=method)
    records = build_generated_records(
        generated_dir,
        start_index=0,
        sample_count=EXPECTED_SAMPLE_COUNT,
        class_count=EXPECTED_CLASS_COUNT,
    )
    cohort = analyze_cohort(
        records,
        cohort_id=method,
        cohort_kind="quality_bridge_terminal_generated",
        expected_size=(width, height),
        expected_class_count=EXPECTED_CLASS_COUNT,
        expected_samples_per_class=EXPECTED_SAMPLES_PER_CLASS,
        expected_sample_set_sha256=str(terminal["sample_set_sha256"]),
        nearest_chunk_size=nearest_chunk_size,
    )
    sources = {
        "metrics_report": metrics_identity,
        "sampling_report": sampling_identity,
        "sampling_manifest": recomputed["manifest_identity"],
        "sampling_progress": recomputed["sampling_progress"]["identity"],
        "generated_dir": generated_dir.as_posix(),
        "checkpoint_sha256": recomputed["checkpoint_sha256"],
        "checkpoint_step": recomputed["checkpoint_step"],
        "sample_set_sha256": recomputed["sample_set_sha256"],
        "sampling": sampling,
    }
    return cohort, sources, dict(real_set)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("terminal distribution support requires CUDA_VISIBLE_DEVICES=-1")
    if args.nearest_chunk_size < 1:
        raise ValueError("nearest chunk size must be positive")
    project = args.project.resolve()
    diagnostic_git = _validate_git(
        project,
        expected_revision=args.expected_diagnostic_revision,
        expected_branch=args.expected_diagnostic_branch,
    )
    followup_path = args.followup_decision.resolve()
    result_path = args.quality_bridge_result.resolve()
    followup, followup_identity = replay_followup_decision(
        followup_path,
        expected_sha256=args.expected_followup_decision_sha256,
        expected_revision=args.expected_followup_revision,
        expected_branch=args.expected_followup_branch,
    )
    validate_selected_followup_route(followup)
    result, result_identity = replay_quality_bridge_result(
        result_path,
        expected_sha256=args.expected_quality_bridge_result_sha256,
    )
    if followup.get("source_reports", {}).get("quality_bridge_result") != result_identity:
        raise ValueError("follow-up decision binds another quality bridge result")

    cohorts: dict[str, dict[str, Any]] = {}
    method_sources: dict[str, dict[str, Any]] = {}
    real_set: dict[str, Any] | None = None
    for method in METHODS:
        cohort, sources, declared_real = _load_terminal_method(
            method=method,
            quality_bridge_result=result,
            nearest_chunk_size=args.nearest_chunk_size,
        )
        cohorts[method] = cohort
        method_sources[method] = sources
        if real_set is None:
            real_set = declared_real
        elif declared_real != real_set:
            raise ValueError("terminal methods use different real sets")
    assert real_set is not None
    real_root = reject_symlink_chain(
        str(real_set.get("root", "")), name="terminal real set"
    ).resolve()
    real_records, real_selection = build_balanced_real_records(
        real_root,
        class_count=EXPECTED_CLASS_COUNT,
        samples_per_class=EXPECTED_SAMPLES_PER_CLASS,
        expected_image_count=int(real_set.get("image_count", -1)),
        selection_salt=REAL_SELECTION_SALT,
    )
    cohorts["real_reference"] = analyze_cohort(
        real_records,
        cohort_id="real_reference",
        cohort_kind="balanced_real_reference",
        expected_size=(256, 256),
        expected_class_count=EXPECTED_CLASS_COUNT,
        expected_samples_per_class=EXPECTED_SAMPLES_PER_CLASS,
        nearest_chunk_size=args.nearest_chunk_size,
    )
    runtime = {
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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_requested": False,
    }
    return build_terminal_distribution_support_report(
        followup_decision=followup,
        followup_decision_identity=followup_identity,
        quality_bridge_result=result,
        quality_bridge_result_identity=result_identity,
        method_sources=method_sources,
        real_set=real_set,
        real_reference_selection=real_selection,
        cohorts=cohorts,
        diagnostic_git=diagnostic_git,
        runtime=runtime,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay the selected full-data 100K quality-bridge route and diagnose "
            "exact, dHash, and low-level image-statistic support on the physical "
            "matched terminal 10K cohorts without using a GPU or authorizing work."
        )
    )
    _common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"terminal distribution support report exists: {args.output}"
        )
    report = run(args)
    write_json_report(args.output, report)
    print(json.dumps(report["summary_rows"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
