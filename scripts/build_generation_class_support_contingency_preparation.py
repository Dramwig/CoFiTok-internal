"""Build or replay a non-authorizing frozen class-support preparation.

This entry point performs only CPU/file-system source validation. It hashes the
existing 10K-sample trees from the 100K checkpoints and their reports, but never
loads the classifier, launches evaluation, trains, or samples.
"""

# Parsed artifact type mismatches are schema-value failures, not API misuse.
# ruff: noqa: TRY004

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation_class_support_contingency import (
    CLASSIFIER,
    FROZEN_SOURCE_SPECS,
    METHODS,
    SAMPLE_COUNT,
    SOURCE_CHECKPOINT_STEP,
    build_preparation,
    physical_sample_tree_identity,
    stable_file_identity,
    validate_git_identity,
    validate_preparation,
)


def _exact_int(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _read_json_bound(
    path: Path, *, expected_sha256: str | None, name: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    before = stable_file_identity(path)
    if expected_sha256 is not None and before["sha256"] != expected_sha256:
        raise ValueError(f"{name} SHA256 differs")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{name} must contain an object")
    after = stable_file_identity(path)
    if after != before:
        raise RuntimeError(f"{name} changed while it was read")
    return payload, before


def _checkout_identity(project_root: Path, script: Path) -> dict[str, Any]:
    root = project_root.resolve(strict=True)
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    if status:
        raise ValueError("preparation checkout must be completely clean")
    branch = subprocess.check_output(
        ["git", "-C", str(root), "branch", "--show-current"], text=True
    ).strip()
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD^{commit}"], text=True
    ).strip()
    tree = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True
    ).strip()
    if not branch:
        raise ValueError("preparation checkout must be on a named branch")
    resolved_script = script.resolve(strict=True)
    try:
        relative = resolved_script.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("preparation script must be inside the checkout") from error
    subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
        check=True,
        capture_output=True,
    )
    committed = subprocess.check_output(
        ["git", "-C", str(root), "show", f"HEAD:{relative}"]
    )
    if committed != resolved_script.read_bytes():
        raise ValueError("preparation script differs from the exact HEAD blob")
    return validate_git_identity(
        {
            "revision": revision,
            "tree": tree,
            "branch": branch,
            "tracked_dirty": False,
        },
        "preparation Git",
    )


def _classifier_contract(report: dict[str, Any]) -> bool:
    classifier = report.get("classifier")
    if not isinstance(classifier, dict):
        return False
    return (
        classifier.get("name") == CLASSIFIER["name"]
        and classifier.get("weights_enum") == CLASSIFIER["weights_enum"]
        and type(classifier.get("weights_bytes")) is int
        and classifier["weights_bytes"] == CLASSIFIER["weights_bytes"]
        and classifier.get("weights_sha256") == CLASSIFIER["weights_sha256"]
        and classifier.get("categories_sha256") == CLASSIFIER["categories_sha256"]
        and type(classifier.get("num_classes")) is int
        and classifier["num_classes"] == CLASSIFIER["num_classes"]
        and classifier.get("preprocessing") == CLASSIFIER["preprocessing"]
    )


def _collect_source(
    *,
    method: str,
    sample_dir: Path,
    sampling_report_path: Path,
    sampling_manifest_path: Path,
    sampling_progress_path: Path,
    class_fidelity_report_path: Path,
    generation_metrics_report_path: Path,
) -> dict[str, Any]:
    spec = FROZEN_SOURCE_SPECS[method]
    report, report_id = _read_json_bound(
        sampling_report_path,
        expected_sha256=spec["sampling_report_sha256"],
        name=f"{method} sampling report",
    )
    manifest, manifest_id = _read_json_bound(
        sampling_manifest_path,
        expected_sha256=spec["sampling_manifest_sha256"],
        name=f"{method} sampling manifest",
    )
    progress, progress_id = _read_json_bound(
        sampling_progress_path,
        expected_sha256=spec["sampling_progress_sha256"],
        name=f"{method} sampling progress",
    )
    fidelity, fidelity_id = _read_json_bound(
        class_fidelity_report_path,
        expected_sha256=spec["class_fidelity_report_sha256"],
        name=f"{method} class-fidelity report",
    )
    metrics, metrics_id = _read_json_bound(
        generation_metrics_report_path,
        expected_sha256=spec["generation_metrics_report_sha256"],
        name=f"{method} generation-metrics report",
    )
    tree = physical_sample_tree_identity(sample_dir)
    prefix = spec["prefix_budget"]
    sampling = report.get("sampling")
    if not isinstance(sampling, dict):
        raise ValueError(f"{method} sampling report lacks a protocol")
    if (
        report.get("status") != "completed"
        or report.get("weights") != "ema"
        or _exact_int(
            report.get("checkpoint_step"), f"{method} checkpoint step", minimum=1
        )
        != SOURCE_CHECKPOINT_STEP
        or report.get("checkpoint_sha256") != spec["checkpoint_sha256"]
        or report.get("sampling_manifest_sha256") != manifest_id["sha256"]
        or report.get("sample_sets", {}).get(str(prefix))
        != {
            "count": 10_000,
            "sha256": spec["sample_set_sha256"],
        }
        or Path(report.get("output_dirs", {}).get(str(prefix), "")).resolve()
        != sample_dir.resolve()
        or progress.get("status") != "completed"
        or _exact_int(
            progress.get("completed_samples"),
            f"{method} completed sample count",
            minimum=1,
        )
        != SAMPLE_COUNT
        or progress.get("sampling_manifest_sha256") != manifest_id["sha256"]
        or progress.get("sample_sets") != report.get("sample_sets")
        or tree["sample_set_sha256"] != spec["sample_set_sha256"]
    ):
        raise ValueError(f"{method} physical sampling chain differs")
    for field in (
        "git",
        "checkpoint",
        "checkpoint_sha256",
        "checkpoint_integrity_manifest",
        "checkpoint_step",
        "weights",
        "sampling",
        "output_dirs",
        "runtime_environment",
        "runtime_environment_sha256",
    ):
        if manifest.get(field) != report.get(field):
            raise ValueError(f"{method} manifest/report field differs: {field}")
    fidelity_source = fidelity.get("sample_provenance")
    metric_source = metrics.get("sample_provenance")
    if (
        fidelity.get("status") != "completed"
        or not _classifier_contract(fidelity)
        or not isinstance(fidelity_source, dict)
        or fidelity_source.get("report_identity") != report_id
        or fidelity_source.get("manifest_identity") != manifest_id
        or fidelity_source.get("sample_set_sha256") != tree["sample_set_sha256"]
        or fidelity_source.get("checkpoint_sha256") != spec["checkpoint_sha256"]
        or not isinstance(metric_source, dict)
        or metric_source.get("report_identity") != report_id
        or metric_source.get("manifest_identity") != manifest_id
        or metric_source.get("sample_set_sha256") != tree["sample_set_sha256"]
        or metric_source.get("checkpoint_sha256") != spec["checkpoint_sha256"]
    ):
        raise ValueError(f"{method} evaluator provenance differs from sampling")
    existing_fidelity = fidelity.get("metrics")
    existing_metrics = metrics.get("metrics")
    if not isinstance(existing_fidelity, dict) or not isinstance(
        existing_metrics, dict
    ):
        raise ValueError(f"{method} existing metrics are missing")
    return {
        "identities": {
            "sampling_report": report_id,
            "sampling_manifest": manifest_id,
            "sampling_progress": progress_id,
            "class_fidelity_report": fidelity_id,
            "generation_metrics_report": metrics_id,
        },
        "sample_tree": tree,
        "sampling": sampling,
        "sampling_git": report["git"],
        "checkpoint_step": _exact_int(
            report["checkpoint_step"], f"{method} checkpoint step", minimum=1
        ),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "prefix_budget": prefix,
        "weights": report["weights"],
        "existing_class_fidelity": {
            key: existing_fidelity[key]
            for key in (
                "sample_count",
                "num_classes",
                "requested_class_count",
                "requested_count_min",
                "requested_count_max",
                "top1_accuracy",
                "top5_accuracy",
                "predicted_class_fraction",
                "normalized_predicted_class_entropy",
            )
        },
        "existing_generation_metrics": {
            key: existing_metrics[key]
            for key in (
                "frechet_inception_distance",
                "precision",
                "recall",
            )
        },
    }


def _exclusive_write(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    if not path.is_absolute():
        raise ValueError("preparation output path must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        )
        + "\n"
    ).encode("ascii")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return stable_file_identity(path)


def _add_sources(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--causal-discriminator", type=Path, required=True)
    parser.add_argument("--causal-discriminator-sha256", required=True)
    parser.add_argument("--causal-validation", type=Path, required=True)
    parser.add_argument("--causal-validation-sha256", required=True)
    parser.add_argument("--classifier", type=Path, required=True)
    for method in METHODS:
        flag = method.replace("_", "-")
        parser.add_argument(f"--{flag}-sample-dir", type=Path, required=True)
        parser.add_argument(f"--{flag}-sampling-report", type=Path, required=True)
        parser.add_argument(f"--{flag}-sampling-manifest", type=Path, required=True)
        parser.add_argument(f"--{flag}-sampling-progress", type=Path, required=True)
        parser.add_argument(f"--{flag}-class-fidelity-report", type=Path, required=True)
        parser.add_argument(
            f"--{flag}-generation-metrics-report", type=Path, required=True
        )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--output-root", required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    _add_sources(build)
    build.add_argument("--output", type=Path, required=True)
    validate = commands.add_parser("validate")
    _add_sources(validate)
    validate.add_argument("--preparation", type=Path, required=True)
    validate.add_argument("--preparation-sha256", required=True)
    return parser


def _kwargs(args: argparse.Namespace) -> dict[str, Any]:
    decision, decision_id = _read_json_bound(
        args.causal_discriminator,
        expected_sha256=args.causal_discriminator_sha256,
        name="causal discriminator",
    )
    validation, validation_id = _read_json_bound(
        args.causal_validation,
        expected_sha256=args.causal_validation_sha256,
        name="causal discriminator validation",
    )
    classifier_id = stable_file_identity(args.classifier)
    sources = {}
    for method in METHODS:
        sources[method] = _collect_source(
            method=method,
            sample_dir=getattr(args, f"{method}_sample_dir"),
            sampling_report_path=getattr(args, f"{method}_sampling_report"),
            sampling_manifest_path=getattr(args, f"{method}_sampling_manifest"),
            sampling_progress_path=getattr(args, f"{method}_sampling_progress"),
            class_fidelity_report_path=getattr(args, f"{method}_class_fidelity_report"),
            generation_metrics_report_path=getattr(
                args, f"{method}_generation_metrics_report"
            ),
        )
    return {
        "causal_discriminator": decision,
        "causal_discriminator_identity": decision_id,
        "causal_validation": validation,
        "causal_validation_identity": validation_id,
        "classifier_identity": classifier_id,
        "sources": sources,
        "preparation_git": _checkout_identity(args.project_root, args.script),
        "output_root": args.output_root,
    }


def main() -> int:
    args = _parser().parse_args()
    expected = build_preparation(**_kwargs(args))
    if args.command == "build":
        if args.output.exists():
            raise FileExistsError(f"refusing to overwrite preparation: {args.output}")
        identity = _exclusive_write(args.output, expected)
    else:
        actual, identity = _read_json_bound(
            args.preparation,
            expected_sha256=args.preparation_sha256,
            name="class-support contingency preparation",
        )
        validate_preparation(actual, expected_output_root=args.output_root)
        if actual != expected:
            raise ValueError("preparation differs from physical source replay")
    print(
        json.dumps(
            {
                "command": args.command,
                "status": "pass",
                "scientific_status": "diagnostic_not_executed",
                "preparation": identity,
                "classifier_inference_performed": False,
                "execution_authorized": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
