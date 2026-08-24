from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping

from cofitok.configs import config_from_dict, config_to_dict
from cofitok.generation.conditioning_ranking_probe import (
    validate_standing_experiment_authorization,
)
from cofitok.generation.exposure_semantic_trajectory import (
    CHECKPOINT_STEPS,
    CLAIM_BOUNDARY,
    DATASET_IDENTITY_SHA256,
    DESCRIPTIVE_TIMESTEPS,
    ELIGIBLE_TIMESTEPS,
    EVALUATION_REQUEST,
    EXECUTION_BOUNDARY,
    EXPECTED_CHECKPOINTS,
    EXPECTED_PARAMETER_COUNTS,
    EXPECTED_RUN_FILES,
    METHOD_RUN_DIRS,
    OUTPUT_ROOT,
    PAIR_ROOT,
    PREPARATION_ROLE,
    REJECTED_POSTEVALUATION_PATH,
    REJECTED_POSTEVALUATION_SHA256,
    RUNTIME_ENVIRONMENT_SHA256,
    SCHEMA_VERSION,
    SCOPE,
    STAGE,
    STANDING_AUTHORIZATION_PATH,
    STANDING_AUTHORIZATION_SHA256,
    TRAINING_REVISION,
    validate_preparation,
    validate_rejected_postevaluation,
)
from cofitok.generation_pair import generation_pair_contract
from cofitok.inference_replay import (
    file_identity,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.reporting import git_provenance
from cofitok.training.checkpointing import (
    checkpoint_integrity_path,
    verify_training_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_AUXILIARY_WEIGHTS = (
    "class_conditioning_ranking_weight",
    "class_conditioning_residual_alignment_weight",
    "class_conditioning_residual_alignment_reconstruction_weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the source-bound CPU-only semantic exposure trajectory over "
            "the existing matched 5K checkpoints."
        )
    )
    parser.add_argument(
        "--standing-authorization",
        type=Path,
        default=Path(STANDING_AUTHORIZATION_PATH),
    )
    parser.add_argument(
        "--rejected-postevaluation",
        type=Path,
        default=Path(REJECTED_POSTEVALUATION_PATH),
    )
    parser.add_argument("--pair-root", type=Path, default=Path(PAIR_ROOT))
    parser.add_argument("--output-root", default=OUTPUT_ROOT)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=PROJECT_ROOT / "scripts/evaluate_generation_conditioning_sensitivity.py",
    )
    parser.add_argument(
        "--report-builder",
        type=Path,
        default=PROJECT_ROOT
        / "scripts/build_generation_exposure_semantic_trajectory_report.py",
    )
    parser.add_argument("--runbook", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def full_git() -> dict[str, Any]:
    provenance = git_provenance(PROJECT_ROOT)
    status = subprocess.check_output(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    if provenance["tracked_dirty"] or status:
        raise ValueError("exposure-trajectory preparation requires a clean checkout")
    return {
        **provenance,
        "tree": subprocess.check_output(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            text=True,
        ).strip(),
    }


def _exact_source(
    path: Path,
    *,
    expected_path: str,
    expected_sha256: str,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = reject_symlink_chain(path, name=label).resolve()
    identity = file_identity(source)
    if source.as_posix() != expected_path or identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} identity differs")
    return read_json_object(source, name=label), identity


def _load_metrics(path: Path) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    metrics_path = reject_symlink_chain(path, name="5K training metrics").resolve()
    rows = []
    with metrics_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid metrics JSON at line {line_number}") from error
            if not isinstance(row, Mapping):
                raise ValueError("training metric row is not a mapping")
            rows.append(dict(row))
    if not rows:
        raise ValueError("training metrics are empty")
    steps = [int(row["step"]) for row in rows]
    strictly_increasing = all(right > left for left, right in zip(steps, steps[1:]))
    exposure_bound = all(
        int(row["samples_seen"]) == int(row["step"]) * 64 for row in rows
    )
    finite = all(
        math.isfinite(float(value))
        for row in rows
        for value in row.values()
        if isinstance(value, float)
    )
    selected = {int(row["step"]): row for row in rows if int(row["step"]) in CHECKPOINT_STEPS}
    if (
        len(rows) != 201
        or steps[0] != 1
        or steps[-1] != 5_000
        or not strictly_increasing
        or not exposure_bound
        or not finite
        or sorted(selected) != CHECKPOINT_STEPS
    ):
        raise ValueError("5K metrics do not satisfy the exposure contract")
    return {
        "identity": file_identity(metrics_path),
        "row_count": len(rows),
        "first_step": steps[0],
        "last_step": steps[-1],
        "strictly_increasing": strictly_increasing,
        "samples_seen_equals_step_times_64": exposure_bound,
        "all_float_values_finite": finite,
    }, selected


def _method_evidence(method: str, pair_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    run_dir = reject_symlink_chain(
        pair_root / METHOD_RUN_DIRS[method],
        name=f"{method} 5K run directory",
    ).resolve()
    expected_dir = f"{PAIR_ROOT}/{METHOD_RUN_DIRS[method]}"
    if run_dir.as_posix() != expected_dir:
        raise ValueError(f"{method} 5K run path differs")

    file_sources: dict[str, Any] = {}
    payloads: dict[str, Any] = {}
    for filename, expected_sha in EXPECTED_RUN_FILES[method].items():
        path = reject_symlink_chain(run_dir / filename, name=f"{method} {filename}")
        identity = file_identity(path)
        if identity["sha256"] != expected_sha:
            raise ValueError(f"{method} {filename} SHA256 differs")
        file_sources[filename] = identity
        if filename.endswith(".json"):
            payloads[filename] = read_json_object(path, name=f"{method} {filename}")

    report = payloads["training_report.json"]
    latest = payloads["latest.json"]
    run_manifest = payloads["run_manifest.json"]
    config = config_to_dict(config_from_dict(report.get("config", {})))
    loss = config["loss"]
    semantic_weights_zero = all(
        float(loss.get(field, 0.0)) == 0.0 for field in SEMANTIC_AUXILIARY_WEIGHTS
    )
    git = {"revision": TRAINING_REVISION, "branch": "", "dirty": False}
    if (
        report.get("training_complete") is not True
        or int(report.get("completed_steps", -1)) != 5_000
        or int(report.get("target_steps", -1)) != 5_000
        or report.get("git") != git
        or run_manifest.get("git") != git
        or int(report.get("parameter_count", -1)) != EXPECTED_PARAMETER_COUNTS[method]
        or report.get("runtime_environment_sha256") != RUNTIME_ENVIRONMENT_SHA256
        or report.get("dataset_provenance", {}).get("identity_sha256")
        != DATASET_IDENTITY_SHA256
        or config["data"]["dataset"] != "imagenet_256_10pct"
        or int(config["data"]["batch_size"])
        * int(config["optimization"]["gradient_accumulation_steps"])
        != 64
        or not semantic_weights_zero
        or int(report.get("final_metrics", {}).get("step", -1)) != 5_000
        or int(report.get("final_metrics", {}).get("samples_seen", -1)) != 320_000
        or latest.get("checkpoint") != "checkpoint_step_00005000.pt"
        or int(latest.get("step", -1)) != 5_000
        or latest.get("checkpoint_sha256")
        != EXPECTED_CHECKPOINTS[method][5_000]["sha256"]
        or latest.get("dataset_identity_sha256") != DATASET_IDENTITY_SHA256
        or latest.get("runtime_environment_sha256") != RUNTIME_ENVIRONMENT_SHA256
        or latest.get("git_revision") != TRAINING_REVISION
        or latest.get("git_branch") != ""
        or latest.get("git_dirty") is not False
    ):
        raise ValueError(f"{method} 5K training contract differs")

    metrics_summary, selected_metrics = _load_metrics(run_dir / "train_metrics.jsonl")
    if metrics_summary["identity"] != file_sources["train_metrics.jsonl"]:
        raise ValueError(f"{method} metrics identity changed while reading")

    checkpoints = []
    for step in CHECKPOINT_STEPS:
        checkpoint = reject_symlink_chain(
            run_dir / f"checkpoint_step_{step:08d}.pt",
            name=f"{method} checkpoint {step}",
        ).resolve()
        sidecar = reject_symlink_chain(
            checkpoint_integrity_path(checkpoint),
            name=f"{method} checkpoint sidecar {step}",
        ).resolve()
        integrity = verify_training_checkpoint(checkpoint)
        expected = EXPECTED_CHECKPOINTS[method][step]
        sidecar_identity = file_identity(sidecar)
        if (
            int(integrity.get("step", -1)) != step
            or int(integrity.get("checkpoint_bytes", -1)) != expected["bytes"]
            or integrity.get("checkpoint_sha256") != expected["sha256"]
            or integrity.get("dataset_identity_sha256") != DATASET_IDENTITY_SHA256
            or integrity.get("runtime_environment_sha256")
            != RUNTIME_ENVIRONMENT_SHA256
            or integrity.get("git_revision") != TRAINING_REVISION
            or integrity.get("git_branch") != ""
            or integrity.get("git_dirty") is not False
            or sidecar_identity["sha256"] != expected["sidecar_sha256"]
            or int(selected_metrics[step]["samples_seen"]) != step * 64
        ):
            raise ValueError(f"{method} checkpoint {step} physical binding differs")
        checkpoints.append(
            {
                "step": step,
                "samples_seen": int(selected_metrics[step]["samples_seen"]),
                "validation_epsilon_mse": float(
                    selected_metrics[step]["validation_epsilon_mse"]
                ),
                "checkpoint": {
                    "path": checkpoint.as_posix(),
                    "bytes": int(integrity["checkpoint_bytes"]),
                    "sha256": str(integrity["checkpoint_sha256"]),
                },
                "sidecar": sidecar_identity,
                "physical_sha256_verified": True,
                "sidecar_verified": True,
            }
        )
    return {
        "run_dir": run_dir.as_posix(),
        "training_revision": TRAINING_REVISION,
        "dataset_identity_sha256": DATASET_IDENTITY_SHA256,
        "runtime_environment_sha256": RUNTIME_ENVIRONMENT_SHA256,
        "parameter_count": EXPECTED_PARAMETER_COUNTS[method],
        "training_complete": True,
        "completed_steps": 5_000,
        "samples_seen": 320_000,
        "metrics_strictly_increasing": True,
        "metrics_exposure_bound": True,
        "semantic_auxiliary_weights_zero": True,
        "source_files": file_sources,
        "metrics_summary": metrics_summary,
        "checkpoints": checkpoints,
    }, config


def main() -> None:
    args = parse_args()
    if args.output_root != OUTPUT_ROOT:
        raise ValueError("exposure-trajectory output root differs")
    pair_root = reject_symlink_chain(args.pair_root, name="matched 5K pair root").resolve()
    if pair_root.as_posix() != PAIR_ROOT:
        raise ValueError("matched 5K pair root differs")
    standing, standing_identity = _exact_source(
        args.standing_authorization,
        expected_path=STANDING_AUTHORIZATION_PATH,
        expected_sha256=STANDING_AUTHORIZATION_SHA256,
        label="standing authorization",
    )
    rejected, rejected_identity = _exact_source(
        args.rejected_postevaluation,
        expected_path=REJECTED_POSTEVALUATION_PATH,
        expected_sha256=REJECTED_POSTEVALUATION_SHA256,
        label="rejected residual-alignment postevaluation",
    )
    validate_standing_experiment_authorization(standing)
    validate_rejected_postevaluation(rejected)

    methods: dict[str, Any] = {}
    configs: dict[str, dict[str, Any]] = {}
    for method in METHOD_RUN_DIRS:
        methods[method], configs[method] = _method_evidence(method, pair_root)
    pair_contract = generation_pair_contract(configs["cofitok"], configs["dense_identity"])
    if pair_contract.get("issues") != []:
        raise ValueError(
            "matched 5K pair contract failed: " + "; ".join(pair_contract["issues"])
        )

    git = full_git()
    report = {
        "schema_version": SCHEMA_VERSION,
        "role": PREPARATION_ROLE,
        "status": "pass",
        "scope": SCOPE,
        "stage": STAGE,
        "git": git,
        "output_root": OUTPUT_ROOT,
        "pair_root": PAIR_ROOT,
        "evaluation_request": EVALUATION_REQUEST,
        "eligible_timesteps": ELIGIBLE_TIMESTEPS,
        "descriptive_timesteps": DESCRIPTIVE_TIMESTEPS,
        "checkpoint_steps": CHECKPOINT_STEPS,
        "sources": {
            "standing_authorization": standing_identity,
            "rejected_postevaluation": rejected_identity,
            "evaluator": file_identity(
                reject_symlink_chain(args.evaluator, name="trajectory evaluator")
            ),
            "report_builder": file_identity(
                reject_symlink_chain(args.report_builder, name="trajectory report builder")
            ),
            "preparation_builder": file_identity(Path(__file__).resolve()),
            "runbook": file_identity(
                reject_symlink_chain(args.runbook, name="trajectory runbook")
            ),
        },
        "pair_contract": pair_contract,
        "methods": methods,
        "execution_boundary": EXECUTION_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "generation_advantage_proven": False,
    }
    validate_preparation(
        report,
        expected_revision=git["revision"],
        expected_tree=git["tree"],
        expected_branch=git["branch"],
    )
    identity = prepare_manifest(
        args.output,
        report,
        resume=args.resume,
        overwrite=False,
    )
    print(identity["sha256"])


if __name__ == "__main__":
    main()
