from __future__ import annotations

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation import GenerationRequest, GenerationSession, save_tensor_png
from cofitok.generation.protocol import INFERENCE_API, SAMPLING_PROTOCOL_SCHEMA
from cofitok.inference_replay import (
    INFERENCE_MANIFEST_ROLE,
    INFERENCE_MANIFEST_SCHEMA_VERSION,
    INFERENCE_REPORT_SCHEMA_VERSION,
    file_identity,
    load_progress,
    prepare_manifest,
    read_json_object,
    reject_symlink_chain,
    reusable_completed_report,
    validate_report_binding,
    validate_output_directory_layout,
    write_progress,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _integers(raw: str, *, name: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if not values:
        raise ValueError(f"{name} must not be empty")
    return values


def _resolve_seeds(raw: str, *, seed: int, num_images: int) -> list[int]:
    if raw.strip():
        seeds = _integers(raw, name="seeds")
    else:
        if num_images < 1:
            raise ValueError("num-images must be positive")
        seeds = [seed + index for index in range(num_images)]
    if any(value < 0 or value >= 2**63 for value in seeds):
        raise ValueError("seeds must be in [0, 2^63)")
    return seeds


def _resolve_labels(raw: str, *, count: int, num_classes: int) -> list[int] | None:
    if num_classes <= 0:
        if raw.strip():
            raise ValueError("unconditional checkpoint does not accept class IDs")
        return None
    if not raw.strip():
        raise ValueError("class-conditional checkpoint requires --class-ids")
    labels = _integers(raw, name="class-ids")
    if len(labels) == 1:
        labels *= count
    if len(labels) != count:
        raise ValueError("class-ids must contain one value or one value per seed")
    if any(label < 0 or label >= num_classes for label in labels):
        raise ValueError("class ID is outside the checkpoint class range")
    return labels


def _resolve_budgets(raw: str, *, token_count: int) -> list[int]:
    if not raw.strip():
        return [token_count]
    budgets = sorted(set(_integers(raw, name="prefix-budgets")))
    if any(budget < 1 or budget > token_count for budget in budgets):
        raise ValueError("prefix budget is outside the checkpoint token range")
    return budgets


def build_parser(*, released: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate from a terminal-completion-authorized CoFiTok EMA artifact."
            if released
            else "Generate class/seed/prefix-controlled images from a CoFiTok checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--class-ids", default="")
    parser.add_argument("--seeds", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--prefix-budgets", default="")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--cfg-batch-mode", choices=["batched", "sequential"], default="batched")
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument(
        "--weights",
        choices=["ema"] if released else ["ema", "model"],
        default="ema",
    )
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--completion-receipt",
        required=released,
        default=None if released else "",
        help="Terminal generation release receipt for consumer-verifiable inference.",
    )
    if released:
        parser.set_defaults(
            require_release_authorization=True,
            require_completion_authorization=True,
        )
    else:
        parser.add_argument("--require-release-authorization", action="store_true")
        parser.add_argument(
            "--require-completion-authorization",
            action="store_true",
            help=(
                "Reject inference unless the terminal completion receipt "
                "authorizes this artifact."
            ),
        )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _checkpoint_metadata(session: GenerationSession) -> dict[str, Any]:
    loaded = session.loaded
    return {
        "checkpoint": loaded.checkpoint_path.as_posix(),
        "checkpoint_sha256": loaded.checkpoint_sha256,
        "checkpoint_integrity_manifest": (
            loaded.checkpoint_integrity_manifest.as_posix()
        ),
        "checkpoint_step": loaded.checkpoint_step,
        "weights": loaded.weights,
        "artifact_type": loaded.artifact_type,
        "source_checkpoint_sha256": loaded.source_checkpoint_sha256,
        "source_runtime_environment_sha256": (
            loaded.source_runtime_environment_sha256
        ),
        "source_git": loaded.source_git_provenance,
        "training_authorization": loaded.training_authorization,
        "release_authorization": loaded.release_authorization,
        "release_authorization_required": (
            loaded.release_authorization_required
        ),
        "completion_authorization": loaded.completion_authorization,
        "completion_authorization_required": (
            loaded.completion_authorization_required
        ),
    }


def _expected_outputs(
    output_dir: Path,
    *,
    seeds: list[int],
    labels: list[int] | None,
    budgets: list[int],
) -> list[dict[str, Any]]:
    outputs = []
    for budget in budgets:
        for index, seed in enumerate(seeds):
            label = labels[index] if labels is not None else None
            class_tag = f"class_{label:04d}" if label is not None else "unconditional"
            filename = f"seed_{seed:019d}_{class_tag}_prefix_{budget:02d}.png"
            outputs.append(
                {
                    "path": (output_dir / filename).resolve().as_posix(),
                    "filename": filename,
                    "seed": seed,
                    "class_id": label,
                    "prefix_budget": budget,
                }
            )
    return outputs


def _report_base(
    manifest: dict[str, Any],
    manifest_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": INFERENCE_REPORT_SCHEMA_VERSION,
        "git": manifest["git"],
        "runtime_environment": manifest["runtime_environment"],
        "runtime_environment_sha256": manifest["runtime_environment_sha256"],
        "inference_api": manifest["inference_api"],
        "sampling_protocol_schema": manifest["sampling_protocol_schema"],
        "checkpoint": manifest["checkpoint"],
        "request": manifest["request"],
        "manifest": manifest_identity,
    }


def _run_inference_locked(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    resume = bool(getattr(args, "resume", False))
    overwrite = bool(getattr(args, "overwrite", False))
    if resume and overwrite:
        raise ValueError("inference --resume and --overwrite are mutually exclusive")
    session = GenerationSession.from_checkpoint(
        args.checkpoint,
        weights=args.weights,
        require_release_authorization=getattr(
            args,
            "require_release_authorization",
            False,
        ),
        completion_receipt=(getattr(args, "completion_receipt", "") or None),
        require_completion_authorization=getattr(
            args,
            "require_completion_authorization",
            False,
        ),
    )
    runtime_environment = capture_runtime_environment(
        session.device,
        project_root=PROJECT_ROOT,
    )
    runtime_environment_sha = runtime_environment_sha256(runtime_environment)
    execution_git = git_provenance(PROJECT_ROOT)
    seeds = _resolve_seeds(args.seeds, seed=args.seed, num_images=args.num_images)
    labels = _resolve_labels(args.class_ids, count=len(seeds), num_classes=session.num_classes)
    budgets = _resolve_budgets(args.prefix_budgets, token_count=session.token_count)
    identities = list(zip(seeds, labels if labels is not None else [None] * len(seeds)))
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate seed/class requests would overwrite the same image")
    output_dir = reject_symlink_chain(args.output_dir, name="inference output directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = output_dir.resolve()
    report_path = (
        reject_symlink_chain(args.report, name="inference report")
        if getattr(args, "report", "")
        else output_dir / "inference_report.json"
    )
    report_path = report_path.resolve()
    manifest_path = output_dir / "inference_manifest.json"
    progress_path = output_dir / "inference_progress.json"
    if report_path in {manifest_path, progress_path}:
        raise ValueError("inference report must differ from manifest and progress paths")
    expected_outputs = _expected_outputs(
        output_dir,
        seeds=seeds,
        labels=labels,
        budgets=budgets,
    )
    validate_output_directory_layout(
        output_dir,
        expected_outputs=expected_outputs,
    )
    if report_path.exists() and not report_path.is_file():
        raise ValueError(f"inference report is not a regular file: {report_path}")
    if not resume and not overwrite:
        existing = [
            path
            for path in (manifest_path, progress_path, report_path)
            if path.exists() or path.is_symlink()
        ]
        existing.extend(
            Path(row["path"])
            for row in expected_outputs
            if Path(row["path"]).exists() or Path(row["path"]).is_symlink()
        )
        if existing:
            raise FileExistsError(
                "Inference outputs already exist: "
                + ", ".join(path.as_posix() for path in existing)
            )
    request = {
        "seeds": seeds,
        "class_ids": labels,
        "prefix_budgets": budgets,
        "batch_size": args.batch_size,
        "sample_steps": args.sample_steps,
        "actual_timesteps": select_sampling_timesteps(
            session.schedule.num_train_timesteps,
            args.sample_steps,
        ),
        "guidance_scale": args.guidance_scale,
        "guidance_rescale": args.guidance_rescale,
        "cfg_batch_mode": args.cfg_batch_mode,
        "eta": args.eta,
        "clip_x0": True,
        "precision": args.precision,
        "image_shape": [
            session.loaded.config.model.image_channels,
            session.loaded.config.model.image_size,
            session.loaded.config.model.image_size,
        ],
    }
    manifest = {
        "schema_version": INFERENCE_MANIFEST_SCHEMA_VERSION,
        "role": INFERENCE_MANIFEST_ROLE,
        "git": execution_git,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha,
        "inference_api": dict(INFERENCE_API),
        "sampling_protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "checkpoint": _checkpoint_metadata(session),
        "request": request,
        "output_root": output_dir.as_posix(),
        "report": report_path.as_posix(),
        "expected_outputs": expected_outputs,
    }
    manifest_identity = prepare_manifest(
        manifest_path,
        manifest,
        resume=resume,
        overwrite=overwrite,
    )
    progress_existed = progress_path.is_file()
    state = load_progress(
        progress_path,
        manifest_identity=manifest_identity,
        expected_outputs=expected_outputs,
        resume=resume,
        overwrite=overwrite,
    )
    existing_report = None
    if resume and report_path.is_file():
        if not progress_existed:
            raise ValueError("existing inference report is missing its progress evidence")
        existing_report = read_json_object(report_path, name="inference report")
        validate_report_binding(
            existing_report,
            manifest_identity=manifest_identity,
            manifest=manifest,
        )
        if (
            existing_report.get("status") == "completed"
            and existing_report.get("progress") != file_identity(progress_path)
        ):
            raise ValueError("completed inference progress identity differs")
        if reusable_completed_report(
            existing_report,
            progress_path=progress_path,
            progress_status=state["status"],
            expected_outputs=expected_outputs,
            valid_outputs=state["outputs"],
        ):
            return {**existing_report, "reused": True}

    valid_outputs = dict(state["outputs"])
    for output in expected_outputs:
        path = reject_symlink_chain(output["path"], name="inference output")
        if path.exists() and not path.is_file():
            raise ValueError(f"inference output is not a regular file: {path}")
        if path.exists() and output["filename"] not in valid_outputs and not (
            resume or overwrite
        ):
            raise FileExistsError(f"Inference output already exists: {path}")

    attempt_count = int(state["attempt_count"]) + 1
    prior_elapsed = float(state["cumulative_elapsed_seconds"])
    started = time.perf_counter()

    def ordered_outputs() -> list[dict[str, Any]]:
        return [
            valid_outputs[row["filename"]]
            for row in expected_outputs
            if row["filename"] in valid_outputs
        ]

    progress_identity = write_progress(
        progress_path,
        manifest_identity=manifest_identity,
        expected_output_count=len(expected_outputs),
        outputs=ordered_outputs(),
        status="running",
        attempt_count=attempt_count,
        cumulative_elapsed_seconds=prior_elapsed,
        updated_at=_now(),
    )
    write_json_report(
        report_path,
        {
            **_report_base(manifest, manifest_identity),
            "status": "running",
            "progress": progress_identity,
            "attempt_count": attempt_count,
            "expected_output_count": len(expected_outputs),
            "completed_output_count": len(valid_outputs),
            "outputs": ordered_outputs(),
            "updated_at": _now(),
        },
    )
    try:
        for budget in budgets:
            missing = [
                output
                for output in expected_outputs
                if output["prefix_budget"] == budget
                and output["filename"] not in valid_outputs
            ]
            for start in range(0, len(missing), args.batch_size):
                batch = missing[start : start + args.batch_size]
                batch_seeds = tuple(int(output["seed"]) for output in batch)
                batch_labels = (
                    tuple(int(output["class_id"]) for output in batch)
                    if labels is not None
                    else None
                )
                result = session.generate(
                    GenerationRequest(
                        seeds=batch_seeds,
                        class_labels=batch_labels,
                        sample_steps=args.sample_steps,
                        prefix_budget=budget,
                        guidance_scale=args.guidance_scale,
                        guidance_rescale=args.guidance_rescale,
                        cfg_batch_mode=args.cfg_batch_mode,
                        eta=args.eta,
                        precision=args.precision,
                    )
                )
                if len(result.images) != len(batch):
                    raise RuntimeError("inference batch output count differs")
                for output, image in zip(batch, result.images, strict=True):
                    path = save_tensor_png(
                        image,
                        output["path"],
                        overwrite=resume or overwrite,
                    )
                    valid_outputs[output["filename"]] = {
                        **output,
                        "path": path.resolve().as_posix(),
                        "sha256": file_sha256(path),
                    }
                    progress_identity = write_progress(
                        progress_path,
                        manifest_identity=manifest_identity,
                        expected_output_count=len(expected_outputs),
                        outputs=ordered_outputs(),
                        status="running",
                        attempt_count=attempt_count,
                        cumulative_elapsed_seconds=(
                            prior_elapsed + time.perf_counter() - started
                        ),
                        updated_at=_now(),
                    )
        outputs = ordered_outputs()
        if len(outputs) != len(expected_outputs):
            raise RuntimeError("inference did not produce every expected output")
    except BaseException as error:
        cumulative_elapsed = prior_elapsed + time.perf_counter() - started
        progress_identity = write_progress(
            progress_path,
            manifest_identity=manifest_identity,
            expected_output_count=len(expected_outputs),
            outputs=ordered_outputs(),
            status="failed",
            attempt_count=attempt_count,
            cumulative_elapsed_seconds=cumulative_elapsed,
            updated_at=_now(),
            error=error,
        )
        failure = {
            **_report_base(manifest, manifest_identity),
            "status": "failed",
            "progress": progress_identity,
            "attempt_count": attempt_count,
            "expected_output_count": len(expected_outputs),
            "completed_output_count": len(valid_outputs),
            "outputs": ordered_outputs(),
            "elapsed_seconds": cumulative_elapsed,
            "attempt_elapsed_seconds": cumulative_elapsed - prior_elapsed,
            "error_type": type(error).__name__,
            "error": str(error),
            "updated_at": _now(),
        }
        write_json_report(report_path, failure)
        raise
    cumulative_elapsed = prior_elapsed + time.perf_counter() - started
    progress_identity = write_progress(
        progress_path,
        manifest_identity=manifest_identity,
        expected_output_count=len(expected_outputs),
        outputs=outputs,
        status="completed",
        attempt_count=attempt_count,
        cumulative_elapsed_seconds=cumulative_elapsed,
        updated_at=_now(),
    )
    report = {
        **_report_base(manifest, manifest_identity),
        "status": "completed",
        "progress": progress_identity,
        "attempt_count": attempt_count,
        "output_count": len(outputs),
        "outputs": outputs,
        "elapsed_seconds": cumulative_elapsed,
        "attempt_elapsed_seconds": cumulative_elapsed - prior_elapsed,
        "reused": False,
        "updated_at": _now(),
    }
    write_json_report(report_path, report)
    return report


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    with exclusive_output_lock(
        args.output_dir,
        role="generation_inference",
    ):
        return _run_inference_locked(args)


def main() -> None:
    args = parse_args()
    report = run_inference(args)
    print(Path(args.report) if args.report else Path(args.output_dir) / "inference_report.json")
    print(f"generated {report['output_count']} images")


if __name__ == "__main__":
    main()
