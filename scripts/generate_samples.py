from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from torchvision.utils import save_image

from cofitok.diffusion import select_sampling_timesteps
from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.generation import (
    INFERENCE_API,
    SAMPLING_MANIFEST_SCHEMA_VERSION,
    SAMPLING_PROTOCOL_SCHEMA,
    SAMPLING_REPORT_SCHEMA_VERSION,
    GenerationRequest,
    GenerationSession,
)
from cofitok.image_integrity import is_valid_png, sample_set_sha256
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import file_sha256, git_provenance, write_json_report
from cofitok.sampling_progress import (
    build_sampling_progress,
    load_sampling_progress_state,
    write_sampling_progress,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reproducible CoFiTok samples from EMA weights.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sample-steps", type=int, default=250)
    parser.add_argument("--prefix-budgets", default="", help="Comma-separated token budgets; default K.")
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument(
        "--cfg-batch-mode",
        choices=["batched", "sequential"],
        default="batched",
        help="Evaluate conditional/unconditional CFG branches together or separately.",
    )
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _parse_budgets(raw: str, token_count: int) -> list[int]:
    if not raw.strip():
        return [token_count]
    budgets = sorted({int(value.strip()) for value in raw.split(",") if value.strip()})
    if any(budget < 1 or budget > token_count for budget in budgets):
        raise ValueError("prefix budget is outside the model token range")
    return budgets


def _labels(start: int, count: int, num_classes: int, device: torch.device) -> torch.Tensor | None:
    if num_classes <= 0:
        return None
    return torch.arange(start, start + count, device=device, dtype=torch.long) % num_classes


def _sample_seed(seed: int, global_index: int) -> int:
    return (seed + global_index) % (2**63)


def _sample_generators(
    seed: int,
    start: int,
    count: int,
    device: torch.device,
) -> list[torch.Generator]:
    return [
        torch.Generator(device=device).manual_seed(_sample_seed(seed, index))
        for index in range(start, start + count)
    ]


def _save_batch(
    images: torch.Tensor,
    directory: Path,
    start: int,
    *,
    overwrite: bool,
    skip_existing: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for offset, image in enumerate(images):
        path = directory / f"{start + offset:06d}.png"
        image_size = int(image.shape[-1])
        image_channels = int(image.shape[-3])
        if path.exists():
            if skip_existing and is_valid_png(
                path,
                width=image_size,
                height=image_size,
                channels=image_channels,
            ):
                continue
            if not overwrite and not skip_existing:
                raise FileExistsError(f"Refusing to overwrite existing sample {path}")
        temporary = path.with_name(f".{path.name}.part")
        try:
            save_image(
                (image.float().clamp(-1.0, 1.0) + 1.0) * 0.5,
                temporary,
                format="png",
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _batch_complete(
    directory: Path,
    start: int,
    count: int,
    *,
    image_size: int,
    image_channels: int,
) -> bool:
    return all(
        is_valid_png(
            directory / f"{index:06d}.png",
            width=image_size,
            height=image_size,
            channels=image_channels,
        )
        for index in range(start, start + count)
    )


def _has_images(directories: list[Path]) -> bool:
    return any(any(directory.glob("*.png")) for directory in directories if directory.is_dir())


def _validate_numbered_output(
    directory: Path,
    start: int,
    stop: int,
    *,
    image_size: int,
    image_channels: int,
) -> None:
    expected = {f"{index:06d}.png" for index in range(start, stop)}
    actual = {path.name for path in directory.glob("*.png") if path.is_file()}
    if actual != expected:
        missing = len(expected - actual)
        extra = len(actual - expected)
        raise RuntimeError(
            f"Incomplete numbered sample set in {directory}: missing={missing}, extra={extra}"
        )
    invalid = [
        name
        for name in sorted(expected)
        if not is_valid_png(
            directory / name,
            width=image_size,
            height=image_size,
            channels=image_channels,
        )
    ]
    if invalid:
        raise RuntimeError(
            f"Invalid PNG samples in {directory}: count={len(invalid)}, first={invalid[0]}"
        )


def _prepare_sampling_manifest(
    path: Path,
    manifest: dict[str, object],
    *,
    resume: bool,
    has_existing_images: bool,
) -> None:
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError("Existing sampling manifest does not match this invocation")
        if not resume:
            raise FileExistsError("Sampling manifest already exists; pass --resume to continue")
        return
    if has_existing_images:
        raise FileExistsError("Generated images exist without a sampling manifest")
    write_json_report(path, manifest)


def _run_sampling(args: argparse.Namespace) -> None:
    if args.num_samples < 1 or args.batch_size < 1:
        raise ValueError("num-samples and batch-size must be positive")
    if args.start_index < 0:
        raise ValueError("start-index must be non-negative")
    if args.resume and args.overwrite:
        raise ValueError("resume and overwrite are mutually exclusive")
    session = GenerationSession.from_checkpoint(args.checkpoint, weights=args.weights)
    loaded = session.loaded
    checkpoint_path = loaded.checkpoint_path
    checkpoint_hash = loaded.checkpoint_sha256
    checkpoint_step = loaded.checkpoint_step
    config = loaded.config
    device = loaded.device
    runtime_environment = capture_runtime_environment(
        device,
        project_root=PROJECT_ROOT,
    )
    runtime_environment_sha = runtime_environment_sha256(runtime_environment)
    budgets = _parse_budgets(args.prefix_budgets, config.model.token_count)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stop_index = args.start_index + args.num_samples
    code_git = git_provenance(PROJECT_ROOT)
    sampling = {
        "protocol_schema": SAMPLING_PROTOCOL_SCHEMA,
        "inference_api": INFERENCE_API,
        "sampler": "ddim",
        "num_samples": args.num_samples,
        "start_index": args.start_index,
        "batch_size": args.batch_size,
        "sample_steps": args.sample_steps,
        "num_train_timesteps": session.schedule.num_train_timesteps,
        "actual_timesteps": select_sampling_timesteps(
            session.schedule.num_train_timesteps,
            args.sample_steps,
        ),
        "prefix_budgets": budgets,
        "guidance_scale": args.guidance_scale,
        "guidance_rescale": args.guidance_rescale,
        "cfg_batch_mode": args.cfg_batch_mode,
        "eta": args.eta,
        "clip_x0": True,
        "seed": args.seed,
        "precision": args.precision,
        "image_shape": [
            config.model.image_channels,
            config.model.image_size,
            config.model.image_size,
        ],
        "class_schedule": "balanced_modulo" if config.model.num_classes > 0 else None,
        "random_stream": {
            "scope": "per_global_sample_index",
            "seed_formula": "(seed + global_index) mod 2^63",
            "prefix_budgets_share_stream": True,
            "batch_size_invariant": True,
            "resume_index_invariant": True,
        },
        "sample_set_digest": {
            "algorithm": "sha256",
            "framing": "filename_utf8_nul_file_bytes_nul",
        },
    }
    output_dirs = {
        str(budget): str((output_dir / f"prefix_{budget}").resolve()) for budget in budgets
    }
    manifest = {
        "schema_version": SAMPLING_MANIFEST_SCHEMA_VERSION,
        "git": code_git,
        "runtime_environment": runtime_environment,
        "runtime_environment_sha256": runtime_environment_sha,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_integrity_manifest": str(loaded.checkpoint_integrity_manifest),
        "checkpoint_step": checkpoint_step,
        "weights": args.weights,
        "sampling": sampling,
        "output_dirs": output_dirs,
    }
    budget_directories = [output_dir / f"prefix_{budget}" for budget in budgets]
    sampling_manifest_path = output_dir / "sampling_manifest.json"
    _prepare_sampling_manifest(
        sampling_manifest_path,
        manifest,
        resume=args.resume,
        has_existing_images=_has_images(budget_directories),
    )
    sampling_manifest_sha256 = file_sha256(sampling_manifest_path)
    progress_path = output_dir / "sampling_progress.json"
    progress_state = load_sampling_progress_state(
        progress_path,
        sampling_manifest_sha256=sampling_manifest_sha256,
        total_samples=args.num_samples,
        start_index=args.start_index,
        prefix_budgets=budgets,
    )
    start_time = time.time()
    completed = int(progress_state["prior_completed_samples"])
    write_sampling_progress(
        progress_path,
        build_sampling_progress(
            progress_state,
            status="running",
            completed_samples=completed,
            invocation_elapsed_seconds=0.0,
        ),
    )

    try:
        for batch_start in range(args.start_index, stop_index, args.batch_size):
            count = min(args.batch_size, stop_index - batch_start)
            labels = _labels(batch_start, count, config.model.num_classes, device)
            for budget in budgets:
                budget_directory = output_dir / f"prefix_{budget}"
                if args.resume and _batch_complete(
                    budget_directory,
                    batch_start,
                    count,
                    image_size=config.model.image_size,
                    image_channels=config.model.image_channels,
                ):
                    continue
                class_labels = (
                    tuple(int(value) for value in labels.detach().cpu().tolist())
                    if labels is not None
                    else None
                )
                request = GenerationRequest(
                    seeds=tuple(
                        _sample_seed(args.seed, index)
                        for index in range(batch_start, batch_start + count)
                    ),
                    class_labels=class_labels,
                    sample_steps=args.sample_steps,
                    prefix_budget=budget,
                    guidance_scale=args.guidance_scale,
                    guidance_rescale=args.guidance_rescale,
                    cfg_batch_mode=args.cfg_batch_mode,
                    eta=args.eta,
                    clip_x0=True,
                    precision=args.precision,
                )
                samples = session.generate(request).images
                _save_batch(
                    samples,
                    budget_directory,
                    batch_start,
                    overwrite=args.overwrite,
                    skip_existing=args.resume,
                )
            completed = max(completed, batch_start + count - args.start_index)
            invocation_elapsed = time.time() - start_time
            write_sampling_progress(
                progress_path,
                build_sampling_progress(
                    progress_state,
                    status="running",
                    completed_samples=completed,
                    invocation_elapsed_seconds=invocation_elapsed,
                ),
            )
            print(f"generated {completed}/{args.num_samples}")

        sample_sets = {}
        for budget, budget_directory in zip(budgets, budget_directories):
            _validate_numbered_output(
                budget_directory,
                args.start_index,
                stop_index,
                image_size=config.model.image_size,
                image_channels=config.model.image_channels,
            )
            sample_paths = [
                budget_directory / f"{index:06d}.png"
                for index in range(args.start_index, stop_index)
            ]
            sample_sets[str(budget)] = {
                "count": len(sample_paths),
                "sha256": sample_set_sha256(sample_paths),
            }

        invocation_elapsed = time.time() - start_time
        completed_progress = build_sampling_progress(
            progress_state,
            status="completed",
            completed_samples=args.num_samples,
            invocation_elapsed_seconds=invocation_elapsed,
            sample_sets=sample_sets,
        )
        report = {
            "schema_version": SAMPLING_REPORT_SCHEMA_VERSION,
            "status": "completed",
            "git": code_git,
            "runtime_environment": runtime_environment,
            "runtime_environment_sha256": runtime_environment_sha,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_integrity_manifest": str(loaded.checkpoint_integrity_manifest),
            "checkpoint_step": checkpoint_step,
            "weights": args.weights,
            "sampling": sampling,
            "sampling_manifest_sha256": sampling_manifest_sha256,
            "sampling_progress": progress_path.resolve().as_posix(),
            "output_dirs": output_dirs,
            "sample_sets": sample_sets,
            "invocation_elapsed_seconds": invocation_elapsed,
            "elapsed_seconds": completed_progress["cumulative_elapsed_seconds"],
            "torch_version": torch.__version__,
            "device": str(device),
        }
        write_json_report(output_dir / "sampling_report.json", report)
        write_sampling_progress(progress_path, completed_progress)
        print(f"wrote {output_dir / 'sampling_report.json'}")
    except BaseException as error:
        invocation_elapsed = time.time() - start_time
        write_sampling_progress(
            progress_path,
            build_sampling_progress(
                progress_state,
                status="failed",
                completed_samples=completed,
                invocation_elapsed_seconds=invocation_elapsed,
                error=error,
            ),
        )
        raise


def main() -> None:
    args = parse_args()
    with exclusive_output_lock(
        args.output_dir,
        role="generation_sampling",
    ):
        _run_sampling(args)


if __name__ == "__main__":
    main()
