from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from cofitok.generation import GenerationRequest, GenerationSession, save_tensor_png
from cofitok.reporting import file_sha256, write_json_report


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate class/seed/prefix-controlled images from a CoFiTok checkpoint."
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
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run_inference(args: argparse.Namespace) -> dict[str, Any]:
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    session = GenerationSession.from_checkpoint(args.checkpoint, weights=args.weights)
    seeds = _resolve_seeds(args.seeds, seed=args.seed, num_images=args.num_images)
    labels = _resolve_labels(args.class_ids, count=len(seeds), num_classes=session.num_classes)
    budgets = _resolve_budgets(args.prefix_budgets, token_count=session.token_count)
    identities = list(zip(seeds, labels if labels is not None else [None] * len(seeds)))
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate seed/class requests would overwrite the same image")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = (
        Path(args.report)
        if getattr(args, "report", "")
        else output_dir / "inference_report.json"
    )
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(f"Inference report already exists: {report_path}")
    started = time.perf_counter()
    outputs = []
    last_metadata = None
    try:
        for budget in budgets:
            for start in range(0, len(seeds), args.batch_size):
                stop = min(len(seeds), start + args.batch_size)
                batch_seeds = tuple(seeds[start:stop])
                batch_labels = tuple(labels[start:stop]) if labels is not None else None
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
                last_metadata = result.metadata
                for offset, image in enumerate(result.images):
                    index = start + offset
                    label = labels[index] if labels is not None else None
                    class_tag = f"class_{label:04d}" if label is not None else "unconditional"
                    filename = (
                        f"seed_{seeds[index]:019d}_{class_tag}_prefix_{budget:02d}.png"
                    )
                    path = save_tensor_png(
                        image,
                        output_dir / filename,
                        overwrite=args.overwrite,
                    )
                    outputs.append(
                        {
                            "path": path.resolve().as_posix(),
                            "filename": filename,
                            "sha256": file_sha256(path),
                            "seed": seeds[index],
                            "class_id": label,
                            "prefix_budget": budget,
                        }
                    )
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "failed",
            "checkpoint": str(Path(args.checkpoint).resolve()),
            "completed_output_count": len(outputs),
            "error_type": type(error).__name__,
            "error": str(error),
        }
        write_json_report(report_path, failure)
        raise
    if last_metadata is None:
        raise RuntimeError("inference produced no output")
    report = {
        "schema_version": 1,
        "status": "completed",
        "checkpoint": {
            key: last_metadata[key]
            for key in (
                "checkpoint",
                "checkpoint_sha256",
                "checkpoint_integrity_manifest",
                "checkpoint_step",
                "weights",
                "artifact_type",
                "source_checkpoint_sha256",
                "source_runtime_environment_sha256",
                "source_git",
                "training_authorization",
                "release_authorization",
            )
        },
        "request": {
            "seeds": seeds,
            "class_ids": labels,
            "prefix_budgets": budgets,
            "batch_size": args.batch_size,
            "sample_steps": args.sample_steps,
            "guidance_scale": args.guidance_scale,
            "guidance_rescale": args.guidance_rescale,
            "cfg_batch_mode": args.cfg_batch_mode,
            "eta": args.eta,
            "precision": args.precision,
        },
        "output_count": len(outputs),
        "outputs": outputs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_report(report_path, report)
    return report


def main() -> None:
    args = parse_args()
    report = run_inference(args)
    print(Path(args.report) if args.report else Path(args.output_dir) / "inference_report.json")
    print(f"generated {report['output_count']} images")


if __name__ == "__main__":
    main()
