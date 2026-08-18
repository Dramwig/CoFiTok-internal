from __future__ import annotations

import argparse
import gc
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from cofitok.inference_replay import (
    file_identity,
    read_json_object,
    reject_symlink_chain,
)
from cofitok.models import CoFiTokTiny
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from scripts.evaluate_generation_conditioning_sensitivity import (
    _checkpoint_identity,
    _load_checkpoint_payload,
    evaluate_sensitivity,
    select_samples,
    summarize_timestep_rows,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_conditioning_gain_sweep"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep an inference-only gain on class-embedding displacement from "
            "the classifier-free null embedding while holding checkpoint, "
            "images, noise, and timesteps fixed."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--start-label", type=int, default=0)
    parser.add_argument("--wrong-label-offset", type=int, default=500)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[500])
    parser.add_argument(
        "--gains",
        type=float,
        nargs="+",
        default=[0.0, 0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument("--noise-seed", type=int, default=102030)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--dataset-root")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_samples < 1:
        raise ValueError("num_samples must be positive")
    if args.start_label < 0:
        raise ValueError("start_label must be non-negative")
    if args.wrong_label_offset < 1:
        raise ValueError("wrong_label_offset must be positive")
    if args.noise_seed < 0:
        raise ValueError("noise_seed must be non-negative")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if not args.timesteps or len(set(args.timesteps)) != len(args.timesteps):
        raise ValueError("timesteps must be non-empty and unique")
    if not args.gains or len(set(args.gains)) != len(args.gains):
        raise ValueError("gains must be non-empty and unique")
    if any(not math.isfinite(gain) or gain < 0.0 for gain in args.gains):
        raise ValueError("gains must be finite and non-negative")


def apply_class_embedding_gain(
    model: CoFiTokTiny,
    base_weight: torch.Tensor,
    gain: float,
) -> None:
    class_embed = getattr(model.predictor, "class_embed", None)
    null_label = int(getattr(model.predictor, "null_class", -1))
    if class_embed is None or null_label < 1:
        raise ValueError("Model does not expose class conditioning")
    if tuple(class_embed.weight.shape) != tuple(base_weight.shape):
        raise ValueError("Class embedding shape changed during gain sweep")
    null = base_weight[null_label : null_label + 1]
    scaled = base_weight.clone()
    scaled[:null_label] = null + gain * (base_weight[:null_label] - null)
    scaled[null_label] = null[0]
    with torch.no_grad():
        class_embed.weight.copy_(scaled)


def summarize_gain_rows(
    rows_by_gain: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    result = {}
    for gain, rows in rows_by_gain.items():
        if not rows:
            raise ValueError("Gain sweep row set is empty")
        count = len(rows)
        result[gain] = {
            "row_count": count,
            "correct_better_count": {
                "than_wrong": sum(
                    bool(row["correct_better"]["than_wrong"]) for row in rows
                ),
                "than_null": sum(
                    bool(row["correct_better"]["than_null"]) for row in rows
                ),
            },
            "correct_relative_mse_improvement_mean": {
                comparison: sum(
                    float(row["correct_relative_mse_improvement"][comparison])
                    for row in rows
                )
                / count
                for comparison in ("versus_wrong", "versus_null")
            },
            "relative_delta_to_correct_rms_mean": {
                comparison: sum(
                    float(row["relative_delta_to_correct_rms"][comparison])
                    for row in rows
                )
                / count
                for comparison in (
                    "correct_vs_wrong",
                    "correct_vs_null",
                    "wrong_vs_null",
                )
            },
        }
    return result


@torch.no_grad()
def run_gain_sweep(
    *,
    model: CoFiTokTiny,
    base_weight: torch.Tensor,
    config: Any,
    samples: Sequence[Mapping[str, Any]],
    timesteps: Sequence[int],
    gains: Sequence[float],
    wrong_label_offset: int,
    noise_seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    rows_by_gain: dict[str, list[dict[str, Any]]] = {}
    timestep_summaries: dict[str, Any] = {}
    try:
        for gain in gains:
            apply_class_embedding_gain(model, base_weight, gain)
            rows = evaluate_sensitivity(
                model=model,
                config=config,
                samples=samples,
                timesteps=timesteps,
                wrong_label_offset=wrong_label_offset,
                noise_seed=noise_seed,
            )
            key = format(gain, ".12g")
            rows_by_gain[key] = rows
            timestep_summaries[key] = summarize_timestep_rows(rows, timesteps)
    finally:
        apply_class_embedding_gain(model, base_weight, 1.0)
    return rows_by_gain, timestep_summaries


def _request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "weights": str(args.weights),
        "num_samples": int(args.num_samples),
        "start_label": int(args.start_label),
        "wrong_label_offset": int(args.wrong_label_offset),
        "timesteps": [int(value) for value in args.timesteps],
        "gains": [float(value) for value in args.gains],
        "noise_seed": int(args.noise_seed),
        "threads": int(args.threads),
    }


def _validate_completed_report(
    report: Mapping[str, Any],
    *,
    git: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    request: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("git") != git
        or report.get("checkpoint") != checkpoint
        or report.get("request") != request
        or report.get("dataset") != dataset
    ):
        raise ValueError("Completed conditioning gain sweep binding differs")
    runtime = report.get("runtime")
    summary = report.get("gain_summary")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("device") != "cpu"
        or not math.isfinite(float(runtime.get("elapsed_seconds", -1.0)))
        or float(runtime["elapsed_seconds"]) <= 0.0
        or not isinstance(summary, Mapping)
        or len(summary) != len(request["gains"])
    ):
        raise ValueError("Completed conditioning gain sweep is malformed")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output = reject_symlink_chain(args.output, name="conditioning gain sweep output")
    if output.exists() and not output.is_file():
        raise ValueError(f"Conditioning gain sweep output is not a file: {output}")
    with exclusive_output_lock(output, role=REPORT_ROLE):
        started = time.monotonic()
        torch.set_num_threads(args.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        checkpoint_path, _integrity_path, checkpoint = _checkpoint_identity(
            args.checkpoint
        )
        payload, config = _load_checkpoint_payload(checkpoint_path, checkpoint)
        if args.start_label + args.num_samples > config.model.num_classes:
            raise ValueError("requested sample labels exceed checkpoint num_classes")
        if any(
            timestep < 0 or timestep >= config.diffusion.num_train_timesteps
            for timestep in args.timesteps
        ):
            raise ValueError("requested timestep is outside the diffusion schedule")
        dataset_root = Path(args.dataset_root) if args.dataset_root else (
            Path(config.data.root) / config.data.dataset
        )
        label_map, samples = select_samples(
            dataset_root=dataset_root,
            num_classes=config.model.num_classes,
            start_label=args.start_label,
            num_samples=args.num_samples,
        )
        dataset = {
            "alias": config.data.dataset,
            "root": reject_symlink_chain(
                dataset_root,
                name="conditioning gain dataset root",
            ).resolve().as_posix(),
            "label_map": label_map,
            "samples": samples,
        }
        git = git_provenance(PROJECT_ROOT)
        request = _request(args)
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "Conditioning gain sweep exists; pass --resume to validate it"
                )
            report = read_json_object(output, name="conditioning gain sweep")
            _validate_completed_report(
                report,
                git=git,
                checkpoint=checkpoint,
                request=request,
                dataset=dataset,
            )
            print(output.resolve().as_posix())
            return

        model = CoFiTokTiny(config.model)
        raw_state = payload["model"]
        ema_payload = payload["ema"]
        if not isinstance(raw_state, Mapping) or not isinstance(ema_payload, Mapping):
            raise ValueError("Checkpoint model or EMA state is malformed")
        ema_state = ema_payload.get("shadow")
        if not isinstance(ema_state, Mapping):
            raise ValueError("Checkpoint EMA shadow state is malformed")
        selected_state = ema_state if args.weights == "ema" else raw_state
        model.load_state_dict(selected_state, strict=True)
        model.eval()
        class_embed = getattr(model.predictor, "class_embed", None)
        if class_embed is None:
            raise ValueError("Checkpoint predictor has no class embedding")
        base_weight = class_embed.weight.detach().clone()
        del payload, raw_state, ema_payload, ema_state, selected_state
        gc.collect()

        rows_by_gain, timestep_summaries = run_gain_sweep(
            model=model,
            base_weight=base_weight,
            config=config,
            samples=samples,
            timesteps=args.timesteps,
            gains=args.gains,
            wrong_label_offset=args.wrong_label_offset,
            noise_seed=args.noise_seed,
        )
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "role": REPORT_ROLE,
            "status": "completed",
            "git": git,
            "checkpoint": checkpoint,
            "request": request,
            "dataset": dataset,
            "model": {
                "name": config.name,
                "predictor_type": config.model.predictor_type,
                "synthesis_mode": config.model.synthesis_mode,
                "token_count": config.model.token_count,
                "num_classes": config.model.num_classes,
            },
            "gain_summary": summarize_gain_rows(rows_by_gain),
            "timestep_summaries": timestep_summaries,
            "rows_by_gain": rows_by_gain,
            "runtime": {
                "device": "cpu",
                "threads": args.threads,
                "elapsed_seconds": time.monotonic() - started,
                "torch_version": torch.__version__,
            },
            "claim_boundary": {
                "diagnostic_only": True,
                "changes_checkpoint_bytes": False,
                "authorizes_training": False,
                "authorizes_sampling": False,
                "replaces_formal_quality_gate": False,
            },
        }
        write_json_report(output, report)
        _validate_completed_report(
            report,
            git=git,
            checkpoint=checkpoint,
            request=request,
            dataset=dataset,
        )
        print(output.resolve().as_posix())


if __name__ == "__main__":
    main()
