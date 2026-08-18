from __future__ import annotations

import argparse
import gc
import math
import statistics
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
from cofitok.models.scalable_unet import ConditionedResBlock
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report
from scripts.evaluate_generation_conditioning_sensitivity import (
    _checkpoint_identity,
    _load_checkpoint_payload,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_conditioning_path_activation_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit time/class embedding and conditioned-ResBlock modulation "
            "scales in a frozen class-conditional generation checkpoint."
        )
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--num-labels", type=int, default=8)
    parser.add_argument("--start-label", type=int, default=0)
    parser.add_argument("--wrong-label-offset", type=int, default=500)
    parser.add_argument("--timesteps", type=int, nargs="+", default=[100, 500, 900])
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.num_labels < 1:
        raise ValueError("num_labels must be positive")
    if args.start_label < 0:
        raise ValueError("start_label must be non-negative")
    if args.wrong_label_offset < 1:
        raise ValueError("wrong_label_offset must be positive")
    if args.threads < 1:
        raise ValueError("threads must be positive")
    if not args.timesteps or len(set(args.timesteps)) != len(args.timesteps):
        raise ValueError("timesteps must be non-empty and unique")


def _tensor_rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt())


def _tensor_norm(value: torch.Tensor) -> float:
    return float(value.detach().float().norm())


def _cosine(first: torch.Tensor, second: torch.Tensor) -> float | None:
    first_value = first.detach().float().flatten()
    second_value = second.detach().float().flatten()
    denominator = float(first_value.norm() * second_value.norm())
    if denominator == 0.0:
        return None
    return float(torch.dot(first_value, second_value) / denominator)


def _relative_delta(first: torch.Tensor, second: torch.Tensor) -> float:
    return _tensor_rms(first.detach().float() - second.detach().float()) / max(
        _tensor_rms(first), 1e-12
    )


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty sequence")
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "population_std": statistics.pstdev(values),
    }


def _conditioned_blocks(model: CoFiTokTiny) -> list[tuple[str, ConditionedResBlock]]:
    blocks = [
        (name, module)
        for name, module in model.predictor.named_modules()
        if isinstance(module, ConditionedResBlock)
    ]
    if not blocks:
        raise ValueError("Checkpoint predictor has no conditioned residual blocks")
    return blocks


@torch.no_grad()
def audit_conditioning_path(
    *,
    model: CoFiTokTiny,
    labels: Sequence[int],
    wrong_label_offset: int,
    timesteps: Sequence[int],
) -> dict[str, Any]:
    predictor = model.predictor
    class_embed = getattr(predictor, "class_embed", None)
    time_embed = getattr(predictor, "time_embed", None)
    null_label = int(getattr(predictor, "null_class", -1))
    if class_embed is None or time_embed is None or null_label < 1:
        raise ValueError("Predictor does not expose class/time conditioning")
    blocks = _conditioned_blocks(model)
    rows = []
    for timestep in timesteps:
        timestep_tensor = torch.tensor([timestep], dtype=torch.long)
        time_value = time_embed(timestep_tensor)[0].float()
        null_value = class_embed(
            torch.tensor([null_label], dtype=torch.long)
        )[0].float()
        for correct_label in labels:
            wrong_label = (int(correct_label) + wrong_label_offset) % null_label
            if wrong_label == correct_label:
                raise ValueError("wrong label resolved to the correct label")
            correct_value = class_embed(
                torch.tensor([correct_label], dtype=torch.long)
            )[0].float()
            wrong_value = class_embed(
                torch.tensor([wrong_label], dtype=torch.long)
            )[0].float()
            combined = {
                "correct": time_value + correct_value,
                "wrong": time_value + wrong_value,
                "null": time_value + null_value,
            }
            block_rows = []
            for block_name, block in blocks:
                modulation = {
                    condition: block.embedding(value.unsqueeze(0))[0].float()
                    for condition, value in combined.items()
                }
                block_rows.append(
                    {
                        "block": block_name,
                        "modulation_rms": {
                            condition: _tensor_rms(value)
                            for condition, value in modulation.items()
                        },
                        "relative_delta": {
                            "correct_vs_wrong": _relative_delta(
                                modulation["correct"], modulation["wrong"]
                            ),
                            "correct_vs_null": _relative_delta(
                                modulation["correct"], modulation["null"]
                            ),
                            "wrong_vs_null": _relative_delta(
                                modulation["wrong"], modulation["null"]
                            ),
                        },
                    }
                )
            rows.append(
                {
                    "timestep": timestep,
                    "correct_label": int(correct_label),
                    "wrong_label": wrong_label,
                    "null_label": null_label,
                    "embedding": {
                        "time": {"rms": _tensor_rms(time_value), "norm": _tensor_norm(time_value)},
                        "correct_class": {
                            "rms": _tensor_rms(correct_value),
                            "norm": _tensor_norm(correct_value),
                        },
                        "wrong_class": {
                            "rms": _tensor_rms(wrong_value),
                            "norm": _tensor_norm(wrong_value),
                        },
                        "null_class": {
                            "rms": _tensor_rms(null_value),
                            "norm": _tensor_norm(null_value),
                        },
                        "class_to_time_norm_ratio": {
                            "correct": _tensor_norm(correct_value)
                            / max(_tensor_norm(time_value), 1e-12),
                            "wrong": _tensor_norm(wrong_value)
                            / max(_tensor_norm(time_value), 1e-12),
                            "null": _tensor_norm(null_value)
                            / max(_tensor_norm(time_value), 1e-12),
                        },
                        "cosine_with_time": {
                            "correct": _cosine(correct_value, time_value),
                            "wrong": _cosine(wrong_value, time_value),
                            "null": _cosine(null_value, time_value),
                        },
                        "combined_rms": {
                            condition: _tensor_rms(value)
                            for condition, value in combined.items()
                        },
                        "relative_delta": {
                            "correct_vs_wrong": _relative_delta(
                                combined["correct"], combined["wrong"]
                            ),
                            "correct_vs_null": _relative_delta(
                                combined["correct"], combined["null"]
                            ),
                            "wrong_vs_null": _relative_delta(
                                combined["wrong"], combined["null"]
                            ),
                        },
                    },
                    "blocks": block_rows,
                }
            )
    return {
        "row_count": len(rows),
        "block_count": len(blocks),
        "block_names": [name for name, _block in blocks],
        "rows": rows,
        "summary": summarize_path_rows(rows, timesteps),
    }


def summarize_path_rows(
    rows: Sequence[Mapping[str, Any]],
    timesteps: Sequence[int],
) -> list[dict[str, Any]]:
    summaries = []
    for timestep in timesteps:
        selected = [row for row in rows if int(row["timestep"]) == timestep]
        if not selected:
            raise ValueError(f"No conditioning path rows for timestep {timestep}")
        block_rows = [block for row in selected for block in row["blocks"]]
        summaries.append(
            {
                "timestep": timestep,
                "label_count": len(selected),
                "class_to_time_norm_ratio": {
                    condition: _summary(
                        [
                            float(
                                row["embedding"]["class_to_time_norm_ratio"][
                                    condition
                                ]
                            )
                            for row in selected
                        ]
                    )
                    for condition in ("correct", "wrong", "null")
                },
                "combined_embedding_relative_delta": {
                    comparison: _summary(
                        [
                            float(row["embedding"]["relative_delta"][comparison])
                            for row in selected
                        ]
                    )
                    for comparison in (
                        "correct_vs_wrong",
                        "correct_vs_null",
                        "wrong_vs_null",
                    )
                },
                "block_modulation_relative_delta": {
                    comparison: _summary(
                        [
                            float(block["relative_delta"][comparison])
                            for block in block_rows
                        ]
                    )
                    for comparison in (
                        "correct_vs_wrong",
                        "correct_vs_null",
                        "wrong_vs_null",
                    )
                },
            }
        )
    return summaries


def _request(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "weights": str(args.weights),
        "num_labels": int(args.num_labels),
        "start_label": int(args.start_label),
        "wrong_label_offset": int(args.wrong_label_offset),
        "timesteps": [int(value) for value in args.timesteps],
        "threads": int(args.threads),
    }


def _validate_completed_report(
    report: Mapping[str, Any],
    *,
    git: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    request: Mapping[str, Any],
) -> None:
    if (
        report.get("schema_version") != REPORT_SCHEMA_VERSION
        or report.get("role") != REPORT_ROLE
        or report.get("status") != "completed"
        or report.get("git") != git
        or report.get("checkpoint") != checkpoint
        or report.get("request") != request
    ):
        raise ValueError("Completed conditioning path audit binding differs")
    runtime = report.get("runtime")
    audit = report.get("audit")
    if (
        not isinstance(runtime, Mapping)
        or runtime.get("device") != "cpu"
        or not math.isfinite(float(runtime.get("elapsed_seconds", -1.0)))
        or float(runtime["elapsed_seconds"]) <= 0.0
        or not isinstance(audit, Mapping)
        or int(audit.get("row_count", -1))
        != int(request["num_labels"]) * len(request["timesteps"])
    ):
        raise ValueError("Completed conditioning path audit is malformed")


def main() -> None:
    args = parse_args()
    _validate_args(args)
    output = reject_symlink_chain(args.output, name="conditioning path audit output")
    if output.exists() and not output.is_file():
        raise ValueError(f"Conditioning path audit output is not a file: {output}")
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
        if args.start_label + args.num_labels > config.model.num_classes:
            raise ValueError("requested labels exceed checkpoint num_classes")
        if any(
            timestep < 0 or timestep >= config.diffusion.num_train_timesteps
            for timestep in args.timesteps
        ):
            raise ValueError("requested timestep is outside the diffusion schedule")
        git = git_provenance(PROJECT_ROOT)
        request = _request(args)
        if output.exists():
            if not args.resume:
                raise FileExistsError(
                    "Conditioning path audit exists; pass --resume to validate it"
                )
            report = read_json_object(output, name="conditioning path audit")
            _validate_completed_report(
                report,
                git=git,
                checkpoint=checkpoint,
                request=request,
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
        del payload, raw_state, ema_payload, ema_state, selected_state
        gc.collect()
        labels = list(range(args.start_label, args.start_label + args.num_labels))
        audit = audit_conditioning_path(
            model=model,
            labels=labels,
            wrong_label_offset=args.wrong_label_offset,
            timesteps=args.timesteps,
        )
        report = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "role": REPORT_ROLE,
            "status": "completed",
            "git": git,
            "checkpoint": checkpoint,
            "request": request,
            "model": {
                "name": config.name,
                "predictor_type": config.model.predictor_type,
                "synthesis_mode": config.model.synthesis_mode,
                "token_count": config.model.token_count,
                "num_classes": config.model.num_classes,
            },
            "audit": audit,
            "runtime": {
                "device": "cpu",
                "threads": args.threads,
                "elapsed_seconds": time.monotonic() - started,
                "torch_version": torch.__version__,
            },
            "claim_boundary": {
                "diagnostic_only": True,
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
        )
        print(output.resolve().as_posix())


if __name__ == "__main__":
    main()
