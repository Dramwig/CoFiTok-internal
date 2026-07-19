from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report


EMA_EVAL = Path("checkpoint_eval_ema_t500_512/checkpoint_evaluation_report.json")
MODEL_EVAL = Path("checkpoint_eval_model_t500_512/checkpoint_evaluation_report.json")


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _parse_candidate(raw: str) -> tuple[str, Path]:
    name, separator, run_dir = raw.partition("=")
    if not separator or not name.strip() or not run_dir.strip():
        raise ValueError(f"candidate must use NAME=RUN_DIR: {raw!r}")
    return name.strip(), Path(run_dir).resolve()


def ema_weight_profile(*, steps: int, decay: float, warmup_steps: int) -> dict[str, Any]:
    if steps < 1 or not 0.0 <= decay < 1.0 or warmup_steps < 0:
        raise ValueError("EMA profile parameters are invalid")
    contributions: list[tuple[int, float]] = []
    survival = 1.0
    for step in range(steps, 0, -1):
        ramp = 1.0 if warmup_steps == 0 else min(1.0, step / warmup_steps)
        effective_decay = decay * ramp
        contributions.append((step, (1.0 - effective_decay) * survival))
        survival *= effective_decay
    contributions.append((0, survival))
    total = sum(weight for _, weight in contributions)
    mean_step = sum(step * weight for step, weight in contributions) / total
    cumulative = 0.0
    median_step = 0
    for step, weight in sorted(contributions):
        cumulative += weight
        if cumulative >= 0.5 * total:
            median_step = step
            break
    return {
        "training_steps": steps,
        "target_decay": decay,
        "warmup_steps": warmup_steps,
        "weighted_mean_step": mean_step,
        "weighted_mean_age_steps": steps - mean_step,
        "median_step": median_step,
        "initial_model_weight": survival,
        "post_warmup_weight": sum(
            weight for step, weight in contributions if step >= warmup_steps
        ),
        "last_1000_steps_weight": sum(
            weight for step, weight in contributions if step >= max(1, steps - 999)
        ),
    }


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _metrics(report: dict[str, Any], *, candidate: str, weights: str) -> dict[str, Any]:
    if report.get("status") != "completed" or report.get("weights") != weights:
        raise ValueError(f"{candidate} {weights} checkpoint evaluation is invalid")
    metrics = report.get("metrics", {})
    ordered = metrics.get("orders", {}).get("ordered", {})
    if (
        int(report.get("checkpoint_step", -1)) != 5_000
        or int(metrics.get("evaluated_images", -1)) != 512
        or int(metrics.get("timestep", -1)) != 500
        or int(metrics.get("order_count", -1)) != 18
    ):
        raise ValueError(f"{candidate} {weights} evaluation protocol differs")
    return {
        "endpoint_clean_mse": _finite(
            ordered.get("endpoint_clean_mse"),
            name=f"{candidate}.{weights}.endpoint_clean_mse",
        ),
        "endpoint_clean_psnr": _finite(
            ordered.get("endpoint_clean_psnr"),
            name=f"{candidate}.{weights}.endpoint_clean_psnr",
        ),
        "ordered_prefix_path_mse_auc": _finite(
            ordered.get("prefix_path_mse_auc"),
            name=f"{candidate}.{weights}.prefix_path_mse_auc",
        ),
        "ordered_rank_by_path_auc": int(metrics["ordered_rank_by_path_auc"]),
        "component_energy": [
            _finite(value, name=f"{candidate}.{weights}.component_energy")
            for value in metrics["component_energy"]
        ],
    }


def _candidate(name: str, run_dir: Path) -> dict[str, Any]:
    training_path = run_dir / "training_report.json"
    ema_path = run_dir / EMA_EVAL
    model_path = run_dir / MODEL_EVAL
    training = _read_object(training_path)
    ema_report = _read_object(ema_path)
    model_report = _read_object(model_path)
    if (
        training.get("training_complete") is not True
        or int(training.get("completed_steps", -1)) != 5_000
        or int(training.get("target_steps", -1)) != 5_000
    ):
        raise ValueError(f"{name} training is not an exact completed 5K probe")
    if (
        ema_report.get("checkpoint_sha256") != model_report.get("checkpoint_sha256")
        or ema_report.get("config") != model_report.get("config")
        or ema_report.get("git") != model_report.get("git")
        or ema_report.get("git") != training.get("git")
    ):
        raise ValueError(f"{name} model/EMA evaluation identity differs")
    ema_metrics = _metrics(ema_report, candidate=name, weights="ema")
    model_metrics = _metrics(model_report, candidate=name, weights="model")
    optimization = training["config"]["optimization"]
    profile = ema_weight_profile(
        steps=5_000,
        decay=float(optimization["ema_decay"]),
        warmup_steps=int(optimization["ema_warmup_steps"]),
    )
    return {
        "name": name,
        "run_dir": run_dir.as_posix(),
        "git": training["git"],
        "checkpoint_sha256": ema_report["checkpoint_sha256"],
        "ema_profile": profile,
        "ema": ema_metrics,
        "model": model_metrics,
        "comparison": {
            "ema_to_model_endpoint_mse_ratio": (
                ema_metrics["endpoint_clean_mse"]
                / model_metrics["endpoint_clean_mse"]
            ),
            "ema_minus_model_path_auc": (
                ema_metrics["ordered_prefix_path_mse_auc"]
                - model_metrics["ordered_prefix_path_mse_auc"]
            ),
            "ordered_rank_agrees": (
                ema_metrics["ordered_rank_by_path_auc"]
                == model_metrics["ordered_rank_by_path_auc"]
            ),
        },
        "sources": {
            "training": _source(training_path),
            "ema_checkpoint_evaluation": _source(ema_path),
            "model_checkpoint_evaluation": _source(model_path),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit short-probe EMA lag against raw model weights."
    )
    parser.add_argument("--candidate", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    parsed = [_parse_candidate(raw) for raw in args.candidate]
    if len(parsed) < 2 or len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("EMA audit requires at least two uniquely named candidates")
    candidates = [_candidate(name, run_dir) for name, run_dir in parsed]
    report = {
        "schema_version": 1,
        "status": "completed",
        "role": "non_formal_short_probe_ema_audit",
        "formal_claim_allowed": False,
        "automatic_50k_or_300k_launch_allowed": False,
        "candidate_selection_rule": (
            "Use raw-model diagnostics to detect short-horizon EMA lag; require the "
            "formal 50K and all promotion/final sampling to use EMA weights."
        ),
        "candidates": candidates,
    }
    write_json_report(args.output, report)
    print(json.dumps({"status": report["status"], "candidates": len(candidates)}))


if __name__ == "__main__":
    main()
