from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report


CHECKPOINT_EVAL = Path(
    "checkpoint_eval_ema_t500_512/checkpoint_evaluation_report.json"
)
SAMPLE_ROOT = Path("samples_probe512_ddim50_cfg15")
SAMPLING_REPORT = SAMPLE_ROOT / "sampling_report.json"
METRICS_REPORT = SAMPLE_ROOT / "metrics" / "generation_metrics_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize non-formal rank-recovery generation probes."
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="Candidate in NAME=RUN_DIR form; pass at least two.",
    )
    parser.add_argument("--legacy-checkpoint-eval", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
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


def _finite(value: object, *, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} is not finite")
    return result


def _candidate_summary(name: str, run_dir: Path) -> dict[str, Any]:
    training_path = run_dir / "training_report.json"
    checkpoint_eval_path = run_dir / CHECKPOINT_EVAL
    sampling_path = run_dir / SAMPLING_REPORT
    metrics_path = run_dir / METRICS_REPORT
    training = _read_json(training_path)
    checkpoint_eval = _read_json(checkpoint_eval_path)
    sampling = _read_json(sampling_path)
    generation = _read_json(metrics_path)

    if (
        training.get("training_complete") is not True
        or int(training.get("completed_steps", -1)) != 5_000
        or int(training.get("target_steps", -1)) != 5_000
    ):
        raise ValueError(f"candidate training is not an exact 5K probe: {run_dir}")
    if checkpoint_eval.get("status") != "completed":
        raise ValueError(f"candidate checkpoint evaluation is incomplete: {run_dir}")
    if sampling.get("status") != "completed":
        raise ValueError(f"candidate sampling is incomplete: {run_dir}")
    if generation.get("status") != "completed":
        raise ValueError(f"candidate generation metrics are incomplete: {run_dir}")
    if int(sampling.get("sampling", {}).get("num_samples", -1)) != 512:
        raise ValueError(f"candidate sampling is not the 512-image probe: {run_dir}")

    mechanism = checkpoint_eval["metrics"]
    ordered = mechanism["orders"]["ordered"]
    generated_metrics = generation["metrics"]
    return {
        "name": name,
        "run_dir": run_dir.as_posix(),
        "git": training.get("git", {}),
        "checkpoint_sha256": checkpoint_eval["checkpoint_sha256"],
        "endpoint_clean_mse": _finite(
            ordered["endpoint_clean_mse"], name=f"{name}.endpoint_clean_mse"
        ),
        "endpoint_clean_psnr": _finite(
            ordered["endpoint_clean_psnr"], name=f"{name}.endpoint_clean_psnr"
        ),
        "ordered_prefix_path_mse_auc": _finite(
            ordered["prefix_path_mse_auc"],
            name=f"{name}.ordered_prefix_path_mse_auc",
        ),
        "ordered_rank_by_path_auc": int(mechanism["ordered_rank_by_path_auc"]),
        "component_energy": [
            _finite(value, name=f"{name}.component_energy")
            for value in mechanism["component_energy"]
        ],
        "diagnostic_fid_512": _finite(
            generated_metrics["frechet_inception_distance"],
            name=f"{name}.diagnostic_fid_512",
        ),
        "diagnostic_inception_score_512": _finite(
            generated_metrics["inception_score_mean"],
            name=f"{name}.diagnostic_inception_score_512",
        ),
        "sources": {
            "training": _source(training_path),
            "checkpoint_evaluation": _source(checkpoint_eval_path),
            "sampling": _source(sampling_path),
            "generation_metrics": _source(metrics_path),
        },
    }


def main() -> None:
    args = parse_args()
    parsed = [_parse_candidate(raw) for raw in args.candidate]
    if len(parsed) < 2 or len({name for name, _ in parsed}) != len(parsed):
        raise ValueError("rank-recovery probe requires at least two uniquely named candidates")

    legacy_path = Path(args.legacy_checkpoint_eval).resolve()
    legacy = _read_json(legacy_path)
    legacy_ordered = legacy["metrics"]["orders"]["ordered"]
    legacy_endpoint = _finite(
        legacy_ordered["endpoint_clean_mse"], name="legacy.endpoint_clean_mse"
    )
    candidates = [_candidate_summary(name, run_dir) for name, run_dir in parsed]
    ranked = sorted(
        candidates,
        key=lambda row: (
            row["ordered_rank_by_path_auc"] != 1,
            row["endpoint_clean_mse"],
            row["diagnostic_fid_512"],
        ),
    )
    best = ranked[0]
    mechanism_ready = (
        best["ordered_rank_by_path_auc"] == 1
        and best["endpoint_clean_mse"] <= 0.95 * legacy_endpoint
    )
    report = {
        "schema_version": 1,
        "status": "completed",
        "role": "non_formal_rank_recovery_probe",
        "formal_claim_allowed": False,
        "protocol": {
            "training_steps": 5_000,
            "sample_count": 512,
            "sample_steps": 50,
            "guidance_scale": 1.5,
            "checkpoint_evaluation_images": 512,
            "checkpoint_evaluation_timestep": 500,
            "note": (
                "The 512-image FID/IS values are directional diagnostics only and "
                "cannot replace the 10K promotion gate."
            ),
        },
        "legacy_rank_deficient_reference": {
            "endpoint_clean_mse": legacy_endpoint,
            "ordered_rank_by_path_auc": int(
                legacy["metrics"]["ordered_rank_by_path_auc"]
            ),
            "component_energy": legacy["metrics"]["component_energy"],
            "source": _source(legacy_path),
        },
        "candidates": candidates,
        "selection": {
            "best_mechanism_candidate": best["name"],
            "mechanism_ready_for_visual_review": mechanism_ready,
            "decision": (
                "manual_visual_review_required"
                if mechanism_ready
                else "revise_architecture_or_objective"
            ),
            "automatic_50k_or_300k_launch_allowed": False,
        },
    }
    write_json_report(args.output, report)
    print(json.dumps(report["selection"], sort_keys=True))


if __name__ == "__main__":
    main()
