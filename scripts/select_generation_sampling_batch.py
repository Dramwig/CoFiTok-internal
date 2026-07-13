from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from cofitok.reporting import write_json_report
from cofitok.training.checkpointing import checkpoint_integrity_path


def parse_candidates(value: str, *, baseline_batch_size: int) -> list[int]:
    candidates = []
    for raw in value.split(","):
        stripped = raw.strip()
        if not stripped.isdigit() or int(stripped) < 1:
            raise ValueError(f"invalid sampling batch candidate: {raw}")
        batch_size = int(stripped)
        if batch_size not in candidates:
            candidates.append(batch_size)
    if not candidates:
        raise ValueError("at least one sampling batch candidate is required")
    if baseline_batch_size not in candidates:
        raise ValueError("sampling candidates must retain the baseline batch size")
    return candidates


def _protocol(report: dict[str, Any]) -> dict[str, Any]:
    request = report.get("request", {})
    ignored = {"batch_size", "effective_model_batch_size", "prefix_budget", "token_count"}
    return {key: value for key, value in request.items() if key not in ignored}


def select_sampling_batch(
    candidates: list[dict[str, Any]],
    *,
    baseline_batch_size: int,
    max_memory_fraction: float,
) -> dict[str, Any]:
    if not 0.0 < max_memory_fraction < 1.0:
        raise ValueError("max_memory_fraction must be between zero and one")
    eligible = []
    normalized = []
    for row in candidates:
        candidate = dict(row)
        batch_size = int(candidate.get("batch_size", -1))
        methods = candidate.get("methods", {})
        reasons = []
        throughputs = []
        memory_fractions = []
        protocols = []
        checkpoint_steps = []
        for method in ("cofitok", "dense_identity"):
            report = methods.get(method, {})
            if report.get("status") != "passed":
                reasons.append(f"{method}_preflight_incomplete")
                continue
            request = report.get("request", {})
            result = report.get("result", {})
            if int(request.get("batch_size", -1)) != batch_size:
                reasons.append(f"{method}_batch_mismatch")
            throughput = float(result.get("output_images_per_second", math.nan))
            if not math.isfinite(throughput) or throughput <= 0.0:
                reasons.append(f"{method}_invalid_throughput")
            else:
                throughputs.append(throughput)
            peak_memory = result.get("cuda_memory_after_forward") or {}
            peak = int(peak_memory.get("peak_allocated_bytes", 0))
            total = int(result.get("device_total_memory_bytes", 0))
            if peak <= 0 or total <= 0 or peak > total:
                reasons.append(f"{method}_invalid_memory_accounting")
            else:
                fraction = peak / total
                memory_fractions.append(fraction)
                if fraction > max_memory_fraction:
                    reasons.append(f"{method}_memory_headroom")
            protocols.append(_protocol(report))
            checkpoint_steps.append(int(report.get("checkpoint_step", -1)))
        if len(protocols) == 2 and protocols[0] != protocols[1]:
            reasons.append("matched_sampling_protocol_mismatch")
        if len(checkpoint_steps) == 2 and checkpoint_steps[0] != checkpoint_steps[1]:
            reasons.append("matched_checkpoint_step_mismatch")
        candidate["eligible"] = not reasons
        candidate["ineligible_reasons"] = sorted(set(reasons))
        candidate["selection_score_images_per_second"] = (
            min(throughputs) if len(throughputs) == 2 else None
        )
        candidate["max_memory_fraction"] = (
            max(memory_fractions) if len(memory_fractions) == 2 else None
        )
        normalized.append(candidate)
        if candidate["eligible"]:
            eligible.append(candidate)
    if not eligible:
        raise ValueError("no shared generation sampling batch passed")
    baseline = next(
        (row for row in normalized if int(row["batch_size"]) == baseline_batch_size),
        None,
    )
    if baseline is None or not baseline["eligible"]:
        raise ValueError("conservative sampling batch baseline did not pass")
    selected = max(
        eligible,
        key=lambda row: (
            float(row["selection_score_images_per_second"]),
            -int(row["batch_size"]),
        ),
    )
    speedup = float(selected["selection_score_images_per_second"]) / float(
        baseline["selection_score_images_per_second"]
    )
    return {
        "schema_version": 1,
        "status": "selected",
        "policy": {
            "shared_candidate_required": True,
            "score": "maximize_worst_method_output_images_per_second",
            "baseline_batch_size": baseline_batch_size,
            "max_memory_fraction": max_memory_fraction,
            "batch_size_invariant_random_stream_required": True,
        },
        "selected": {
            "batch_size": int(selected["batch_size"]),
            "selection_score_images_per_second": float(
                selected["selection_score_images_per_second"]
            ),
            "max_memory_fraction": float(selected["max_memory_fraction"]),
            "estimated_speedup_over_baseline": speedup,
        },
        "candidates": normalized,
    }


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _checkpoint_identity(path: Path) -> dict[str, Any]:
    integrity_path = checkpoint_integrity_path(path)
    integrity = _read(integrity_path)
    if integrity.get("checkpoint") != path.name:
        raise ValueError(f"checkpoint integrity sidecar names another file: {path}")
    return {
        "path": path.resolve().as_posix(),
        "sha256": integrity["checkpoint_sha256"],
        "step": int(integrity["step"]),
        "integrity_manifest": integrity_path.resolve().as_posix(),
    }


def _archive_stale_run(run_dir: Path) -> None:
    if not run_dir.exists():
        return
    suffix = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    archived = run_dir.with_name(f"{run_dir.name}.stale.{suffix}")
    counter = 1
    while archived.exists():
        archived = run_dir.with_name(f"{run_dir.name}.stale.{suffix}.{counter}")
        counter += 1
    run_dir.rename(archived)


def _preflight_matches(
    report: dict[str, Any],
    *,
    checkpoint_identity: dict[str, Any],
    expected_revision: str,
    batch_size: int,
    prefix_budget: int,
    guidance_scale: float,
    guidance_rescale: float,
    cfg_batch_mode: str,
    weights: str,
    precision: str,
    warmup_forwards: int,
    measured_forwards: int,
) -> bool:
    request = report.get("request", {})
    return (
        report.get("status") == "passed"
        and report.get("git_revision") == expected_revision
        and report.get("checkpoint_sha256") == checkpoint_identity["sha256"]
        and int(report.get("checkpoint_step", -1)) == checkpoint_identity["step"]
        and int(request.get("batch_size", -1)) == batch_size
        and int(request.get("prefix_budget", -1)) == prefix_budget
        and float(request.get("guidance_scale", math.nan)) == guidance_scale
        and float(request.get("guidance_rescale", math.nan)) == guidance_rescale
        and request.get("cfg_batch_mode") == cfg_batch_mode
        and report.get("weights") == weights
        and request.get("precision") == precision
        and int(request.get("warmup_forwards", -1)) == warmup_forwards
        and int(request.get("measured_forwards", -1)) == measured_forwards
    )


def _run_preflight(
    *,
    method: str,
    checkpoint: Path,
    checkpoint_identity: dict[str, Any],
    expected_revision: str,
    batch_size: int,
    prefix_budget: int,
    output_root: Path,
    preflight_script: Path,
    project_root: Path,
    guidance_scale: float,
    guidance_rescale: float,
    cfg_batch_mode: str,
    weights: str,
    precision: str,
    warmup_forwards: int,
    measured_forwards: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    run_dir = output_root / method / f"batch_{batch_size}"
    report_path = run_dir / "sampling_preflight.json"
    expected = {
        "checkpoint_identity": checkpoint_identity,
        "expected_revision": expected_revision,
        "batch_size": batch_size,
        "prefix_budget": prefix_budget,
        "guidance_scale": guidance_scale,
        "guidance_rescale": guidance_rescale,
        "cfg_batch_mode": cfg_batch_mode,
        "weights": weights,
        "precision": precision,
        "warmup_forwards": warmup_forwards,
        "measured_forwards": measured_forwards,
    }
    if report_path.is_file():
        existing = _read(report_path)
        if _preflight_matches(existing, **expected):
            return existing
        _archive_stale_run(run_dir)
    elif run_dir.exists():
        _archive_stale_run(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(preflight_script),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(report_path),
        "--batch-size",
        str(batch_size),
        "--prefix-budget",
        str(prefix_budget),
        "--guidance-scale",
        str(guidance_scale),
        "--guidance-rescale",
        str(guidance_rescale),
        "--cfg-batch-mode",
        cfg_batch_mode,
        "--weights",
        weights,
        "--precision",
        precision,
        "--warmup-forwards",
        str(warmup_forwards),
        "--measured-forwards",
        str(measured_forwards),
    ]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    try:
        result = subprocess.run(
            command,
            cwd=project_root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "status": "failed",
            "failure_type": "timeout",
            "timeout_seconds": timeout_seconds,
            "stdout": error.stdout,
            "stderr": error.stderr,
        }
    if report_path.is_file():
        report = _read(report_path)
        if report.get("status") == "passed" and not _preflight_matches(
            report, **expected
        ):
            raise ValueError(f"{method} batch {batch_size} preflight provenance mismatch")
        return report
    return {
        "status": "failed",
        "failure_type": "process_error",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4_000:],
        "stderr_tail": result.stderr[-4_000:],
    }


def _git_revision(project_root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select one shared, memory-safe batch for matched generation sampling."
    )
    parser.add_argument("--cofitok-checkpoint", required=True)
    parser.add_argument("--dense-checkpoint", required=True)
    parser.add_argument("--cofitok-prefix-budget", type=int, required=True)
    parser.add_argument("--dense-prefix-budget", type=int, required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", default="16,32,64,128")
    parser.add_argument("--baseline-batch-size", type=int, default=32)
    parser.add_argument("--guidance-scale", type=float, default=1.5)
    parser.add_argument("--guidance-rescale", type=float, default=0.0)
    parser.add_argument("--cfg-batch-mode", choices=["batched", "sequential"], default="batched")
    parser.add_argument("--weights", choices=["ema", "model"], default="ema")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--warmup-forwards", type=int, default=2)
    parser.add_argument("--measured-forwards", type=int, default=5)
    parser.add_argument("--max-memory-fraction", type=float, default=0.90)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--preflight-script", default="scripts/preflight_generation_sampling.py")
    args = parser.parse_args()
    if args.warmup_forwards < 0 or args.measured_forwards < 1 or args.timeout_seconds < 1:
        raise ValueError("sampling selection measurement settings are invalid")

    project_root = Path(args.project_root).resolve()
    cofitok_checkpoint = Path(args.cofitok_checkpoint).resolve()
    dense_checkpoint = Path(args.dense_checkpoint).resolve()
    output_root = Path(args.output_root).resolve()
    preflight_script = (project_root / args.preflight_script).resolve()
    revision = _git_revision(project_root)
    candidates = parse_candidates(
        args.candidates,
        baseline_batch_size=args.baseline_batch_size,
    )
    identities = {
        "cofitok": _checkpoint_identity(cofitok_checkpoint),
        "dense_identity": _checkpoint_identity(dense_checkpoint),
    }
    rows = []
    for batch_size in candidates:
        rows.append(
            {
                "batch_size": batch_size,
                "methods": {
                    "cofitok": _run_preflight(
                        method="cofitok",
                        checkpoint=cofitok_checkpoint,
                        checkpoint_identity=identities["cofitok"],
                        expected_revision=revision,
                        batch_size=batch_size,
                        prefix_budget=args.cofitok_prefix_budget,
                        output_root=output_root,
                        preflight_script=preflight_script,
                        project_root=project_root,
                        guidance_scale=args.guidance_scale,
                        guidance_rescale=args.guidance_rescale,
                        cfg_batch_mode=args.cfg_batch_mode,
                        weights=args.weights,
                        precision=args.precision,
                        warmup_forwards=args.warmup_forwards,
                        measured_forwards=args.measured_forwards,
                        timeout_seconds=args.timeout_seconds,
                    ),
                    "dense_identity": _run_preflight(
                        method="dense_identity",
                        checkpoint=dense_checkpoint,
                        checkpoint_identity=identities["dense_identity"],
                        expected_revision=revision,
                        batch_size=batch_size,
                        prefix_budget=args.dense_prefix_budget,
                        output_root=output_root,
                        preflight_script=preflight_script,
                        project_root=project_root,
                        guidance_scale=args.guidance_scale,
                        guidance_rescale=args.guidance_rescale,
                        cfg_batch_mode=args.cfg_batch_mode,
                        weights=args.weights,
                        precision=args.precision,
                        warmup_forwards=args.warmup_forwards,
                        measured_forwards=args.measured_forwards,
                        timeout_seconds=args.timeout_seconds,
                    ),
                },
            }
        )
    selection = select_sampling_batch(
        rows,
        baseline_batch_size=args.baseline_batch_size,
        max_memory_fraction=args.max_memory_fraction,
    )
    selection["git_revision"] = revision
    selection["checkpoints"] = identities
    selection["benchmark_root"] = output_root.as_posix()
    write_json_report(args.output, selection)
    print(selection["selected"]["batch_size"])


if __name__ == "__main__":
    main()
