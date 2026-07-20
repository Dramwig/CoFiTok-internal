from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch

from cofitok.configs import config_to_dict, load_config
from cofitok.data.provenance import validate_dataset_provenance
from cofitok.environment import capture_runtime_environment, runtime_environment_sha256
from cofitok.reporting import file_sha256, git_provenance, write_json_report


CANDIDATE_PATTERN = re.compile(r"^(?P<micro>[1-9][0-9]*)x(?P<accum>[1-9][0-9]*)$")


def _validated_runtime_environment_sha(report: dict[str, Any]) -> str | None:
    environment = report.get("runtime_environment")
    if not isinstance(environment, dict):
        return None
    actual = runtime_environment_sha256(environment)
    return actual if report.get("runtime_environment_sha256") == actual else None


def _validated_dataset_identity_sha(report: dict[str, Any]) -> str | None:
    provenance = report.get("dataset_provenance")
    if not isinstance(provenance, dict):
        return None
    config = report.get("config")
    config_data = config.get("data") if isinstance(config, dict) else None
    expected_dataset = (
        str(config_data.get("dataset", ""))
        if isinstance(config_data, dict) and config_data.get("dataset")
        else str(provenance.get("dataset", ""))
    )
    try:
        return validate_dataset_provenance(
            provenance,
            expected_dataset=expected_dataset,
        )["identity_sha256"]
    except (TypeError, ValueError):
        return None


def parse_candidates(value: str, *, expected_effective_batch: int) -> list[tuple[int, int]]:
    candidates = []
    for raw in value.split(","):
        match = CANDIDATE_PATTERN.fullmatch(raw.strip())
        if match is None:
            raise ValueError(f"invalid runtime candidate: {raw}")
        candidate = (int(match.group("micro")), int(match.group("accum")))
        if candidate[0] * candidate[1] != expected_effective_batch:
            raise ValueError(f"runtime candidate {raw} changes effective batch")
        if candidate not in candidates:
            candidates.append(candidate)
    if not candidates:
        raise ValueError("at least one runtime candidate is required")
    if (16, 4) not in candidates:
        raise ValueError("runtime candidates must retain the conservative 16x4 baseline")
    return candidates


def select_runtime_candidate(
    candidates: list[dict[str, Any]],
    *,
    expected_effective_batch: int,
    max_memory_fraction: float,
) -> dict[str, Any]:
    if not 0.0 < max_memory_fraction < 1.0:
        raise ValueError("max_memory_fraction must be between zero and one")
    eligible = []
    normalized = []
    environment_provenance_issues = []
    observed_method_environment_shas = []
    dataset_provenance_issues = []
    observed_method_dataset_shas = []
    for row in candidates:
        candidate = dict(row)
        methods = candidate.get("methods", {})
        reasons = []
        scores = []
        memory_fractions = []
        environment_shas = []
        dataset_shas = []
        for method in ("cofitok", "dense_identity"):
            report = methods.get(method, {})
            if report.get("status") != "completed":
                reasons.append(f"{method}_benchmark_incomplete")
                continue
            if int(report.get("effective_batch_size", -1)) != expected_effective_batch:
                reasons.append(f"{method}_effective_batch_mismatch")
            environment_sha = _validated_runtime_environment_sha(report)
            if environment_sha is None:
                reason = f"{method}_invalid_runtime_environment"
                reasons.append(reason)
                environment_provenance_issues.append(reason)
            else:
                environment_shas.append(environment_sha)
                observed_method_environment_shas.append(environment_sha)
            dataset_sha = _validated_dataset_identity_sha(report)
            if dataset_sha is None:
                reason = f"{method}_invalid_dataset_provenance"
                reasons.append(reason)
                dataset_provenance_issues.append(reason)
            else:
                dataset_shas.append(dataset_sha)
                observed_method_dataset_shas.append(dataset_sha)
            seconds = float(report.get("mean_optimizer_step_seconds", math.nan))
            throughput = float(report.get("images_per_second", math.nan))
            if not math.isfinite(seconds) or seconds <= 0.0:
                reasons.append(f"{method}_invalid_step_time")
            if not math.isfinite(throughput) or throughput <= 0.0:
                reasons.append(f"{method}_invalid_throughput")
            if math.isfinite(seconds) and seconds > 0.0:
                scores.append(seconds)
            peak = int(report.get("peak_vram_bytes", 0))
            total = int(report.get("device_total_memory_bytes", 0))
            if peak <= 0 or total <= 0 or peak > total:
                reasons.append(f"{method}_invalid_memory_accounting")
            else:
                fraction = peak / total
                memory_fractions.append(fraction)
                if fraction > max_memory_fraction:
                    reasons.append(f"{method}_memory_headroom")
        if len(environment_shas) == 2 and len(set(environment_shas)) != 1:
            reasons.append("matched_runtime_environment_mismatch")
            environment_provenance_issues.append(
                "matched_runtime_environment_mismatch"
            )
        if len(dataset_shas) == 2 and len(set(dataset_shas)) != 1:
            reasons.append("matched_dataset_identity_mismatch")
            dataset_provenance_issues.append("matched_dataset_identity_mismatch")
        candidate["runtime_environment_sha256"] = (
            environment_shas[0]
            if len(environment_shas) == 2 and len(set(environment_shas)) == 1
            else None
        )
        candidate["dataset_identity_sha256"] = (
            dataset_shas[0]
            if len(dataset_shas) == 2 and len(set(dataset_shas)) == 1
            else None
        )
        candidate["eligible"] = not reasons
        candidate["ineligible_reasons"] = sorted(set(reasons))
        candidate["selection_score_seconds"] = max(scores) if len(scores) == 2 else None
        candidate["max_memory_fraction"] = (
            max(memory_fractions) if len(memory_fractions) == 2 else None
        )
        normalized.append(candidate)
        if candidate["eligible"]:
            eligible.append(candidate)
    if environment_provenance_issues:
        raise ValueError(
            "runtime benchmark environment provenance failed: "
            + ", ".join(sorted(set(environment_provenance_issues)))
        )
    if dataset_provenance_issues:
        raise ValueError(
            "runtime benchmark dataset provenance failed: "
            + ", ".join(sorted(set(dataset_provenance_issues)))
        )
    observed_environment_shas = {
        str(value) for value in observed_method_environment_shas
    }
    if len(observed_environment_shas) > 1:
        raise ValueError("runtime environment changed across benchmark candidates")
    observed_dataset_shas = {str(value) for value in observed_method_dataset_shas}
    if len(observed_dataset_shas) > 1:
        raise ValueError("dataset identity changed across benchmark candidates")
    if not eligible:
        raise ValueError("no shared generation runtime candidate passed")
    selected = min(
        eligible,
        key=lambda row: (
            float(row["selection_score_seconds"]),
            int(row["micro_batch_size"]),
        ),
    )
    baseline = next(
        (
            row
            for row in normalized
            if int(row["micro_batch_size"]) == 16
            and int(row["gradient_accumulation_steps"]) == 4
        ),
        None,
    )
    if baseline is None or not baseline["eligible"]:
        raise ValueError("conservative 16x4 runtime baseline did not pass")
    speedup = float(baseline["selection_score_seconds"]) / float(
        selected["selection_score_seconds"]
    )
    return {
        "schema_version": 2,
        "status": "selected",
        "policy": {
            "shared_candidate_required": True,
            "score": "minimize_worst_method_mean_optimizer_step_seconds",
            "expected_effective_batch_size": expected_effective_batch,
            "max_memory_fraction": max_memory_fraction,
        },
        "selected": {
            "micro_batch_size": int(selected["micro_batch_size"]),
            "gradient_accumulation_steps": int(
                selected["gradient_accumulation_steps"]
            ),
            "effective_batch_size": expected_effective_batch,
            "selection_score_seconds": float(selected["selection_score_seconds"]),
            "max_memory_fraction": float(selected["max_memory_fraction"]),
            "estimated_speedup_over_16x4": speedup,
        },
        "runtime_environment_sha256": selected["runtime_environment_sha256"],
        "dataset_identity_sha256": selected["dataset_identity_sha256"],
        "candidates": normalized,
    }


def _expected_config(path: Path, micro_batch: int, accumulation: int) -> dict[str, Any]:
    config = load_config(path)
    config = replace(config, data=replace(config.data, batch_size=micro_batch))
    config = replace(
        config,
        optimization=replace(
            config.optimization,
            gradient_accumulation_steps=accumulation,
        ),
    )
    return config_to_dict(config)


def _training_state(run_dirs: list[Path]) -> dict[str, Any]:
    runs = []
    for run_dir in run_dirs:
        if run_dir.exists() and not run_dir.is_dir():
            raise ValueError(f"training run path is not a directory: {run_dir}")
        entries = sorted(item.name for item in run_dir.iterdir()) if run_dir.is_dir() else []
        runs.append(
            {
                "path": run_dir.as_posix(),
                "started": bool(entries),
                "entry_count": len(entries),
                "entries_preview": entries[:20],
            }
        )
    return {"started": any(run["started"] for run in runs), "runs": runs}


def _selection_contract(
    *,
    run_dirs: list[Path],
    candidates: list[tuple[int, int]],
    expected_effective_batch: int,
    benchmark_steps: int,
    warmup_steps: int,
    max_memory_fraction: float,
    target_steps: int,
    revision: str,
    branch: str,
    config_sha256: dict[str, str],
    benchmark_root: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "freeze_on_training_state",
        "training_run_dirs": [path.as_posix() for path in run_dirs],
        "candidates": [
            {
                "micro_batch_size": micro_batch,
                "gradient_accumulation_steps": accumulation,
            }
            for micro_batch, accumulation in candidates
        ],
        "expected_effective_batch_size": expected_effective_batch,
        "benchmark_steps": benchmark_steps,
        "warmup_steps": warmup_steps,
        "max_memory_fraction": max_memory_fraction,
        "training_target_steps": target_steps,
        "git": {
            "revision": revision,
            "branch": branch,
            "tracked_dirty": False,
        },
        "config_sha256": dict(config_sha256),
        "benchmark_root": benchmark_root.as_posix(),
    }


def _current_runtime_environment_sha(
    config_path: Path,
    *,
    project_root: Path,
) -> str:
    config = load_config(config_path)
    device = torch.device(config.runtime.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable for formal runtime selection")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = config.runtime.allow_tf32
        torch.backends.cudnn.allow_tf32 = config.runtime.allow_tf32
        torch.backends.cudnn.benchmark = config.runtime.cudnn_benchmark
    torch.set_float32_matmul_precision("high")
    environment = capture_runtime_environment(device, project_root=project_root)
    return runtime_environment_sha256(environment)


def validate_frozen_runtime_selection(
    selection: dict[str, Any],
    *,
    expected_contract: dict[str, Any],
    cofitok_config: Path,
    dense_config: Path,
    current_runtime_environment_sha256: str,
) -> tuple[int, int]:
    if selection.get("schema_version") != 2 or selection.get("status") != "selected":
        raise ValueError("frozen runtime selection is incomplete or unsupported")
    if selection.get("selection_lock") != expected_contract:
        raise ValueError("frozen runtime selection contract changed")
    if selection.get("git_revision") != expected_contract["git"]["revision"]:
        raise ValueError("frozen runtime selection revision changed")
    if selection.get("config_sha256") != expected_contract["config_sha256"]:
        raise ValueError("frozen runtime selection config identity changed")
    if selection.get("benchmark_root") != expected_contract["benchmark_root"]:
        raise ValueError("frozen runtime selection benchmark root changed")
    if selection.get("runtime_environment_sha256") != current_runtime_environment_sha256:
        raise ValueError("current runtime environment differs from frozen selection")

    candidates = selection.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("frozen runtime selection candidates are malformed")
    observed_pairs = [
        (
            int(candidate.get("micro_batch_size", -1)),
            int(candidate.get("gradient_accumulation_steps", -1)),
        )
        for candidate in candidates
    ]
    expected_pairs = [
        (
            int(candidate["micro_batch_size"]),
            int(candidate["gradient_accumulation_steps"]),
        )
        for candidate in expected_contract["candidates"]
    ]
    if observed_pairs != expected_pairs:
        raise ValueError("frozen runtime selection candidate set changed")

    revision = expected_contract["git"]["revision"]
    benchmark_steps = int(expected_contract["benchmark_steps"])
    warmup_steps = int(expected_contract["warmup_steps"])
    for candidate, (micro_batch, accumulation) in zip(
        candidates, observed_pairs, strict=True
    ):
        methods = candidate.get("methods")
        if not isinstance(methods, dict):
            raise ValueError("frozen runtime benchmark methods are malformed")
        for method, config_path in (
            ("cofitok", cofitok_config),
            ("dense_identity", dense_config),
        ):
            report = methods.get(method)
            if not isinstance(report, dict):
                raise ValueError(f"frozen {method} benchmark is missing")
            if report.get("status") != "completed":
                continue
            if not _benchmark_matches(
                report,
                expected_config=_expected_config(
                    config_path, micro_batch, accumulation
                ),
                expected_revision=revision,
                benchmark_steps=benchmark_steps,
                warmup_steps=warmup_steps,
            ):
                raise ValueError(
                    f"frozen {method} {micro_batch}x{accumulation} benchmark changed"
                )

    recomputed = select_runtime_candidate(
        deepcopy(candidates),
        expected_effective_batch=int(
            expected_contract["expected_effective_batch_size"]
        ),
        max_memory_fraction=float(expected_contract["max_memory_fraction"]),
    )
    for key in (
        "schema_version",
        "status",
        "policy",
        "selected",
        "runtime_environment_sha256",
        "dataset_identity_sha256",
        "candidates",
    ):
        if selection.get(key) != recomputed.get(key):
            raise ValueError(f"frozen runtime selection {key} is not reproducible")
    selected = recomputed["selected"]
    return (
        int(selected["micro_batch_size"]),
        int(selected["gradient_accumulation_steps"]),
    )


def reuse_runtime_selection_after_training_start(
    *,
    selection_path: Path,
    run_dirs: list[Path],
    expected_contract: dict[str, Any],
    cofitok_config: Path,
    dense_config: Path,
    current_runtime_environment_sha256: str,
) -> tuple[int, int] | None:
    state = _training_state(run_dirs)
    if not state["started"]:
        return None
    if not selection_path.is_file():
        started = [run["path"] for run in state["runs"] if run["started"]]
        raise FileNotFoundError(
            "training state exists without a frozen runtime selection: "
            + ", ".join(started)
        )
    return validate_frozen_runtime_selection(
        _read(selection_path),
        expected_contract=expected_contract,
        cofitok_config=cofitok_config,
        dense_config=dense_config,
        current_runtime_environment_sha256=current_runtime_environment_sha256,
    )


def _read(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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


def _benchmark_matches(
    report: dict[str, Any],
    *,
    expected_config: dict[str, Any],
    expected_revision: str,
    benchmark_steps: int,
    warmup_steps: int,
) -> bool:
    return (
        report.get("status") == "completed"
        and report.get("config") == expected_config
        and report.get("git", {}).get("revision") == expected_revision
        and report.get("git", {}).get("dirty") is False
        and _validated_runtime_environment_sha(report) is not None
        and _validated_dataset_identity_sha(report) is not None
        and int(report.get("benchmark_steps", -1)) == benchmark_steps
        and int(report.get("warmup_steps", -1)) == warmup_steps
        and report.get("checkpoint_written") is False
    )


def _run_benchmark(
    *,
    method: str,
    config_path: Path,
    micro_batch: int,
    accumulation: int,
    output_root: Path,
    train_script: Path,
    project_root: Path,
    expected_revision: str,
    benchmark_steps: int,
    warmup_steps: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    tag = f"mb{micro_batch}_ga{accumulation}"
    run_dir = output_root / method / tag
    report_path = run_dir / "benchmark_report.json"
    expected_config = _expected_config(config_path, micro_batch, accumulation)
    if report_path.is_file():
        existing = _read(report_path)
        if _benchmark_matches(
            existing,
            expected_config=expected_config,
            expected_revision=expected_revision,
            benchmark_steps=benchmark_steps,
            warmup_steps=warmup_steps,
        ):
            return existing
        _archive_stale_run(run_dir)
    elif run_dir.exists():
        _archive_stale_run(run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        str(train_script),
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
        "--micro-batch-size",
        str(micro_batch),
        "--gradient-accumulation-steps",
        str(accumulation),
        "--benchmark-steps",
        str(benchmark_steps),
        "--benchmark-warmup-steps",
        str(warmup_steps),
        "--benchmark-output",
        str(report_path),
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
    if result.returncode != 0:
        combined = f"{result.stdout}\n{result.stderr}".lower()
        return {
            "status": "failed",
            "failure_type": "cuda_oom" if "out of memory" in combined else "process_error",
            "returncode": result.returncode,
            "stdout_tail": result.stdout[-4_000:],
            "stderr_tail": result.stderr[-4_000:],
        }
    report = _read(report_path)
    if not _benchmark_matches(
        report,
        expected_config=expected_config,
        expected_revision=expected_revision,
        benchmark_steps=benchmark_steps,
        warmup_steps=warmup_steps,
    ):
        raise ValueError(f"{method} {tag} benchmark report provenance mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a shared, memory-safe runtime for matched full generation training."
    )
    parser.add_argument("--cofitok-config", required=True)
    parser.add_argument("--dense-config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--training-run-dir",
        action="append",
        required=True,
        help=(
            "Formal training run directory to protect; repeat for both matched methods. "
            "Any existing run state freezes the selection."
        ),
    )
    parser.add_argument("--candidates", default="16x4,32x2,64x1")
    parser.add_argument("--effective-batch-size", type=int, default=64)
    parser.add_argument("--benchmark-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--max-memory-fraction", type=float, default=0.90)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--train-script", default="scripts/train_generation.py")
    args = parser.parse_args()
    if args.benchmark_steps <= args.warmup_steps or args.timeout_seconds < 1:
        raise ValueError("runtime benchmark steps or timeout are invalid")

    project_root = Path(args.project_root).resolve()
    cofitok_config = Path(args.cofitok_config).resolve()
    dense_config = Path(args.dense_config).resolve()
    output_root = Path(args.output_root).resolve()
    output_path = Path(args.output).resolve()
    run_dirs = [Path(value).resolve() for value in args.training_run_dir]
    if len(run_dirs) != 2 or len(set(run_dirs)) != 2:
        raise ValueError("exactly two distinct matched training run directories are required")
    train_script = (project_root / args.train_script).resolve()
    git = git_provenance(project_root)
    if git["tracked_dirty"]:
        raise ValueError("runtime selection requires a clean tracked worktree")
    revision = str(git["revision"])
    candidates = parse_candidates(
        args.candidates,
        expected_effective_batch=args.effective_batch_size,
    )
    cofitok_base = load_config(cofitok_config)
    dense_base = load_config(dense_config)
    if cofitok_base.runtime.steps != dense_base.runtime.steps:
        raise ValueError("matched runtime configs have different training horizons")
    config_sha256 = {
        "cofitok": file_sha256(cofitok_config),
        "dense_identity": file_sha256(dense_config),
    }
    selection_contract = _selection_contract(
        run_dirs=run_dirs,
        candidates=candidates,
        expected_effective_batch=args.effective_batch_size,
        benchmark_steps=args.benchmark_steps,
        warmup_steps=args.warmup_steps,
        max_memory_fraction=args.max_memory_fraction,
        target_steps=cofitok_base.runtime.steps,
        revision=revision,
        branch=str(git["branch"]),
        config_sha256=config_sha256,
        benchmark_root=output_root,
    )
    current_environment_sha = _current_runtime_environment_sha(
        cofitok_config,
        project_root=project_root,
    )
    reused = reuse_runtime_selection_after_training_start(
        selection_path=output_path,
        run_dirs=run_dirs,
        expected_contract=selection_contract,
        cofitok_config=cofitok_config,
        dense_config=dense_config,
        current_runtime_environment_sha256=current_environment_sha,
    )
    if reused is not None:
        print(f"{reused[0]} {reused[1]}")
        return

    rows = []
    for micro_batch, accumulation in candidates:
        rows.append(
            {
                "micro_batch_size": micro_batch,
                "gradient_accumulation_steps": accumulation,
                "effective_batch_size": micro_batch * accumulation,
                "methods": {
                    "cofitok": _run_benchmark(
                        method="cofitok",
                        config_path=cofitok_config,
                        micro_batch=micro_batch,
                        accumulation=accumulation,
                        output_root=output_root,
                        train_script=train_script,
                        project_root=project_root,
                        expected_revision=revision,
                        benchmark_steps=args.benchmark_steps,
                        warmup_steps=args.warmup_steps,
                        timeout_seconds=args.timeout_seconds,
                    ),
                    "dense_identity": _run_benchmark(
                        method="dense_identity",
                        config_path=dense_config,
                        micro_batch=micro_batch,
                        accumulation=accumulation,
                        output_root=output_root,
                        train_script=train_script,
                        project_root=project_root,
                        expected_revision=revision,
                        benchmark_steps=args.benchmark_steps,
                        warmup_steps=args.warmup_steps,
                        timeout_seconds=args.timeout_seconds,
                    ),
                },
            }
        )
    selection = select_runtime_candidate(
        rows,
        expected_effective_batch=args.effective_batch_size,
        max_memory_fraction=args.max_memory_fraction,
    )
    selection["git_revision"] = revision
    selection["config_sha256"] = config_sha256
    selection["benchmark_root"] = output_root.as_posix()
    selection["selection_lock"] = selection_contract
    if selection["runtime_environment_sha256"] != current_environment_sha:
        raise ValueError("runtime benchmark environment differs from selector environment")
    write_json_report(output_path, selection)
    selected = selection["selected"]
    print(
        f"{selected['micro_batch_size']} "
        f"{selected['gradient_accumulation_steps']}"
    )


if __name__ == "__main__":
    main()
