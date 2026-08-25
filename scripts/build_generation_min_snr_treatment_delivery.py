from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEMANTIC_SOURCE_PATHS = (
    "scripts/train_generation.py",
    "src/cofitok/diffusion/schedule.py",
    "src/cofitok/training/losses.py",
)
METHODS = ("cofitok", "dense_identity")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze a source-bound, single-arm Min-SNR treatment-delivery "
            "diagnostic through an exact training step."
        )
    )
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cutoff-step", type=int, required=True)
    parser.add_argument("--expected-gamma", type=float, required=True)
    parser.add_argument("--expected-effective-batch", type=int, required=True)
    parser.add_argument("--expected-scheduler-horizon", type=int, required=True)
    parser.add_argument("--expected-seed", type=int, required=True)
    parser.add_argument("--expected-dataset", required=True)
    parser.add_argument("--expected-training-revision", required=True)
    parser.add_argument("--expected-training-tree", required=True)
    parser.add_argument("--expected-training-branch", required=True)
    parser.add_argument("--expected-builder-revision", required=True)
    parser.add_argument("--expected-builder-tree", required=True)
    parser.add_argument("--expected-builder-branch", required=True)
    parser.add_argument("--metrics-origin")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=PROJECT_ROOT,
        check=check,
        capture_output=True,
        text=True,
    )


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _source(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def _is_hex_digest(value: Any, *, length: int) -> bool:
    if not isinstance(value, str) or len(value) != length:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _read_metrics_prefix(path: Path, *, cutoff_step: int) -> dict[str, Any]:
    if cutoff_step < 1:
        raise ValueError("cutoff_step must be positive")
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    prefix_parts: list[bytes] = []
    seen_after_cutoff = False
    for line_number, raw_line in enumerate(raw.splitlines(keepends=True), 1):
        if not raw_line.strip():
            raise ValueError(f"{path}:{line_number} is blank")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        try:
            step = int(row["step"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}:{line_number} has an invalid step") from error
        if step <= cutoff_step:
            if seen_after_cutoff:
                raise ValueError(f"{path} contains a cutoff row after a later step")
            rows.append(row)
            prefix_parts.append(raw_line)
        else:
            seen_after_cutoff = True
    if not rows or int(rows[-1]["step"]) != cutoff_step:
        raise ValueError(f"{path} does not contain the exact cutoff step {cutoff_step}")
    prefix = b"".join(prefix_parts)
    return {
        "rows": rows,
        "raw_prefix": prefix,
        "identity": {
            "row_count": len(rows),
            "last_step": cutoff_step,
            "bytes": len(prefix),
            "sha256": hashlib.sha256(prefix).hexdigest(),
        },
        "observed_file": _source(path),
    }


def _atomic_write_new(path: Path, content: bytes) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing evidence: {path}")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _builder_identity(
    *,
    expected_revision: str,
    expected_tree: str,
    expected_branch: str,
    training_revision: str,
    training_tree: str,
) -> dict[str, Any]:
    revision = _git("rev-parse", "HEAD").stdout.strip()
    tree = _git("rev-parse", "HEAD^{tree}").stdout.strip()
    branch = _git("branch", "--show-current").stdout.strip()
    tracked_dirty = bool(
        _git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    )
    if (
        revision != expected_revision
        or tree != expected_tree
        or branch != expected_branch
        or tracked_dirty
    ):
        raise ValueError("builder checkout identity differs")
    if _git("merge-base", "--is-ancestor", training_revision, revision, check=False).returncode:
        raise ValueError("training revision is not an ancestor of the builder revision")
    observed_training_tree = _git(
        "rev-parse", f"{training_revision}^{{tree}}"
    ).stdout.strip()
    if observed_training_tree != training_tree:
        raise ValueError("training revision tree differs")
    semantic_diff = _git(
        "diff",
        "--quiet",
        training_revision,
        revision,
        "--",
        *SEMANTIC_SOURCE_PATHS,
        check=False,
    )
    if semantic_diff.returncode != 0:
        raise ValueError("Min-SNR training semantics changed after the training revision")
    return {
        "revision": revision,
        "tree": tree,
        "branch": branch,
        "tracked_dirty": False,
        "training_revision_is_ancestor": True,
        "training_tree_verified": True,
        "training_semantic_sources_unchanged": True,
    }


def _validate_manifest(
    manifest: dict[str, Any],
    *,
    method: str,
    expected_gamma: float,
    expected_effective_batch: int,
    expected_scheduler_horizon: int,
    expected_seed: int,
    expected_dataset: str,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
) -> dict[str, Any]:
    git = manifest.get("git")
    if not isinstance(git, dict):
        raise ValueError("manifest lacks Git identity")
    if (
        git.get("revision") != expected_training_revision
        or git.get("branch") != expected_training_branch
        or git.get("dirty") is not False
    ):
        raise ValueError("manifest training Git identity differs")
    provenance = manifest.get("dataset_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("status") != "pass"
        or provenance.get("formal") is not True
        or provenance.get("dataset") != expected_dataset
        or not _is_hex_digest(provenance.get("identity_sha256"), length=64)
    ):
        raise ValueError("manifest dataset provenance differs")
    runtime_identity = manifest.get("runtime_environment_sha256")
    if not _is_hex_digest(runtime_identity, length=64):
        raise ValueError("manifest runtime identity is invalid")
    if int(manifest.get("parameter_count", 0)) < 1:
        raise ValueError("manifest parameter count is invalid")
    config = manifest.get("config")
    if not isinstance(config, dict):
        raise ValueError("manifest lacks resolved config")
    data = config.get("data")
    diffusion = config.get("diffusion")
    loss = config.get("loss")
    model = config.get("model")
    optimization = config.get("optimization")
    runtime = config.get("runtime")
    if not all(
        isinstance(value, dict)
        for value in (data, diffusion, loss, model, optimization, runtime)
    ):
        raise ValueError("manifest resolved config is incomplete")
    gamma = _finite_number(loss.get("min_snr_gamma"), label="min_snr_gamma")
    if gamma != expected_gamma or gamma <= 0.0:
        raise ValueError("manifest Min-SNR gamma differs")
    effective_batch = int(data.get("batch_size", 0)) * int(
        optimization.get("gradient_accumulation_steps", 0)
    )
    if effective_batch != expected_effective_batch:
        raise ValueError("manifest effective batch differs")
    if (
        data.get("dataset") != expected_dataset
        or diffusion.get("prediction_target") != "epsilon"
        or int(runtime.get("steps", -1)) != expected_scheduler_horizon
        or int(runtime.get("seed", -1)) != expected_seed
    ):
        raise ValueError("manifest pilot contract differs")
    if method == "cofitok":
        if (
            model.get("synthesis_mode") == "dense_identity"
            or int(model.get("token_count", 0)) <= 1
        ):
            raise ValueError("manifest does not describe a CoFiTok arm")
    elif (
        model.get("synthesis_mode") != "dense_identity"
        or int(model.get("token_count", 0)) != 1
    ):
        raise ValueError("manifest does not describe a dense identity arm")
    log_interval = int(optimization.get("log_interval", 0))
    if log_interval < 1:
        raise ValueError("manifest log interval is invalid")
    return {
        "git": {
            "revision": expected_training_revision,
            "tree": expected_training_tree,
            "branch": expected_training_branch,
            "dirty": False,
        },
        "dataset": expected_dataset,
        "dataset_identity_sha256": provenance["identity_sha256"],
        "runtime_environment_sha256": runtime_identity,
        "parameter_count": int(manifest["parameter_count"]),
        "prediction_target": "epsilon",
        "min_snr_gamma": gamma,
        "effective_batch_size": effective_batch,
        "scheduler_horizon_steps": expected_scheduler_horizon,
        "seed": expected_seed,
        "log_interval": log_interval,
        "model": {
            "token_count": int(model["token_count"]),
            "synthesis_mode": model["synthesis_mode"],
        },
    }


def _validate_rows(
    rows: list[dict[str, Any]],
    *,
    cutoff_step: int,
    log_interval: int,
    effective_batch: int,
) -> dict[str, Any]:
    if cutoff_step % log_interval:
        raise ValueError("cutoff_step must be divisible by the log interval")
    expected_steps = [1, *range(log_interval, cutoff_step + 1, log_interval)]
    expected_steps = list(dict.fromkeys(expected_steps))
    observed_steps = [int(row.get("step", -1)) for row in rows]
    if observed_steps != expected_steps:
        raise ValueError("metrics do not follow the exact logging schedule")
    weights = []
    epsilon_ratios = []
    downweighted_weight_rows = 0
    downweighted_epsilon_rows = 0
    elapsed_values = []
    required_fields = (
        "total",
        "epsilon",
        "epsilon_unweighted",
        "min_snr_weight_mean",
        "learning_rate",
        "grad_norm",
        "cumulative_elapsed_seconds",
    )
    for row in rows:
        step = int(row["step"])
        if int(row.get("samples_seen", -1)) != step * effective_batch:
            raise ValueError(f"metrics samples_seen differs at step {step}")
        for field, value in row.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                _finite_number(value, label=f"step {step} {field}")
        values = {
            field: _finite_number(row.get(field), label=f"step {step} {field}")
            for field in required_fields
        }
        if any(
            values[field] < 0.0
            for field in (
                "total",
                "epsilon",
                "epsilon_unweighted",
                "learning_rate",
                "grad_norm",
            )
        ):
            raise ValueError(f"metrics contain a negative loss/runtime value at step {step}")
        if values["cumulative_elapsed_seconds"] <= 0.0:
            raise ValueError(f"cumulative elapsed time is not positive at step {step}")
        weight = values["min_snr_weight_mean"]
        if not 0.0 < weight <= 1.0:
            raise ValueError(f"Min-SNR weight is outside (0, 1] at step {step}")
        if values["epsilon"] > values["epsilon_unweighted"] + 1e-12:
            raise ValueError(f"weighted epsilon exceeds unweighted epsilon at step {step}")
        weights.append(weight)
        elapsed_values.append(values["cumulative_elapsed_seconds"])
        if weight < 1.0 - 1e-8:
            downweighted_weight_rows += 1
        if values["epsilon"] < values["epsilon_unweighted"] - 1e-12:
            downweighted_epsilon_rows += 1
        if values["epsilon_unweighted"] > 0.0:
            epsilon_ratios.append(values["epsilon"] / values["epsilon_unweighted"])
    if any(current <= previous for previous, current in zip(elapsed_values, elapsed_values[1:])):
        raise ValueError("cumulative elapsed time is not strictly increasing")
    if downweighted_weight_rows < 1 or downweighted_epsilon_rows < 1:
        raise ValueError("metrics do not demonstrate active Min-SNR downweighting")
    endpoint = rows[-1]
    return {
        "row_count": len(rows),
        "first_step": observed_steps[0],
        "last_step": observed_steps[-1],
        "samples_seen": int(endpoint["samples_seen"]),
        "steps_exact": True,
        "samples_seen_exact": True,
        "all_numeric_metrics_finite": True,
        "cumulative_elapsed_seconds_strictly_increasing": True,
        "required_treatment_fields_present": True,
        "downweighted_weight_row_count": downweighted_weight_rows,
        "downweighted_epsilon_row_count": downweighted_epsilon_rows,
        "all_logged_rows_have_weight_below_one": downweighted_weight_rows == len(rows),
        "all_logged_rows_have_weighted_epsilon_below_unweighted": (
            downweighted_epsilon_rows == len(rows)
        ),
        "min_snr_weight_mean": {
            "minimum": min(weights),
            "maximum": max(weights),
            "mean": statistics.mean(weights),
            "median": statistics.median(weights),
        },
        "weighted_to_unweighted_epsilon_ratio": {
            "minimum": min(epsilon_ratios),
            "maximum": max(epsilon_ratios),
            "mean": statistics.mean(epsilon_ratios),
            "median": statistics.median(epsilon_ratios),
        },
        "endpoint": {
            "step": int(endpoint["step"]),
            "samples_seen": int(endpoint["samples_seen"]),
            "epsilon": float(endpoint["epsilon"]),
            "epsilon_unweighted": float(endpoint["epsilon_unweighted"]),
            "min_snr_weight_mean": float(endpoint["min_snr_weight_mean"]),
            "total": float(endpoint["total"]),
            "grad_norm": float(endpoint["grad_norm"]),
            "learning_rate": float(endpoint["learning_rate"]),
            "cumulative_elapsed_seconds": float(
                endpoint["cumulative_elapsed_seconds"]
            ),
        },
    }


def build_report(
    *,
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    method: str,
    cutoff_step: int,
    expected_gamma: float,
    expected_effective_batch: int,
    expected_scheduler_horizon: int,
    expected_seed: int,
    expected_dataset: str,
    expected_training_revision: str,
    expected_training_tree: str,
    expected_training_branch: str,
    builder_identity: dict[str, Any],
) -> dict[str, Any]:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    if not _is_hex_digest(expected_training_revision, length=40):
        raise ValueError("expected training revision must be a full Git revision")
    contract = _validate_manifest(
        manifest,
        method=method,
        expected_gamma=expected_gamma,
        expected_effective_batch=expected_effective_batch,
        expected_scheduler_horizon=expected_scheduler_horizon,
        expected_seed=expected_seed,
        expected_dataset=expected_dataset,
        expected_training_revision=expected_training_revision,
        expected_training_tree=expected_training_tree,
        expected_training_branch=expected_training_branch,
    )
    trajectory = _validate_rows(
        rows,
        cutoff_step=cutoff_step,
        log_interval=contract["log_interval"],
        effective_batch=expected_effective_batch,
    )
    return {
        "schema_version": 1,
        "status": "pass",
        "role": "single_arm_min_snr_treatment_delivery_diagnostic",
        "method": method,
        "cutoff_step": cutoff_step,
        "contract": contract,
        "trajectory": trajectory,
        "builder_git": builder_identity,
        "scientific_interpretation": {
            "treatment_delivery_verified": True,
            "logged_numerical_stability_through_cutoff_verified": True,
            "matched_method_comparison_present": False,
            "dense_same_cutoff_evidence_required_for_matched_comparison": True,
            "sample_quality_metrics_present": False,
            "generation_advantage_proven": False,
        },
        "claim_boundary": {
            "supports": [
                "single-arm Min-SNR gamma configuration through the cutoff",
                "source-bound observation of active epsilon-loss downweighting at logged steps",
                "finite logged training metrics and exact exposure accounting through the cutoff",
            ],
            "does_not_support": [
                "matched CoFiTok-versus-dense comparison",
                "training convergence",
                "sample quality or class fidelity",
                "generation advantage",
                "continuation beyond the bounded pilot",
            ],
            "quality_claim_allowed": False,
            "matched_advantage_claim_allowed": False,
            "formal_50k_result_substitute": False,
            "promotion_authorization_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "release_allowed": False,
        },
        "authorization_boundary": {
            "training_launch_allowed": False,
            "sampling_launch_allowed": False,
            "evaluation_launch_allowed": False,
            "continuation_beyond_50000_allowed": False,
            "full_training_launch_allowed": False,
            "full_300k_launch_allowed": False,
            "promotion_allowed": False,
            "inference_export_allowed": False,
            "release_allowed": False,
            "process_signals_allowed": False,
            "gpu_execution_allowed": False,
        },
    }


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing evidence: {args.output}")
    builder_identity = _builder_identity(
        expected_revision=args.expected_builder_revision,
        expected_tree=args.expected_builder_tree,
        expected_branch=args.expected_builder_branch,
        training_revision=args.expected_training_revision,
        training_tree=args.expected_training_tree,
    )
    metrics = _read_metrics_prefix(args.metrics, cutoff_step=args.cutoff_step)
    _atomic_write_new(args.snapshot, metrics["raw_prefix"])
    snapshot = _read_metrics_prefix(args.snapshot, cutoff_step=args.cutoff_step)
    if snapshot["identity"] != metrics["identity"]:
        raise ValueError("metrics snapshot identity differs from the observed prefix")
    manifest = _read_object(args.manifest)
    report = build_report(
        rows=snapshot["rows"],
        manifest=manifest,
        method=args.method,
        cutoff_step=args.cutoff_step,
        expected_gamma=args.expected_gamma,
        expected_effective_batch=args.expected_effective_batch,
        expected_scheduler_horizon=args.expected_scheduler_horizon,
        expected_seed=args.expected_seed,
        expected_dataset=args.expected_dataset,
        expected_training_revision=args.expected_training_revision,
        expected_training_tree=args.expected_training_tree,
        expected_training_branch=args.expected_training_branch,
        builder_identity=builder_identity,
    )
    builder_path = Path(__file__).resolve()
    report["builder"] = {
        "path": "scripts/build_generation_min_snr_treatment_delivery.py",
        "bytes": builder_path.stat().st_size,
        "sha256": file_sha256(builder_path),
        "semantic_sources": {
            path: _source(PROJECT_ROOT / path) for path in SEMANTIC_SOURCE_PATHS
        },
    }
    report["sources"] = {
        "metrics": {
            "origin": args.metrics_origin,
            "observed_file": metrics["observed_file"],
            "bound_prefix": metrics["identity"],
            "snapshot": _source(args.snapshot),
        },
        "manifest": _source(args.manifest),
    }
    write_json_report(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "status": report["status"],
                "method": report["method"],
                "cutoff_step": report["cutoff_step"],
                "sha256": file_sha256(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
