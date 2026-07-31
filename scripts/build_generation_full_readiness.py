from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.configs import load_config
from cofitok.generation_authorization import capture_generation_gate_binding
from cofitok.generation_gate_sources import verify_generation_gate_source_reports
from cofitok.reporting import (
    file_sha256,
    git_provenance,
    to_jsonable,
    write_json_report,
)

try:
    from scripts.build_generation_large_capacity_deployment_receipt import (
        verify_deployment_receipt,
    )
    from scripts.select_generation_training_runtime import (
        _current_runtime_environment_sha,
        _selection_contract,
        validate_frozen_runtime_selection,
    )
    from scripts.validate_generation_configs import validate_pair
except ModuleNotFoundError:
    from build_generation_large_capacity_deployment_receipt import (
        verify_deployment_receipt,
    )
    from select_generation_training_runtime import (
        _current_runtime_environment_sha,
        _selection_contract,
        validate_frozen_runtime_selection,
    )
    from validate_generation_configs import validate_pair


READINESS_SCHEMA_VERSION = 1
READINESS_ROLE = "stability_full_training_readiness"
RUNTIME_CANDIDATES = ((1, 64), (2, 32), (4, 16), (8, 8), (16, 4))
RUNTIME_BASELINE = (1, 64)
EXPECTED_EFFECTIVE_BATCH = 64
EXPECTED_COFITOK_PARAMETERS = 250_153_763
EXPECTED_DENSE_PARAMETERS = 250_135_043


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"readiness source is missing: {resolved}")
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def source_identities(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    return {name: source_identity(path) for name, path in paths.items()}


def require_absent_training_state(training_run_dirs: list[Path]) -> None:
    for run_dir in training_run_dirs:
        if run_dir.exists() and not run_dir.is_dir():
            raise ValueError(f"training run path is not a directory: {run_dir}")
        entries = sorted(run_dir.iterdir()) if run_dir.is_dir() else []
        if entries:
            preview = ", ".join(item.name for item in entries[:20])
            raise ValueError(
                "stability full readiness requires absent training state: "
                f"{run_dir} contains {preview}"
            )


def validate_full_storage_capacity(
    report: dict[str, Any],
    *,
    expected_revision: str,
    expected_branch: str,
    expected_path: Path,
) -> dict[str, Any]:
    if (
        report.get("schema_version") != 2
        or report.get("role") != "generation_storage_capacity_preflight"
        or report.get("stage") != "full_training"
        or report.get("status") != "pass"
    ):
        raise ValueError("stability full storage capacity report is invalid")
    if report.get("git") != {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }:
        raise ValueError("stability full storage capacity Git identity differs")
    filesystem = report.get("filesystem", {})
    if Path(str(filesystem.get("path", ""))).resolve() != expected_path.resolve():
        raise ValueError("stability full storage capacity path differs")
    plan = report.get("plan", {})
    minimums = {
        "checkpoint_count": 16,
        "sample_count": 16_384,
        "estimated_sample_bytes_each": 256 * 1024,
        "additional_bytes": 16 * 1024**3,
        "safety_margin_bytes": 64 * 1024**3,
    }
    for key, minimum in minimums.items():
        if int(plan.get(key, -1)) < minimum:
            raise ValueError(f"stability full storage reserve {key} was weakened")
    reference_bytes = int(plan.get("reference_checkpoint_bytes_each", -1))
    multiplier = float(plan.get("checkpoint_size_multiplier", math.nan))
    checkpoint_bytes = int(plan.get("checkpoint_bytes_each", -1))
    if reference_bytes < 1 or not math.isfinite(multiplier) or multiplier < 4.0:
        raise ValueError("stability full checkpoint scaling was weakened")
    if checkpoint_bytes != math.ceil(reference_bytes * multiplier):
        raise ValueError("stability full checkpoint scaling arithmetic differs")
    checkpoint_reserve = int(plan["checkpoint_count"]) * checkpoint_bytes
    sample_reserve = int(plan["sample_count"]) * int(
        plan["estimated_sample_bytes_each"]
    )
    if checkpoint_reserve != int(plan.get("checkpoint_reserve_bytes", -1)):
        raise ValueError("stability full checkpoint reserve arithmetic differs")
    if sample_reserve != int(plan.get("sample_reserve_bytes", -1)):
        raise ValueError("stability full sample reserve arithmetic differs")
    required = (
        checkpoint_reserve
        + sample_reserve
        + int(plan["additional_bytes"])
        + int(plan["safety_margin_bytes"])
    )
    if required != int(plan.get("required_free_bytes", -1)):
        raise ValueError("stability full storage requirement arithmetic differs")
    total = int(filesystem.get("total_bytes", -1))
    used = int(filesystem.get("used_bytes", -1))
    free = int(filesystem.get("free_bytes", -1))
    if total < 1 or used < 0 or free < required or used + free > total:
        raise ValueError("stability full storage filesystem headroom is invalid")
    if int(report.get("headroom_bytes", -1)) != free - required:
        raise ValueError("stability full storage headroom arithmetic differs")
    return {
        "reference_checkpoint_bytes_each": reference_bytes,
        "checkpoint_size_multiplier": multiplier,
        "checkpoint_bytes_each": checkpoint_bytes,
        "checkpoint_count": int(plan["checkpoint_count"]),
        "required_free_bytes": required,
        "free_bytes": free,
        "headroom_bytes": free - required,
    }


def validate_full_config_contract(
    report: dict[str, Any],
    *,
    cofitok_config_path: Path,
    dense_config_path: Path,
) -> dict[str, Any]:
    recomputed = to_jsonable(
        validate_pair(
            load_config(cofitok_config_path),
            load_config(dense_config_path),
            max_parameter_gap=0.02,
            stage="stability_full",
        )
    )
    if report != recomputed:
        raise ValueError("stability full config validation is not reproducible")
    if (
        report.get("status") != "pass"
        or report.get("mismatches") != []
        or report.get("cofitok", {}).get("parameter_count")
        != EXPECTED_COFITOK_PARAMETERS
        or report.get("dense", {}).get("parameter_count")
        != EXPECTED_DENSE_PARAMETERS
        or report.get("matched_backbone", {}).get("base_channels") != 256
        or report.get("training_recipe", {}).get("stage") != "stability_full"
        or report.get("training_recipe", {}).get("valid") is not True
    ):
        raise ValueError("stability full config contract is invalid")
    return {
        "cofitok_parameter_count": EXPECTED_COFITOK_PARAMETERS,
        "dense_parameter_count": EXPECTED_DENSE_PARAMETERS,
        "relative_parameter_gap": float(report["relative_parameter_gap"]),
        "base_channels": 256,
        "recipe_schema": report["training_recipe"]["schema"],
        "recipe_stage": "stability_full",
    }


def validate_full_runtime_selection(
    selection: dict[str, Any],
    *,
    cofitok_config_path: Path,
    dense_config_path: Path,
    training_run_dirs: list[Path],
    benchmark_root: Path,
    expected_revision: str,
    expected_branch: str,
    project_root: Path,
    require_current_runtime_environment: bool,
) -> dict[str, Any]:
    config_sha256 = {
        "cofitok": file_sha256(cofitok_config_path),
        "dense_identity": file_sha256(dense_config_path),
    }
    contract = _selection_contract(
        run_dirs=training_run_dirs,
        candidates=list(RUNTIME_CANDIDATES),
        baseline_candidate=RUNTIME_BASELINE,
        expected_effective_batch=EXPECTED_EFFECTIVE_BATCH,
        benchmark_steps=8,
        warmup_steps=2,
        max_memory_fraction=0.9,
        target_steps=300_000,
        revision=expected_revision,
        branch=expected_branch,
        config_sha256=config_sha256,
        benchmark_root=benchmark_root,
    )
    current_environment_sha = str(selection.get("runtime_environment_sha256", ""))
    if require_current_runtime_environment:
        current_environment_sha = _current_runtime_environment_sha(
            cofitok_config_path,
            project_root=project_root,
        )
    micro_batch, accumulation = validate_frozen_runtime_selection(
        selection,
        expected_contract=contract,
        cofitok_config=cofitok_config_path,
        dense_config=dense_config_path,
        current_runtime_environment_sha256=current_environment_sha,
    )
    if selection.get("schema_version") != 3 or selection.get("baseline") is None:
        raise ValueError("stability full runtime selection lacks the baseline contract")
    return {
        "micro_batch_size": micro_batch,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": micro_batch * accumulation,
        "baseline": {
            "micro_batch_size": RUNTIME_BASELINE[0],
            "gradient_accumulation_steps": RUNTIME_BASELINE[1],
        },
        "candidate_count": len(RUNTIME_CANDIDATES),
        "runtime_environment_sha256": current_environment_sha,
        "dataset_identity_sha256": selection["dataset_identity_sha256"],
        "estimated_speedup_over_baseline": float(
            selection["selected"]["estimated_speedup_over_baseline"]
        ),
    }


def build_readiness_report(
    *,
    source_paths: dict[str, Path],
    training_run_dirs: list[Path],
    benchmark_root: Path,
    storage_path: Path,
    project_root: Path,
    expected_revision: str,
    expected_branch: str,
    require_current_runtime_environment: bool,
    require_current_formal_repository: bool = True,
    require_current_git: bool = True,
    require_training_state_absent: bool = False,
) -> dict[str, Any]:
    expected_git = {
        "revision": expected_revision,
        "branch": expected_branch,
        "tracked_dirty": False,
    }
    if require_current_git and git_provenance(project_root) != expected_git:
        raise ValueError("stability full readiness Git identity differs")
    if require_training_state_absent:
        require_absent_training_state(training_run_dirs)
    sources = source_identities(source_paths)
    deployment = verify_deployment_receipt(
        _read_json(source_paths["deployment_receipt"]),
        receipt_path=source_paths["deployment_receipt"],
        expected_receipt_sha256=sources["deployment_receipt"]["sha256"],
        require_current_formal_repository=require_current_formal_repository,
    )
    deployment_checkout = deployment["checkout"]["git"]
    if deployment_checkout != expected_git:
        raise ValueError("stability full deployment checkout identity differs")
    if require_current_git and (
        Path(deployment["checkout"]["path"]).resolve() != project_root.resolve()
    ):
        raise ValueError("stability full readiness is running outside the deployed checkout")
    if (
        deployment.get("readiness_execution_allowed") is not True
        or deployment.get("readiness_executed") is not False
        or deployment.get("full_training_launch_allowed") is not False
    ):
        raise ValueError("stability full deployment authorization boundary differs")
    gate = _read_json(source_paths["promotion_gate"])
    gate_sources = verify_generation_gate_source_reports(gate)
    if gate_sources.get("source_profile") != "stability_scaling":
        raise ValueError("stability full readiness uses the wrong gate source profile")
    authorization = capture_generation_gate_binding(
        source_paths["promotion_gate"],
        expected_stage="scaling",
    )
    config_contract = validate_full_config_contract(
        _read_json(source_paths["config_validation"]),
        cofitok_config_path=source_paths["cofitok_config"],
        dense_config_path=source_paths["dense_config"],
    )
    storage = validate_full_storage_capacity(
        _read_json(source_paths["storage_capacity"]),
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_path=storage_path,
    )
    runtime = validate_full_runtime_selection(
        _read_json(source_paths["runtime_selection"]),
        cofitok_config_path=source_paths["cofitok_config"],
        dense_config_path=source_paths["dense_config"],
        training_run_dirs=training_run_dirs,
        benchmark_root=benchmark_root,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        project_root=project_root,
        require_current_runtime_environment=require_current_runtime_environment,
    )
    return {
        "schema_version": READINESS_SCHEMA_VERSION,
        "status": "pass",
        "role": READINESS_ROLE,
        "stage": "stability_full",
        "git": expected_git,
        "source_reports": sources,
        "deployment": {
            "role": deployment["role"],
            "receipt_sha256": sources["deployment_receipt"]["sha256"],
            "formal_repository_git": deployment["formal_repository"]["git"],
            "checkout_git": deployment_checkout,
            "pytest_tests": deployment["validation"]["pytest"]["tests"],
            "runbook_count": deployment["validation"]["runbook_syntax"][
                "checked_count"
            ],
        },
        "promotion_authorization": authorization,
        "gate_source_profile": gate_sources["source_profile"],
        "config_contract": config_contract,
        "storage_capacity": storage,
        "runtime_selection": runtime,
        "training_run_dirs": [path.resolve().as_posix() for path in training_run_dirs],
        "training_state_absent_at_build": True,
        "benchmark_root": benchmark_root.resolve().as_posix(),
        "full_training_launch_allowed": True,
        "formal_generation_completion_claimed": False,
    }


def verify_readiness_report(
    report: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    expected = build_readiness_report(**kwargs)
    if report != expected:
        raise ValueError("stability full readiness report is not reproducible")
    return expected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the source-bound 250M stability-full training readiness artifact."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--promotion-gate", type=Path, required=True)
    parser.add_argument("--deployment-receipt", type=Path, required=True)
    parser.add_argument("--expected-deployment-receipt-sha256", required=True)
    parser.add_argument("--cofitok-config", type=Path, required=True)
    parser.add_argument("--dense-config", type=Path, required=True)
    parser.add_argument("--config-validation", type=Path, required=True)
    parser.add_argument("--storage-capacity", type=Path, required=True)
    parser.add_argument("--runtime-selection", type=Path, required=True)
    parser.add_argument("--training-run-dir", action="append", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--storage-path", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", required=True)
    parser.add_argument("--expected-gate-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        raise FileExistsError(f"readiness output already exists: {args.output}")
    if len(args.training_run_dir) != 2 or len(set(args.training_run_dir)) != 2:
        raise ValueError("exactly two distinct matched training run directories are required")
    if file_sha256(args.promotion_gate) != args.expected_gate_sha256:
        raise ValueError("stability scaling gate SHA256 differs")
    if (
        file_sha256(args.deployment_receipt)
        != args.expected_deployment_receipt_sha256
    ):
        raise ValueError("large-capacity deployment receipt SHA256 differs")
    source_paths = {
        "deployment_receipt": args.deployment_receipt,
        "promotion_gate": args.promotion_gate,
        "cofitok_config": args.cofitok_config,
        "dense_config": args.dense_config,
        "config_validation": args.config_validation,
        "storage_capacity": args.storage_capacity,
        "runtime_selection": args.runtime_selection,
    }
    report = build_readiness_report(
        source_paths=source_paths,
        training_run_dirs=args.training_run_dir,
        benchmark_root=args.benchmark_root,
        storage_path=args.storage_path,
        project_root=Path(args.project_root),
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        require_current_runtime_environment=True,
        require_training_state_absent=True,
    )
    write_json_report(args.output, report)
    print(args.output)


if __name__ == "__main__":
    main()
