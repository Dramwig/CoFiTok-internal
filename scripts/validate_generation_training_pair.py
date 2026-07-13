from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.generation_pair import MATCHED_CONFIG_SECTIONS, generation_pair_contract
from cofitok.generation_recipe import generation_training_recipe_contract


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _validate_report(
    report: dict[str, Any],
    *,
    label: str,
    expected_steps: int,
    expected_revision: str,
    expected_branch: str,
    expected_dataset: str,
) -> dict[str, Any]:
    if report.get("training_complete") is not True:
        raise ValueError(f"{label} training is incomplete")
    completed_steps = int(report.get("completed_steps", -1))
    target_steps = int(report.get("target_steps", -1))
    if completed_steps != expected_steps or target_steps != expected_steps:
        raise ValueError(f"{label} training did not finish exactly {expected_steps} steps")
    git = report.get("git", {})
    if git.get("dirty") is not False:
        raise ValueError(f"{label} training used a dirty tracked worktree")
    if git.get("revision") != expected_revision:
        raise ValueError(f"{label} training revision does not match the pinned queue")
    if git.get("branch") != expected_branch:
        raise ValueError(f"{label} training branch does not match {expected_branch}")
    config = report.get("config", {})
    if config.get("data", {}).get("dataset") != expected_dataset:
        raise ValueError(f"{label} training dataset does not match {expected_dataset}")
    parameter_count = int(report.get("parameter_count", 0))
    if parameter_count < 1:
        raise ValueError(f"{label} parameter count is missing")
    latest = report.get("latest_checkpoint", {})
    expected_checkpoint = f"checkpoint_step_{expected_steps:08d}.pt"
    if latest.get("checkpoint") != expected_checkpoint:
        raise ValueError(f"{label} latest checkpoint is not {expected_checkpoint}")
    if int(latest.get("step", -1)) != expected_steps:
        raise ValueError(f"{label} latest checkpoint step does not match training completion")
    return {
        "completed_steps": completed_steps,
        "revision": git["revision"],
        "branch": git["branch"],
        "dataset": expected_dataset,
        "parameter_count": parameter_count,
        "latest_checkpoint": expected_checkpoint,
    }


def validate_training_pair(
    cofitok: dict[str, Any],
    dense: dict[str, Any],
    *,
    expected_steps: int,
    expected_revision: str,
    expected_branch: str = "scale/generative-system",
    expected_dataset: str = "imagenet_256_10pct",
    max_parameter_gap: float = 0.02,
    expected_recipe_stage: str | None = None,
) -> dict[str, Any]:
    if expected_steps < 1:
        raise ValueError("expected_steps must be positive")
    if len(expected_revision) != 40:
        raise ValueError("expected_revision must be a full 40-character revision")
    if not math.isfinite(max_parameter_gap) or max_parameter_gap < 0.0:
        raise ValueError("max_parameter_gap must be finite and non-negative")
    validated_cofitok = _validate_report(
        cofitok,
        label="cofitok",
        expected_steps=expected_steps,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_dataset=expected_dataset,
    )
    validated_dense = _validate_report(
        dense,
        label="dense",
        expected_steps=expected_steps,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_dataset=expected_dataset,
    )
    pair_contract = generation_pair_contract(cofitok["config"], dense["config"])
    if not pair_contract["valid"]:
        raise ValueError("training pair contract failed: " + "; ".join(pair_contract["issues"]))
    recipe_contract = None
    if expected_recipe_stage is not None:
        recipe_contract = generation_training_recipe_contract(
            cofitok["config"],
            dense["config"],
            stage=expected_recipe_stage,
        )
        if recipe_contract["valid"] is not True:
            raise ValueError(
                "training recipe contract failed: "
                + "; ".join(recipe_contract["issues"])
            )
    dense_parameters = validated_dense["parameter_count"]
    parameter_gap = (
        validated_cofitok["parameter_count"] - dense_parameters
    ) / dense_parameters
    if abs(parameter_gap) > max_parameter_gap:
        raise ValueError(
            f"training pair parameter gap {parameter_gap:.6f} exceeds {max_parameter_gap:.6f}"
        )
    return {
        "schema_version": 1,
        "status": "pass",
        "expected_steps": expected_steps,
        "expected_revision": expected_revision,
        "expected_branch": expected_branch,
        "expected_dataset": expected_dataset,
        "matched_config_sections": list(MATCHED_CONFIG_SECTIONS),
        "pair_contract": pair_contract,
        "training_recipe": recipe_contract,
        "relative_parameter_gap": parameter_gap,
        "max_parameter_gap": max_parameter_gap,
        "cofitok": validated_cofitok,
        "dense": validated_dense,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a completed matched generation training pair."
    )
    parser.add_argument("--cofitok-training", required=True)
    parser.add_argument("--dense-training", required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", default="scale/generative-system")
    parser.add_argument("--expected-dataset", default="imagenet_256_10pct")
    parser.add_argument("--max-parameter-gap", type=float, default=0.02)
    parser.add_argument("--expected-recipe-stage", choices=("scaling", "full"))
    args = parser.parse_args()
    report = validate_training_pair(
        _read(args.cofitok_training),
        _read(args.dense_training),
        expected_steps=args.expected_steps,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
        expected_dataset=args.expected_dataset,
        max_parameter_gap=args.max_parameter_gap,
        expected_recipe_stage=args.expected_recipe_stage,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
