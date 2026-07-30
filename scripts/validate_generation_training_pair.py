from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from cofitok.data.provenance import validate_dataset_provenance
from cofitok.generation_pair import MATCHED_CONFIG_SECTIONS, generation_pair_contract
from cofitok.generation_recipe import generation_training_recipe_contract
from cofitok.training.authorization import validate_generation_training_authorization


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
    allow_legacy_missing_dataset_provenance: bool,
    expected_authorization_gate: dict[str, Any] | None,
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
    dataset_provenance = report.get("dataset_provenance")
    provenance_evidence = None
    legacy_warning = None
    if dataset_provenance is None:
        if not allow_legacy_missing_dataset_provenance:
            raise ValueError(f"{label} training lacks formal dataset provenance")
        legacy_warning = "pinned legacy training report lacks embedded dataset provenance"
    else:
        if not isinstance(dataset_provenance, dict):
            raise ValueError(f"{label} dataset provenance is malformed")
        provenance_evidence = validate_dataset_provenance(
            dataset_provenance,
            expected_dataset=expected_dataset,
        )
        if latest.get("dataset_identity_sha256") != provenance_evidence[
            "identity_sha256"
        ]:
            raise ValueError(
                f"{label} checkpoint pointer lacks the training dataset identity"
            )
    training_authorization = report.get("training_authorization")
    authorization_evidence = None
    if expected_dataset == "imagenet_256" and not isinstance(
        training_authorization, dict
    ):
        raise ValueError(f"{label} full training lacks scaling-gate authorization")
    if training_authorization is not None:
        if not isinstance(training_authorization, dict):
            raise ValueError(f"{label} training authorization is malformed")
        authorization_evidence = validate_generation_training_authorization(
            training_authorization,
            expected_gate=expected_authorization_gate,
        )
        latest_authorization = {
            "stage": latest.get("authorization_stage"),
            "decision": latest.get("authorization_decision"),
            "gate_bytes": latest.get("authorization_gate_bytes"),
            "gate_sha256": latest.get("authorization_gate_sha256"),
            "gate_identity_sha256": latest.get(
                "authorization_gate_identity_sha256"
            ),
        }
        expected_latest_authorization = {
            key: authorization_evidence[key]
            for key in (
                "stage",
                "decision",
                "gate_bytes",
                "gate_sha256",
                "gate_identity_sha256",
            )
        }
        if latest_authorization != expected_latest_authorization:
            raise ValueError(
                f"{label} checkpoint pointer lacks the scaling-gate authorization binding"
            )
    return {
        "completed_steps": completed_steps,
        "revision": git["revision"],
        "branch": git["branch"],
        "dataset": expected_dataset,
        "parameter_count": parameter_count,
        "latest_checkpoint": expected_checkpoint,
        "dataset_provenance": provenance_evidence,
        "dataset_provenance_warning": legacy_warning,
        "training_authorization": authorization_evidence,
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
    allow_legacy_missing_dataset_provenance: bool = False,
    expected_authorization_gate: dict[str, Any] | None = None,
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
        allow_legacy_missing_dataset_provenance=allow_legacy_missing_dataset_provenance,
        expected_authorization_gate=expected_authorization_gate,
    )
    validated_dense = _validate_report(
        dense,
        label="dense",
        expected_steps=expected_steps,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_dataset=expected_dataset,
        allow_legacy_missing_dataset_provenance=allow_legacy_missing_dataset_provenance,
        expected_authorization_gate=expected_authorization_gate,
    )
    provenance_states = {
        validated_cofitok["dataset_provenance"] is None,
        validated_dense["dataset_provenance"] is None,
    }
    if len(provenance_states) != 1:
        raise ValueError("matched training pair mixes legacy and bound dataset provenance")
    if validated_cofitok["dataset_provenance"] is not None and (
        validated_cofitok["dataset_provenance"]["identity_sha256"]
        != validated_dense["dataset_provenance"]["identity_sha256"]
    ):
        raise ValueError("matched training pair used different dataset identities")
    authorization_states = {
        validated_cofitok["training_authorization"] is None,
        validated_dense["training_authorization"] is None,
    }
    if len(authorization_states) != 1:
        raise ValueError("matched training pair mixes bound and unbound authorization")
    if (
        validated_cofitok["training_authorization"] is not None
        and validated_cofitok["training_authorization"]
        != validated_dense["training_authorization"]
    ):
        raise ValueError("matched training pair used different promotion authorizations")
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
        "schema_version": 3,
        "status": "pass",
        "expected_steps": expected_steps,
        "expected_revision": expected_revision,
        "expected_branch": expected_branch,
        "expected_dataset": expected_dataset,
        "matched_config_sections": list(MATCHED_CONFIG_SECTIONS),
        "pair_contract": pair_contract,
        "training_recipe": recipe_contract,
        "legacy_dataset_provenance_allowed": allow_legacy_missing_dataset_provenance,
        "dataset_identity_sha256": (
            None
            if validated_cofitok["dataset_provenance"] is None
            else validated_cofitok["dataset_provenance"]["identity_sha256"]
        ),
        "authorization_gate_identity_sha256": (
            None
            if validated_cofitok["training_authorization"] is None
            else validated_cofitok["training_authorization"][
                "gate_identity_sha256"
            ]
        ),
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
    parser.add_argument(
        "--expected-recipe-stage",
        choices=(
            "legacy_scaling",
            "scaling",
            "full",
            "stability_scaling",
            "stability_full",
        ),
    )
    parser.add_argument(
        "--allow-legacy-missing-dataset-provenance",
        action="store_true",
        help="Allow both pinned legacy reports to omit dataset provenance.",
    )
    parser.add_argument(
        "--authorization-gate",
        default="",
        help="Scaling promotion gate that must authorize both full training reports.",
    )
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
        allow_legacy_missing_dataset_provenance=(
            args.allow_legacy_missing_dataset_provenance
        ),
        expected_authorization_gate=(
            _read(args.authorization_gate) if args.authorization_gate else None
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
