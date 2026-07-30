from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report
from scripts.validate_generation_training_pair import validate_training_pair


EXPECTED_STEPS = 50_000
EXPECTED_EFFECTIVE_BATCH = 64
EXPECTED_IMAGES = EXPECTED_STEPS * EXPECTED_EFFECTIVE_BATCH


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the audited summary for a stability-authorized matched 50K pair."
    )
    parser.add_argument("--cofitok-training", type=Path, required=True)
    parser.add_argument("--dense-training", type=Path, required=True)
    parser.add_argument("--decision-validation", type=Path, required=True)
    parser.add_argument("--config-validation", type=Path, required=True)
    parser.add_argument("--expected-revision", required=True)
    parser.add_argument("--expected-branch", default="scale/generation-stability")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
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


def build_summary(
    *,
    cofitok_training: dict[str, Any],
    dense_training: dict[str, Any],
    decision_validation: dict[str, Any],
    config_validation: dict[str, Any],
    expected_revision: str,
    expected_branch: str,
) -> dict[str, Any]:
    if (
        decision_validation.get("status") != "pass"
        or decision_validation.get("decision")
        != "authorize_fresh_matched_50k_preparation"
        or decision_validation.get("authorized_next_stage")
        != "fresh_matched_50k_preparation"
        or decision_validation.get("robust_seeds") != [2029, 2039]
        or int(decision_validation.get("min_robust_images", 0)) < 64
    ):
        raise ValueError("stability decision validation does not authorize this 50K pair")
    recipe = config_validation.get("training_recipe")
    if (
        config_validation.get("status") != "pass"
        or not isinstance(recipe, dict)
        or recipe.get("stage") != "stability_scaling"
        or recipe.get("valid") is not True
        or recipe.get("schema") != "cofitok_generation_training_recipe_v4"
    ):
        raise ValueError("stability 50K config validation did not pass")

    pair = validate_training_pair(
        cofitok_training,
        dense_training,
        expected_steps=EXPECTED_STEPS,
        expected_revision=expected_revision,
        expected_branch=expected_branch,
        expected_dataset="imagenet_256_10pct",
        max_parameter_gap=0.02,
        expected_recipe_stage="stability_scaling",
    )
    for method, report in (
        ("cofitok", cofitok_training),
        ("dense_identity", dense_training),
    ):
        final_metrics = report.get("final_metrics")
        if (
            not isinstance(final_metrics, dict)
            or int(final_metrics.get("samples_seen", -1)) != EXPECTED_IMAGES
        ):
            raise ValueError(f"{method} did not record exactly {EXPECTED_IMAGES} images")
        config = report.get("config", {})
        effective_batch = int(config.get("data", {}).get("batch_size", 0)) * int(
            config.get("optimization", {}).get(
                "gradient_accumulation_steps",
                0,
            )
        )
        if effective_batch != EXPECTED_EFFECTIVE_BATCH:
            raise ValueError(
                f"{method} effective batch differs from {EXPECTED_EFFECTIVE_BATCH}"
            )

    return {
        "schema_version": 1,
        "status": "completed",
        "stage": "stability_matched_50k",
        "completed_steps_per_method": EXPECTED_STEPS,
        "images_seen_per_method": EXPECTED_IMAGES,
        "effective_batch_size": EXPECTED_EFFECTIVE_BATCH,
        "git": {
            "revision": expected_revision,
            "branch": expected_branch,
        },
        "stability_authorization": decision_validation,
        "config_contract": {
            "schema": recipe["schema"],
            "stage": recipe["stage"],
            "valid": recipe["valid"],
        },
        "training_pair": pair,
        "formal_300k_authorization_allowed": False,
        "formal_ema_sampling_gate_required": True,
    }


def main() -> None:
    args = _parse_args()
    cofitok_training = _read(args.cofitok_training)
    dense_training = _read(args.dense_training)
    decision_validation = _read(args.decision_validation)
    config_validation = _read(args.config_validation)
    report = build_summary(
        cofitok_training=cofitok_training,
        dense_training=dense_training,
        decision_validation=decision_validation,
        config_validation=config_validation,
        expected_revision=args.expected_revision,
        expected_branch=args.expected_branch,
    )
    report["sources"] = {
        "cofitok_training": _source(args.cofitok_training),
        "dense_training": _source(args.dense_training),
        "decision_validation": _source(args.decision_validation),
        "config_validation": _source(args.config_validation),
    }
    write_json_report(args.output, report)
    print(json.dumps({"output": str(args.output), "status": report["status"]}, indent=2))


if __name__ == "__main__":
    main()
