from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from cofitok.configs import ExperimentConfig, load_config
from cofitok.generation_recipe import (
    generation_training_recipe_contract,
    infer_generation_training_stage,
)
from cofitok.models import CoFiTokTiny
from cofitok.reporting import write_json_report


BACKBONE_FIELDS = (
    "image_channels",
    "image_size",
    "base_channels",
    "predictor_type",
    "predictor_channel_multipliers",
    "predictor_num_res_blocks",
    "predictor_attention_resolutions",
    "predictor_num_heads",
    "predictor_dropout",
    "predictor_gradient_checkpointing",
    "num_classes",
    "class_dropout_prob",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a matched CoFiTok/dense generation pair.")
    parser.add_argument("--cofitok-config", required=True)
    parser.add_argument("--dense-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-parameter-gap", type=float, default=0.02)
    parser.add_argument("--stage", choices=("scaling", "full"))
    return parser.parse_args()


def _parameter_count(config: ExperimentConfig) -> int:
    model = CoFiTokTiny(config.model)
    return sum(parameter.numel() for parameter in model.parameters())


def validate_pair(
    cofitok: ExperimentConfig,
    dense: ExperimentConfig,
    max_parameter_gap: float,
    stage: str | None = None,
) -> dict[str, object]:
    mismatches = []
    for section in ("data", "diffusion", "runtime", "optimization"):
        if getattr(cofitok, section) != getattr(dense, section):
            mismatches.append(section)
    for field in BACKBONE_FIELDS:
        if getattr(cofitok.model, field) != getattr(dense.model, field):
            mismatches.append(f"model.{field}")
    if cofitok.model.synthesis_mode not in {"restricted", "fixed_basis"}:
        mismatches.append("cofitok synthesis is not a restricted token-only operator")
    if dense.model.synthesis_mode != "dense_identity":
        mismatches.append("dense synthesis is not dense_identity")
    if not cofitok.data.class_conditional or cofitok.model.num_classes <= 0:
        mismatches.append("class conditioning is disabled")

    cofitok_parameters = _parameter_count(cofitok)
    dense_parameters = _parameter_count(dense)
    relative_gap = (cofitok_parameters - dense_parameters) / dense_parameters
    if abs(relative_gap) > max_parameter_gap:
        mismatches.append("parameter_gap")
    cofitok_config = asdict(cofitok)
    dense_config = asdict(dense)
    resolved_stage = stage or infer_generation_training_stage(
        cofitok_config,
        dense_config,
    )
    recipe = generation_training_recipe_contract(
        cofitok_config,
        dense_config,
        stage=resolved_stage,
    )
    if recipe["valid"] is not True:
        mismatches.append("training_recipe")
    return {
        "status": "pass" if not mismatches else "fail",
        "mismatches": mismatches,
        "cofitok": {
            "name": cofitok.name,
            "parameter_count": cofitok_parameters,
            "synthesis_mode": cofitok.model.synthesis_mode,
            "token_count": cofitok.model.token_count,
        },
        "dense": {
            "name": dense.name,
            "parameter_count": dense_parameters,
            "synthesis_mode": dense.model.synthesis_mode,
            "token_count": dense.model.token_count,
        },
        "relative_parameter_gap": relative_gap,
        "max_parameter_gap": max_parameter_gap,
        "training_recipe": recipe,
        "matched_backbone": {field: getattr(cofitok.model, field) for field in BACKBONE_FIELDS},
        "matched_data": asdict(cofitok.data),
        "matched_diffusion": asdict(cofitok.diffusion),
        "matched_runtime": asdict(cofitok.runtime),
        "matched_optimization": asdict(cofitok.optimization),
    }


def main() -> None:
    args = parse_args()
    report = validate_pair(
        load_config(args.cofitok_config),
        load_config(args.dense_config),
        args.max_parameter_gap,
        stage=args.stage,
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
