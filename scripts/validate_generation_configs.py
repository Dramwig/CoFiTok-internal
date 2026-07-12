from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from cofitok.configs import ExperimentConfig, load_config
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
    return parser.parse_args()


def _parameter_count(config: ExperimentConfig) -> int:
    model = CoFiTokTiny(config.model)
    return sum(parameter.numel() for parameter in model.parameters())


def validate_pair(
    cofitok: ExperimentConfig,
    dense: ExperimentConfig,
    max_parameter_gap: float,
) -> dict[str, object]:
    mismatches = []
    for section in ("data", "diffusion", "runtime", "optimization"):
        if getattr(cofitok, section) != getattr(dense, section):
            mismatches.append(section)
    for field in BACKBONE_FIELDS:
        if getattr(cofitok.model, field) != getattr(dense.model, field):
            mismatches.append(f"model.{field}")
    if cofitok.model.synthesis_mode != "restricted":
        mismatches.append("cofitok synthesis is not restricted")
    if dense.model.synthesis_mode != "dense_identity":
        mismatches.append("dense synthesis is not dense_identity")
    if not cofitok.data.class_conditional or cofitok.model.num_classes <= 0:
        mismatches.append("class conditioning is disabled")

    cofitok_parameters = _parameter_count(cofitok)
    dense_parameters = _parameter_count(dense)
    relative_gap = (cofitok_parameters - dense_parameters) / dense_parameters
    if abs(relative_gap) > max_parameter_gap:
        mismatches.append("parameter_gap")
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
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
