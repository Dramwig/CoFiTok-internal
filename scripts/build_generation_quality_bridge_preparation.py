from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.configs import config_to_dict, load_config
from cofitok.generation.quality_bridge import build_quality_bridge_preparation
from cofitok.generation_gate_sources import (
    gate_source_report_identity,
    verify_generation_gate_source_reports,
)
from cofitok.reporting import write_json_report
from scripts.validate_generation_configs import validate_pair


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_from_paths(
    *,
    promotion_gate_path: str | Path,
    expected_promotion_gate_sha256: str,
    cofitok_config_path: str | Path,
    dense_config_path: str | Path,
) -> dict[str, Any]:
    promotion_gate_identity = gate_source_report_identity(promotion_gate_path)
    if promotion_gate_identity["sha256"] != expected_promotion_gate_sha256:
        raise ValueError("promotion gate SHA256 differs from the frozen quality hold")
    promotion_gate = _read(promotion_gate_path)
    gate_source_verification = verify_generation_gate_source_reports(promotion_gate)

    cofitok_config = load_config(cofitok_config_path)
    dense_config = load_config(dense_config_path)
    cofitok_dict = config_to_dict(cofitok_config)
    dense_dict = config_to_dict(dense_config)
    config_validation = validate_pair(
        cofitok_config,
        dense_config,
        max_parameter_gap=0.02,
        stage="stability_quality_bridge",
    )

    source_reports = promotion_gate["source_reports"]
    return build_quality_bridge_preparation(
        promotion_gate=promotion_gate,
        promotion_gate_identity=promotion_gate_identity,
        gate_source_verification=gate_source_verification,
        source_cofitok_training=_read(source_reports["cofitok_training"]["path"]),
        source_dense_training=_read(source_reports["dense_training"]["path"]),
        cofitok_config=cofitok_dict,
        dense_config=dense_dict,
        cofitok_config_identity=gate_source_report_identity(cofitok_config_path),
        dense_config_identity=gate_source_report_identity(dense_config_path),
        config_validation=config_validation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the deterministic, non-authorizing full-data 100K quality "
            "bridge preparation report."
        )
    )
    parser.add_argument("--promotion-gate", required=True)
    parser.add_argument("--expected-promotion-gate-sha256", required=True)
    parser.add_argument("--cofitok-config", required=True)
    parser.add_argument("--dense-config", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_from_paths(
        promotion_gate_path=args.promotion_gate,
        expected_promotion_gate_sha256=args.expected_promotion_gate_sha256,
        cofitok_config_path=args.cofitok_config,
        dense_config_path=args.dense_config,
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
