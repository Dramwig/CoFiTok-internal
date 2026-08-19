from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.reporting import file_sha256, write_json_report
from cofitok.generation.quality_bridge import validate_quality_bridge_preparation
from cofitok.training_exposure import (
    TRAINING_EXPOSURE_SCHEMA_VERSION,
    compare_training_exposures,
    training_exposure_summary,
)


REPORT_SCHEMA_VERSION = 1
REPORT_ROLE = "generation_training_exposure_audit"


def _source_identity(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": resolved.as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": file_sha256(resolved),
    }


def parse_training_spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise ValueError("training input must use LABEL=PATH")
    return label.strip(), Path(raw_path.strip())


def build_report(
    training_reports: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    *,
    quality_bridge_preparation: tuple[dict[str, Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not training_reports:
        raise ValueError("at least one training report is required")
    rows: dict[str, Any] = {}
    sources: dict[str, Any] = {}
    for label, (report, identity) in training_reports.items():
        if not label or label in rows:
            raise ValueError("training report labels must be unique and non-empty")
        rows[label] = training_exposure_summary(report)
        sources[label] = dict(identity)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "exposure_schema_version": TRAINING_EXPOSURE_SCHEMA_VERSION,
        "status": "pass",
        "role": REPORT_ROLE,
        "sources": sources,
        "rows": rows,
        "comparison": compare_training_exposures(rows),
        "claim_boundary": {
            "training_scale_claim_allowed": True,
            "sample_quality_claim_allowed": False,
            "method_quality_ranking_allowed": False,
            "formal_gate_substitute": False,
        },
    }
    if quality_bridge_preparation is not None:
        preparation, identity = quality_bridge_preparation
        validated = validate_quality_bridge_preparation(preparation)
        report["quality_bridge_plan"] = {
            "source": dict(identity),
            "training_exposure": validated["training_exposure"],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="repeatable source-bound training report",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quality-bridge-preparation", type=Path)
    args = parser.parse_args()

    reports: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    for raw in args.training:
        label, path = parse_training_spec(raw)
        if label in reports:
            raise ValueError(f"duplicate training label: {label}")
        if not path.is_file():
            raise FileNotFoundError(path)
        reports[label] = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    preparation = None
    if args.quality_bridge_preparation is not None:
        path = args.quality_bridge_preparation
        if not path.is_file():
            raise FileNotFoundError(path)
        preparation = (
            json.loads(path.read_text(encoding="utf-8")),
            _source_identity(path),
        )
    write_json_report(
        args.output,
        build_report(reports, quality_bridge_preparation=preparation),
    )


if __name__ == "__main__":
    main()
