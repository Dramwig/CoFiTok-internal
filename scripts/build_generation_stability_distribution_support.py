from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from cofitok.generation.distribution_support import (
    build_stability_distribution_support_qualification,
)
from cofitok.generation_gate import (
    STABILITY_SCALING_MAX_PRECISION_REGRESSION,
    STABILITY_SCALING_MAX_RECALL_REGRESSION,
    STABILITY_SCALING_MIN_PRECISION,
    STABILITY_SCALING_MIN_RECALL,
)
from cofitok.generation_gate_sources import (
    gate_source_report_identity,
    verify_generation_gate_source_reports,
)
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-authorizing precision/recall qualification from an "
            "immutable stability-scaling gate and its bound 10K metrics."
        )
    )
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--min-precision", type=float, default=STABILITY_SCALING_MIN_PRECISION
    )
    parser.add_argument(
        "--min-recall", type=float, default=STABILITY_SCALING_MIN_RECALL
    )
    parser.add_argument(
        "--max-precision-regression",
        type=float,
        default=STABILITY_SCALING_MAX_PRECISION_REGRESSION,
    )
    parser.add_argument(
        "--max-recall-regression",
        type=float,
        default=STABILITY_SCALING_MAX_RECALL_REGRESSION,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def _read_json_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError(f"JSON report is not an object: {path}")
    return decoded, payload


def _read_bound_json(identity: dict[str, Any]) -> dict[str, Any]:
    path = Path(str(identity["path"]))
    report, payload = _read_json_bytes(path)
    actual = {
        "path": path.resolve().as_posix(),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if actual != identity:
        raise ValueError(f"bound source changed while reading: {path}")
    return report


def build_bound_report(
    gate_path: Path,
    *,
    min_precision: float = STABILITY_SCALING_MIN_PRECISION,
    min_recall: float = STABILITY_SCALING_MIN_RECALL,
    max_precision_regression: float = (
        STABILITY_SCALING_MAX_PRECISION_REGRESSION
    ),
    max_recall_regression: float = STABILITY_SCALING_MAX_RECALL_REGRESSION,
    builder_git: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate, gate_bytes = _read_json_bytes(gate_path)
    gate_identity = {
        "path": gate_path.resolve().as_posix(),
        "bytes": len(gate_bytes),
        "sha256": hashlib.sha256(gate_bytes).hexdigest(),
    }
    verified = verify_generation_gate_source_reports(gate)
    bound = verified["source_reports"]
    cofitok_identity = bound["cofitok_generation"]
    dense_identity = bound["dense_generation"]
    cofitok_generation = _read_bound_json(cofitok_identity)
    dense_generation = _read_bound_json(dense_identity)
    report = build_stability_distribution_support_qualification(
        base_gate=gate,
        cofitok_generation=cofitok_generation,
        dense_generation=dense_generation,
        sources={
            "promotion_gate": gate_identity,
            "cofitok_generation": cofitok_identity,
            "dense_generation": dense_identity,
        },
        builder_git=builder_git or git_provenance(PROJECT_ROOT),
        min_precision=min_precision,
        min_recall=min_recall,
        max_precision_regression=max_precision_regression,
        max_recall_regression=max_recall_regression,
    )
    if gate_source_report_identity(gate_path) != gate_identity:
        raise ValueError("base gate changed while building distribution-support report")
    if verify_generation_gate_source_reports(gate) != verified:
        raise ValueError("bound gate sources changed while building qualification")
    return report


def main() -> int:
    args = parse_args()
    with exclusive_output_lock(
        args.output,
        role="generation_stability_distribution_support_qualification",
    ):
        report = build_bound_report(
            args.gate,
            min_precision=args.min_precision,
            min_recall=args.min_recall,
            max_precision_regression=args.max_precision_regression,
            max_recall_regression=args.max_recall_regression,
        )
        if args.output.exists():
            if not args.resume:
                raise FileExistsError(
                    "distribution-support report already exists; pass --resume to verify it"
                )
            existing, _ = _read_json_bytes(args.output)
            if existing != report:
                raise ValueError(
                    "existing distribution-support report differs from exact replay"
                )
            print(f"reused {args.output}")
        else:
            write_json_report(args.output, report)
            print(f"wrote {args.output}")
    return 2 if args.require_pass and report["status"] != "pass" else 0


if __name__ == "__main__":
    raise SystemExit(main())
