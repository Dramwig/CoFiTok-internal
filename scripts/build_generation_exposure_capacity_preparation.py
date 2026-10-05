from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation.exposure_capacity import build_preparation
from cofitok.path_security import reject_symlink_chain
from cofitok.reporting import file_sha256, git_provenance, write_json_report


PROJECT_ROOT = reject_symlink_chain(
    Path(__file__),
    name="exposure/capacity preparation script",
).parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build non-authorizing exposure/capacity disambiguation preparation."
    )
    for name in (
        "objective_reassessment",
        "quality_bridge_result",
        "post_reconciliation_decision",
        "cross_protocol_reconciliation",
        "sampling_recovery_result",
        "min_snr_result",
        "cofitok_training_report",
        "dense_training_report",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--pair-monitor", type=Path, required=True)
    parser.add_argument("--cofitok-metrics", type=Path, required=True)
    parser.add_argument("--dense-metrics", type=Path, required=True)
    parser.add_argument("--exposure-output-root", required=True)
    parser.add_argument("--capacity-output-root", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="preparation source")
    if not source.is_file():
        raise FileNotFoundError(f"preparation source is missing: {source}")
    value = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON source is not an object: {source}")
    return value


def _read_metrics_tail(path: Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="metrics source")
    if not source.is_file():
        raise FileNotFoundError(f"metrics source is missing: {source}")
    lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line]
    if not lines:
        raise ValueError(f"metrics source is empty: {source}")
    value = json.loads(lines[-1])
    if not isinstance(value, dict):
        raise ValueError(f"metrics tail is not an object: {source}")
    return value


def _identity(path: Path) -> dict[str, Any]:
    source = reject_symlink_chain(path, name="preparation source")
    if not source.is_file():
        raise FileNotFoundError(f"preparation source is missing: {source}")
    return {
        "path": str(source),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _builder_git() -> dict[str, Any]:
    provenance = git_provenance(PROJECT_ROOT)
    return {
        **provenance,
        "tree": subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
    }


def main() -> None:
    args = _parse_args()
    output = reject_symlink_chain(args.output, name="preparation output")
    paths = {
        "objective_reassessment": args.objective_reassessment,
        "quality_bridge_result": args.quality_bridge_result,
        "post_reconciliation_decision": args.post_reconciliation_decision,
        "cross_protocol_reconciliation": args.cross_protocol_reconciliation,
        "sampling_recovery_result": args.sampling_recovery_result,
        "min_snr_result": args.min_snr_result,
        "pair_monitor": args.pair_monitor,
        "cofitok_training_report": args.cofitok_training_report,
        "dense_training_report": args.dense_training_report,
        "cofitok_metrics": args.cofitok_metrics,
        "dense_metrics": args.dense_metrics,
    }
    if not paths["post_reconciliation_decision"].exists():
        raise FileNotFoundError("post-reconciliation decision source does not exist")
    report = build_preparation(
        objective_reassessment=_read(args.objective_reassessment),
        quality_bridge_result=_read(args.quality_bridge_result),
        post_reconciliation_decision=_read(args.post_reconciliation_decision),
        cross_protocol_reconciliation=_read(args.cross_protocol_reconciliation),
        sampling_recovery_result=_read(args.sampling_recovery_result),
        min_snr_result=_read(args.min_snr_result),
        pair_monitor=_read(args.pair_monitor),
        cofitok_metrics=_read_metrics_tail(args.cofitok_metrics),
        dense_metrics=_read_metrics_tail(args.dense_metrics),
        cofitok_training_report=_read(args.cofitok_training_report),
        dense_training_report=_read(args.dense_training_report),
        source_identities={name: _identity(path) for name, path in paths.items()},
        builder_git=_builder_git(),
        exposure_output_root=args.exposure_output_root,
        capacity_output_root=args.capacity_output_root,
    )
    write_json_report(output, report)
    print(file_sha256(output))


if __name__ == "__main__":
    main()
