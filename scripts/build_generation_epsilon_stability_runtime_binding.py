from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from cofitok.environment import (
    capture_runtime_environment,
    runtime_environment_sha256,
)
from cofitok.generation import EPSILON_STABILITY_RUNTIME_BINDING_SCHEMA
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.reporting import write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_verified(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    identity = gate_source_report_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from the bound source")
    with Path(identity["path"]).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} is not a JSON object")
    environment = payload.get("runtime_environment")
    if not isinstance(environment, dict):
        raise ValueError(f"{label} runtime environment is missing")
    actual = runtime_environment_sha256(environment)
    if payload.get("runtime_environment_sha256") != actual:
        raise ValueError(f"{label} runtime environment does not replay")
    return environment


def build_from_paths(
    *,
    sampling_report_path: str | Path,
    expected_sampling_report_sha256: str,
    metrics_report_path: str | Path,
    expected_metrics_report_sha256: str,
    class_fidelity_report_path: str | Path,
    expected_class_fidelity_report_sha256: str,
) -> dict[str, Any]:
    sampling = _read_verified(
        sampling_report_path,
        expected_sampling_report_sha256,
        label="sampling runtime source",
    )
    metrics = _read_verified(
        metrics_report_path,
        expected_metrics_report_sha256,
        label="metrics runtime source",
    )
    class_fidelity = _read_verified(
        class_fidelity_report_path,
        expected_class_fidelity_report_sha256,
        label="class-fidelity runtime source",
    )
    artifact = capture_runtime_environment(
        torch.device("cpu"),
        project_root=PROJECT_ROOT,
    )
    environments = {
        "sampling": sampling,
        "metrics": metrics,
        "class_fidelity": class_fidelity,
        "artifact": artifact,
    }
    return {
        "schema": EPSILON_STABILITY_RUNTIME_BINDING_SCHEMA,
        "status": "pass",
        "runtime_environments": environments,
        "runtime_environment_sha256s": {
            role: runtime_environment_sha256(environment)
            for role, environment in environments.items()
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the source-replayed epsilon-stability runtime binding."
    )
    parser.add_argument("--sampling-report", required=True)
    parser.add_argument("--expected-sampling-report-sha256", required=True)
    parser.add_argument("--metrics-report", required=True)
    parser.add_argument("--expected-metrics-report-sha256", required=True)
    parser.add_argument("--class-fidelity-report", required=True)
    parser.add_argument("--expected-class-fidelity-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_from_paths(
        sampling_report_path=args.sampling_report,
        expected_sampling_report_sha256=args.expected_sampling_report_sha256,
        metrics_report_path=args.metrics_report,
        expected_metrics_report_sha256=args.expected_metrics_report_sha256,
        class_fidelity_report_path=args.class_fidelity_report,
        expected_class_fidelity_report_sha256=(
            args.expected_class_fidelity_report_sha256
        ),
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
