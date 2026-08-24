from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation import build_epsilon_stability_case_observation
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.reporting import write_json_report


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return payload


def _source(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    identity = gate_source_report_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from the bound source")
    return {"identity": identity, "payload": _read(identity["path"])}


def build_from_paths(
    *,
    design_path: str | Path,
    expected_design_sha256: str,
    execution_authorization_path: str | Path,
    expected_execution_authorization_sha256: str,
    case_id: str,
    method: str,
    sampling_report_path: str | Path,
    expected_sampling_report_sha256: str,
    metrics_report_path: str | Path,
    expected_metrics_report_sha256: str,
    class_fidelity_report_path: str | Path,
    expected_class_fidelity_report_sha256: str,
    artifact_report_path: str | Path,
    expected_artifact_report_sha256: str,
) -> dict[str, Any]:
    design = _source(
        design_path,
        expected_design_sha256,
        label="sampling design",
    )
    execution = _source(
        execution_authorization_path,
        expected_execution_authorization_sha256,
        label="execution authorization",
    )
    reports = {
        "sampling_report": _source(
            sampling_report_path,
            expected_sampling_report_sha256,
            label="sampling report",
        ),
        "metrics_report": _source(
            metrics_report_path,
            expected_metrics_report_sha256,
            label="metrics report",
        ),
        "class_fidelity_report": _source(
            class_fidelity_report_path,
            expected_class_fidelity_report_sha256,
            label="class-fidelity report",
        ),
        "artifact_report": _source(
            artifact_report_path,
            expected_artifact_report_sha256,
            label="artifact report",
        ),
    }
    return build_epsilon_stability_case_observation(
        design=design["payload"],
        design_identity=design["identity"],
        execution_authorization=execution["payload"],
        execution_authorization_identity=execution["identity"],
        case_id=case_id,
        method=method,
        report_sources=reports,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build one source-bound epsilon-stability case observation from "
            "native reports."
        )
    )
    parser.add_argument("--design", required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--expected-execution-authorization-sha256", required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--method", choices=("cofitok", "dense_identity"), required=True)
    parser.add_argument("--sampling-report", required=True)
    parser.add_argument("--expected-sampling-report-sha256", required=True)
    parser.add_argument("--metrics-report", required=True)
    parser.add_argument("--expected-metrics-report-sha256", required=True)
    parser.add_argument("--class-fidelity-report", required=True)
    parser.add_argument("--expected-class-fidelity-report-sha256", required=True)
    parser.add_argument("--artifact-report", required=True)
    parser.add_argument("--expected-artifact-report-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_from_paths(
        design_path=args.design,
        expected_design_sha256=args.expected_design_sha256,
        execution_authorization_path=args.execution_authorization,
        expected_execution_authorization_sha256=(
            args.expected_execution_authorization_sha256
        ),
        case_id=args.case_id,
        method=args.method,
        sampling_report_path=args.sampling_report,
        expected_sampling_report_sha256=args.expected_sampling_report_sha256,
        metrics_report_path=args.metrics_report,
        expected_metrics_report_sha256=args.expected_metrics_report_sha256,
        class_fidelity_report_path=args.class_fidelity_report,
        expected_class_fidelity_report_sha256=(
            args.expected_class_fidelity_report_sha256
        ),
        artifact_report_path=args.artifact_report,
        expected_artifact_report_sha256=args.expected_artifact_report_sha256,
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
