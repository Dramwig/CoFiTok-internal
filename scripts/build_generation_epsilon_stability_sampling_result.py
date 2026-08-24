from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from cofitok.generation import (
    EPSILON_STABILITY_OBSERVATION_MANIFEST_SCHEMA,
    EPSILON_STABILITY_REAL_ARTIFACT_REFERENCE_SCHEMA,
    build_epsilon_stability_sampling_result,
)
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.reporting import write_json_report


REPORT_IDENTITY_KEYS = (
    "sampling_report",
    "metrics_report",
    "class_fidelity_report",
    "artifact_report",
)


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return payload


def _verified_identity(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    identity = gate_source_report_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from the bound source")
    return identity


def _bound_source(expected: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(expected, dict) or set(expected) != {
        "path",
        "bytes",
        "sha256",
    }:
        raise ValueError(f"{label} identity is malformed")
    actual = gate_source_report_identity(str(expected["path"]))
    if actual != expected:
        raise ValueError(f"{label} changed after binding")
    return {"identity": actual, "payload": _read(actual["path"])}


def _observation_sources(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema") != EPSILON_STABILITY_OBSERVATION_MANIFEST_SCHEMA:
        raise ValueError("epsilon-stability observation manifest schema mismatch")
    entries = manifest.get("observations")
    if not isinstance(entries, list):
        raise ValueError("epsilon-stability observation manifest is malformed")
    sources = []
    for index, expected in enumerate(entries):
        if not isinstance(expected, dict) or set(expected) != {
            "case_id",
            "method",
            "observation",
            "source_reports",
        }:
            raise ValueError(f"observation manifest row {index} is malformed")
        reports = expected["source_reports"]
        if not isinstance(reports, dict) or set(reports) != set(
            REPORT_IDENTITY_KEYS
        ):
            raise ValueError(
                f"observation manifest row {index} report set is malformed"
            )
        observation = _bound_source(
            expected["observation"],
            label=f"case observation {index}",
        )
        sources.append(
            {
                **observation,
                "reports": {
                    key: _bound_source(
                        reports[key],
                        label=f"case observation {index} {key}",
                    )
                    for key in REPORT_IDENTITY_KEYS
                },
            }
        )
    return sources


def _real_artifact_source(
    *,
    identity: dict[str, Any],
    payload: dict[str, Any],
) -> dict[str, Any]:
    reports = payload.get("source_reports")
    if not isinstance(reports, dict) or set(reports) != {
        "artifact_report",
        "subset_manifest",
    }:
        raise ValueError("real artifact reference source reports are incomplete")
    return {
        "identity": identity,
        "payload": payload,
        "artifact_report": _bound_source(
            reports["artifact_report"],
            label="real artifact report",
        ),
    }


def build_from_paths(
    *,
    design_path: str | Path,
    expected_design_sha256: str,
    execution_authorization_path: str | Path,
    expected_execution_authorization_sha256: str,
    observation_manifest_path: str | Path,
    expected_observation_manifest_sha256: str,
    real_artifact_reference_path: str | Path,
    expected_real_artifact_reference_sha256: str,
) -> dict[str, Any]:
    design_identity = _verified_identity(
        design_path,
        expected_design_sha256,
        label="sampling design",
    )
    execution_identity = _verified_identity(
        execution_authorization_path,
        expected_execution_authorization_sha256,
        label="execution authorization",
    )
    manifest_identity = _verified_identity(
        observation_manifest_path,
        expected_observation_manifest_sha256,
        label="observation manifest",
    )
    real_identity = _verified_identity(
        real_artifact_reference_path,
        expected_real_artifact_reference_sha256,
        label="real artifact reference",
    )
    observation_manifest = _read(observation_manifest_path)
    real_payload = _read(real_artifact_reference_path)
    if real_payload.get("schema") != EPSILON_STABILITY_REAL_ARTIFACT_REFERENCE_SCHEMA:
        raise ValueError("real artifact reference schema mismatch")
    if real_payload.get("status") != "pass":
        raise ValueError("real artifact reference is incomplete")
    return build_epsilon_stability_sampling_result(
        design=_read(design_path),
        design_identity=design_identity,
        execution_authorization=_read(execution_authorization_path),
        execution_authorization_identity=execution_identity,
        observation_sources=_observation_sources(observation_manifest),
        observation_manifest={
            "identity": manifest_identity,
            "payload": observation_manifest,
        },
        real_artifact_reference=_real_artifact_source(
            identity=real_identity,
            payload=real_payload,
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a source-bound, permanently non-authorizing epsilon-stability "
            "1K screening result."
        )
    )
    parser.add_argument("--design", required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--expected-execution-authorization-sha256", required=True)
    parser.add_argument("--observation-manifest", required=True)
    parser.add_argument("--expected-observation-manifest-sha256", required=True)
    parser.add_argument("--real-artifact-reference", required=True)
    parser.add_argument("--expected-real-artifact-reference-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_from_paths(
        design_path=args.design,
        expected_design_sha256=args.expected_design_sha256,
        execution_authorization_path=args.execution_authorization,
        expected_execution_authorization_sha256=(
            args.expected_execution_authorization_sha256
        ),
        observation_manifest_path=args.observation_manifest,
        expected_observation_manifest_sha256=(
            args.expected_observation_manifest_sha256
        ),
        real_artifact_reference_path=args.real_artifact_reference,
        expected_real_artifact_reference_sha256=(
            args.expected_real_artifact_reference_sha256
        ),
    )
    write_json_report(Path(args.output), result)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
