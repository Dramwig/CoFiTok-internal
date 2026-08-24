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
from cofitok.generation import (
    artifact_sample_set_sha256,
    build_epsilon_stability_artifact_report,
    compute_epsilon_stability_artifact_metrics,
    numbered_epsilon_stability_pngs,
)
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.output_lock import exclusive_output_lock
from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return payload


def _verified_source(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = gate_source_report_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from the bound source")
    return identity, _read(identity["path"])


def _real_set_contract(payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload.get("real_set", payload)
    if not isinstance(candidate, dict) or set(candidate) != {
        "digest_schema",
        "sha256",
        "root",
        "image_count",
    }:
        raise ValueError("real-set contract is malformed")
    return candidate


def _generated_source_binding(
    *,
    image_dir: Path,
    sampling_report_path: str,
    expected_sampling_report_sha256: str,
) -> tuple[dict[str, Any], int, int, str]:
    identity, report = _verified_source(
        sampling_report_path,
        expected_sampling_report_sha256,
        label="sampling report",
    )
    sampling = report.get("sampling")
    output_dirs = report.get("output_dirs")
    sample_sets = report.get("sample_sets")
    if (
        report.get("schema_version") != 6
        or report.get("status") != "completed"
        or not isinstance(sampling, dict)
        or not isinstance(output_dirs, dict)
        or not isinstance(sample_sets, dict)
        or len(output_dirs) != 1
        or set(output_dirs) != set(sample_sets)
    ):
        raise ValueError("sampling report contract differs")
    budget_key = next(iter(output_dirs))
    if Path(str(output_dirs[budget_key])).resolve() != image_dir.resolve():
        raise ValueError("artifact image directory differs from sampling report")
    sample_count = int(sampling.get("num_samples", -1))
    start_index = int(sampling.get("start_index", -1))
    if sample_count != 1_000 or start_index != 0:
        raise ValueError("artifact sampling range must be exactly 0..999")
    sample_set = sample_sets[budget_key]
    if (
        not isinstance(sample_set, dict)
        or int(sample_set.get("count", -1)) != sample_count
    ):
        raise ValueError("sampling report sample-set count differs")
    return (
        {
            "sampling_report": identity,
            "generated_dir": image_dir.resolve().as_posix(),
            "sample_set_sha256": str(sample_set.get("sha256", "")),
        },
        sample_count,
        start_index,
        str(sample_set.get("sha256", "")),
    )


def build_from_paths(
    *,
    image_dir: str | Path,
    source_kind: str,
    sampling_report_path: str = "",
    expected_sampling_report_sha256: str = "",
    real_set_contract_path: str = "",
    expected_real_set_contract_sha256: str = "",
    subset_manifest_path: str = "",
    expected_subset_manifest_sha256: str = "",
) -> dict[str, Any]:
    root = Path(image_dir)
    subset_payload: dict[str, Any] = {}
    if source_kind == "generated":
        source_binding, sample_count, start_index, expected_sample_sha256 = (
            _generated_source_binding(
                image_dir=root,
                sampling_report_path=sampling_report_path,
                expected_sampling_report_sha256=(
                    expected_sampling_report_sha256
                ),
            )
        )
    elif source_kind == "real_reference":
        real_identity, real_payload = _verified_source(
            real_set_contract_path,
            expected_real_set_contract_sha256,
            label="real-set contract",
        )
        subset_identity, subset_payload = _verified_source(
            subset_manifest_path,
            expected_subset_manifest_sha256,
            label="real artifact subset manifest",
        )
        sample_count = 1_000
        start_index = 0
        expected_sample_sha256 = ""
        source_binding = {
            "real_set_identity": real_identity,
            "real_set": _real_set_contract(real_payload),
            "subset_manifest_identity": subset_identity,
        }
    else:
        raise ValueError("artifact source kind is unsupported")

    images = numbered_epsilon_stability_pngs(
        root,
        sample_count=sample_count,
        start_index=start_index,
    )
    sample_sha256 = artifact_sample_set_sha256(images)
    if expected_sample_sha256 and sample_sha256 != expected_sample_sha256:
        raise ValueError("physical artifact sample set differs from sampling report")
    if source_kind == "real_reference" and (
        subset_payload.get("schema")
        != "cofitok_epsilon_stability_real_artifact_subset_v1"
        or subset_payload.get("status") != "pass"
        or Path(str(subset_payload.get("image_dir", ""))).resolve()
        != root.resolve()
        or subset_payload.get("sample_count") != 1_000
        or subset_payload.get("start_index") != 0
        or subset_payload.get("sample_set_sha256") != sample_sha256
        or subset_payload.get("real_set_identity") != source_binding[
            "real_set_identity"
        ]
    ):
        raise ValueError("real artifact subset manifest does not replay")
    metrics = compute_epsilon_stability_artifact_metrics(images)
    environment = capture_runtime_environment(
        torch.device("cpu"),
        project_root=PROJECT_ROOT,
    )
    environment_sha256 = runtime_environment_sha256(environment)
    return build_epsilon_stability_artifact_report(
        source_kind=source_kind,
        git=git_provenance(PROJECT_ROOT),
        runtime_environment=environment,
        runtime_environment_sha256_value=environment_sha256,
        sample_count=sample_count,
        sample_set_sha256_value=sample_sha256,
        source_binding=source_binding,
        metrics=metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute source-bound CPU artifact statistics for the matched "
            "epsilon-stability diagnostic."
        )
    )
    parser.add_argument("--image-dir", required=True)
    parser.add_argument(
        "--source-kind",
        choices=("generated", "real_reference"),
        required=True,
    )
    parser.add_argument("--sampling-report", default="")
    parser.add_argument("--expected-sampling-report-sha256", default="")
    parser.add_argument("--real-set-contract", default="")
    parser.add_argument("--expected-real-set-contract-sha256", default="")
    parser.add_argument("--subset-manifest", default="")
    parser.add_argument("--expected-subset-manifest-sha256", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    with exclusive_output_lock(
        output,
        role="generation_epsilon_stability_artifact_statistics",
    ):
        report = build_from_paths(
            image_dir=args.image_dir,
            source_kind=args.source_kind,
            sampling_report_path=args.sampling_report,
            expected_sampling_report_sha256=(
                args.expected_sampling_report_sha256
            ),
            real_set_contract_path=args.real_set_contract,
            expected_real_set_contract_sha256=(
                args.expected_real_set_contract_sha256
            ),
            subset_manifest_path=args.subset_manifest,
            expected_subset_manifest_sha256=(
                args.expected_subset_manifest_sha256
            ),
        )
        if output.is_file():
            if not args.resume:
                raise FileExistsError(
                    "artifact report exists; pass --resume to validate it"
                )
            if _read(output) != report:
                raise ValueError("existing artifact report does not replay")
            print(f"reused {output}")
            return
        write_json_report(output, report)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
