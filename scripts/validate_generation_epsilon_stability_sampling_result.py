from __future__ import annotations

import argparse
import json
from pathlib import Path

from cofitok.reporting import file_sha256
from scripts.build_generation_epsilon_stability_sampling_result import build_from_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay and validate an epsilon-stability 1K screening result."
    )
    parser.add_argument("--result", required=True)
    parser.add_argument("--expected-result-sha256", required=True)
    parser.add_argument("--design", required=True)
    parser.add_argument("--expected-design-sha256", required=True)
    parser.add_argument("--execution-authorization", required=True)
    parser.add_argument("--expected-execution-authorization-sha256", required=True)
    parser.add_argument("--observation-manifest", required=True)
    parser.add_argument("--expected-observation-manifest-sha256", required=True)
    parser.add_argument("--real-artifact-reference", required=True)
    parser.add_argument("--expected-real-artifact-reference-sha256", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_path = Path(args.result)
    if file_sha256(result_path) != args.expected_result_sha256:
        raise ValueError("epsilon-stability result SHA256 differs from evidence")
    with result_path.open("r", encoding="utf-8") as handle:
        actual = json.load(handle)
    expected = build_from_paths(
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
    if actual != expected:
        raise ValueError("epsilon-stability result does not deterministically replay")
    print(json.dumps(actual, sort_keys=True))


if __name__ == "__main__":
    main()
