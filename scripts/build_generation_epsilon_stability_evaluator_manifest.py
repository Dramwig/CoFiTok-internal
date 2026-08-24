from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src/cofitok/generation/quality_repair.py",
    "src/cofitok/generation/quality_repair_artifact.py",
    "src/cofitok/generation/quality_repair_gate.py",
    "src/cofitok/generation/quality_repair_result.py",
    "scripts/generate_samples.py",
    "scripts/evaluate_generation_metrics.py",
    "scripts/evaluate_generation_class_fidelity.py",
    "scripts/evaluate_generation_epsilon_stability_artifacts.py",
    "scripts/prepare_generation_epsilon_stability_real_artifact_subset.py",
    "scripts/build_generation_epsilon_stability_case_observation.py",
    "scripts/build_generation_epsilon_stability_evaluator_manifest.py",
    "scripts/build_generation_epsilon_stability_execution_authorization.py",
    "scripts/build_generation_epsilon_stability_preparation.py",
    "scripts/build_generation_epsilon_stability_runtime_binding.py",
    "scripts/build_generation_epsilon_stability_sampling_result.py",
    "scripts/build_generation_epsilon_stability_user_authorization.py",
    "scripts/validate_generation_epsilon_stability_execution_authorization.py",
    "scripts/validate_generation_epsilon_stability_preparation.py",
    "scripts/validate_generation_epsilon_stability_sampling_result.py",
    "scripts/run_generation_epsilon_stability_sampling_recovery.py",
    "artifacts/runbooks/generation_epsilon_stability_sampling_recovery_v1.sh",
)


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), *args],
        text=True,
    ).strip()


def build_manifest() -> dict[str, Any]:
    git = git_provenance(PROJECT_ROOT)
    if git["tracked_dirty"]:
        raise ValueError("epsilon-stability evaluator checkout is tracked-dirty")
    return {
        "schema": "cofitok_matched_epsilon_stability_evaluator_manifest_v1",
        "status": "pass",
        "git": {
            **git,
            "tree": _git("rev-parse", "HEAD^{tree}"),
        },
        "sources": {
            path: gate_source_report_identity(PROJECT_ROOT / path)
            for path in SOURCE_PATHS
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the epsilon-stability evaluator source manifest."
    )
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_json_report(Path(args.output), build_manifest())
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
