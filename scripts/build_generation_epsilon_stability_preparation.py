from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from cofitok.generation import build_epsilon_stability_preparation
from cofitok.generation_gate_sources import gate_source_report_identity
from cofitok.reporting import git_provenance, write_json_report


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_NAMES = (
    "design",
    "decision",
    "decision-verification",
    "reconciliation",
    "quality-bridge-result",
    "training-pair-report",
    "cofitok-checkpoint",
    "cofitok-sidecar",
    "cofitok-latest",
    "cofitok-training-report",
    "dense-checkpoint",
    "dense-sidecar",
    "dense-latest",
    "dense-training-report",
    "dataset-manifest",
    "real-set-source",
    "runtime-binding",
    "evaluator-manifest",
    "classifier-weights",
    "classifier-report",
)


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON source is not an object: {path}")
    return payload


def _identity(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    identity = gate_source_report_identity(path)
    if identity["sha256"] != expected_sha256:
        raise ValueError(f"{label} SHA256 differs from the bound source")
    return identity


def _source(
    path: str | Path,
    expected_sha256: str,
    *,
    label: str,
) -> dict[str, Any]:
    identity = _identity(path, expected_sha256, label=label)
    return {"identity": identity, "payload": _read(identity["path"])}


def _repair_git() -> dict[str, Any]:
    git = git_provenance(PROJECT_ROOT)
    if git["tracked_dirty"]:
        raise ValueError("epsilon-stability preparation checkout is tracked-dirty")
    tree = subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD^{tree}"],
        text=True,
    ).strip()
    return {**git, "tree": tree}


def build_from_paths(
    *,
    design_path: str | Path,
    expected_design_sha256: str,
    decision_path: str | Path,
    expected_decision_sha256: str,
    decision_verification_path: str | Path,
    expected_decision_verification_sha256: str,
    reconciliation_path: str | Path,
    expected_reconciliation_sha256: str,
    quality_bridge_result_path: str | Path,
    expected_quality_bridge_result_sha256: str,
    training_pair_report_path: str | Path,
    expected_training_pair_report_sha256: str,
    cofitok_checkpoint_path: str | Path,
    expected_cofitok_checkpoint_sha256: str,
    cofitok_sidecar_path: str | Path,
    expected_cofitok_sidecar_sha256: str,
    cofitok_latest_path: str | Path,
    expected_cofitok_latest_sha256: str,
    cofitok_training_report_path: str | Path,
    expected_cofitok_training_report_sha256: str,
    dense_checkpoint_path: str | Path,
    expected_dense_checkpoint_sha256: str,
    dense_sidecar_path: str | Path,
    expected_dense_sidecar_sha256: str,
    dense_latest_path: str | Path,
    expected_dense_latest_sha256: str,
    dense_training_report_path: str | Path,
    expected_dense_training_report_sha256: str,
    dataset_manifest_path: str | Path,
    expected_dataset_manifest_sha256: str,
    real_set_source_path: str | Path,
    expected_real_set_source_sha256: str,
    runtime_binding_path: str | Path,
    expected_runtime_binding_sha256: str,
    evaluator_manifest_path: str | Path,
    expected_evaluator_manifest_sha256: str,
    classifier_weights_path: str | Path,
    expected_classifier_weights_sha256: str,
    classifier_report_path: str | Path,
    expected_classifier_report_sha256: str,
    seed: int,
    random_stream_namespace: str,
    output_root: str,
    require_output_absent: bool = True,
) -> dict[str, Any]:
    output_path = Path(output_root)
    if require_output_absent and (
        output_path.exists() or Path(f"{output_root}.lock").exists()
    ):
        raise FileExistsError("epsilon-stability output root or lock already exists")
    return build_epsilon_stability_preparation(
        design_source=_source(
            design_path, expected_design_sha256, label="sampling design"
        ),
        decision_source=_source(
            decision_path,
            expected_decision_sha256,
            label="post-reconciliation decision",
        ),
        decision_verification_source=_source(
            decision_verification_path,
            expected_decision_verification_sha256,
            label="post-reconciliation decision verification",
        ),
        reconciliation_source=_source(
            reconciliation_path,
            expected_reconciliation_sha256,
            label="cross-protocol reconciliation",
        ),
        quality_bridge_result_source=_source(
            quality_bridge_result_path,
            expected_quality_bridge_result_sha256,
            label="quality bridge result",
        ),
        training_pair_report_source=_source(
            training_pair_report_path,
            expected_training_pair_report_sha256,
            label="training pair report",
        ),
        method_sources={
            "cofitok": {
                "checkpoint": _identity(
                    cofitok_checkpoint_path,
                    expected_cofitok_checkpoint_sha256,
                    label="CoFiTok checkpoint",
                ),
                "integrity_sidecar": _source(
                    cofitok_sidecar_path,
                    expected_cofitok_sidecar_sha256,
                    label="CoFiTok checkpoint sidecar",
                ),
                "latest": _source(
                    cofitok_latest_path,
                    expected_cofitok_latest_sha256,
                    label="CoFiTok latest",
                ),
                "training_report": _source(
                    cofitok_training_report_path,
                    expected_cofitok_training_report_sha256,
                    label="CoFiTok training report",
                ),
            },
            "dense_identity": {
                "checkpoint": _identity(
                    dense_checkpoint_path,
                    expected_dense_checkpoint_sha256,
                    label="dense checkpoint",
                ),
                "integrity_sidecar": _source(
                    dense_sidecar_path,
                    expected_dense_sidecar_sha256,
                    label="dense checkpoint sidecar",
                ),
                "latest": _source(
                    dense_latest_path,
                    expected_dense_latest_sha256,
                    label="dense latest",
                ),
                "training_report": _source(
                    dense_training_report_path,
                    expected_dense_training_report_sha256,
                    label="dense training report",
                ),
            },
        },
        dataset_identity=_identity(
            dataset_manifest_path,
            expected_dataset_manifest_sha256,
            label="dataset manifest",
        ),
        real_set_source=_source(
            real_set_source_path,
            expected_real_set_source_sha256,
            label="real set source",
        ),
        runtime_binding_source=_source(
            runtime_binding_path,
            expected_runtime_binding_sha256,
            label="runtime binding",
        ),
        evaluator_source=_source(
            evaluator_manifest_path,
            expected_evaluator_manifest_sha256,
            label="evaluator manifest",
        ),
        classifier_identity=_identity(
            classifier_weights_path,
            expected_classifier_weights_sha256,
            label="classifier weights",
        ),
        classifier_report_source=_source(
            classifier_report_path,
            expected_classifier_report_sha256,
            label="classifier report",
        ),
        repair_git=_repair_git(),
        seed=seed,
        random_stream_namespace=random_stream_namespace,
        output_root=output_root,
    )


def add_source_args(parser: argparse.ArgumentParser) -> None:
    for name in SOURCE_NAMES:
        destination = name.replace("-", "_")
        parser.add_argument(
            f"--{name}",
            dest=f"{destination}_path",
            required=True,
        )
        parser.add_argument(
            f"--expected-{name}-sha256",
            dest=f"expected_{destination}_sha256",
            required=True,
        )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--random-stream-namespace", required=True)
    parser.add_argument("--output-root", required=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the source-bound, non-authorizing epsilon-stability "
            "preparation."
        )
    )
    add_source_args(parser)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_from_paths(
        **{
            key: value
            for key, value in vars(args).items()
            if key != "output"
        },
        require_output_absent=True,
    )
    write_json_report(Path(args.output), report)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
